#!/usr/bin/env python3
"""A retained prefix cache as a file: the state a decode snapshot resumes from.

An image lists prompts and, oldest first, the cache entries that hold their
complete blocks: which prompt, which block of it, and the block id of every
layer. Installing it into a fresh placement (``install_prefix_image``)
reproduces the pool layout, holes included, and the cache's recency order.
Entry identity is recomputed from the prompt tokens at installation with the
placement's own key function, so an image never carries a key the resumed run
could not have produced itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from serveloop.contracts import RequestSpec, ServeLoopError, canonical_sha256
from serveloop.io import load_json_object, write_json_atomic
from serveloop.placement import ServeLoopPlacement


PREFIX_IMAGE_SCHEMA = {"name": "serveloop.prefix_image", "version": 1}


@dataclass(frozen=True)
class ImagePrompt:
    request_id: str
    model_id: str
    cache_salt: str
    token_ids: tuple[int, ...]

    def request(self) -> RequestSpec:
        """The prompt as a request, which is all a block key depends on."""
        return RequestSpec(
            request_id=self.request_id, arrival_ns=0.0, model_id=self.model_id,
            prompt_tokens=len(self.token_ids), output_tokens=1,
            token_ids=self.token_ids, cache_salt=self.cache_salt)

    def canonical(self) -> dict[str, Any]:
        return {"request_id": self.request_id, "model_id": self.model_id,
                "cache_salt": self.cache_salt, "token_ids": list(self.token_ids)}


@dataclass(frozen=True)
class ImageEntry:
    prompt: int
    block: int
    blocks: tuple[int, ...]  # one block per layer
    expires_ns: float | None  # from the resumed run's start

    def canonical(self) -> dict[str, Any]:
        return {"prompt": self.prompt, "block": self.block,
                "blocks": list(self.blocks), "expires_ns": self.expires_ns}


@dataclass(frozen=True)
class PrefixImage:
    source: Mapping[str, Any]
    kv_block_tokens: int
    prompts: tuple[ImagePrompt, ...]
    entries: tuple[ImageEntry, ...]

    def __post_init__(self) -> None:
        if isinstance(self.kv_block_tokens, bool) or not isinstance(self.kv_block_tokens, int) \
                or self.kv_block_tokens <= 0:
            raise ServeLoopError("a prefix image needs a positive KV block size")
        if len({prompt.request_id for prompt in self.prompts}) != len(self.prompts):
            raise ServeLoopError("prefix image prompts repeat a request id")
        for entry in self.entries:
            if not 0 <= entry.prompt < len(self.prompts):
                raise ServeLoopError("prefix image entry names a missing prompt")
            prompt = self.prompts[entry.prompt]
            if not 0 <= entry.block < len(prompt.token_ids) // self.kv_block_tokens:
                raise ServeLoopError("prefix image entry names a block the prompt does not complete")
            if not entry.blocks or any(isinstance(b, bool) or not isinstance(b, int) or b < 0 for b in entry.blocks):
                raise ServeLoopError("prefix image entry needs one nonnegative block id per layer")
            if entry.expires_ns is not None and (not math.isfinite(entry.expires_ns) or entry.expires_ns <= 0):
                raise ServeLoopError("prefix image expiry must be a positive time or null")
        if len({(entry.prompt, entry.block) for entry in self.entries}) != len(self.entries):
            raise ServeLoopError("prefix image repeats a prompt block")

    @property
    def blocks(self) -> int:
        """KV blocks the image occupies, over every layer."""
        return sum(len(entry.blocks) for entry in self.entries)

    def canonical(self) -> dict[str, Any]:
        return {
            "schema": PREFIX_IMAGE_SCHEMA,
            "source": dict(self.source),
            "kv_block_tokens": self.kv_block_tokens,
            "prompts": [prompt.canonical() for prompt in self.prompts],
            "entries": [entry.canonical() for entry in self.entries],
        }

    @cached_property
    def digest(self) -> str:
        return canonical_sha256(self.canonical())


def load_prefix_image(path: Path) -> PrefixImage:
    value = load_json_object(path, "prefix image")
    if value.get("schema") != PREFIX_IMAGE_SCHEMA or set(value) != {"schema", "source", "kv_block_tokens", "prompts", "entries"}:
        raise ServeLoopError(f"{path} is not a {PREFIX_IMAGE_SCHEMA['name']} v{PREFIX_IMAGE_SCHEMA['version']} file")
    if not isinstance(value["source"], Mapping) or not isinstance(value["prompts"], list) \
            or not isinstance(value["entries"], list):
        raise ServeLoopError("prefix image fields have the wrong shape")
    try:
        prompts = tuple(
            ImagePrompt(request_id=str(row["request_id"]), model_id=str(row["model_id"]),
                        cache_salt=str(row["cache_salt"]), token_ids=tuple(int(t) for t in row["token_ids"]))
            for row in value["prompts"])
        entries = tuple(
            ImageEntry(prompt=int(row["prompt"]), block=int(row["block"]),
                       blocks=tuple(int(b) for b in row["blocks"]),
                       expires_ns=None if row["expires_ns"] is None else float(row["expires_ns"]))
            for row in value["entries"])
    except (KeyError, TypeError, ValueError) as error:
        raise ServeLoopError(f"malformed prefix image: {error}") from error
    return PrefixImage(source=value["source"], kv_block_tokens=int(value["kv_block_tokens"]),
                       prompts=prompts, entries=entries)


def write_prefix_image(path: Path, image: PrefixImage) -> None:
    write_json_atomic(path, image.canonical())


def install_prefix_image(placement: ServeLoopPlacement, image: PrefixImage) -> dict[str, Any]:
    """Retain the image in ``placement``'s prefix cache and return what was installed."""
    if image.kv_block_tokens != placement.spec.kv_block_tokens:
        raise ServeLoopError(
            f"the prefix image holds {image.kv_block_tokens}-token blocks but the placement "
            f"uses {placement.spec.kv_block_tokens}-token blocks")
    keys = [placement.prefix_keys_for(prompt.request()) for prompt in image.prompts]
    entries = [(keys[entry.prompt][entry.block], entry.blocks, entry.expires_ns) for entry in image.entries]
    if len({key for key, _, _ in entries}) != len(entries):
        raise ServeLoopError("prefix image entries resolve to a repeated block key")
    placement.install_prefix_image(entries)
    return {
        "sha256": image.digest,
        "prompts": len(image.prompts),
        "entries": len(image.entries),
        "blocks": image.blocks,
        "bytes": image.blocks * placement.kv_block_bytes,
        "source": dict(image.source),
    }
