"""Bounded, read-only object chunks cached from HBF in a fixed HBM partition."""

from collections import Counter, OrderedDict

from hbfsim_client.transaction_protocol import hbf_link_bytes_by_stack
from serveloop.contracts import ServeLoopError, canonical_sha256


class HbfWeightCache:
    """LRU admission in logical access order; physical slot reuse waits for readers.

    ServeLoop executes complete batches serially. Residency persists across batches,
    while their completed read/install fences can be discarded at the next batch.
    Each line belongs to one immutable object; a partial final line is never read
    past the object's end and still occupies one full cache slot.
    """

    def __init__(self, *, begin, capacity, chunk_bytes, geometry):
        if capacity < chunk_bytes or capacity % chunk_bytes:
            raise ServeLoopError("HBF weight cache must contain whole cache slots")
        if geometry is None:
            raise ServeLoopError("HBF weight cache requires physical link geometry")
        if chunk_bytes % geometry.page_size_bytes:
            raise ServeLoopError("weight-cache slots must align to HBF pages")
        self.begin, self.capacity, self.chunk_bytes = begin, capacity, chunk_bytes
        self.geometry = geometry
        self.lines = OrderedDict()
        self.next_slot = 0
        self.totals = Counter()
        self.begin_batch()

    def begin_batch(self):
        self.readers = {}
        self.ready = {}
        self.counters = Counter()

    def account(self, **values):
        self.counters.update(values)
        self.totals.update(values)

    def read(self, *, placed, offset, byte_count, dependencies, emit, hbm_pieces):
        if offset < 0 or byte_count <= 0 or offset + byte_count > placed.bytes:
            raise ServeLoopError("weight cache access escapes its immutable object")
        pieces, mapped_ids = [], []
        cursor, end = offset, offset + byte_count
        while cursor < end:
            index = cursor // self.chunk_bytes
            within = cursor % self.chunk_bytes
            size = min(end - cursor, self.chunk_bytes - within)
            key = (placed.object_id, index)
            if key in self.lines:
                slot = self.lines.pop(key)
                self.account(hit_bytes=size)
            else:
                if len(self.lines) == self.capacity // self.chunk_bytes:
                    _, slot = self.lines.popitem(last=False)
                    self.account(evictions=1)
                else:
                    slot = self.next_slot
                    self.next_slot += 1
                source = placed.addr + index * self.chunk_bytes
                fill_bytes = min(self.chunk_bytes, placed.bytes - index * self.chunk_bytes)
                read = emit(target="HBF_LOGICAL", op="R", addr=source,
                            byte_count=fill_bytes, dependencies=dependencies)
                links = []
                full_pages = fill_bytes // self.geometry.page_size_bytes * self.geometry.page_size_bytes
                stack_bytes = list(hbf_link_bytes_by_stack(source, full_pages, self.geometry))
                if full_pages != fill_bytes:
                    tail_stack = self.geometry.stack_for_logical_page((source + full_pages) // self.geometry.page_size_bytes)
                    stack_bytes[tail_stack] += fill_bytes - full_pages
                for stack, amount in enumerate(stack_bytes):
                    if amount:
                        links.append(emit(target="D2D_HBF_TO_HBM", op="R", addr=0,
                                          byte_count=amount, dependencies=(read,), stack=stack))
                # The backing read may overlap older consumers. Installation
                # must wait for every consumer of this physical cache slot.
                install_deps = (*links, *self.readers.get(slot, ()))
                installs = [emit(target="HBM", op="W", addr=addr,
                                 byte_count=amount, dependencies=install_deps)
                            for addr, amount in hbm_pieces(self.begin + slot * self.chunk_bytes, fill_bytes)]
                self.ready[slot] = tuple(installs)
                self.readers[slot] = []
                self.account(miss_requested_bytes=size, fill_bytes=fill_bytes,
                             hbm_install_bytes=fill_bytes, d2d_bytes=fill_bytes, fills=1)
            self.lines[key] = slot
            local = hbm_pieces(self.begin + slot * self.chunk_bytes + within, size)
            reads = [emit(target="HBM", op="R", addr=addr, byte_count=amount,
                          dependencies=(*dependencies, *self.ready.get(slot, ())))
                     for addr, amount in local]
            self.readers.setdefault(slot, []).extend(reads)
            pieces.extend(local)
            mapped_ids.extend(reads)
            self.account(accessed_bytes=size)
            cursor += size
        return pieces, mapped_ids

    def receipt(self):
        return {
            "policy": "address_only_lru",
            "capacity_bytes": self.capacity,
            "chunk_bytes": self.chunk_bytes,
            "occupied_bytes": len(self.lines) * self.chunk_bytes,
            "resident_chunks": len(self.lines),
            "lru_sha256": canonical_sha256([(key, slot) for key, slot in self.lines.items()]),
            "batch_counters": dict(self.counters),
            "cumulative": dict(self.totals),
            "fill_boundary": "one immutable object; partial final chunk consumes a full slot",
        }
