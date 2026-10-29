# Optional coarse weight scheduling

Generated analytic coverage supports two explicit aggregate weight release policies.
The default is `legacy-prefetch`, preserving the causal-only patch's graph and main's
weight release behavior. `consumer-order` is an opt-in conservative alternative.
Both retain the same KV producer/publication/current-read and route-to-dispatch data
edges, weight payloads, expert choices and original compute/routing durations.

```sh
python -m serveloop run ... --coarse-coverage-policy model-derived \
  --coarse-schedule-policy legacy-prefetch
python -m serveloop run ... --coarse-coverage-policy model-derived \
  --coarse-schedule-policy consumer-order
```

Supported coverage entries are `model-derived`, `dense-16bit-swiglu` and its
`dense-bf16-swiglu` alias. Bare aggregate and captured-profile paths reject an
explicit policy; their existing behavior remains. Shared snapshot input loading
also rejects this field because it does not execute generated coverage.

Alternatively, set `"coarse_schedule_policy": "consumer-order"` or
`"legacy-prefetch"` in run-config JSON. The CLI and JSON cannot both set it.
Resolved configuration, coverage provenance and canonical audit record the policy.
Unknown values are errors. Library users pass `schedule_policy` directly to
`AnalyticCoverageCompiler` or `CatalogCoverageCompiler`, or use the
`coarse_schedule_policy` argument of `run_experiment`. The existing three-value
`run_config_from_dict` timing/scheduler return contract is unchanged.

| Weight group | legacy-prefetch (default) | consumer-order |
|---|---|---|
| Attention aggregate | Existing memory-root release | After previous layer or embedding completion |
| Dense FFN aggregate | Existing early release | After output projection, before post-attention norm |
| Router aggregate | Existing early release | Before router call, after its preceding declared call |
| Shared-expert aggregate | Existing early release | After post-attention norm; independent of current route decision |
| Routed experts | Existing routing-ready dependency | Same routing-ready dependency |

Consumer-order uses the existing aggregate groups, so a group can contain weights
for multiple consumers. Its whole-group/whole-layer gates are explicit conservative
assumptions, not globally measured hardware dependencies. Historical KV read release
still follows the original memory root, so prefetch depth can affect it under either
policy. No weight subranges, packing convention, selected-expert prefetch mechanism,
CTA/instruction arrival model, cache policy or new compute cost is introduced.

Only weight release parents change when selecting consumer-order. All non-weight
operation dependencies, including data-production edges, are preserved. A stable
topological sort keeps the canonical graph valid. Selecting a different policy can
change modeled cache behavior and simulated time; it is not a timing-accuracy claim.
Hardware evidence for one execution window does not qualify other models/contexts.
