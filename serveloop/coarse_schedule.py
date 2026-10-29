"""Optional aggregate weight release, separate from mandatory data causality."""
from dataclasses import replace
import heapq

from .contracts import ServeLoopError
from .coarse_causality import validate_contract

POLICIES = ('legacy-prefetch', 'consumer-order')


def validate_schedule_policy(value):
    if value not in POLICIES:
        raise ServeLoopError('unsupported coarse schedule policy: ' + str(value))
    return value


def apply_weight_schedule(batch, phase, policy):
    """Default legacy preserves the causal-only graph, including weight parents.

    Consumer-order is a conservative aggregate policy, not a recovered GPU DAG.
    Routed experts retain their existing routing-ready dependency under both.
    """
    validate_schedule_policy(policy)
    routing = validate_contract(phase)
    audit = {key: dict(value) for key, value in batch.audit.items()}
    ops = {o.id: o for o in batch.operations}
    if policy == 'consumer-order':
        positions = {tuple(audit[o.id]['consumer_position']): o.id
                     for o in batch.operations if o.role == 'consumer/complete'}
        weights = {}
        for o in batch.memory_operations:
            if o.role.endswith('weights'):
                weights.setdefault((audit[o.id]['layer'], o.role), []).append(o.id)
        embedded = next(o.id for o in batch.operations if o.role == 'embedding/complete')
        computes = {int(o.role.split('/')[1]): o.id for o in batch.operations
                    if o.role.startswith('layer/') and o.role.endswith('/compute')}
        for layer, body in enumerate(phase['layers']):
            previous_layer = computes[layer-1] if layer else embedded
            for identifier in weights.get((layer, 'attention/weights'), []):
                ops[identifier] = replace(ops[identifier], dependencies=(previous_layer,))
            for lane, calls in [('routing', routing[layer]), ('body', body)]:
                for index, call in enumerate(calls):
                    kind = call['semantic']['kind']
                    if kind == 'post_norm':
                        ready = positions[layer, lane, index-1] if index else previous_layer
                        for identifier in weights.get((layer, 'ffn/weights'), []):
                            ops[identifier] = replace(ops[identifier], dependencies=(ready,))
                        # Shared weights are known without this route decision.
                        for identifier in weights.get((layer, 'moe/shared_expert_weights'), []):
                            ops[identifier] = replace(ops[identifier], dependencies=(positions[layer, lane, index],))
                    if kind == 'router':
                        ready = positions[layer, lane, index-1] if index else previous_layer
                        for identifier in weights.get((layer, 'moe/router_weights'), []):
                            ops[identifier] = replace(ops[identifier], dependencies=(ready,))
        # Stable topological order after changing only weight release parents.
        indices = {o.id: i for i, o in enumerate(batch.operations)}
        children = {identifier: [] for identifier in ops}
        degree = {identifier: len(o.dependencies) for identifier, o in ops.items()}
        for identifier, o in ops.items():
            for parent in o.dependencies:
                if parent not in ops:
                    raise ServeLoopError('missing coarse schedule dependency')
                children[parent].append(identifier)
        ready = [(indices[k], k) for k, n in degree.items() if n == 0]
        heapq.heapify(ready); ordered = []
        while ready:
            _, identifier = heapq.heappop(ready); ordered.append(ops[identifier])
            for child in children[identifier]:
                degree[child] -= 1
                if degree[child] == 0:
                    heapq.heappush(ready, (indices[child], child))
        if len(ordered) != len(ops):
            raise ServeLoopError('coarse weight schedule dependency cycle')
        operations = tuple(ordered)
    else:
        operations = batch.operations
    final = batch.operations[-1].id
    audit[final]['coarse_kv_contract'] = dict(audit[final]['coarse_kv_contract'],
        weight_release='unchanged main dependency parents' if policy == 'legacy-prefetch' else 'explicit consumer-order aggregate policy')
    audit[final]['coarse_schedule'] = dict(policy=policy,
        default='legacy-prefetch', scope='aggregate weight release only',
        data_dependencies='unchanged causal producer/publication/current-read and route-to-dispatch edges',
        hardware_accuracy_validated=False,
        limitation='consumer-order is conservative; no instruction arrival or weight subrange inference')
    if policy == 'consumer-order' and 'coarse_compute' in audit[final]:
        audit[final]['coarse_compute'] = dict(audit[final]['coarse_compute'],
            overlap='consumer-order aggregate weight release; existing compute budgets and data causality')
    return replace(batch, operations=operations, audit=audit)
