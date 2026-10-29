"""Bind declared coarse KV producers without changing weight release policies.

Profiles retain their existing serial call envelope and layer timing budgets.
Captured names never supply an implicit semantic/dataflow contract.
"""
from dataclasses import replace
import heapq
from serveloop.contracts import ServeLoopError

SEMANTIC_KINDS = ('operation','input_norm','post_norm','kv_source','kv_copy',
                  'attention','router','route_selection','dispatch','shared_expert')
WEIGHT_ROLES = ('attention/weights','ffn/weights','moe/router_weights',
                'moe/routed_expert_weights','moe/shared_expert_weights')


def validate_contract(phase):
    """Reject incomplete/unknown declarations before inspecting graph members."""
    if not isinstance(phase,dict) or phase.get('consumer_contract')!='model-semantic-v1':
        raise ServeLoopError('causal binding requires an explicit model semantic contract; captured profile names are insufficient')
    layers=phase.get('layers')
    if not isinstance(layers,list) or not layers:
        raise ServeLoopError('semantic contract requires layer call lists')
    routing=phase.get('routing_layers',[[] for _ in layers])
    if not isinstance(routing,list) or len(routing)!=len(layers):
        raise ServeLoopError('semantic routing lane count mismatch')
    for body,route in zip(layers,routing):
        if not isinstance(body,list) or not body or not isinstance(route,list):
            raise ServeLoopError('invalid semantic call list')
        kinds=[]
        for item in route+body:
            if not isinstance(item,dict) or not isinstance(item.get('id'),str):
                raise ServeLoopError('invalid semantic call identity')
            sem=item.get('semantic')
            if not isinstance(sem,dict) or sem.get('kind') not in SEMANTIC_KINDS or 'requires_weights' not in sem:
                raise ServeLoopError('missing/unknown explicit consumer semantic role')
            if set(sem)-{'kind','requires_weights'}:
                raise ServeLoopError('unknown consumer semantic field')
            for key in ('requires_weights',):
                values=sem.get(key,[])
                if not isinstance(values,list) or any(v not in WEIGHT_ROLES for v in values):
                    raise ServeLoopError('unknown semantic weight role')
            kinds.append(sem['kind'])
        if any(kinds.count(kind)!=1 for kind in ('kv_source','attention','post_norm')):
            raise ServeLoopError('semantic contract requires one KV producer, attention and post-norm')
        if kinds.index('kv_source')>kinds.index('attention'):
            raise ServeLoopError('semantic KV producer must precede its consumer')
        if route:
            route_kinds=[item['semantic']['kind'] for item in route]
            if route_kinds.count('route_selection')!=1 or 'kv_source' not in route_kinds or 'dispatch' not in kinds[len(route):]:
                raise ServeLoopError('incomplete semantic routing producer/dispatch contract')
        elif 'route_selection' in kinds or 'dispatch' in kinds:
            raise ServeLoopError('semantic route operation lacks routing lane')
    return routing


def bind_producers(batch, phase):
    """Retain weight parents; bind KV publication and route-to-dispatch causality."""
    routing_layers=validate_contract(phase)
    operations = {o.id:o for o in batch.operations}
    audit = {k:dict(v) for k,v in batch.audit.items()}
    order = [o.id for o in batch.operations]
    embedded = next(o.id for o in batch.operations if o.role=='embedding/complete')
    computes = {int(o.role.split('/')[1]):o.id for o in batch.operations
                if o.role.startswith('layer/') and o.role.endswith('/compute')}
    weights = {}
    kvwrites = {}; kvreads = {}; current_reads = {}; sources = {}; routes = {}
    positions = {}
    for o in batch.operations:
        layer=audit[o.id].get('layer')
        if layer is not None and not o.is_barrier:
            weights.setdefault((layer,o.role),[]).append(o.id)
            if o.role=='attention/kv_write':kvwrites.setdefault(layer,[]).append(o.id)
            if o.role=='attention/kv_read':kvreads.setdefault(layer,[]).append(o.id)
            if o.role=='attention/kv_read_current':current_reads.setdefault(layer,[]).append(o.id)
        if o.role.startswith('layer/') and o.role.endswith('/kv_source_ready'):
            sources[int(o.role.split('/')[1])]=o.id
        if o.role.startswith('layer/') and o.role.endswith('/routing_ready'):
            routes[int(o.role.split('/')[1])]=o.id
        if o.role=='consumer/complete':
            position=audit[o.id].get('consumer_position')
            if not isinstance(position,list) or len(position)!=3 or type(position[0]) is not int or position[1] not in ('routing','body') or type(position[2]) is not int:
                raise ServeLoopError('invalid consumer call reference')
            key=tuple(position)
            if key in positions:raise ServeLoopError('duplicate consumer semantic position')
            positions[key]=o.id

    def dependencies(identifier,parents):
        operations[identifier]=replace(operations[identifier],dependencies=tuple(dict.fromkeys(parents)))

    def members(layer,role):
        result=weights.get((layer,role),[])
        if not result:raise ServeLoopError('semantic contract missing weight role '+role)
        return result

    expected={(layer,lane,index) for layer,body in enumerate(phase['layers'])
              for lane,calls in [('routing',routing_layers[layer]),('body',body)]
              for index in range(len(calls))}
    if set(positions)!=expected or set(computes)!=set(range(len(phase['layers']))):
        raise ServeLoopError('consumer call reference or layer count mismatch')
    for layer,body in enumerate(phase['layers']):
        routing=routing_layers[layer]
        if not kvwrites.get(layer):raise ServeLoopError('semantic contract lacks current KV publication')
        previous_layer=computes[layer-1] if layer else embedded
        previous=previous_layer
        source_done=None;post_done=None;route_done=None
        if bool(routing) != (layer in routes):
            raise ServeLoopError('model semantic routing lane disagrees with layer routing barrier')
        for lane,calls in [('routing',routing),('body',body)]:
            if lane=='body' and routing:
                if route_done is None:raise ServeLoopError('missing router/top-k selection completion')
                dependencies(routes[layer],[previous])
                previous=routes[layer] # Original routing duration remains here.
            for index,call in enumerate(calls):
                key=(layer,lane,index)
                if key not in positions:raise ServeLoopError('missing declared consumer completion')
                done=positions[key];meta=audit[done];sem=call['semantic']
                if meta.get('semantic')!=sem:raise ServeLoopError('consumer semantic annotation drift')
                kind=sem['kind'];parents=[previous]
                for role in sem['requires_weights']:parents.extend(members(layer,role))
                if kind=='attention':
                    if source_done is None:raise ServeLoopError('KV consumer precedes declared producer')
                    parents.extend([*kvreads.get(layer,[]),*current_reads.get(layer,[])])
                if kind=='dispatch' and (not routing or route_done is None):
                    raise ServeLoopError('expert dispatch lacks current routing selection')
                ids=meta.get('memory_ids')
                if not isinstance(ids,list) or any(not isinstance(i,str) or i not in operations or audit[i].get('consumer_position')!=list(key) for i in ids) or len(set(ids))!=len(ids):
                    raise ServeLoopError('invalid consumer memory call reference')
                within=set(ids)
                for identifier in ids:
                    # Preserve any internal multistage edges; replace only the
                    # enclosing old aggregate release with semantic parents.
                    internal=[d for d in operations[identifier].dependencies if d in within]
                    dependencies(identifier,[*internal,*parents])
                dependencies(done,ids or parents)
                if kind=='kv_source':
                    if source_done is not None:raise ServeLoopError('multiple current KV producers')
                    source_done=done
                    gate=sources.get(layer)
                    if gate is not None:dependencies(gate,[done])
                    for identifier in kvwrites.get(layer,[]):dependencies(identifier,[gate or done])
                    # History has no RAW edge from this forward. Current-token
                    # reads keep the corresponding publication edge from coverage.
                if kind=='post_norm':post_done=done
                if kind=='route_selection':
                    if lane!='routing':raise ServeLoopError('route selection outside routing lane')
                    route_done=done
                previous=done
        if source_done is None or post_done is None:
            raise ServeLoopError('incomplete model KV/post-norm semantic contract')
        # The old memory_complete may contain current KV reads and selected
        # weights; it must never be the current KV producer's parent.
        dependencies(computes[layer],[*operations[computes[layer]].dependencies,previous])

    indices={identifier:i for i,identifier in enumerate(order)}
    indegree={identifier:len(o.dependencies) for identifier,o in operations.items()}
    children={identifier:[] for identifier in operations}
    for identifier,o in operations.items():
        for parent in o.dependencies:
            if parent not in operations:raise ServeLoopError('missing consumer dependency '+parent)
            children[parent].append(identifier)
    ready=[(indices[k],k) for k,n in indegree.items() if n==0];heapq.heapify(ready);result=[]
    while ready:
        _,identifier=heapq.heappop(ready);result.append(operations[identifier])
        for child in children[identifier]:
            indegree[child]-=1
            if indegree[child]==0:heapq.heappush(ready,(indices[child],child))
    if len(result)!=len(operations):raise ServeLoopError('consumer dependency cycle')
    payload=lambda xs:sorted((o.id,o.op,o.object_id,o.offset,o.bytes,o.duration_ns) for o in xs if not o.is_barrier)
    if payload(result)!=payload(batch.operations):raise ServeLoopError('causal binding changed memory ranges')
    if sum(o.duration_ns for o in result)!=sum(o.duration_ns for o in batch.operations):
        raise ServeLoopError('causal binding changed original compute/routing duration')
    final=batch.operations[-1].id
    audit[final]['coarse_kv_contract']=dict(audit[final].get('coarse_kv_contract',{}),
        mode='bound-model-semantic-producers',captured_operator_inference=False,
        qualification='actual declared producer completion gates current KV publication/current-token read; zero joins, no hardware timing claim')
    audit[final]['coarse_kv_contract'].update(
        weight_release='unchanged main dependency parents',
        serial_envelope='existing profile call order, not a universally minimal hardware DAG')
    if 'coarse_compute' in audit[final]:
        audit[final]['coarse_compute']['overlap'] = 'main weight prefetch; explicit KV production and route-to-dispatch dependencies'
    return replace(batch,operations=tuple(result),audit=audit)
