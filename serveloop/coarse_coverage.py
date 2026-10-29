"""Digest-bound coarse coverage profiles, independent of GPU compute calibration.

Profiles describe covered shapes only. They are tensor/kernel footprints, not
instruction traces or GPU cache misses. The ideal-temporaries option is an
explicit sensitivity bound, not a cache implementation.
"""
from dataclasses import replace
import json
from pathlib import Path

from serveloop.compiler import ServeLoopCompiler
from serveloop.contracts import ServeLoopError, SemanticOperation, canonical_sha256

SCHEMA = 'serveloop.coarse_coverage.v1'


class CoarseCoverageProfile:
    def __init__(self, document):
        try:
            if document['schema'] != SCHEMA:
                raise ValueError('unsupported schema')
            self.digest = canonical_sha256(document)
            self.document = document
            self.model_digest = document['model_sha256']
            self.request_digest = document['request_trace_sha256']
            if any(not isinstance(d,str) or len(d)!=64 or any(c not in '0123456789abcdef' for c in d)
                   for d in (self.model_digest,self.request_digest)):
                raise ValueError('invalid model/request digest')
            self.workspace_bytes = document['workspace_bytes']
            if type(self.workspace_bytes) is not int or self.workspace_bytes <= 0:
                raise ValueError('invalid workspace span')
            self.objects = document['objects']
            if not self.objects or any(not isinstance(o,str) for o in self.objects):
                raise ValueError('invalid object identities')
            self.phases = document['phases']
            if type(document['layers']) is not int or document['layers'] <= 0:
                raise ValueError('invalid layer count')
            for phase in ('prefill','decode'):
                p = self.phases[phase]
                if len(p['layers']) != document['layers'] or len(p['shape']) != 2:
                    raise ValueError('inconsistent shape/layer coverage')
                if any(type(n) is not int or n < 0 for n in p['shape']) or p['shape'][0] <= 0:
                    raise ValueError('invalid covered shape')
                if type(p['current_token_kv_read']) is not bool:
                    raise ValueError('invalid current-token KV rule')
                routing_layers = p.get('routing_layers', [[] for _ in p['layers']])
                if len(routing_layers) != document['layers']:
                    raise ValueError('inconsistent routing-layer coverage')
                if 'consumer_contract' in p:
                    from serveloop.coarse_causality import validate_contract
                    validate_contract(p)
                    for calls in p['layers'] + routing_layers:
                        for item in calls:
                            if len(item['stages'])!=1 or item['stages'][0]['kernel'] is not None:
                                raise ValueError('model semantic contract requires one declared tensor stage per call')
                for call in p['embedding'] + sum(p['layers'],[]) + sum(routing_layers,[]) + p['tail']:
                    if not isinstance(call['id'],str) or not isinstance(call['operator'],str):
                        raise ValueError('invalid call identity')
                    for stage in call['stages']:
                        if stage['kernel'] is not None and (type(stage['kernel']) is not int or stage['kernel'] < 0):
                            raise ValueError('invalid kernel ordinal')
                        for obj,op,offset,size in stage['ranges']:
                            if any(type(v) is not int for v in (obj,op,offset,size)):
                                raise ValueError('range fields must be integers')
                            if not (0 <= obj < len(self.objects) and op in (0,1) and
                                    offset >= 0 and size > 0 and offset+size <= self.workspace_bytes):
                                raise ValueError('range escapes profile storage')
        except (KeyError,TypeError,ValueError) as error:
            raise ServeLoopError(f'invalid coarse coverage profile: {error}') from error

    @classmethod
    def load(cls,path):
        try:
            return cls(json.loads(Path(path).read_text()))
        except (ValueError,TypeError) as error:
            raise ServeLoopError(f'invalid coarse coverage JSON: {error}') from error


class CoarseCoverageCompiler(ServeLoopCompiler):
    # Captured profiles retain their original memory-only qualification.
    supports_roofline = False
    supports_linear = False
    supports_chunked_prefill = False

    requires_causal_contract = False

    def __init__(self, *, coverage_profile, cache_bound='off', schedule_policy=None, **kwargs):
        if schedule_policy is not None:
            raise ServeLoopError('coarse schedule policy requires generated analytic coverage; captured profiles keep their schedule')
        super().__init__(**kwargs)
        self.coverage = coverage_profile
        if cache_bound not in ('off','ideal-temporaries'):
            raise ServeLoopError('unsupported coarse cache sensitivity bound')
        self.cache_bound = cache_bound
        if self.timing.timing_model != 'memory_only' and not (
                self.supports_roofline and self.timing.timing_model == 'roofline') and not (
                self.supports_linear and self.timing.timing_model == 'linear'):
            raise ServeLoopError('captured coverage requires memory_only; model-derived coverage supports memory_only or roofline or linear')
        if len(self.models) != 1 or next(iter(self.models.values())).digest != self.coverage.model_digest:
            raise ServeLoopError('model does not match the coarse coverage profile')
        if self.request_trace.digest != self.coverage.request_digest:
            raise ServeLoopError('request trace does not match this input-bound coarse coverage profile')

    def coverage_for_batch(self, batch):
        return self.coverage

    def compile(self,batch):
        coverage = self.coverage_for_batch(batch)
        if len(batch.slices) != 1:
            raise ServeLoopError('coarse profile currently covers B1 only')
        s = batch.slices[0]
        phase = coverage.phases.get(s.phase)
        if phase is None or [s.token_count,s.context_tokens_before] != phase['shape']:
            raise ServeLoopError('request shape outside coarse coverage profile; no extrapolation')
        # Tail profile is bound to one output-emitting forward of this shape.
        if not s.emits_output and not (self.supports_chunked_prefill and s.phase == 'prefill'):
            raise ServeLoopError('coarse profile requires an output-emitting forward')
        if self.requires_causal_contract:
            from serveloop.coarse_causality import validate_contract
            validate_contract(phase)
        base = super().compile(batch)
        model = self.models[batch.model_id]
        if phase['current_token_kv_read']:
            # Keep history independent of this forward's KV producer. Only the
            # appended current-token read requires its corresponding KV write.
            ops=[];audit=dict(base.audit)
            for o in base.operations:
                ops.append(o)
                if o.role == 'attention/kv_read':
                    layer = base.audit[o.id]['layer']
                    write=next(w for w in base.memory_operations
                        if w.role=='attention/kv_write' and w.object_id==o.object_id)
                    identifier=o.id+'/current'
                    ops.append(replace(o,id=identifier,role='attention/kv_read_current',
                        offset=o.offset+o.bytes,bytes=s.token_count*model.layers[layer].kv_bytes_per_token,
                        dependencies=(write.id,)))
                    audit[identifier]=dict(audit[o.id],role='attention/kv_read_current',
                        traffic_semantics='current_token_read_after_corresponding_publication')
                    audit[o.id]=dict(audit[o.id],kv_read_scope='historical-only')
            additions={o.id.removesuffix('/current'):o.id for o in ops if o.role=='attention/kv_read_current'}
            ops=[replace(o,dependencies=tuple(dict.fromkeys((*o.dependencies,
                *(additions[d] for d in o.dependencies if d in additions))))) for o in ops]
            base = replace(base,operations=tuple(ops),audit=audit)
        output=[];audit=dict(base.audit);count=0;joins=0

        def stages(calls,dependencies,layer=None,lane=None):
            nonlocal count,joins
            previous=tuple(dependencies)
            for call_index,call in enumerate(calls):
                call_ids=[]
                for stage in call['stages']:
                    ids=[]
                    for obj,op,offset,size in stage['ranges']:
                        if self.cache_bound == 'ideal-temporaries':
                            continue
                        identifier=f'serve/b{batch.batch_id}/coverage{count}';count+=1
                        direction='R' if op==0 else 'W';role='activation_coverage/'+direction
                        output.append(SemanticOperation(id=identifier,op=direction,
                            object_id=f'workspace/{model.model_id}',offset=offset,bytes=size,
                            duration_ns=0.0,dependencies=previous,role=role))
                        ids.append(identifier);call_ids.append(identifier)
                        audit[identifier]=dict(role=role,call=call['id'],operator=call['operator'],
                            observed_object=coverage.objects[obj],attention_kernel=stage['kernel'],
                            traffic_semantics='kernel/tensor coverage; not measured post-cache traffic',
                            coarse_coverage_sha256=coverage.digest)
                        if layer is not None and phase.get('consumer_contract'):
                            audit[identifier]['consumer_position']=[layer,lane,call_index]
                    if ids:
                        if stage['kernel'] is not None:
                            identifier=f'serve/b{batch.batch_id}/attention_join{joins}';joins+=1
                            role='attention_kernel/complete'
                            output.append(SemanticOperation(id=identifier,op=None,object_id=None,offset=0,
                                bytes=0,duration_ns=0.0,dependencies=tuple(ids),role=role))
                            audit[identifier]=dict(role=role,call=call['id'],kernel=stage['kernel'],
                                source='zero-duration dependency join; no extra device resource')
                            previous=(identifier,)
                        else:
                            previous=tuple(ids)
                if layer is not None and phase.get('consumer_contract'):
                    slot='r' if lane=='routing' else 'c'
                    identifier=f'consumer/b{batch.batch_id}/l{layer}/{slot}{call_index}'
                    role='consumer/complete'
                    output.append(SemanticOperation(id=identifier,op=None,object_id=None,offset=0,
                        bytes=0,duration_ns=0.0,dependencies=previous,role=role))
                    audit[identifier]=dict(role=role,layer=layer,call=call['id'],
                        consumer=call['id'],operator=call['operator'],consumer_position=[layer,lane,call_index],
                        semantic=call['semantic'],memory_ids=call_ids,
                        source='explicit model semantic completion; zero added compute')
                    previous=(identifier,)
            return previous

        for operation in base.operations:
            deps=operation.dependencies
            if operation.role=='embedding/complete':
                deps=tuple(dict.fromkeys((*deps,*stages(phase['embedding'],deps))))
            elif operation.role.startswith('layer/') and operation.role.endswith('/routing_ready'):
                layer=int(operation.role.split('/')[1])
                calls=phase.get('routing_layers',[[] for _ in model.layers])[layer]
                deps=tuple(dict.fromkeys((*deps,*stages(calls,deps,layer,'routing'))))
            elif operation.role.startswith('layer/') and operation.role.endswith('/compute'):
                layer=int(operation.role.split('/')[1])
                deps=tuple(dict.fromkeys((*deps,*stages(phase['layers'][layer],deps,layer,'body'))))
            elif operation.role=='tail/compute':
                deps=tuple(dict.fromkeys((*deps,*stages(phase['tail'] if s.emits_output else phase['tail'][:1],deps))))
            output.append(replace(operation,dependencies=deps))
        audit[output[-1].id]=dict(audit[output[-1].id],coarse_coverage_sha256=coverage.digest,
            coarse_cache_bound=self.cache_bound,
            cache_bound_scope='temporary coverage only; weight/KV traffic unchanged',
            coverage_evidence=coverage.document.get('evidence',{}),
            coarse_kv_contract=dict(mode='declared-model-semantics-await-consumer-binding' if phase.get('consumer_contract') else 'abstract-base-KV-floor',
                captured_operator_inference=False,
                base_source_gate_present=any(o.role.endswith('/kv_source_ready') for o in base.operations),
                qualification='consumer compiler binds declared producer joins; captured profiles retain base dependency floor only'))
        if self.timing.timing_model == 'roofline':
            audit[output[-1].id]['coarse_compute'] = dict(
                model='original layer-aggregate FLOP roofline',
                peak_tflops=self.timing.peak_tflops, efficiency=self.timing.efficiency,
                compute_nodes='original durations preserved; MoE routing split counted once',
                overlap='next-layer weight/KV prefetch; covered layer activations precede aggregate compute',
                calibrated=False)
        elif self.timing.timing_model == 'linear':
            audit[output[-1].id]['coarse_compute'] = dict(
                model='original layer-aggregate linear sensitivity',
                provider=self.timing.canonical(), calibrated=False,
                compute_nodes='original durations and MoE routing split preserved',
                overlap='next-layer weight/KV prefetch; covered layer activations precede aggregate compute')
        result = replace(base,operations=tuple(output),audit=audit)
        if phase.get('consumer_contract'):
            from serveloop.coarse_causality import bind_producers
            result = bind_producers(result, phase)
        if hasattr(self, 'schedule_policy'):
            from serveloop.coarse_schedule import apply_weight_schedule
            result = apply_weight_schedule(result, phase, self.schedule_policy)
        return result
