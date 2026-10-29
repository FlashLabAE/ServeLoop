"""Catalog-derived external tensor footprints for dense/MoE GQA and MLA.

Quantized weight/scale bytes and expert choices remain in the original ledger.
External activations are declared 16-bit, with fused quantized GEMMs and fused
latent attention; this is not an implementation-specific kernel trace.
"""
from serveloop.coarse_analytic import build_profile as dense_profile
from serveloop.coarse_coverage import CoarseCoverageCompiler, CoarseCoverageProfile, SCHEMA
from serveloop.contracts import ServeLoopError

POLICY = 'model-derived'
ASSUMPTIONS = (
    '16-bit external activations; original quantized weights/scales and KV ledger unchanged',
    'fused quantized GEMMs; no expanded-weight workspace or activation re-quantization scratch',
    'SwiGLU/RMSNorm; fused attention, no materialized score matrix',
    'MoE top-k token copies in packed buffers; no expert padding/capacity/drop or network traffic',
    'MLA fused compressed-context attention; no external expanded context K/V',
    'no instruction repeats, hardware GPU cache calibration or compute timing',
)


def build_profile(model, request_digest, phase, tokens, context):
    if model.coverage_descriptor is None:
        if model.structure is not None:
            return dense_profile(model, request_digest, phase, tokens, context)
        raise ServeLoopError('model-derived coverage needs a public descriptor or model JSON imported with --coverage-descriptor')
    if (phase not in ('prefill', 'decode') or type(tokens) is not int or tokens <= 0
            or type(context) is not int or context < 0
            or (phase == 'decode' and (tokens != 1 or context == 0))):
        raise ServeLoopError('model-derived coverage supports B1 full/chunked prefill and single-token decode')
    # Preserve the established dense16 numerical path when already qualified.
    if model.structure is not None and not any(l.is_moe for l in model.layers):
        return dense_profile(model, request_digest, phase, tokens, context)
    document = model.coverage_descriptor
    a, precision = document['architecture'], document['precision']
    attn, ffn = a['attention'], a['ffn']; moe = ffn.get('moe')
    h, heads = a['hidden_size'], a['num_attention_heads']; x = 2*tokens*h
    if attn['kind'] not in ('gqa', 'mla'):
        raise ServeLoopError('model-derived coverage supports GQA/MLA attention')
    if precision['non_matrix_weight_bytes'] != 2 or precision['kv_bytes'] != 2:
        raise ServeLoopError('model-derived coverage currently requires16-bit norm/KV payloads')
    widths = [ffn.get('dense_intermediate_size', 0)]
    if moe: widths += [moe['expert_intermediate_size']]
    f = max(widths); top = max(l.top_k for l in model.layers) or 1
    shared = moe['shared_experts_per_layer'] if moe else 0
    maxkv = max(l.kv_bytes_per_token for l in model.layers)*tokens
    q = 2*tokens*heads*(attn['head_dim'] if attn['kind']=='gqa' else
                         attn['qk_nope_head_dim']+attn['qk_rope_head_dim'])
    out = 2*tokens*heads*(attn['head_dim'] if attn['kind']=='gqa' else attn['v_head_dim'])
    rotary = attn.get('rotary_dim', attn.get('head_dim',attn.get('qk_rope_head_dim')))
    rope_row = rotary*(2 if attn.get('rope_table_dtype','float32')!='float32' else 4)
    objects=[]; spans={}; cursor=0
    def tensor(name, size):
        nonlocal cursor
        cursor=(cursor+511)//512*512
        objects.append(name); spans[name]=(len(objects)-1,cursor,size);cursor+=size
    for name,size in [('token_ids',8*tokens),('positions',8*tokens),('rope_rows',tokens*rope_row),
                      ('residual',x),('hidden',x),('norm',x),('attention_output',out)]:tensor(name,size)
    if attn['kind']=='gqa':
        tensor('qkv',q+maxkv);tensor('dense_kv',maxkv)
    else:
        tensor('query_down',2*tokens*attn['q_lora_rank'])
        tensor('query',q);tensor('latent_kv',maxkv)
    tensor('gate_up',4*tokens*f);tensor('silu',2*tokens*f)
    if moe:
        experts=moe['routed_experts_per_layer'];ef=moe['expert_intermediate_size']
        for name,size in [('router_logits',4*tokens*experts),('route_ids',4*tokens*top),
                         ('route_weights',4*tokens*top),('dispatch',x*top),
                         ('expert_gate_up',4*tokens*top*ef),('expert_silu',2*tokens*top*ef),
                         ('expert_output',x*top)]:tensor(name,size)
        if shared:
            tensor('shared_gate_up',4*tokens*shared*ef);tensor('shared_silu',2*tokens*shared*ef)
            tensor('shared_output',x)
    tensor('selected',2*h);tensor('logits_16bit',2*model.vocab_size)
    tensor('logits_fp32',4*model.vocab_size)
    def call(name, accesses):
        ranges=[]
        for access in accesses:
            obj,op,size=access[:3];rel=access[3] if len(access)==4 else 0
            index,offset,capacity=spans[obj]
            if size <= 0 or rel < 0 or rel+size > capacity:
                raise ServeLoopError('catalog coverage tensor exceeds its reserved span')
            ranges.append([index,0 if op=='R' else 1,offset+rel,size])
        return dict(id=name,operator=name,stages=[dict(kernel=None,ranges=ranges)])
    embedding=[call('embedding',[('token_ids','R',8*tokens),('residual','W',x)])]
    layers=[]; routing_layers=[]
    for i,layer in enumerate(model.layers):
        kv=tokens*layer.kv_bytes_per_token
        pre=[('residual','R',x),('norm','W',x)] if i==0 else [
            ('residual','R',x),('norm','R',x),('residual','W',x),('norm','W',x)]
        calls=[call('rmsnorm' if i==0 else 'fused_add_rmsnorm',pre)]
        if attn['kind']=='gqa':
            calls.append(call('qkv_projection',[('norm','R',x),('qkv','W',q+kv)]))
            if attn.get('qk_head_norms',False):
                calls.append(call('qk_head_norm',[('qkv','R',q+kv//2),('qkv','W',q+kv//2)]))
            calls.append(call('rope',[('positions','R',8*tokens),('rope_rows','R',rope_row*tokens),
                                       ('qkv','R',q+kv//2),('qkv','W',q+kv//2)]))
            calls.append(call('kv_store_source',[('qkv','R',kv,q)]))
            if phase=='prefill':
                calls.append(call('kv_contiguous_copy',[('qkv','R',kv,q),('dense_kv','W',kv)]))
                attention=[('qkv','R',q),('dense_kv','R',kv),('attention_output','W',out)]
            else:attention=[('qkv','R',q),('attention_output','W',out)]
        else:
            qdown=2*tokens*attn['q_lora_rank']; latent=2*tokens*attn['kv_lora_rank']
            calls += [call('mla_query_down',[('norm','R',x),('query_down','W',qdown)]),
                      call('mla_query_norm',[('query_down','R',qdown),('query_down','W',qdown)]),
                      call('mla_query_up',[('query_down','R',qdown),('query','W',q)]),
                      call('mla_kv_down',[('norm','R',x),('latent_kv','W',kv)]),
                      call('mla_kv_norm',[('latent_kv','R',latent),('latent_kv','W',latent)]),
                      call('mla_rope',[('positions','R',8*tokens),('rope_rows','R',tokens*rope_row),
                                       ('query','R',q),('query','W',q),
                                       ('latent_kv','R',kv-latent,latent),('latent_kv','W',kv-latent,latent)]),
                      call('kv_store_source',[('latent_kv','R',kv)])]
            attention=[('query','R',q),('attention_output','W',out)]
            if phase=='prefill':attention.insert(1,('latent_kv','R',kv))
        calls += [call('fused_attention' if attn['kind']=='gqa' else 'fused_latent_attention',attention),
                  call('output_projection',[('attention_output','R',out),('hidden','W',x)]),
                  call('fused_add_rmsnorm',[('residual','R',x),('hidden','R',x),
                                           ('residual','W',x),('hidden','W',x)])]
        if not layer.is_moe:
            df=ffn['dense_intermediate_size']
            calls += [call('gate_up_projection',[('hidden','R',x),('gate_up','W',4*tokens*df)]),
                      call('silu_and_mul',[('gate_up','R',4*tokens*df),('silu','W',2*tokens*df)]),
                      call('down_projection',[('silu','R',2*tokens*df),('norm','W',x)])]
            routing_layers.append([])
        else:
            k=layer.top_k;ef=moe['expert_intermediate_size'];routes=4*tokens*k
            calls += [call('router_logits',[('hidden','R',x),('router_logits','W',4*tokens*len(layer.expert_weight_bytes))]),
                      call('router_topk',[('router_logits','R',4*tokens*len(layer.expert_weight_bytes)),
                                          ('route_ids','W',routes),('route_weights','W',routes)])]
            routing_layers.append(calls)
            calls=[call('moe_dispatch',[('hidden','R',x),('route_ids','R',routes),('dispatch','W',x*k)]),
                   call('expert_gate_up',[('dispatch','R',x*k),('expert_gate_up','W',4*tokens*k*ef)]),
                   call('expert_silu_and_mul',[('expert_gate_up','R',4*tokens*k*ef),('expert_silu','W',2*tokens*k*ef)]),
                   call('expert_down',[('expert_silu','R',2*tokens*k*ef),('expert_output','W',x*k)]),
                   call('moe_combine',[('expert_output','R',x*k),('route_weights','R',routes),('norm','W',x)])]
            if shared:
                calls += [call('shared_gate_up',[('hidden','R',x),('shared_gate_up','W',4*tokens*shared*ef)]),
                          call('shared_silu',[('shared_gate_up','R',4*tokens*shared*ef),('shared_silu','W',2*tokens*shared*ef)]),
                          call('shared_down',[('shared_silu','R',2*tokens*shared*ef),('shared_output','W',x)]),
                          call('shared_combine',[('shared_output','R',x),('norm','R',x),('norm','W',x)])]
        # This closed model generator declares roles; consumers never infer
        # captured operator names or treat an offline route as execution.
        route_calls = routing_layers[-1]
        for item in route_calls + calls:
            item['semantic'] = dict(kind='operation', requires_weights=[])
        attention_calls = route_calls or calls
        attention_calls[0]['semantic'] = dict(kind='input_norm', requires_weights=['attention/weights'])
        declared = {'kv_store_source':'kv_source', 'kv_contiguous_copy':'kv_copy',
                    'fused_attention':'attention', 'fused_latent_attention':'attention',
                    'router_topk':'route_selection'}
        for item in attention_calls:
            if item['id'] in declared: item['semantic']['kind'] = declared[item['id']]
        post = next(j for j,c in enumerate(attention_calls) if c['id']=='output_projection')+1
        attention_calls[post]['semantic'] = dict(kind='post_norm', requires_weights=[])
        if layer.is_moe:
            for item in route_calls:
                if item['id']=='router_logits':
                    item['semantic'] = dict(kind='router', requires_weights=['moe/router_weights'])
            calls[0]['semantic'] = dict(kind='dispatch', requires_weights=['moe/routed_expert_weights'])
            for item in calls:
                if item['id']=='shared_gate_up':
                    item['semantic'] = dict(kind='shared_expert', requires_weights=['moe/shared_expert_weights'])
        else:
            attention_calls[post]['semantic'].update(requires_weights=['ffn/weights'])
        layers.append(calls)
    tail=[call('final_add_rmsnorm',[('residual','R',x),('norm','R',x),('residual','W',x),('norm','W',x)]),
          call('last_token_select',[('norm','R',2*h),('selected','W',2*h)]),
          call('lm_head',[('selected','R',2*h),('logits_16bit','W',2*model.vocab_size)]),
          call('logits_cast',[('logits_16bit','R',2*model.vocab_size),('logits_fp32','W',4*model.vocab_size)])]
    p=dict(consumer_contract='model-semantic-v1',shape=[tokens,context],current_token_kv_read=phase=='decode',embedding=embedding,
           layers=layers,routing_layers=routing_layers,tail=tail)
    return CoarseCoverageProfile(dict(schema=SCHEMA,model_sha256=model.digest,request_trace_sha256=request_digest,
        layers=model.num_layers,workspace_bytes=cursor,objects=objects,
        phases={phase:p,('decode' if phase=='prefill' else 'prefill'):p},
        evidence=dict(policy=POLICY,architecture=a,weight_precision=precision,external_activation_bytes=2,
                      assumptions=ASSUMPTIONS,traffic='model-derived tensor footprint, not measured kernel traffic')))


class CatalogCoverageCompiler(CoarseCoverageCompiler):
    requires_causal_contract = True
    supports_roofline = True
    supports_linear = True
    supports_chunked_prefill = True

    def __init__(self, *, schedule_policy='legacy-prefetch', **kwargs):
        from serveloop.coarse_schedule import validate_schedule_policy
        self.schedule_policy = validate_schedule_policy(schedule_policy)
        models=kwargs['models']
        if len(models)!=1:raise ServeLoopError('model-derived coverage currently supports one model')
        model=next(iter(models.values()))
        super().__init__(coverage_profile=build_profile(model,kwargs['request_trace'].digest,'prefill',1,0),**kwargs)

    def coverage_for_batch(self,batch):
        if len(batch.slices)!=1:raise ServeLoopError('model-derived coverage currently supports B1')
        s=batch.slices[0]
        return build_profile(self.models[batch.model_id],self.request_trace.digest,s.phase,s.token_count,s.context_tokens_before)
