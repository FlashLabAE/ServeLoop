"""Model-derived dense BF16/FP16 SwiGLU footprints, with explicit fusion assumptions.

No captured profile is read. This policy describes external tensor footprints,
not a particular attention kernel, repeated loads, GPU cache or compute.
"""
from serveloop.coarse_coverage import CoarseCoverageCompiler, CoarseCoverageProfile, SCHEMA
from serveloop.contracts import ServeLoopError, DenseModelStructure

POLICY = 'dense-16bit-swiglu'
POLICY_ALIASES = (POLICY, 'dense-bf16-swiglu')
ASSUMPTIONS = ('BF16/FP16 dense GQA/SwiGLU; homogeneous weights/activations/KV; fused RoPE and residual/RMSNorm; '
               'fused attention without materialized score matrix; dense prefill K/V copies; '
               'one last-token 16-bit head plus FP32 logits per forward; '
               'attention internal scratch/local frames and GEMM workspace omitted; '
               'no instruction repeats, GPU cache filtering or compute')


def model_geometry(model):
    if model.structure is not None:
        return model.structure.hidden_size, [model.structure.intermediate_size]*model.num_layers
    # Compatibility only for the previously released input family. New models
    # require explicit structure; no general byte-to-architecture inference.
    h, rem = divmod(model.embedding_bytes, 2 * model.vocab_size)
    if rem or h <= 0 or model.final_norm_bytes != 2*h:
        raise ServeLoopError('analytic policy requires consistent BF16 embedding/norm geometry')
    widths = []
    for layer in model.layers:
        f, rem = divmod(layer.ffn_weight_bytes - 2*h, 6*h)
        if layer.is_moe or rem or f <= 0 or layer.kv_bytes_per_token % 4:
            raise ServeLoopError('analytic policy requires dense BF16 SwiGLU and integral KV geometry')
        # Q/K/V and output projections, allowing small norm/bias payloads.
        projection = 2*h*(2*h + layer.kv_bytes_per_token//2)
        if not projection <= layer.attention_weight_bytes <= projection + 16*h:
            raise ServeLoopError('analytic policy requires compatible GQA projection geometry')
        widths.append(f)
    return h, widths


def build_profile(model, request_digest, phase, tokens, context):
    if (phase not in ('prefill','decode') or type(tokens) is not int or tokens <= 0
            or type(context) is not int or context < 0
            or (phase == 'decode' and (tokens != 1 or context == 0))):
        raise ServeLoopError('analytic policy supports B1 full/chunked prefill and single-token decode')
    h, widths = model_geometry(model)
    if model.structure is None and (h!=1536 or any(f!=8960 or l.kv_bytes_per_token!=1024
                      for f,l in zip(widths,model.layers))):
        raise ServeLoopError('analytic policy currently requires Qwen-style H1536/F8960/KV1024 geometry')
    g=model.structure or DenseModelStructure(h,8960,12,2,128,'bfloat16','bfloat16',128)
    rope_row=g.rotary_dim*(4 if g.rope_table_dtype=='float32' else 2)
    qbytes=2*tokens*g.num_attention_heads*g.head_dim
    objects=[]; spans={}; cursor=0
    def tensor(name, size):
        nonlocal cursor
        cursor=(cursor+511)//512*512
        objects.append(name); spans[name]=(len(objects)-1,cursor,size); cursor+=size
    x=2*tokens*h
    tensor('token_ids',8*tokens); tensor('positions',8*tokens)
    tensor('rope_rows',tokens*rope_row)
    for name in ('residual','hidden','norm'): tensor(name,x)
    tensor('attention_output',qbytes)
    # Reuse sequential-layer workspace; sizes cover every layer, no layer-count multiplier.
    tensor('qkv',qbytes+tokens*max(l.kv_bytes_per_token for l in model.layers))
    tensor('dense_kv',tokens*max(l.kv_bytes_per_token for l in model.layers))
    tensor('gate_up',4*tokens*max(widths)); tensor('silu',2*tokens*max(widths))
    tensor('selected',2*h); tensor('logits_16bit',2*model.vocab_size)
    tensor('logits_fp32',4*model.vocab_size)
    def call(name, accesses):
        ranges=[]
        for access in accesses:
            obj,op,size=access[:3]
            relative=access[3] if len(access)==4 else 0
            index,offset,capacity=spans[obj]
            if relative+size > capacity: raise ServeLoopError('analytic tensor exceeds reserved footprint')
            ranges.append([index,0 if op=='R' else 1,offset+relative,size])
        return dict(id=name,operator=name,stages=[dict(kernel=None,ranges=ranges)])
    embedding=[call('embedding',[('token_ids','R',8*tokens),('residual','W',x)])]
    layers=[]
    for i,(layer,f) in enumerate(zip(model.layers,widths)):
        kv=tokens*layer.kv_bytes_per_token
        qkv=qbytes+kv
        pre=[('residual','R',x),('norm','W',x)] if i==0 else [
            ('residual','R',x),('norm','R',x),('residual','W',x),('norm','W',x)]
        calls=[call('rmsnorm' if i==0 else 'fused_add_rmsnorm',pre),
               call('qkv_projection',[('norm','R',x),('qkv','W',qkv)]),
               call('rope',[('positions','R',8*tokens),('rope_rows','R',rope_row*tokens),
                            ('qkv','R',qbytes+kv//2),('qkv','W',qbytes+kv//2)])]
        if g.qk_head_norms:
            calls.insert(2,call('qk_head_norm',[('qkv','R',qbytes+kv//2),('qkv','W',qbytes+kv//2)]))
        calls.append(call('kv_store_source',[('qkv','R',kv,qbytes)]))
        if phase=='prefill':
            calls.append(call('kv_contiguous_copy',[('qkv','R',kv,qbytes),('dense_kv','W',kv)]))
            attn=[('qkv','R',qbytes),('dense_kv','R',kv),('attention_output','W',qbytes)]
        else:
            # Persistent KV read is emitted by the original compiler, augmented below.
            attn=[('qkv','R',qbytes),('attention_output','W',qbytes)]
        calls.extend([call('fused_attention',attn),
            call('output_projection',[('attention_output','R',qbytes),('hidden','W',x)]),
            call('fused_add_rmsnorm',[('residual','R',x),('hidden','R',x),
                                      ('residual','W',x),('hidden','W',x)]),
            call('gate_up_projection',[('hidden','R',x),('gate_up','W',4*tokens*f)]),
            call('silu_and_mul',[('gate_up','R',4*tokens*f),('silu','W',2*tokens*f)]),
            call('down_projection',[('silu','R',2*tokens*f),('norm','W',x)])])
        # Producer-declared semantics, separate from display/operator identities.
        for item in calls:
            item['semantic'] = dict(kind='operation', requires_weights=[])
        calls[0]['semantic'] = dict(kind='input_norm', requires_weights=['attention/weights'])
        declared = {'kv_store_source':'kv_source', 'kv_contiguous_copy':'kv_copy',
                    'fused_attention':'attention'}
        for item in calls:
            if item['id'] in declared: item['semantic']['kind'] = declared[item['id']]
        post = next(j for j,c in enumerate(calls) if c['id']=='output_projection')+1
        calls[post]['semantic'] = dict(kind='post_norm', requires_weights=['ffn/weights'])
        layers.append(calls)
    tail=[call('final_add_rmsnorm',[('residual','R',x),('norm','R',x),
                                   ('residual','W',x),('norm','W',x)]),
          call('last_token_select',[('norm','R',2*h),('selected','W',2*h)]),
          call('lm_head',[('selected','R',2*h),('logits_16bit','W',2*model.vocab_size)]),
          call('logits_cast',[('logits_16bit','R',2*model.vocab_size),
                              ('logits_fp32','W',4*model.vocab_size)])]
    # Placeholder for the other phase is never selected; preserve v1 validation.
    p=dict(consumer_contract='model-semantic-v1',shape=[tokens,context],current_token_kv_read=phase=='decode',
           embedding=embedding,layers=layers,tail=tail)
    return CoarseCoverageProfile(dict(schema=SCHEMA,model_sha256=model.digest,
        request_trace_sha256=request_digest,layers=model.num_layers,
        workspace_bytes=cursor,objects=objects,phases={phase:p,('decode' if phase=='prefill' else 'prefill'):p},
        evidence=dict(policy=POLICY,assumptions=ASSUMPTIONS,structure=g.canonical(),
            geometry_source='explicit model structure' if model.structure else 'legacy qualified1.5B compatibility',traffic='analytic tensor footprint approximation')))


class AnalyticCoverageCompiler(CoarseCoverageCompiler):
    requires_causal_contract = True
    supports_roofline = True
    supports_linear = True
    supports_chunked_prefill = True

    def __init__(self, *, schedule_policy='legacy-prefetch', **kwargs):
        from serveloop.coarse_schedule import validate_schedule_policy
        self.schedule_policy = validate_schedule_policy(schedule_policy)
        models=kwargs['models']
        if len(models)!=1: raise ServeLoopError('analytic coverage currently supports one model')
        model=next(iter(models.values()))
        profile=build_profile(model,kwargs['request_trace'].digest,'prefill',1,0)
        super().__init__(coverage_profile=profile,**kwargs)

    def coverage_for_batch(self,batch):
        if len(batch.slices)!=1: raise ServeLoopError('analytic coverage currently supports B1 only')
        s=batch.slices[0]
        return build_profile(self.models[batch.model_id],self.request_trace.digest,
                             s.phase,s.token_count,s.context_tokens_before)
