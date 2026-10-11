"""Export a compact input-bound coverage profile from a qualified research ledger."""
import argparse
import hashlib
import json
from pathlib import Path

from serveloop.catalog import load_model_any
from serveloop.coarse_coverage import CoarseCoverageProfile, SCHEMA
from serveloop.io import load_request_trace


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('ledger','model','requests','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    if args.out.exists():
        parser.error('output exists; use a fresh path')
    ledger=json.loads(args.ledger.read_text());model=load_model_any(args.model)
    if (model.num_layers!=28 or model.vocab_size!=151936 or any(
            l.attention_weight_bytes!=11017216 or l.ffn_weight_bytes!=82578432 or
            l.kv_bytes_per_token!=1024 or l.is_moe for l in model.layers)):
        parser.error('this exporter currently supports the qualified28-layer SGLang ledger only')
    objects=sorted(ledger['object_offsets']);indices={n:i for i,n in enumerate(objects)}
    def convert(call):
        stages=call.get('kernel_stages',[dict(kernel=None,ranges=call['ranges'])])
        return dict(id=call['id'],operator=call['operator'],stages=[dict(kernel=s['kernel'],ranges=[
            [indices[r['object']],0 if r['op']=='R' else 1,r['offset'],r['bytes']] for r in s['ranges']]) for s in stages])
    profile=dict(schema=SCHEMA,model_sha256=model.digest,
        request_trace_sha256=load_request_trace(args.requests).digest,layers=model.num_layers,
        workspace_bytes=ledger['workspace_bytes'],objects=objects,phases={},
        evidence=dict(source_ledger_sha256=hashlib.sha256(args.ledger.read_bytes()).hexdigest(),
            domain='input-bound B1 P16 + one decode ctx17; no extrapolation',
            traffic='tensor/kernel footprints, not instruction repeats or GPU cache misses'))
    for phase,shape in [('prefill',[16,0]),('decode',[1,16])]:
        calls=ledger['phases'][phase]
        if len(calls)!=(259 if phase=='prefill' else 256) or calls[0]['operator']!='aten.embedding.default':
            parser.error('ledger does not match the qualified call structure')
        if calls[253]['operator']!='sgl_kernel.fused_add_rmsnorm.default':
            parser.error('ledger tail does not match the qualified implementation')
        profile['phases'][phase]=dict(shape=shape,current_token_kv_read=phase=='decode',
            embedding=[convert(calls[0])],layers=[[convert(c) for c in calls[1+9*l:1+9*(l+1)]]
                for l in range(model.num_layers)],tail=[convert(c) for c in calls[253:]])
    CoarseCoverageProfile(profile)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(profile,separators=(',',':'))+'\n')
    print(args.out)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
