from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from serveloop.catalog import convert_descriptor, main as convert_main
from serveloop.coarse_analytic import AnalyticCoverageCompiler, build_profile
from serveloop.contracts import DenseModelStructure, ModelSpec, ServeLoopError
from serveloop.io import load_request_trace
from serveloop.sglang.model import dense_model

ROOT=Path(__file__).resolve().parents[1]


def config(h=3584,f=18944,q=28,k=4,d=128,n=28):
    return dict(model_type='qwen2',hidden_size=h,intermediate_size=f,
                num_attention_heads=q,num_key_value_heads=k,head_dim=d,
                num_hidden_layers=n,vocab_size=152064,tie_word_embeddings=False)


class StructureTests(unittest.TestCase):
    def test_legacy_canonical_digest_and_structure_roundtrip(self):
        from serveloop.catalog import load_model_any
        model=load_model_any(ROOT/'examples/coarse-coverage/model.json')
        self.assertNotIn('structure',model.canonical())
        self.assertEqual(model.digest,ModelSpec.from_dict(model.canonical()).digest)
        g=DenseModelStructure(1536,8960,12,2,128,'bfloat16','bfloat16',128)
        enriched=replace(model,structure=g)
        self.assertEqual(enriched,ModelSpec.from_dict(enriched.canonical()))
        self.assertNotEqual(model.digest,enriched.digest)
        a=build_profile(model,'a'*64,'prefill',16,0)
        b=build_profile(enriched,'a'*64,'prefill',16,0)
        self.assertEqual(a.phases,b.phases)
        self.assertEqual(a.workspace_bytes,b.workspace_bytes)

    def test_gqa_mha_and_nonstandard_query_width_formulas(self):
        for q,k,d,h in [(28,4,128,3584),(16,16,64,1024),(8,2,96,1024)]:
            model=dense_model(config(h,2048,q,k,d,2),'bfloat16','bfloat16',include_structure=True)
            p=build_profile(model,'a'*64,'prefill',128,0)
            ranges=p.phases['prefill']['layers'][0][1]['stages'][0]['ranges']
            self.assertEqual([r[3] for r in ranges],[128*h*2,128*(q+2*k)*d*2])
            out=next(c for c in p.phases['prefill']['layers'][0] if c['id']=='output_projection')
            self.assertEqual([r[3] for r in out['stages'][0]['ranges']],[128*q*d*2,128*h*2])

    def test_fp16_partial_rotary_and_rope_table_dtype(self):
        hf=config();hf.update(partial_rotary_factor=.5,rope_table_dtype='float16')
        model=dense_model(hf,'float16','float16',include_structure=True)
        p=build_profile(model,'a'*64,'prefill',16,0)
        rope=next(c for c in p.phases['prefill']['layers'][0] if c['id']=='rope')
        self.assertEqual(rope['stages'][0]['ranges'][1][3],16*64*2)
        self.assertEqual(p.document['evidence']['structure']['dtype'],'float16')

    def test_unsupported_and_inconsistent_structure_rejected(self):
        g=dense_model(config(),'bfloat16','bfloat16',include_structure=True).structure
        for change in (dict(dtype='float32'),dict(kv_dtype='float16'),dict(num_key_value_heads=3),
                       dict(rotary_dim=129),dict(mlp='gelu'),dict(hidden_size=True)):
            with self.assertRaises(ServeLoopError):replace(g,**change)
        model=dense_model(config(),'bfloat16','bfloat16',include_structure=True)
        with self.assertRaises(ServeLoopError):replace(model,structure=replace(g,num_key_value_heads=7))
        hf=config();hf['num_experts']=4
        with self.assertRaises(ValueError):dense_model(hf,'bfloat16','bfloat16',include_structure=True)

    def test_catalog_qk_norm_and_hf_cli_import(self):
        model=convert_descriptor(ROOT/'models/qwen3-8b-bf16-kv-bf16.json')
        self.assertEqual(model.structure.hidden_size,4096)
        p=build_profile(model,'a'*64,'decode',1,16)
        self.assertTrue(any(c['id']=='qk_head_norm' for c in p.phases['decode']['layers'][0]))
        with tempfile.TemporaryDirectory() as temp:
            cfg=Path(temp)/'config.json';out=Path(temp)/'model.json'
            cfg.write_text(json.dumps(config()))
            self.assertEqual(convert_main([str(cfg),'--hf-config','--dtype','float16','--output',str(out)]),0)
            rebuilt=ModelSpec.from_dict(json.loads(out.read_text()))
            self.assertEqual(rebuilt.structure.dtype,'float16')
            self.assertEqual(rebuilt.num_layers,28)
            trace=load_request_trace(ROOT/'examples/coarse-coverage/requests.json')
            AnalyticCoverageCompiler(models={rebuilt.model_id:rebuilt},request_trace=trace)


if __name__=='__main__':unittest.main()
