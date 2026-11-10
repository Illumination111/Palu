import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from palu.rank_search import allocate_ranks, kv_modules
from palu.model.modules.svd_linear import _per_head_whiten_decomposition_from_weight

torch.manual_seed(42)
# Freezing non-target weights preserves K/V gradients.
m = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
       num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, attention_dropout=0.)).eval()
x = torch.randint(0,32,(1,12))
loss = m(x, labels=x).loss
manual = torch.nn.functional.cross_entropy(m(x).logits[:,:-1].reshape(-1,32), x[:,1:].reshape(-1))
assert torch.allclose(loss, manual)
loss.backward()
expected = {n:v.weight.grad.clone() for n,v in kv_modules(m).items()}
m.zero_grad(set_to_none=True); m.requires_grad_(False)
for v in kv_modules(m).values(): v.weight.requires_grad_(True)
m(x,labels=x).loss.backward()
assert all(torch.allclose(v.weight.grad,expected[n],atol=1e-7) for n,v in kv_modules(m).items())
# Triangular solve orientation and full-rank whitening reconstruction.
w = torch.randn(8,16)
a = torch.randn(16,32); chol = torch.linalg.cholesky(a@a.T)
l,r = _per_head_whiten_decomposition_from_weight(w,chol,8)
assert torch.allclose(l@r,w,atol=1e-5)
for ratio in (.5,.7,1.):
    ranks=allocate_ranks([0,1,100,1e-8]*16,512,ratio)
    assert all(32 <= r <= 512 and r%32==0 for r in ranks)
    assert sum(ranks)==round(64*16*ratio)*32
print(json.dumps({'passed':True,'checks':['HF single shift','frozen K/V gradient equivalence','whiten full-rank reconstruction','bounded integer rank allocation']},indent=2))
