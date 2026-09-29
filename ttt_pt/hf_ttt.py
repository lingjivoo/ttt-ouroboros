"""TTT-ize a pretrained HF model: frozen backbone + meta-learned prime fast
weights in the last quarter of layers, trained with the TTT-E2E objective.

Design:
  - Backbone frozen entirely; outer-trainable params = prime SwiGLU MLPs
    (+ their pre/post RMSNorms) inserted between attention and MLP of each
    suffix layer, exactly TTT-E2E's placement.
  - prime.w2 is ZERO-INITIALIZED, so at init the wrapped model reproduces the
    base model's logits bit-for-bit — the correctness test is exact equality.
  - Prefix (first 3/4 layers) runs under no_grad with a sliding-window
    attention mask: outer gradients only need the suffix graph.
  - Suffix layers are recomputed from HF submodules (input_layernorm,
    self_attn projections, post_attention_layernorm, mlp) chunk-by-chunk with
    a rolling per-layer KV cache (absolute-position RoPE, crop-to-window),
    prime applied between attention and MLP.

Currently supports the Llama family layout (Qwen3/Qwen2.5/Llama/SmolLM/OLMo-2:
q/k/v/o_proj + optional q_norm/k_norm, gate/up/down_proj). Gemma adapter TBD.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rms(x, weight, eps):
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight.float()).to(dt)


class PrimeMLP(nn.Module):
    """Fast-weight SwiGLU with zero-init w2 and its own pre/post norms."""

    def __init__(self, d, inter, eps):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(d, inter) * 0.02)
        self.w2 = nn.Parameter(torch.zeros(inter, d))
        self.w3 = nn.Parameter(torch.randn(d, inter) * 0.02)
        self.pre = nn.Parameter(torch.ones(d))
        self.post = nn.Parameter(torch.ones(d))
        self.eps = eps


def prime_apply(x, w1, w2, w3, pre, post, eps):
    """x [B, T, D]; w* may carry a leading batch dim (per-sequence fast weights)."""
    h = rms(x, pre, eps)
    cd = x.dtype
    if w1.dim() == 3:
        z = torch.einsum("btd,bdi->bti", h.to(cd), w1.to(cd))
        z3 = torch.einsum("btd,bdi->bti", h.to(cd), w3.to(cd))
        o = torch.einsum("bti,bid->btd", F.silu(z) * z3, w2.to(cd))
    else:
        o = (F.silu(h.to(cd) @ w1.to(cd)) * (h.to(cd) @ w3.to(cd))) @ w2.to(cd)
    return x + rms(o, post, eps)


class HFTTT(nn.Module):
    def __init__(
        self,
        model_name,
        suffix_frac=0.25,
        prime_inter=None,
        window=2048,
        chunk=512,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, attn_implementation="sdpa"
        )
        for p in self.base.parameters():
            p.requires_grad_(False)
        cfgb = self.base.config
        self.L = cfgb.num_hidden_layers
        self.n_suffix = max(1, int(self.L * suffix_frac))
        self.n_prefix = self.L - self.n_suffix
        self.d = cfgb.hidden_size
        self.H = cfgb.num_attention_heads
        self.KV = getattr(cfgb, "num_key_value_heads", self.H)
        self.hd = getattr(cfgb, "head_dim", self.d // self.H)
        self.eps = getattr(cfgb, "rms_norm_eps", 1e-6)
        self.window = window
        self.chunk = chunk
        pi = prime_inter or self.d
        self.primes = nn.ModuleList(
            PrimeMLP(self.d, pi, self.eps).to(torch.float32) for _ in range(self.n_suffix)
        )
        # rope from the base model
        self.rotary = self.base.model.rotary_emb

    # ---- fast-weight plumbing (mirrors TTTModel API) ----
    def prime_params(self):
        out = []
        for p in self.primes:
            out.extend([p.w1, p.w2, p.w3])
        return out

    def init_fast_weights(self, B):
        return [w.unsqueeze(0).repeat(B, 1, 1) for w in self.prime_params()]

    def _layers(self):
        return self.base.model.layers

    # ---- attention on one suffix layer with rolling cache ----
    def _suffix_attn(self, layer, h, cache, pos0):
        B, T, _ = h.shape
        a = layer.self_attn
        q = a.q_proj(h).view(B, T, -1, self.hd)
        k = a.k_proj(h).view(B, T, -1, self.hd)
        v = a.v_proj(h).view(B, T, -1, self.hd)
        if hasattr(a, "q_norm"):
            q = rms(q, a.q_norm.weight, self.eps)
            k = rms(k, a.k_norm.weight, self.eps)
        pos = torch.arange(pos0, pos0 + T, device=h.device)[None]
        cos, sin = self.rotary(v, pos)

        # HF llama rope: half-split convention
        def rope(x, cos, sin):
            x1, x2 = x[..., : self.hd // 2], x[..., self.hd // 2 :]
            rot = torch.cat((-x2, x1), -1)
            return (x * cos.unsqueeze(2) + rot * sin.unsqueeze(2)).to(x.dtype)

        q = rope(q, cos, sin)
        k = rope(k, cos, sin)
        pk, pv, ppos = cache  # [B, S, KV, hd] x2, positions [S]
        k_all = torch.cat([pk, k], 1) if pk is not None else k
        v_all = torch.cat([pv, v], 1) if pv is not None else v
        pos_all = torch.cat([ppos, pos[0]]) if ppos is not None else pos[0]
        # GQA expand
        rep = self.H // k_all.shape[2]
        ke = k_all.repeat_interleave(rep, 2) if rep > 1 else k_all
        ve = v_all.repeat_interleave(rep, 2) if rep > 1 else v_all
        qpos = pos[0][:, None]
        kpos = pos_all[None, :]
        mask = (qpos >= kpos) & (qpos - kpos < self.window)
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), ke.transpose(1, 2), ve.transpose(1, 2), attn_mask=mask[None, None]
        )
        o = o.transpose(1, 2).reshape(B, T, -1)
        o = a.o_proj(o)
        keep = min(self.window, k_all.shape[1])
        new_cache = (k_all[:, -keep:].detach(), v_all[:, -keep:].detach(), pos_all[-keep:].detach())
        return o, new_cache

    def init_kv_caches(self, B, device):
        return [(None, None, None) for _ in range(self.n_suffix)]

    # ---- forward paths ----
    @torch.no_grad()
    def prefix_forward(self, input_ids):
        """Frozen prefix under a sliding-window mask. [B, T, D]"""
        m = self.base.model
        B, T = input_ids.shape
        x = m.embed_tokens(input_ids)
        pos = torch.arange(T, device=input_ids.device)[None]
        idx = torch.arange(T, device=input_ids.device)
        mask4 = ((idx[:, None] >= idx[None, :]) & (idx[:, None] - idx[None, :] < self.window))[
            None, None
        ]
        pe = self.rotary(x, pos)
        for i in range(self.n_prefix):
            out = self._layers()[i](
                x, attention_mask=mask4, position_ids=pos, position_embeddings=pe
            )
            x = out[0] if isinstance(out, tuple) else out
        return x

    def suffix_chunk_forward(self, h_chunk, fast, kv_caches, chunk_id):
        """[B, CS, D] through suffix layers with prime fast weights.
        Returns (logits fp32, new_caches)."""
        m = self.base.model
        pos0 = chunk_id * self.chunk
        x = h_chunk
        new_caches = []
        for j in range(self.n_suffix):
            layer = self._layers()[self.n_prefix + j]
            hn = rms(x, layer.input_layernorm.weight, self.eps)
            a, cache = self._suffix_attn(layer, hn, kv_caches[j], pos0)
            x = x + a
            p = self.primes[j]
            w1, w2, w3 = fast[3 * j : 3 * j + 3]
            x = prime_apply(x, w1, w2, w3, p.pre, p.post, self.eps)
            hn = rms(x, layer.post_attention_layernorm.weight, self.eps)
            x = x + layer.mlp(hn)
            new_caches.append(cache)
        x = rms(x, m.norm.weight, self.eps)
        logits = self.base.lm_head(x)
        return logits.float(), new_caches
