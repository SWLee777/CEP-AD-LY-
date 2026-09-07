import os
from typing import Union, List, Any
from pkg_resources import packaging
import torch
import numpy as np
from AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F

_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])


class VisualTokenAdapter(nn.Module):
    """
    Cross-attention adapter that injects image information into the
    learnable visual tokens. Turns a static, image-agnostic token into
    an image-conditional one by attending over all patch features.

    Query  = visual token   (anomaly_vis_token or normal_vis_token)
    Key/Val= patch features (current image, all patches)

    The output is added back to the base token via a learnable scale,
    so training starts close to the original parameter and gradually
    incorporates image evidence.
    """
    def __init__(self, embed_dim=768, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_norm = nn.LayerNorm(embed_dim)
        self.k_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        # Start near-identity so the adapter doesn't disturb early training
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, vis_token, patch_features):
        """
        vis_token:      [C]      learnable visual token
        patch_features: [N, C]   patch features (no CLS) for current sample
        Returns:        [C]      image-conditional version of vis_token
        """
        C = patch_features.shape[1]
        Q = self.q_norm(self.q_proj(vis_token.view(1, 1, C)))           # [1, 1, C]
        K = self.k_norm(self.k_proj(patch_features.unsqueeze(0)))       # [1, N, C]
        V = self.v_proj(patch_features.unsqueeze(0))                    # [1, N, C]

        attn = torch.bmm(Q, K.transpose(1, 2)) * (C ** -0.5)            # [1, 1, N]
        attn = self.dropout(attn.softmax(dim=-1))
        ctx = torch.bmm(attn, V).view(C)                                # [C]
        ctx = self.out_proj(ctx)

        # Residual update: image-conditional adjustment on the base token
        return vis_token + self.res_scale * ctx


class MultiScaleVisualTokenAdapter(nn.Module):
    """
    Multi-scale wrapper around VisualTokenAdapter.

    Runs 4 independent branches in parallel:
        branch 0 : original patches, no aggregation        (37x37 = 1369 tokens)
        branch 1 : 2x2 non-overlapping block average pool  (18x18 =  324 tokens)
        branch 2 : 3x3 non-overlapping block average pool  (12x12 =  144 tokens)
        branch 3 : 4x4 non-overlapping block average pool  (9x9  =   81 tokens)

    "Non-overlapping" => different blocks never reuse the same patch
    (kernel_size = stride = k in avg_pool2d).

    Each branch has its own learnable anomaly_vis_token / normal_vis_token
    and its own VisualTokenAdapter (independent Q/K/V projections), so each
    scale can learn a different "what-to-look-for" semantic.

    Outputs from the 4 branches are fused with FIXED branch weights
    [0.7, 0.1, 0.1, 0.1] — the original (no-aggregation) branch dominates;
    the three coarser scales each contribute 0.1. Stored as a buffer, so
    it lives in state_dict but is not optimised.
    """
    SCALES = (1, 2, 3, 4)  # 1 means "no aggregation"

    def __init__(self, embed_dim=768, dropout=0.1,
                 init_weights=(0.7, 0.1, 0.1, 0.1)):
        super().__init__()
        assert len(init_weights) == len(self.SCALES)
        self.num_branches = len(self.SCALES)
        # Independent visual tokens per branch
        self.anomaly_vis_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim) * 0.02) for _ in self.SCALES
        ])
        self.normal_vis_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim) * 0.02) for _ in self.SCALES
        ])

        self.adapters = nn.ModuleList([
            VisualTokenAdapter(embed_dim=embed_dim, dropout=dropout)
            for _ in self.SCALES
        ])
        # Fixed (non-learnable) branch fusion weights. Registered as a buffer
        # so they move with .to(device) and are saved in state_dict, but the
        # optimiser will not touch them.
        self.register_buffer(
            "branch_weights",
            torch.tensor(list(init_weights), dtype=torch.float32),
        )

    @staticmethod
    def _aggregate(patch_features, k):
        """
        Non-overlapping kxk block average pooling on a square patch grid.
        patch_features: [N, C] with N = h*h.
        Returns:        [N', C] with N' = floor(h/k)^2.
        """
        if k == 1:
            return patch_features
        N, C = patch_features.shape
        h = int(round(N ** 0.5))
        assert h * h == N, f"expected square patch grid, got N={N}"
        x = patch_features.t().contiguous().view(1, C, h, h)
        x = F.avg_pool2d(x, kernel_size=k, stride=k)         # non-overlapping
        h2 = x.shape[-1]
        return x.view(1, C, h2 * h2).squeeze(0).t().contiguous()

    def forward(self, patch_features):
        """
        patch_features: [N, C] all patches of one sample (no CLS token).
        Returns:
            anomaly_dyn: [C]  weighted combination across 4 scales
            normal_dyn:  [C]  weighted combination across 4 scales
        """
        anomaly_outs, normal_outs = [], []
        for i, k in enumerate(self.SCALES):
            agg = self._aggregate(patch_features, k)
            anomaly_outs.append(self.adapters[i](self.anomaly_vis_tokens[i], agg))
            normal_outs.append(self.adapters[i](self.normal_vis_tokens[i],  agg))
        a_stack = torch.stack(anomaly_outs, dim=0)            # [B, C]
        n_stack = torch.stack(normal_outs,  dim=0)            # [B, C]
        w = self.branch_weights.view(-1, 1)                   # [B, 1]
        return (w * a_stack).sum(dim=0), (w * n_stack).sum(dim=0)


class _SingleSCAExpert(nn.Module):
    """One SCA expert: anchor-query cross-attention over all patches."""
    def __init__(self, embed_dim=768, num_anchors=8, max_patches=1369, dropout=0.1):
        super().__init__()
        self.anchor_queries = nn.Parameter(torch.randn(num_anchors, embed_dim) * 0.02)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_patches, embed_dim) * 0.02)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patch_features, fg_score=None, fg_beta=0.0):
        N, C = patch_features.shape
        patch_feat = patch_features.unsqueeze(0)  # [1, N, C]
        Q = self.anchor_queries.unsqueeze(0)  # [1, num_anchors, C]
        K = self.k_norm(self.k_proj(patch_feat + self.pos_encoding[:, :N, :]))  # [1, N, C]
        V = self.v_proj(patch_feat)  # [1, N, C]
        attn = torch.bmm(Q, K.transpose(1, 2)) * (C ** -0.5)  # [1, num_anchors, N]
        if fg_score is not None and fg_beta > 0:
            attn = attn + fg_beta * fg_score.unsqueeze(0).unsqueeze(0)
        attn = self.dropout(attn.softmax(dim=-1))
        out = torch.bmm(attn, V).mean(dim=1)  # [1, C]
        return self.out_proj(out)  # [1, C]


class MoeSCAEnhancedDAP(nn.Module):
    """
    Mixture-of-Experts SCA for Data-dependent Abnormality Prior.

    4 homogeneous SCA experts are soft-combined via a router driven by the
    learnable visual anomaly token. Exposes gate_probs and expert_outputs for
    ETF loss and load-balance loss in the training loop.
    """
    def __init__(self, embed_dim=768, num_experts=4, num_anchors=8, max_patches=1369, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            _SingleSCAExpert(embed_dim, num_anchors, max_patches, dropout)
            for _ in range(num_experts)
        ])
        self.router = nn.Linear(embed_dim, num_experts, bias=False)

    def forward(self, vis_token, patch_features, fg_score=None, fg_beta=0.0):
        """
        vis_token:      [C]    gate signal (learnable anomaly token)
        patch_features: [N, C] all patch features without CLS token
        fg_score:       [N]    gradient foreground score (optional)
        fg_beta:        float  foreground attention bias weight

        Returns:
            bias:           [1, C]            soft-weighted combination of experts
            gate_probs:     [num_experts]     routing weights (sum to 1)
            expert_outputs: [num_experts, C]  per-expert outputs (for ETF loss)
        """
        gate_probs = F.softmax(self.router(vis_token.unsqueeze(0)), dim=-1).squeeze(0)  # [num_experts]
        expert_out_list = [expert(patch_features, fg_score, fg_beta) for expert in self.experts]
        expert_outputs = torch.cat(expert_out_list, dim=0)  # [num_experts, C]
        bias = (gate_probs.unsqueeze(-1) * expert_outputs).sum(dim=0, keepdim=True)  # [1, C]
        return bias, gate_probs, expert_outputs


class CEP(nn.Module):
    def __init__(self, clip_model, design_details):
        super().__init__()
        classnames = ["object"]
        self.n_cls = len(classnames)
        n_ctx_pos = 5
        n_ctx_neg = 2
        self.num_p = 10
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"]
        dtype = clip_model.transformer.get_cast_dtype()

        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.classnames = classnames

        # --- Visual tokens (VisualAD-inspired) ---
        # Replaced by MultiScaleVisualTokenAdapter below: each of the 4
        # branches owns its own anomaly_vis_token / normal_vis_token.

        # --- Multi-scale cross-attention adapter ---
        # 4 parallel branches: no-aggregation + 2x2 + 3x3 + 4x4 non-overlapping
        # block average pooling. Each branch has its own visual tokens and
        # adapter; the per-branch outputs are fused with learnable weights
        # initialised to [0.7, 0.1, 0.1, 0.1].
        self.vis_token_adapter = MultiScaleVisualTokenAdapter(embed_dim=ctx_dim)

        # --- MoE-SCA DAP (4 experts, soft-weighted by anomaly token router) ---
        self.moe_sca_dap = MoeSCAEnhancedDAP(embed_dim=ctx_dim)
        self._gate_probs = None             # [num_experts] — set each forward pass
        self._expert_outputs = None         # [num_experts, C] — set each forward pass
        self._anomaly_vis_dynamic = None    # [C] — image-conditional anomaly token
        self._normal_vis_dynamic  = None    # [C] — image-conditional normal token

        # --- Visual token image-level classification head (D2) ---
        self.vis_cls_head = nn.Linear(ctx_dim, 1)

        # Random Initialization
        print("Initializing class-specific contexts")
        ctx_vectors_pos = torch.empty(self.n_cls, 1, n_ctx_pos, ctx_dim, dtype=dtype)
        ctx_vectors_neg = torch.empty(self.n_cls, self.num_p, n_ctx_neg, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors_pos, std=0.02)
        nn.init.normal_(ctx_vectors_neg, std=0.02)
        prompt_prefix_pos = " ".join(["N"] * n_ctx_pos)
        prompt_prefix_neg = " ".join(["A"] * n_ctx_neg)
        self.compound_prompts_depth = design_details["learnabel_text_embedding_depth"]
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                       for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            print("single_para", single_para.shape)
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(ctx_dim, 896)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        prompts_pos = [prompt_prefix_pos + " " + name + "." for name in classnames]
        prompts_neg = [prompt_prefix_pos + " " + prompt_prefix_neg + " " + "damaged" + " " + name + "." for _ in
                       range(self.num_p) for name in classnames]

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []

        for p_pos in prompts_pos:
            tokenized_prompts_pos.append(tokenize(p_pos))
        for p_neg in prompts_neg:
            tokenized_prompts_neg.append(tokenize(p_neg))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)

        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            n, l, d = embedding_pos.shape
            print("embedding_pos", embedding_pos.shape)
            embedding_pos = embedding_pos.reshape(1, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(self.num_p, self.n_cls, l, d).permute(1, 0, 2, 3)

        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :])
        self.register_buffer("token_suffix_pos", embedding_pos[:, :, 1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx_pos + n_ctx_neg:, :])

        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(1, self.n_cls, d).permute(1, 0, 2)

        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(self.num_p, self.n_cls, d).permute(1, 0, 2)

        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        self.vis_dim = ctx_dim
        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)
        print("tokenized_prompts shape", self.tokenized_prompts_pos.shape, self.tokenized_prompts_neg.shape)

    def forward(self, patch_features=None, fg_score=None, fg_beta=0.0):
        """
        patch_features: [N, C] all patch features for one sample (without CLS), or None
        fg_score:       [N]    gradient foreground score for one sample, or None
        fg_beta:        float  weight for foreground attention bias
        """
        ctx_pos = self.ctx_pos
        prefix_pos = self.token_prefix_pos
        suffix_pos = self.token_suffix_pos

        prompts_pos = torch.cat(
            [
                prefix_pos,   # (n_cls, 1, dim)
                ctx_pos,      # (n_cls, n_ctx, dim)
                suffix_pos,   # (n_cls, *, dim)
            ],
            dim=2,
        )

        ctx_neg = self.ctx_neg
        prefix_neg = self.token_prefix_neg
        suffix_neg = self.token_suffix_neg

        ctx_pos2 = ctx_pos.expand(-1, self.num_p, -1, -1).reshape(-1, self.num_p, self.n_ctx_pos, self.vis_dim)

        bias = 0
        if patch_features is not None:
            # Multi-scale image-conditional visual tokens.
            # Each of 4 branches (1x1 / 2x2 / 3x3 / 4x4 non-overlapping block
            # pool) runs its own VisualTokenAdapter; the outputs are fused with
            # learnable branch weights (init [0.7, 0.1, 0.1, 0.1]).
            anomaly_vis_dyn, normal_vis_dyn = self.vis_token_adapter(patch_features)
            self._anomaly_vis_dynamic = anomaly_vis_dyn
            self._normal_vis_dynamic  = normal_vis_dyn

            # MoE-SCA DAP: router is now driven by the image-aware anomaly token
            bias, self._gate_probs, self._expert_outputs = self.moe_sca_dap(
                anomaly_vis_dyn,
                patch_features,
                fg_score=fg_score,
                fg_beta=fg_beta
            )  # bias: [1, C]
            ctx_neg = ctx_neg + bias

        prompts_neg = torch.cat(
            [
                prefix_neg,   # (n_cls, 1, dim)
                ctx_pos2,
                ctx_neg,      # (n_cls, n_ctx, dim)
                suffix_neg,   # (n_cls, *, dim)
            ],
            dim=2,
        )

        _, _, l, d = prompts_pos.shape
        prompts_pos = prompts_pos.reshape(-1, l, d)
        _, _, l, d = prompts_neg.shape
        prompts_neg = prompts_neg.reshape(-1, l, d)

        _, l, d = self.tokenized_prompts_pos.shape
        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1, d)
        _, l, d = self.tokenized_prompts_neg.shape
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1, d)

        return prompts_pos, prompts_neg, tokenized_prompts_pos, tokenized_prompts_neg, self.compound_prompts_text, bias
