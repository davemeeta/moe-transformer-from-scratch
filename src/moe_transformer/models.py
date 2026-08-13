"""Dense GPT baseline: token embedding, stacked transformer blocks, tied output head.

This is the comparison point for the MoE model built in step 5 -- same
primitives (RMSNorm, RoPE attention, SwiGLU block), just without a router,
so the dense-vs-MoE comparison in step 8 isolates what routing actually buys.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_transformer.config import ModelConfig
from moe_transformer.init import init_weights
from moe_transformer.model.block import MoEBlock, TransformerBlock
from moe_transformer.model.norm import RMSNorm


class DenseGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.final_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the input embedding and output projection share one
        # tensor. Same table for "which vector represents token X" both when
        # reading a token in and when scoring which token to predict.
        self.lm_head.weight = self.token_emb.weight

        self.apply(init_weights)
        self._scale_residual_projections()

    def _scale_residual_projections(self) -> None:
        """GPT-2 init trick: scale each block's residual-stream writes by
        1/sqrt(2 * n_layer), so activation variance doesn't grow with depth.
        Every block writes into the residual stream twice (attention
        out_proj, FFN down_proj), hence the factor of 2.
        """
        scale = 1.0 / math.sqrt(2 * self.config.n_layer)
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                with torch.no_grad():
                    param.mul_(scale)

    def get_num_params(self, exclude_embedding: bool = False) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            n_params -= self.token_emb.weight.numel()
        return n_params

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )

        x = self.dropout(self.token_emb(idx))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train(was_training)
        return idx


class MoEGPT(nn.Module):
    """Same skeleton as DenseGPT, with every block's FFN replaced by a
    routed MoELayer (MoEBlock). config.expert_ffn_hidden_dim (see
    config.py) sizes each expert so top_k active experts do roughly the
    same FFN compute as DenseGPT's single FFN at the same n_embd/n_layer --
    more total parameters, matched active parameters per token.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(MoEBlock(config) for _ in range(config.n_layer))
        self.final_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(init_weights)
        self._scale_residual_projections()

    def _scale_residual_projections(self) -> None:
        # Same GPT-2 init trick as DenseGPT. This also catches every
        # expert's down_proj (name pattern "...experts.N.down_proj.weight"
        # still matches), which is correct: each expert writes into the
        # residual stream just like a dense FFN does.
        scale = 1.0 / math.sqrt(2 * self.config.n_layer)
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                with torch.no_grad():
                    param.mul_(scale)

    def get_num_params(self, exclude_embedding: bool = False) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            n_params -= self.token_emb.weight.numel()
        return n_params

    def get_active_num_params(self, exclude_embedding: bool = False) -> int:
        """Params actually used for one token's forward pass: every dense
        param (attention, router, embedding) plus only top_k/num_experts of
        the expert FFN params, not all of them."""
        expert_param_count = sum(p.numel() for p in self.blocks[0].moe.experts[0].parameters())
        total_expert_params = expert_param_count * self.config.num_experts * self.config.n_layer
        active_expert_params = expert_param_count * self.config.top_k * self.config.n_layer
        n_params = self.get_num_params(exclude_embedding=exclude_embedding)
        return n_params - total_expert_params + active_expert_params

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )

        x = self.dropout(self.token_emb(idx))

        lb_loss = x.new_zeros(())
        z_loss = x.new_zeros(())
        num_dropped_tokens = 0
        for block in self.blocks:
            x, moe_out = block(x)
            lb_loss = lb_loss + moe_out.load_balancing_loss
            z_loss = z_loss + moe_out.z_loss
            num_dropped_tokens += moe_out.num_dropped_tokens
        n_layer = len(self.blocks)
        lb_loss = lb_loss / n_layer
        z_loss = z_loss / n_layer

        x = self.final_norm(x)
        logits = self.lm_head(x)

        ce_loss = None
        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = (
                ce_loss
                + self.config.aux_loss_weight * lb_loss
                + self.config.router_z_loss_weight * z_loss
            )

        aux = {
            "ce_loss": ce_loss,
            "load_balancing_loss": lb_loss,
            "z_loss": z_loss,
            "num_dropped_tokens": num_dropped_tokens,
        }
        return logits, loss, aux

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train(was_training)
        return idx
