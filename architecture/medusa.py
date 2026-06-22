"""
Medusa Speculative Decoding — parallel draft heads for faster autoregressive generation.

Attaches K lightweight "Medusa heads" to a CausalLM model. Each head predicts
a future token from the same hidden state, allowing tree-based verification
of multiple candidate continuations in a single forward pass.

Usage:
    base_model = CausalLM(config)
    medusa = MedusaModel(base_model, num_heads=3)
    output = medusa_generate(medusa, idx, ...)
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedusaHead(nn.Module):
    """
    Single Medusa draft head.

    Architecture: LayerNorm → Linear(emb_dim, emb_dim) → SiLU → Linear(emb_dim, vocab_size)

    Takes the last hidden state from the base model and predicts the next token
    at a specific offset (e.g., head 0 predicts token t+2, head 1 predicts t+3, etc.).
    """

    def __init__(self, emb_dim: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.linear1 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(emb_dim, vocab_size, bias=False)

        # Initialize with small weights so heads start near-random
        nn.init.normal_(self.linear1.weight, std=0.01)
        nn.init.normal_(self.linear2.weight, std=0.01)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, D) — last hidden layer output from base model

        Returns:
            logits: (B, T, vocab_size)
        """
        x = self.norm(hidden_states)
        x = self.act(self.linear1(x))
        return self.linear2(x)


class MedusaModel(nn.Module):
    """
    Wraps a CausalLM with K Medusa heads for speculative decoding.

    During generation:
      1. Base model produces hidden states + token t+1 prediction
      2. Head 0 predicts t+2, Head 1 predicts t+3, ..., Head K-1 predicts t+K+1
      3. All K+1 candidates are verified in a single forward pass

    Args:
        base_model: A CausalLM model (will not be modified)
        num_heads: Number of Medusa heads (default 3)
        vocab_size: Vocabulary size (auto-detected from base_model if None)
        emb_dim: Embedding dimension (auto-detected from base_model if None)
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_heads: int = 3,
        vocab_size: Optional[int] = None,
        emb_dim: Optional[int] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_heads = num_heads

        # Auto-detect dimensions from base model config
        if vocab_size is None:
            vocab_size = base_model.config.vocab_size
        if emb_dim is None:
            emb_dim = base_model.config.emb_dim

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        # Create Medusa heads
        self.medusa_heads = nn.ModuleList([
            MedusaHead(emb_dim, vocab_size) for _ in range(num_heads)
        ])

    def get_hidden_states(
        self,
        idx: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the base model and return both logits and last hidden states.

        This requires access to the intermediate hidden states before the lm_head.
        We run the base model's components manually to extract them.
        """
        base = self.base_model
        raw = base._orig_mod if hasattr(base, "_orig_mod") else base

        B, T = idx.shape
        x = raw.tok_emb(idx)

        if raw.pos_emb is not None:
            if position_ids is not None:
                x = x + raw.pos_emb(position_ids)
            else:
                positions = torch.arange(T, device=idx.device)
                x = x + raw.pos_emb(positions)

        x = raw.emb_dropout(x)

        for block in raw.blocks:
            if use_cache:
                x, _ = block(x, position_ids=position_ids, use_cache=True)
            else:
                x = block(x, position_ids=position_ids)

        hidden_states = raw.final_norm(x)
        logits = raw.lm_head(hidden_states)

        return logits, hidden_states

    def forward(
        self,
        idx: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass returning base model logits + all Medusa head logits.

        Returns:
            base_logits: (B, T, vocab_size)
            head_logits: list of K tensors, each (B, T, vocab_size)
        """
        base_logits, hidden_states = self.get_hidden_states(
            idx, position_ids=position_ids, use_cache=use_cache
        )

        head_logits = [head(hidden_states) for head in self.medusa_heads]
        return base_logits, head_logits

    def save_heads(self, path: str):
        """Save only the Medusa heads (not the base model)."""
        torch.save({
            "num_heads": self.num_heads,
            "vocab_size": self.vocab_size,
            "emb_dim": self.emb_dim,
            "state_dict": self.medusa_heads.state_dict(),
        }, path)

    def load_heads(self, path: str, map_location: str = "cpu"):
        """Load pre-trained Medusa heads."""
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        self.medusa_heads.load_state_dict(ckpt["state_dict"])


# ===========================================================================
#  Medusa generation loop
# ===========================================================================

def medusa_generate(
    medusa_model: MedusaModel,
    idx: torch.Tensor,
    max_new_tokens: int = 100,
    context_size: int = 2048,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    eos_id: Optional[int] = None,
    metrics: Optional[dict] = None,
) -> torch.Tensor:
    """
    Speculative decoding using Medusa heads.

    Strategy:
      1. Run base model + Medusa heads → get K+1 candidate next tokens
      2. Construct verification batch: [token_0], [token_0, token_1], ...
      3. Run base model to verify all candidates in parallel
      4. Accept the longest prefix that matches the base model's predictions
      5. Advance by accepted_length tokens instead of just 1

    For simplicity, this implementation uses a flat (non-tree) candidate strategy
    where each Medusa head greedily picks its top token. A tree-based strategy
    would be more efficient but significantly more complex.

    Args:
        medusa_model: MedusaModel wrapping a CausalLM
        idx: (1, seq_len) input token ids
        max_new_tokens: number of tokens to generate
        Other args: same as standard generate()

    Returns:
        (1, seq_len + generated) token ids
    """
    import time

    medusa_model.eval()
    base = medusa_model.base_model
    raw_base = base._orig_mod if hasattr(base, "_orig_mod") else base

    if hasattr(raw_base, "reset_caches"):
        raw_base.reset_caches()

    B, T_prompt = idx.shape
    device = idx.device
    num_heads = medusa_model.num_heads

    # Pre-create sampling helpers
    eos_tensor = torch.tensor([eos_id], device=device) if eos_id is not None else None

    def _sample_token(logits: torch.Tensor) -> torch.Tensor:
        """Sample a single token from logits (B, vocab)."""
        if temperature > 0.0:
            logits = logits / temperature
        if top_k is not None:
            top_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            min_val = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)
        if temperature > 0.0:
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(logits, dim=-1, keepdim=True)

    generated = 0
    t0 = time.perf_counter()
    ttft = 0.0

    with torch.no_grad():
        while generated < max_new_tokens:
            step_start = time.perf_counter()

            # Step 1: Run base model + Medusa heads
            base_logits, head_logits_list = medusa_model(
                idx[:, -context_size:],
                use_cache=False,
            )

            # Get the base model's next token prediction
            base_next_logits = base_logits[:, -1, :]  # (B, vocab)

            if repetition_penalty != 1.0:
                unique_tokens = idx[0].unique()
                score = base_next_logits[0, unique_tokens]
                score = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
                base_next_logits[0, unique_tokens] = score

            base_token = _sample_token(base_next_logits)  # (B, 1)

            if generated == 0:
                ttft = time.perf_counter() - step_start
                if metrics is not None:
                    metrics["ttft"] = ttft

            # EOS check
            if eos_tensor is not None and (base_token == eos_tensor).any():
                break

            # Step 2: Get Medusa head predictions (greedy for simplicity)
            draft_tokens = [base_token]  # t+1 from base model
            for head_logits in head_logits_list:
                head_next = head_logits[:, -1, :]  # (B, vocab)
                draft_token = _sample_token(head_next)
                draft_tokens.append(draft_token)  # t+2, t+3, ...

            # Step 3: Verify draft tokens
            # Construct input: [base_token, draft_1, draft_2, ...]
            candidate_seq = torch.cat(draft_tokens, dim=1)  # (B, K+1)
            verify_input = torch.cat([idx, candidate_seq], dim=1)

            # Run verification through base model only
            verify_logits = raw_base(verify_input[:, -context_size:])
            if isinstance(verify_logits, tuple):
                verify_logits = verify_logits[0]

            # Step 4: Check which draft tokens match base model's predictions
            # The base model's prediction for position i should match draft token i+1
            accepted = 1  # We always accept the base model's own prediction
            verify_start = verify_input.shape[1] - candidate_seq.shape[1] - 1

            for k in range(len(draft_tokens) - 1):
                # Base model's prediction at position where draft_token[k] was input
                verify_pos = verify_start + k + 1
                verify_token_logits = verify_logits[:, verify_pos, :]
                verified_token = _sample_token(verify_token_logits)

                if (verified_token == draft_tokens[k + 1]).all():
                    accepted += 1
                else:
                    # Replace the rejected token with the base model's prediction
                    draft_tokens[k + 1] = verified_token
                    break

            # Step 5: Append accepted tokens
            accepted_tokens = torch.cat(draft_tokens[:accepted], dim=1)  # (B, accepted)

            # EOS check in accepted tokens
            if eos_tensor is not None:
                eos_mask = (accepted_tokens == eos_tensor)
                if eos_mask.any():
                    eos_pos = eos_mask.nonzero(as_tuple=True)[1][0].item()
                    accepted_tokens = accepted_tokens[:, :eos_pos]
                    idx = torch.cat([idx, accepted_tokens], dim=1)
                    generated += accepted_tokens.shape[1]
                    break

            idx = torch.cat([idx, accepted_tokens], dim=1)
            generated += accepted

    total_time = time.perf_counter() - t0
    if metrics is not None:
        metrics["decode_time"] = max(0.0, total_time - ttft)
        metrics["accepted_per_step"] = generated / max(1, generated)  # avg acceptance rate

    return idx
