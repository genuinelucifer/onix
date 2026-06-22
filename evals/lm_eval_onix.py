#!/usr/bin/env python3
"""
lm_eval wrapper for Onix CausalLM models.

Allows running EleutherAI's lm-evaluation-harness on models trained
with the Onix framework, loaded from native .pt checkpoints.

Usage:
    lm_eval --model onix \
        --model_args model_name=llama1b-8192-v6,device=cuda:0,dtype=bfloat16 \
        --tasks blimp \
        --batch_size 16

    # Or from Python:
    import lm_eval
    results = lm_eval.simple_evaluate(
        model="onix",
        model_args="model_name=llama1b-8192-v6,device=cuda:0",
        tasks=["blimp"],
    )
"""

import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator

# Add Onix root to path so we can import architecture + model
_ONIX_ROOT = Path(__file__).resolve().parent.parent
if str(_ONIX_ROOT) not in sys.path:
    sys.path.insert(0, str(_ONIX_ROOT))

from architecture import ModelConfig, CausalLM
from architecture.generate import generate as onix_generate
from model import load_model_config, MODELS_DIR

# GPT-2 BPE EOT token id
EOT_TOKEN_ID = 50256


@register_model("onix")
class OnixLM(TemplateLM):
    """lm_eval wrapper for Onix CausalLM models."""

    backend = "causal"

    @classmethod
    def create_from_arg_obj(cls, arg_dict: dict, additional_config: dict | None = None) -> "OnixLM":
        additional_config = additional_config or {}
        # Merge dictionaries safely, letting additional_config (parsed command CLI args) take precedence
        merged = {**arg_dict}
        for k, v in additional_config.items():
            if v is not None:
                merged[k] = v
        return cls(**merged)

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        batch_size: int | str = 1,
        max_gen_toks: int = 256,
        checkpoint_tag: str = "latest",
    ):
        super().__init__()

        self._model_name = model_name
        self._device = torch.device(device)
        self._dtype = getattr(torch, dtype, torch.bfloat16)
        self._batch_size = int(batch_size) if str(batch_size).isdigit() else batch_size
        self._max_gen_toks = max_gen_toks

        # Load tokenizer (tiktoken GPT-2 BPE — same as training)
        self._tokenizer = tiktoken.get_encoding("gpt2")

        # Load model config from models/<model_name>/config.json
        full_cfg = load_model_config(model_name)
        arch_cfg = full_cfg["architecture"]
        self._model_config = ModelConfig.from_dict(arch_cfg)

        # Build model and load checkpoint (directly on target device to save RAM)
        print(f"[OnixLM] Loading model '{model_name}' on {device} ({dtype})")
        model = CausalLM(self._model_config)

        # Load checkpoint
        model_dir = MODELS_DIR / model_name
        ckpt_path = model_dir / f"checkpoint_{checkpoint_tag}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)
        state_dict = ckpt["model_state_dict"]

        # Strip torch.compile prefix if present
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.to(device=self._device, dtype=self._dtype)
        model.eval()

        self._model = model
        self._max_length = self._model_config.context_length

        print(f"[OnixLM] Model loaded: {self._model_config.name}")
        print(f"[OnixLM]   params: {model.param_count():,}")
        print(f"[OnixLM]   context: {self._max_length}")
        print(f"[OnixLM]   device: {self._device}, dtype: {self._dtype}")

    # ------------------------------------------------------------------
    #  Properties required by TemplateLM
    # ------------------------------------------------------------------

    @property
    def eot_token_id(self) -> int:
        return EOT_TOKEN_ID

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def device(self):
        return self._device

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    # ------------------------------------------------------------------
    #  Tokenization
    # ------------------------------------------------------------------

    def tok_encode(self, string: str, add_special_tokens: bool | None = None, **kwargs) -> list[int]:
        """Tokenize a string using tiktoken GPT-2 BPE."""
        return self._tokenizer.encode(string, allowed_special={"<|endoftext|>"})

    def tok_decode(self, tokens, skip_special_tokens: bool = True) -> str:
        """Decode token ids back to string."""
        if isinstance(tokens, int):
            tokens = [tokens]
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self._tokenizer.decode(tokens)

    # ------------------------------------------------------------------
    #  Core model calls
    # ------------------------------------------------------------------

    def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return logits (batch, seq, vocab)."""
        with torch.no_grad(), torch.autocast(
            device_type=self._device.type, dtype=self._dtype
        ):
            logits = self._model(inps)
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int | None = None,
    ) -> list[tuple[float, bool]]:
        """Score (context, continuation) pairs.

        For each request we get (context_enc, continuation_enc) token lists.
        We concatenate them, run through the model, and extract log-probs
        for the continuation tokens.
        """
        res = []

        # Sort by descending length for efficient batching
        def _collate(req):
            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(requests, sort_fn=_collate, group_by=None)

        # Resolve batch size: "auto" isn't handled by Collator, use a sensible default
        if override_bs is not None:
            bs = override_bs
        elif isinstance(self._batch_size, int):
            bs = self._batch_size
        else:
            # batch_size="auto" — default to 32 (safe for most GPU memory configs)
            bs = 32

        chunks = re_ord.get_batched(n=bs)

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        )

        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            padding_len = 0

            for _, context_enc, continuation_enc in chunk:
                # Truncate from the left if too long
                full = (context_enc + continuation_enc)[-(self._max_length + 1):][:-1]
                inp = torch.tensor(full, dtype=torch.long, device=self._device)
                inplen = inp.shape[0]

                inps.append(inp)
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)
                padding_len = max(padding_len, inplen)

            # Pad to same length (right-pad with EOT)
            batched = torch.full(
                (len(inps), padding_len),
                EOT_TOKEN_ID,
                dtype=torch.long,
                device=self._device,
            )
            for i, inp in enumerate(inps):
                batched[i, : inp.shape[0]] = inp

            # Forward pass
            logits = self._model_call(batched)
            log_probs = F.log_softmax(logits, dim=-1)

            for (request_str, ctx_tokens, _), lp, inplen, cont_toks in zip(
                chunk, log_probs, inplens, cont_toks_list
            ):
                contlen = len(cont_toks)
                # Extract log-probs for continuation tokens
                # logits at position i predict token i+1, so continuation
                # log-probs are at positions [inplen-contlen : inplen]
                cont_lp = lp[inplen - contlen : inplen]  # (contlen, vocab)

                # Check greedy
                greedy_tokens = cont_lp.argmax(dim=-1)
                cont_toks_tensor = torch.tensor(
                    cont_toks, dtype=torch.long, device=self._device
                )
                max_equal = (greedy_tokens == cont_toks_tensor).all().item()

                # Gather log-probs for actual continuation tokens
                cont_log_probs = torch.gather(
                    cont_lp, 1, cont_toks_tensor.unsqueeze(-1)
                ).squeeze(-1)

                answer = (float(cont_log_probs.sum()), bool(max_equal))
                res.append(answer)

                if request_str is not None:
                    self.cache_hook.add_partial("loglikelihood", request_str, answer)

                pbar.update(1)

        pbar.close()
        return re_ord.get_original(res)

    # ------------------------------------------------------------------
    #  Rolling log-likelihood (for perplexity tasks)
    # ------------------------------------------------------------------

    def loglikelihood_rolling(
        self, requests, disable_tqdm: bool = False
    ) -> list[float]:
        loglikelihoods = []

        for (string,) in tqdm(
            [req.args for req in requests],
            disable=disable_tqdm,
            desc="Running rolling loglikelihood",
        ):
            token_list = self.tok_encode(string)

            # Build rolling windows
            rolling_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=token_list,
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self._max_length,
                        context_len=1,
                    ),
                )
            )

            # Score each window
            windows_as_requests = [(None, ctx, cont) for ctx, cont in rolling_windows]
            window_results = self._loglikelihood_tokens(
                windows_as_requests,
                disable_tqdm=True,
                override_bs=len(windows_as_requests),
            )

            total_ll = sum(ll for ll, _ in window_results)
            loglikelihoods.append(total_ll)

            self.cache_hook.add_partial(
                "loglikelihood_rolling", (string,), total_ll
            )

        return loglikelihoods

    # ------------------------------------------------------------------
    #  Text generation (for generative tasks)
    # ------------------------------------------------------------------

    def generate_until(
        self, requests, disable_tqdm: bool = False
    ) -> list[str]:
        res = []

        for request in tqdm(requests, disable=disable_tqdm, desc="Running generate_until"):
            context, gen_kwargs = request.args
            if isinstance(gen_kwargs, dict):
                until = gen_kwargs.get("until", [])
                max_gen = gen_kwargs.get("max_gen_toks", self._max_gen_toks)
                temperature = gen_kwargs.get("temperature", 0.0)
                top_k = gen_kwargs.get("top_k", None)
                top_p = gen_kwargs.get("top_p", None)
                do_sample = gen_kwargs.get("do_sample", False)
            else:
                until = []
                max_gen = self._max_gen_toks
                temperature = 0.0
                top_k = None
                top_p = None
                do_sample = False

            if isinstance(until, str):
                until = [until]

            # Encode context
            context_enc = self.tok_encode(context)

            # Truncate context to leave room for generation
            max_ctx = self._max_length - max_gen
            if len(context_enc) > max_ctx:
                context_enc = context_enc[-max_ctx:]

            idx = torch.tensor(
                [context_enc], dtype=torch.long, device=self._device
            )

            # Use Onix generate (without KV cache for simplicity, works for all cases)
            with torch.no_grad(), torch.autocast(
                device_type=self._device.type, dtype=self._dtype
            ):
                gen_ids = onix_generate(
                    self._model,
                    idx,
                    max_new_tokens=max_gen,
                    context_size=self._max_length,
                    temperature=temperature if (temperature > 0 and do_sample) else 0.0,
                    top_k=top_k,
                    top_p=top_p,
                    eos_id=EOT_TOKEN_ID,
                    use_kv_cache=False,
                )

            # Decode only the generated part
            gen_tokens = gen_ids[0, len(context_enc):].tolist()
            gen_text = self.tok_decode(gen_tokens)

            # Truncate at first stop sequence
            for stop_seq in until:
                if stop_seq in gen_text:
                    gen_text = gen_text[:gen_text.index(stop_seq)]

            res.append(gen_text)
            self.cache_hook.add_partial(
                "generate_until", (context, gen_kwargs), gen_text
            )

        return res
