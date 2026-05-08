#!/usr/bin/env python3
"""CLI wrapper for NB3 logic — trains a DPO adapter.

Usage:
    python scripts/train_dpo.py
    python scripts/train_dpo.py --beta 0.05 --output-dir adapters/dpo-b0.05
    python scripts/train_dpo.py --beta 0.5  --output-dir adapters/dpo-b0.50

Mirrors `notebooks/03_dpo_train.py`. Used by `make beta-sweep` for the rigor
add-on +6 (β-sweep mini-experiment).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def patch_xformers_attention_for_t4(torch_module):
    """Use PyTorch SDPA when xformers has no T4 backward kernel for GQA tensors."""
    major, minor = torch_module.cuda.get_device_capability()
    if major >= 8:
        return

    import torch.nn.functional as F

    def sdpa_attention(query, key, value, attn_bias=None, p=0.0, scale=None, op=None, output_dtype=None):
        original_shape = query.shape
        if query.ndim == 5:
            # xformers BMGHK -> PyTorch (B*G, H, M, K)
            bsz, q_len, groups, heads, head_dim = query.shape
            kv_len = key.shape[1]
            query_4d = query.permute(0, 2, 3, 1, 4).reshape(bsz * groups, heads, q_len, head_dim)
            key_4d = key.permute(0, 2, 3, 1, 4).reshape(bsz * groups, heads, kv_len, head_dim)
            value_4d = value.permute(0, 2, 3, 1, 4).reshape(bsz * groups, heads, kv_len, head_dim)
            out = F.scaled_dot_product_attention(
                query_4d,
                key_4d,
                value_4d,
                attn_mask=None,
                dropout_p=p,
                is_causal=attn_bias is not None,
                scale=scale,
            )
            out = out.reshape(bsz, groups, heads, q_len, head_dim).permute(0, 3, 1, 2, 4)
        elif query.ndim == 4:
            # xformers BMHK -> PyTorch (B, H, M, K)
            bsz, q_len, heads, head_dim = query.shape
            query_4d = query.permute(0, 2, 1, 3)
            key_4d = key.permute(0, 2, 1, 3)
            value_4d = value.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(
                query_4d,
                key_4d,
                value_4d,
                attn_mask=None,
                dropout_p=p,
                is_causal=attn_bias is not None,
                scale=scale,
            ).permute(0, 2, 1, 3)
            out = out.reshape(bsz, q_len, heads, head_dim)
        else:
            raise NotImplementedError(f"Unsupported attention rank for SDPA fallback: {query.ndim}")

        if output_dtype is not None:
            out = out.to(output_dtype)
        return out.reshape(original_shape)

    import xformers.ops

    xformers.ops.memory_efficient_attention = sdpa_attention
    if hasattr(xformers.ops, "fmha"):
        xformers.ops.fmha.memory_efficient_attention = sdpa_attention
    try:
        import unsloth.utils.attention_dispatch as attention_dispatch

        attention_dispatch.xformers_attention = sdpa_attention
    except Exception:
        pass

    print(f"Patched xformers attention fallback for CUDA capability {major}.{minor} (T4-compatible SDPA).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--sft-path", default=str(REPO / "adapters" / "sft-mini"))
    parser.add_argument("--pref-path", default=str(REPO / "data" / "pref" / "train.parquet"))
    parser.add_argument("--output-dir", default=str(REPO / "adapters" / "dpo"))
    args = parser.parse_args()

    tier = os.environ.get("COMPUTE_TIER", "T4").upper()
    if tier == "T4":
        base_model = "unsloth/Qwen2.5-3B-bnb-4bit"
        max_len, max_prompt = 512, 256
        batch, grad_accum = 1, 8
    else:
        base_model = "unsloth/Qwen2.5-7B-bnb-4bit"
        max_len, max_prompt = 1024, 512
        batch, grad_accum = 1, 4

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    report_to = "wandb" if use_wandb else "none"
    run_name = f"lab22-dpo-b{args.beta}-{tier.lower()}"

    print(f"Tier:       {tier}")
    print(f"Base:       {base_model}")
    print(f"Beta / LR:  {args.beta} / {args.lr}")
    print(f"Report to:  {report_to}")
    print(f"Output:     {output}")

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from trl import DPOConfig, DPOTrainer
    from unsloth import FastLanguageModel

    patch_xformers_attention_for_t4(torch)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model, max_seq_length=max_len, dtype=None, load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, args.sft_path, is_trainable=True)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42, use_rslora=False, loftq_config=None,
    )

    config = DPOConfig(
        output_dir=str(output.parent / f"{output.name}-checkpoints"),
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=max_len,
        max_prompt_length=max_prompt,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        optim="adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        seed=42,
        loss_type="sigmoid",
        report_to=report_to,
        run_name=run_name,
    )

    pref = Dataset.from_parquet(args.pref_path)
    trainer = DPOTrainer(
        model=model, ref_model=None, args=config,
        train_dataset=pref, tokenizer=tokenizer,
    )
    train_result = trainer.train()

    trainer.model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))

    # Headline metrics
    import pandas as pd

    logs = pd.DataFrame(trainer.state.log_history)
    chosen_col = "rewards/chosen" if "rewards/chosen" in logs.columns else None
    rejected_col = "rewards/rejected" if "rewards/rejected" in logs.columns else None

    metrics = {
        "compute_tier": tier,
        "base_model": base_model,
        "beta": args.beta,
        "lr": args.lr,
        "epochs": args.epochs,
        "final_train_loss": float(train_result.training_loss),
        "end_chosen_reward": float(logs[chosen_col].iloc[-5:].mean()) if chosen_col else None,
        "end_rejected_reward": float(logs[rejected_col].iloc[-5:].mean()) if rejected_col else None,
    }
    if metrics["end_chosen_reward"] is not None and metrics["end_rejected_reward"] is not None:
        metrics["end_reward_gap"] = metrics["end_chosen_reward"] - metrics["end_rejected_reward"]

    (output / "dpo_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nFinal loss:     {train_result.training_loss:.4f}")
    if "end_reward_gap" in metrics:
        print(f"End reward gap: {metrics['end_reward_gap']:+.3f}")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
