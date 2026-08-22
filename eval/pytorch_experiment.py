"""PyTorch 소규모 실험 — 2-layer MLP on synthetic data + optional Transformers adapter.

Dry-run: imports 검증 + 1 forward pass + config 로그 (무거운 학습 스킵).
Full run: 100 steps synthetic 분류, loss 로그, out/pytorch_experiment_log.json 저장.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import random
import sys
import time


def _reconfigure_utf8() -> None:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def build_mlp(input_dim: int = 16, hidden: int = 32, output_dim: int = 2):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, output_dim),
    )


def synthetic_batch(batch_size: int = 16, input_dim: int = 16, seed: int = 42):
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    x = torch.randn(batch_size, input_dim, generator=g)
    # Simple rule: label = 1 if sum(x) > 0 else 0
    y = (x.sum(dim=1) > 0).long()
    return x, y


def train_loop(
    epochs: int = 2,
    steps_per_epoch: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
):
    import torch
    import torch.nn as nn

    _reconfigure_utf8()
    random.seed(seed)

    torch.manual_seed(seed)
    model = build_mlp()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    logs: list[dict] = []
    best_loss = float("inf")
    start = time.perf_counter()

    total_steps = epochs * steps_per_epoch
    for step in range(total_steps):
        x, y = synthetic_batch(batch_size=batch_size, seed=seed + step)
        x, y = x.to(device), y.to(device)
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        if loss_val < best_loss:
            best_loss = loss_val

        # Log every 10 steps and on last
        if step % 10 == 0 or step == total_steps - 1:
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            entry = {
                "step": step + 1,
                "epoch": step // steps_per_epoch + 1,
                "loss": round(loss_val, 4),
                "acc": round(acc, 4),
                "lr": lr,
            }
            logs.append(entry)
            print(f"step {entry['step']}/{total_steps} epoch {entry['epoch']} loss {entry['loss']:.4f} acc {entry['acc']:.4f}")

    elapsed = time.perf_counter() - start
    print(f"Training done: {total_steps} steps, best_loss {best_loss:.4f}, elapsed {elapsed:.1f}s")
    return logs, best_loss


def dry_run(seed: int = 42) -> dict:
    _reconfigure_utf8()
    print("[pytorch_experiment] --dry-run: verifying imports and single forward pass")
    try:
        import torch

        print(f"  torch {torch.__version__} available, device cpu")
    except ImportError as e:
        print(f"  torch import failed: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    try:
        import transformers

        print(f"  transformers {transformers.__version__} available")
    except ImportError:
        print("  transformers not installed (optional, continuing)")

    # Single forward
    import torch.nn as nn  # noqa: F401

    model = build_mlp()
    x, y = synthetic_batch(batch_size=4, seed=seed)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y).item()
    print(f"  dry-run forward: input {tuple(x.shape)} -> logits {tuple(logits.shape)} loss {loss:.4f}")
    print("  dry-run PASS — imports ok, forward ok, no training executed")

    result = {
        "mode": "dry-run",
        "torch_version": torch.__version__,
        "input_shape": list(x.shape),
        "logits_shape": list(logits.shape),
        "loss": round(float(loss), 4),
        "seed": seed,
    }
    return result


def main() -> None:
    _reconfigure_utf8()
    parser = argparse.ArgumentParser(description="PyTorch tiny experiment (MLP on synthetic)")
    parser.add_argument("--dry-run", action="store_true", help="Verify imports + single forward, skip training")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=50, help="Steps per epoch")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="out/pytorch_experiment_log.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.dry_run:
        result = dry_run(seed=args.seed)
        # Also write dry-run log for traceability
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Dry-run log → {out_path}")
        return

    logs, best = train_loop(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "train",
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": args.device,
        "best_loss": round(float(best), 4),
        "logs": logs,
    }
    # Also capture torch/transformers versions
    try:
        import torch

        payload["torch_version"] = torch.__version__
    except Exception:
        pass
    try:
        import transformers

        payload["transformers_version"] = transformers.__version__
    except Exception:
        pass

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Experiment log → {out_path}")


if __name__ == "__main__":
    main()
