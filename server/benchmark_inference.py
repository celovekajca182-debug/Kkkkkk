"""Measure the speedup from the SWIN Transformer's KV-cache on CPU."""
import argparse
import statistics
import time

import torch

from transformer import ByteTokenizer, load_transformer


@torch.inference_mode()
def generate_without_cache(model, prompt: str, max_new_tokens: int) -> str:
    """Reference greedy decoder that recomputes the full context each step."""
    token_ids = ByteTokenizer.encode(prompt) or [32]
    generated = []
    for _ in range(max_new_tokens):
        context = torch.tensor(
            [token_ids[-model.block_size:]],
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
        logits, _ = model(context)
        next_id = int(torch.argmax(logits[0, -1, :]).item())
        generated.append(next_id)
        token_ids.append(next_id)
    return ByteTokenizer.decode(generated)


def timed(callable_, repeats: int):
    measurements = []
    result = ""
    for _ in range(repeats):
        started = time.perf_counter()
        result = callable_()
        measurements.append(time.perf_counter() - started)
    return result, statistics.mean(measurements), statistics.stdev(measurements) if repeats > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Benchmark SWIN cached inference against full recomputation.")
    parser.add_argument("--checkpoint", default="models/swin-transformer.pt")
    parser.add_argument("--prompt", default="Пользователь: Как исправить ошибку Godot?\nSWIN:")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    if not 1 <= args.tokens <= 512 or not 1 <= args.repeats <= 10:
        parser.error("--tokens должен быть 1..512, --repeats — 1..10.")
    torch.set_num_threads(max(1, min(8, args.threads)))
    model = load_transformer(args.checkpoint).to("cpu").eval()

    # Warm up both code paths before timing.
    generate_without_cache(model, args.prompt, min(8, args.tokens))
    model.generate(args.prompt, max_new_tokens=min(8, args.tokens), temperature=0)

    reference, full_mean, full_std = timed(
        lambda: generate_without_cache(model, args.prompt, args.tokens),
        args.repeats,
    )
    cached, cached_mean, cached_std = timed(
        lambda: model.generate(
            args.prompt,
            max_new_tokens=args.tokens,
            temperature=0,
            stop_at_newline=False,
        ),
        args.repeats,
    )
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Prompt: {len(ByteTokenizer.encode(args.prompt))} UTF-8 bytes")
    print(f"Generated: {args.tokens} bytes")
    print(f"Full recompute: {full_mean:.4f}s ± {full_std:.4f}s ({args.tokens / full_mean:.1f} bytes/s)")
    print(f"KV cache:       {cached_mean:.4f}s ± {cached_std:.4f}s ({args.tokens / cached_mean:.1f} bytes/s)")
    print(f"Speedup:        {full_mean / max(cached_mean, 1e-9):.2f}x")
    print(f"Outputs match:  {reference == cached}")
    if reference != cached:
        raise SystemExit("Cached greedy output differs from full recomputation.")


if __name__ == "__main__":
    main()
