from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic diverse PartiPrompts splits")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", default="configs/prompts/parti")
    parser.add_argument("--train-size", type=int, default=96)
    parser.add_argument("--heldout-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=3505)
    return parser.parse_args()


def _balanced_order(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        strata[(row["Category"], row["Challenge"])].append(row)
    queues = {}
    for key, values in strata.items():
        rng.shuffle(values)
        queues[key] = deque(values)
    keys = sorted(queues)
    rng.shuffle(keys)
    result = []
    while queues:
        for key in list(keys):
            queue = queues.get(key)
            if queue:
                result.append(queue.popleft())
            if not queue:
                queues.pop(key, None)
                keys.remove(key)
    return result


def _counts(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(row[field] for row in rows).items()))


def main() -> int:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    raw = source.read_bytes()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"Prompt", "Category", "Challenge"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"PartiPrompts TSV must contain {sorted(required)}")
    unique = {}
    for row in rows:
        prompt = row["Prompt"].strip()
        if prompt:
            unique.setdefault(prompt, row)
    ordered = _balanced_order(list(unique.values()), args.seed)
    needed = args.train_size + args.heldout_size
    if args.train_size <= 0 or args.heldout_size < 32 or len(ordered) < needed:
        raise ValueError("Need positive training size, at least 32 held-out prompts, and enough rows")
    heldout = ordered[: args.heldout_size]
    train = ordered[args.heldout_size : needed]
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "train.txt").write_text(
        "\n".join(row["Prompt"].strip() for row in train) + "\n", encoding="utf-8"
    )
    (output / "heldout.txt").write_text(
        "\n".join(row["Prompt"].strip() for row in heldout) + "\n", encoding="utf-8"
    )
    manifest = {
        "source": source.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_rows": len(rows),
        "unique_prompts": len(unique),
        "seed": args.seed,
        "method": "round-robin over Category x Challenge strata",
        "train": {
            "size": len(train),
            "categories": _counts(train, "Category"),
            "challenges": _counts(train, "Challenge"),
        },
        "heldout": {
            "size": len(heldout),
            "categories": _counts(heldout, "Category"),
            "challenges": _counts(heldout, "Challenge"),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
