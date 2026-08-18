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
    parser.add_argument("--validation-size", type=int, default=48)
    parser.add_argument("--heldout-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=3505)
    parser.add_argument(
        "--exclude-file",
        action="append",
        default=[],
        help="Prompt text file to exclude; may be supplied more than once",
    )
    parser.add_argument(
        "--ensure-axis-coverage",
        action="store_true",
        help="Ensure every available Category and Challenge appears in each split",
    )
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


def _write_split(output: Path, name: str, rows: list[dict[str, str]]) -> None:
    (output / f"{name}.txt").write_text(
        "\n".join(row["Prompt"].strip() for row in rows) + "\n", encoding="utf-8"
    )
    with (output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            handle.write(
                json.dumps(
                    {
                        "prompt_index": index,
                        "prompt": row["Prompt"].strip(),
                        "category": row["Category"].strip(),
                        "challenge": row["Challenge"].strip(),
                    }
                )
                + "\n"
            )


def _manifest_split(rows: list[dict[str, str]]) -> dict:
    return {
        "size": len(rows),
        "categories": _counts(rows, "Category"),
        "challenges": _counts(rows, "Challenge"),
    }


def _take_split(
    ordered: list[dict[str, str]], size: int, *, ensure_axis_coverage: bool
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    if ensure_axis_coverage:
        missing_categories = {row["Category"] for row in ordered}
        missing_challenges = {row["Challenge"] for row in ordered}
        while missing_categories or missing_challenges:
            best_index = max(
                range(len(ordered)),
                key=lambda index: (
                    int(ordered[index]["Category"] in missing_categories)
                    + int(ordered[index]["Challenge"] in missing_challenges),
                    -index,
                ),
            )
            row = ordered.pop(best_index)
            selected.append(row)
            missing_categories.discard(row["Category"])
            missing_challenges.discard(row["Challenge"])
            if len(selected) > size:
                raise ValueError("Split is too small to cover all category/challenge axes")
    needed = size - len(selected)
    selected.extend(ordered[:needed])
    del ordered[:needed]
    return selected


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
    excluded_prompts = set()
    for value in args.exclude_file:
        excluded_prompts.update(
            line.strip()
            for line in Path(value).expanduser().resolve().read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    available = [
        row for prompt, row in unique.items() if prompt not in excluded_prompts
    ]
    ordered = _balanced_order(available, args.seed)
    needed = args.train_size + args.validation_size + args.heldout_size
    if (
        args.train_size <= 0
        or args.validation_size <= 0
        or args.heldout_size < 32
        or len(ordered) < needed
    ):
        raise ValueError("Need positive training size, at least 32 held-out prompts, and enough rows")
    heldout = _take_split(
        ordered, args.heldout_size, ensure_axis_coverage=args.ensure_axis_coverage
    )
    train = _take_split(
        ordered, args.train_size, ensure_axis_coverage=args.ensure_axis_coverage
    )
    validation = _take_split(
        ordered, args.validation_size, ensure_axis_coverage=args.ensure_axis_coverage
    )
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_split(output, "train", train)
    _write_split(output, "validation", validation)
    _write_split(output, "heldout", heldout)
    manifest = {
        "source": source.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_rows": len(rows),
        "unique_prompts": len(unique),
        "excluded_prompts": len(excluded_prompts),
        "exclude_files": [Path(value).name for value in args.exclude_file],
        "available_prompts": len(available),
        "seed": args.seed,
        "method": (
            "round-robin over Category x Challenge strata with axis coverage"
            if args.ensure_axis_coverage
            else "round-robin over Category x Challenge strata"
        ),
        "train": _manifest_split(train),
        "validation": _manifest_split(validation),
        "heldout": _manifest_split(heldout),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
