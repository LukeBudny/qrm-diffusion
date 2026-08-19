from __future__ import annotations

import argparse
import json
from pathlib import Path

from qrm_diffusion.agents.diagnostics import summarize_policy_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize timestep-policy evaluation diagnostics")
    parser.add_argument("--evaluation-jsonl", required=True)
    parser.add_argument("--prompt-metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--update-index", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    return parser.parse_args()


def _read_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    args = parse_args()
    records = _read_jsonl(args.evaluation_jsonl)
    if args.repeat is not None:
        records = [record for record in records if int(record.get("repeat", 0)) == args.repeat]
    if args.update_index is not None:
        records = [
            record
            for record in records
            if int(record.get("update_index", -1)) == args.update_index
        ]
    if args.prompt_metadata:
        metadata = {
            int(item["prompt_index"]): item for item in _read_jsonl(args.prompt_metadata)
        }
        for record in records:
            item = metadata.get(int(record["prompt_index"]), {})
            record.setdefault("prompt", item.get("prompt"))
            record.setdefault("category", item.get("category"))
            record.setdefault("challenge", item.get("challenge"))
    summary = summarize_policy_records(
        records,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_confidence=args.bootstrap_confidence,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["reward"], indent=2))
    print(json.dumps(summary["critic"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
