# from pathlib import Path
# import torch, json

# OUT_DIR = Path("cached_ir_prompts")
# LOGICAL_BATCH_SIZE = 133  # the intended prompts-per-file
# prompts = [rec["text"] for rec in json.load(open("annotations/refl_data.json","r"))]
# prompts = list(dict.fromkeys(prompts))  # same dedupe you used

# def check_batch(batch_id: int) -> bool:
#     f = OUT_DIR / f"batch_{batch_id:05d}.pt"
#     if not f.exists():
#         print(f"missing {f}")
#         return False
#     blob = torch.load(f, map_location="cpu")            # {prompt: {...}}
#     stored_prompts = list(blob.keys())
#     i0 = batch_id * LOGICAL_BATCH_SIZE
#     expected_slice = prompts[i0 : i0 + LOGICAL_BATCH_SIZE]
#     ok = stored_prompts == expected_slice
#     if not ok:
#         # Print a tiny diff to see the misalignment
#         print(f"[MISMATCH] batch {batch_id}:")
#         print(" first stored:", stored_prompts[:1])
#         print(" first expect:", expected_slice[:1])
#         print(" count stored/expect:", len(stored_prompts), len(expected_slice))
#     return ok

# # Check a small window around where you resumed, e.g. 20..36
# for bid in range(20, 37):
#     check_batch(bid)

import re, torch, json
from pathlib import Path

OUT_DIR = Path("cached_ir_prompts")
OUT_DIR_FIX = Path("cached_ir_prompts_fixed")
OUT_DIR_FIX.mkdir(parents=True, exist_ok=True)

LOGICAL_BATCH_SIZE = 133
FIRST_BAD = 20

# Load deduped prompts exactly as you did for precompute
prompts = [rec["text"] for rec in json.load(open("annotations/refl_data.json","r"))]
prompts = list(dict.fromkeys(prompts))

# Build a prompt -> (file_path) index for shards >= FIRST_BAD (lightweight)
rx = re.compile(r"batch_(\d{5})\.pt$")
files = [f for f in sorted(OUT_DIR.glob("batch_*.pt")) if int(rx.fullmatch(f.name).group(1)) >= FIRST_BAD]

prompt2file = {}
for f in files:
    blob = torch.load(f, map_location="cpu")
    for p in blob.keys():
        prompt2file[p] = f  # last wins; fine for our case

# Re-shard into correct logical batches
i = FIRST_BAD * LOGICAL_BATCH_SIZE
bid = FIRST_BAD
while i < len(prompts):
    slice_prompts = prompts[i : i + LOGICAL_BATCH_SIZE]
    if not slice_prompts:
        break
    batch = {}
    # Lazy load per-file to keep RAM low
    cache = {}
    for p in slice_prompts:
        src = prompt2file.get(p)
        if src is None:
            raise RuntimeError(f"Missing cond for prompt starting at {i}: {p[:80]}…")
        if src not in cache:
            cache[src] = torch.load(src, map_location="cpu")
        batch[p] = cache[src][p]
    tmp = OUT_DIR_FIX / f"batch_{bid:05d}.tmp"
    final = OUT_DIR_FIX / f"batch_{bid:05d}.pt"
    torch.save(batch, tmp)
    tmp.replace(final)
    i += LOGICAL_BATCH_SIZE
    bid += 1
