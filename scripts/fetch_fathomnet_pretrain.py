#!/usr/bin/env python3
"""
Build a marine image-CLASSIFICATION dataset from FathomNet for pretraining your
model, organized as a torchvision ImageFolder:  out/train/<class>/*.jpg  +  out/val/...

Each saved image is a CROP of one annotated organism (using the bounding boxes),
so it matches the cropped single-organism style of AQUA20 -- your fine-tuning target.
Pretrain on this, then fine-tune on AQUA20.

Examples
--------
# auto-pick the 60 most-populous FathomNet concepts, up to 800 imgs each
python fetch_fathomnet_pretrain.py --out-root /scratch/fathomnet_cls --top-k 60 --max-per-class 800

# use your own concept list (one concept per line)
python fetch_fathomnet_pretrain.py --out-root /scratch/fathomnet_cls --concepts-file concepts.txt
"""
import argparse, io, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from fathomnet.api import images, boundingboxes
try:
    from fathomnet.dto import GeoImageConstraints          # current package layout
except Exception:                                          # pragma: no cover
    from fathomnet.models import GeoImageConstraints       # older fallback


def safe_name(c):
    return c.strip().replace("/", "-").replace(" ", "_")


def pick_concepts(args):
    if args.concepts_file:
        cs = [l.strip() for l in open(args.concepts_file) if l.strip()]
        print(f"{len(cs)} concepts from file")
        return cs
    print("Fetching concept counts to pick top-K (one API call)...")
    counts = [c for c in boundingboxes.count_total_by_concept() if c.count >= args.min_count]
    counts.sort(key=lambda c: c.count, reverse=True)
    cs = [c.concept for c in counts[:args.top_k]]
    if counts:
        print(f"Top {len(cs)} concepts by #boxes; largest = {counts[0].concept} ({counts[0].count})")
    return cs


def fetch_records(concept, max_per_class):
    try:
        return images.find(GeoImageConstraints(concept=concept, limit=max_per_class))
    except Exception as e:
        print(f"  [warn] {concept}: query failed ({e})")
        return []


def download_bytes(session, url, retries=3):
    for k in range(retries):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            return r.content
        except Exception:
            time.sleep(1.5 * (k + 1))
    return None


def process_record(session, rec, concept, out_dir, min_box):
    """Download one image, crop every box matching `concept`, save crops. Returns #saved."""
    data = download_bytes(session, rec.url)
    if data is None:
        return 0
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return 0
    W, H = im.size
    saved = 0
    for j, b in enumerate(rec.boundingBoxes or []):
        if (b.concept or "").lower() != concept.lower():
            continue
        if None in (b.x, b.y, b.width, b.height):
            continue
        if b.width < min_box or b.height < min_box:
            continue
        x0, y0 = max(0, int(b.x)), max(0, int(b.y))
        x1, y1 = min(W, int(b.x + b.width)), min(H, int(b.y + b.height))
        if x1 <= x0 or y1 <= y0:
            continue
        im.crop((x0, y0, x1, y1)).save(out_dir / f"{rec.uuid}_{j}.jpg", quality=92)
        saved += 1
    return saved


def split_records(recs, val_frac):
    random.shuffle(recs)
    n_val = max(1, int(len(recs) * val_frac)) if len(recs) > 10 else 0
    return {"val": recs[:n_val], "train": recs[n_val:]}


def main():
    ap = argparse.ArgumentParser(description="Build a FathomNet classification dataset (ImageFolder).")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--top-k", type=int, default=60)
    ap.add_argument("--concepts-file")
    ap.add_argument("--min-count", type=int, default=200, help="min #boxes for a concept (top-k mode)")
    ap.add_argument("--max-per-class", type=int, default=800)
    ap.add_argument("--min-box", type=int, default=32, help="skip boxes smaller than this (px)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)

    out = Path(a.out_root)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)

    concepts = pick_concepts(a)
    session = requests.Session()
    session.headers.update({"User-Agent": "fathomnet-pretrain-builder"})

    grand = 0
    summary = []
    for ci, concept in enumerate(concepts, 1):
        recs = fetch_records(concept, a.max_per_class)
        recs = [r for r in recs
                if any((b.concept or "").lower() == concept.lower() for b in (r.boundingBoxes or []))]
        if not recs:
            print(f"[{ci}/{len(concepts)}] {concept}: no usable records, skip")
            continue
        cname = safe_name(concept)
        csaved = 0
        for split, items in split_records(recs, a.val_frac).items():
            d = out / split / cname
            d.mkdir(parents=True, exist_ok=True)
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                futs = [ex.submit(process_record, session, r, concept, d, a.min_box) for r in items]
                for f in as_completed(futs):
                    csaved += f.result()
        grand += csaved
        summary.append((concept, csaved))
        print(f"[{ci}/{len(concepts)}] {concept:30s} -> {csaved} crops")

    print(f"\nDone. {grand} crops across {len(summary)} classes -> {out}")
    print(f"Use: torchvision.datasets.ImageFolder('{out}/train')")
    tiny = [c for c, n in summary if n < 10]
    if tiny:
        print(f"[note] {len(tiny)} classes have <10 crops; raise --max-per-class or lower --min-box.")


if __name__ == "__main__":
    main()
