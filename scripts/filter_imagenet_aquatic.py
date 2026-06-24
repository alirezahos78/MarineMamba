#!/usr/bin/env python3
"""
Filter ImageNet-1k to aquatic / underwater animal classes for domain-relevant pretraining.

Selects ImageNet classes that are descendants (in WordNet) of aquatic roots:
fish, aquatic vertebrates/mammals, crustaceans, echinoderms, cnidarians
(coral/anemone/jellyfish), mollusks, and sea turtles.

You do NOT need to know the dataset path: with no --imagenet-root / --train-dir,
the script auto-searches common locations for an ImageNet 'train' folder.

Examples
--------
# just locate ImageNet on this machine
python filter_imagenet_aquatic.py --find

# list selected classes (auto-finds dataset; nothing written)
python filter_imagenet_aquatic.py --list-only

# point it explicitly (either the parent that contains train/, or the train dir itself)
python filter_imagenet_aquatic.py --imagenet-root /data/imagenet --list-only
python filter_imagenet_aquatic.py --train-dir /data/imagenet/ILSVRC/Data/CLS-LOC/train --list-only

# build the filtered subset via symlinks (works directly with torchvision ImageFolder)
python filter_imagenet_aquatic.py --out-root /data/imagenet_aquatic
"""
import argparse, os, sys
from pathlib import Path
from nltk.corpus import wordnet as wn

AQUATIC_ROOTS = ['aquatic_vertebrate.n.01','fish.n.01','aquatic_mammal.n.01',
    'crustacean.n.01','echinoderm.n.01','coelenterate.n.01','mollusk.n.01','sea_turtle.n.01']
EXCLUDE = set()  # add wnids to force-drop, e.g. land snails: 'n01944390'

# ---------- WordNet aquatic checker ----------
def ensure_wordnet():
    try: wn.synset('fish.n.01')
    except LookupError:
        import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')

def wnid_to_synset(w): return wn.synset_from_pos_and_offset(w[0], int(w[1:]))

def build_checker():
    roots=[wn.synset(r) for r in AQUATIC_ROOTS]
    def is_aquatic(w):
        try: syn=wnid_to_synset(w)
        except Exception: return False
        if syn is None: return False
        h=set()
        for p in syn.hypernym_paths(): h.update(p)
        return any(r in h for r in roots)
    return is_aquatic

# ---------- dataset location ----------
def is_wnid(name): return len(name)==9 and name[0]=='n' and name[1:].isdigit()

def count_wnids(d, cap=60):
    c=0
    try:
        for x in os.scandir(d):
            if x.is_dir(follow_symlinks=True) and is_wnid(x.name):
                c+=1
                if c>=cap: break
    except (PermissionError, FileNotFoundError, NotADirectoryError):
        return 0
    return c

def looks_like_train_dir(d, min_wnids=50): return count_wnids(d) >= min_wnids

DEFAULT_SEARCH_ROOTS = [
    Path.home(), Path("/local-scratch"), Path("/localhome"), Path("/scratch"),
    Path("/datasets"), Path("/data"), Path("/mnt"), Path.cwd(),
]

def find_imagenet(search_roots, max_depth=6, min_wnids=50):
    found=[]
    for base in search_roots:
        base=Path(base)
        if not base.exists(): continue
        base_n=len(base.parts)
        for dirpath, dirnames, _ in os.walk(base, followlinks=False):
            depth=len(Path(dirpath).parts)-base_n
            if depth>max_depth:
                dirnames[:]=[]; continue
            # never descend into wnid folders (they hold thousands of images)
            dirnames[:]=[d for d in dirnames if not is_wnid(d)]
            if looks_like_train_dir(dirpath, min_wnids):
                found.append(Path(dirpath))   # this dir IS the wnid container (train split)
                dirnames[:]=[]                # stop descending below a found split
    # dedupe, prefer dirs literally named 'train'
    uniq=[]
    for p in found:
        if p not in uniq: uniq.append(p)
    uniq.sort(key=lambda p:(p.name.lower()!='train', str(p)))
    return uniq

def resolve_train_dir(args):
    if args.train_dir:
        td=Path(args.train_dir)
        if looks_like_train_dir(td): return td
        sys.exit(f"--train-dir has no wnid (n########) subfolders: {td}")
    if args.imagenet_root:
        td=Path(args.imagenet_root)/'train'
        if looks_like_train_dir(td): return td
        # maybe they passed the train dir itself as root
        if looks_like_train_dir(Path(args.imagenet_root)): return Path(args.imagenet_root)
        sys.exit(f"No valid train/ with wnid folders under: {args.imagenet_root}")
    # auto-discover
    roots = [Path(r) for r in args.search_root] if args.search_root else DEFAULT_SEARCH_ROOTS
    print("No path given -> auto-searching for ImageNet (this may take a moment)...")
    cands=find_imagenet(roots, max_depth=args.max_depth)
    if not cands:
        sys.exit("Could not auto-find ImageNet. Re-run with --search-root <dir> "
                 "or --train-dir <path-to-train-with-wnid-folders>.")
    if len(cands)>1:
        print("Multiple candidates found:")
        for i,c in enumerate(cands): print(f"  [{i}] {c}  ({count_wnids(c)}+ classes)")
        print(f"Using [0]. Override with --train-dir to pick another.")
    chosen=cands[0]
    print(f"Using train dir: {chosen}\n")
    return chosen

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser(description="Filter ImageNet-1k to aquatic classes.")
    ap.add_argument('--imagenet-root', help="parent folder containing train/ (and val/)")
    ap.add_argument('--train-dir', help="the train folder that directly holds n######## subfolders")
    ap.add_argument('--search-root', nargs='+', help="where to auto-search (default: home, /local-scratch, ...)")
    ap.add_argument('--max-depth', type=int, default=6)
    ap.add_argument('--out-root', default=None, help="build the filtered subset here (symlinks)")
    ap.add_argument('--find', action='store_true', help="only locate ImageNet and exit")
    ap.add_argument('--list-only', action='store_true', help="print selected classes, write nothing")
    a=ap.parse_args()

    if a.find:
        roots=[Path(r) for r in a.search_root] if a.search_root else DEFAULT_SEARCH_ROOTS
        cands=find_imagenet(roots, max_depth=a.max_depth)
        if not cands: print("No ImageNet train folder found.")
        else:
            print("Candidate ImageNet train dirs:")
            for c in cands: print(f"  {c}   ({count_wnids(c)}+ classes)")
        return

    ensure_wordnet(); is_aquatic=build_checker()
    train=resolve_train_dir(a)

    allw=sorted([x.name for x in Path(train).iterdir() if x.is_dir() and is_wnid(x.name)])
    sel=[w for w in allw if is_aquatic(w) and w not in EXCLUDE]
    print(f"Found {len(allw)} classes; {len(sel)} aquatic.\n")
    tot=0
    for w in sel:
        name=wnid_to_synset(w).name().split('.')[0]
        n=len(list((train/w).glob('*'))); tot+=n
        print(f"  {w}  {name:24s} {n:6d} imgs")
    print(f"\nTotal aquatic training images: {tot}")

    if a.list_only or a.out_root is None:
        print("\n(list-only) nothing written. add --out-root to build the subset."); return

    # build subset; mirror sibling splits (train/val) if present
    out=Path(a.out_root); root=train.parent
    splits=[train.name]
    if (root/'val').exists() and looks_like_train_dir(root/'val', min_wnids=1): splits.append('val')
    for sp in splits:
        ss=root/sp
        for w in sel:
            src=ss/w
            if not src.exists(): continue
            dst=out/sp/w; dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists(): os.symlink(src.resolve(),dst)
    print(f"\nSymlinked subset -> {out}\nUse: torchvision.datasets.ImageFolder('{out}/{train.name}')")

if __name__=='__main__': main()