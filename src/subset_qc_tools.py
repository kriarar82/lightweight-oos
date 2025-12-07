
import os, glob, json, argparse

def count_boxes(txt_path: str) -> int:
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return sum(1 for ln in f if ln.strip())
    except FileNotFoundError:
        return 0

def parse_bins(spec: str):
    # "0-10,11-30,31-80,81-150,151-9999"
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.append((int(lo), int(hi)))
        else:
            k = int(part); out.append((k, k))
    return out

def which_bin(n, bins):
    for i,(a,b) in enumerate(bins):
        if a <= n <= b:
            return i
    return None

def counts_by_bin(root: str, bins):
    report = {}
    for split in ["train","val","test"]:
        lbl_dir = os.path.join(root, "labels", split)
        if not os.path.isdir(lbl_dir):
            report[split] = {"total": 0, "bins": [0]*len(bins)}
            continue
        totals = [0]*len(bins)
        total_files = 0
        for p in glob.glob(os.path.join(lbl_dir, "*.txt")):
            n = count_boxes(p)
            idx = which_bin(n, bins)
            if idx is not None:
                totals[idx] += 1
                total_files += 1
        report[split] = {"total": total_files, "bins": totals}
    return report

def parity_check(root: str):
    out = {}
    for split in ["train","val","test"]:
        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)
        imgs = set(os.path.splitext(os.path.basename(p))[0] 
                   for p in glob.glob(os.path.join(img_dir, "*.jpg")))
        lbls = set(os.path.splitext(os.path.basename(p))[0] 
                   for p in glob.glob(os.path.join(lbl_dir, "*.txt")))
        out[split] = {
            "num_images": len(imgs),
            "num_labels": len(lbls),
            "missing_labels": sorted(list(imgs - lbls))[:10],  # show only first 10
            "num_missing_labels": len(imgs - lbls),
            "orphan_labels": sorted(list(lbls - imgs))[:10],
            "num_orphan_labels": len(lbls - imgs),
        }
    return out

def make_manifest(root: str, out_path: str):
    man = {}
    for split in ["train","val","test"]:
        img_dir = os.path.join(root, "images", split)
        files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(img_dir, "*.jpg")))
        man[split] = files
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
    return out_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Subset QC: bin coverage, image/label parity, manifest export.")
    ap.add_argument("--root", required=True, help=r"Subset root (e.g., data\sku110k_subset_strat)")
    ap.add_argument("--bins", default="0-10,11-30,31-80,81-150,151-9999", help="Density bins for YOLO box counts")
    ap.add_argument("--manifest_out", default="outputs/subset_manifest.json", help="Where to write the manifest JSON")
    args = ap.parse_args()

    bins = parse_bins(args.bins)

    print("== 1) Coverage by density bins ==")
    cov = counts_by_bin(args.root, bins)
    for split, info in cov.items():
        print(f"\n[{split.upper()}] total label files: {info['total']}")
        for i,(a,b) in enumerate(bins):
            print(f"  bin {i} [{a}-{b}]: {info['bins'][i]}")

    print("\n== 2) Image/label parity ==")
    par = parity_check(args.root)
    for split, info in par.items():
        print(f"\n[{split.upper()}] imgs={info['num_images']} labels={info['num_labels']} missing_labels={info['num_missing_labels']} orphan_labels={info['num_orphan_labels']}")
        if info['num_missing_labels']>0:
            print("  e.g., missing (first 10):", info['missing_labels'])
        if info['num_orphan_labels']>0:
            print("  e.g., orphans (first 10):", info['orphan_labels'])

    print("\n== 3) Manifest export ==")
    mp = make_manifest(args.root, args.manifest_out)
    print("[OK] Wrote manifest to", mp)
