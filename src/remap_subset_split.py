
import os, argparse, shutil, glob

def ensure(d):
    os.makedirs(d, exist_ok=True)

def copy_tree(src, dst):
    ensure(dst)
    if not os.path.isdir(src):
        return 0
    n = 0
    for p in glob.glob(os.path.join(src, "*")):
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(dst, os.path.basename(p))); n += 1
    return n

def main():
    ap = argparse.ArgumentParser(description="Remap subset split: copy images/<src_split> -> images/<dst_split> and labels/<src_split> -> labels/<dst_split>.")
    ap.add_argument("--root", required=True, help=r"Subset root (e.g., data\sku110k_eval_stress)")
    ap.add_argument("--src_split", default="train", help="Source split name (default: train)")
    ap.add_argument("--dst_split", default="test", help="Destination split name (default: test)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files in destination")
    args = ap.parse_args()

    img_src = os.path.join(args.root, "images", args.src_split)
    lbl_src = os.path.join(args.root, "labels", args.src_split)
    img_dst = os.path.join(args.root, "images", args.dst_split)
    lbl_dst = os.path.join(args.root, "labels", args.dst_split)

    # Optionally clean destination
    if args.overwrite:
        for d in (img_dst, lbl_dst):
            if os.path.isdir(d):
                for f in glob.glob(os.path.join(d, "*")):
                    try: os.remove(f)
                    except: pass

    n_i = copy_tree(img_src, img_dst)
    n_l = copy_tree(lbl_src, lbl_dst)

    print(f"[OK] Copied {n_i} images {args.src_split} -> {args.dst_split}")
    print(f"[OK] Copied {n_l} labels {args.src_split} -> {args.dst_split}")
    print(f"Images dst: {img_dst}")
    print(f"Labels dst: {lbl_dst}")

if __name__ == "__main__":
    main()
