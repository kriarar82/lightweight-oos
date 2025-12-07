
import os, csv, json, argparse, glob

def main():
    ap = argparse.ArgumentParser(description="Convert a simple CSV of OOS boxes to GT JSON.")
    ap.add_argument("--csv_path", required=True, help="CSV with rows: filename,x1,y1,x2,y2")
    ap.add_argument("--out_json", required=True, help="Output JSON path, e.g., data\\oos_gt_main.json")
    ap.add_argument("--images_dir", default="", help="Optional; include all images from this dir as keys (empty list if no boxes)")
    args = ap.parse_args()

    gt = {}
    # read CSV rows
    with open(args.csv_path, "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row or len(row) < 5:
                continue
            fn = row[0].strip()
            try:
                x1, y1, x2, y2 = map(float, row[1:5])
            except Exception:
                # skip malformed numeric fields
                continue
            gt.setdefault(fn, []).append([x1, y1, x2, y2])

    # ensure all images appear (even if no boxes)
    if args.images_dir:
        exts = ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG")
        imgs = []
        for e in exts:
            imgs += glob.glob(os.path.join(args.images_dir, e))
        for p in imgs:
            fn = os.path.basename(p)
            gt.setdefault(fn, [])

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)
    print(f"[OK] Wrote GT JSON for {len(gt)} images -> {args.out_json}")

if __name__ == "__main__":
    main()
