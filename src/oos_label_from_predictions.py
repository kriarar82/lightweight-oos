
import argparse, json, os, sys, csv

HELP = """
Prediction Reviewer (No-GUI)
--------------------------------------------
Iterates predicted OOS gaps per image and lets you accept (y) or reject (n).
Writes CSV: filename,x1,y1,x2,y2

Controls:
  y : accept current box
  n : reject current box
  a : accept ALL remaining boxes for this image
  r : reject ALL remaining boxes for this image
  s : save accepted boxes for this image (partial save)
  ENTER : save & NEXT image
  b : save & go BACK one image
  q : save & QUIT
  h : help

Tip: Open overlays in a viewer (e.g., outputs\\oos_vis_main) while answering here.
"""

def load_pred(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def overwrite_for_image(csv_path, filename, rows):
    """Replace rows for 'filename' in CSV with 'rows' (list of [fn,x1,y1,x2,y2])."""
    existing = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            rdr = csv.reader(f)
            for r in rdr:
                if r and len(r) >= 5:
                    existing.append(r)
    keep = [r for r in existing if r[0] != filename]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for r in keep: w.writerow(r)
        for r in rows: w.writerow(r)

def main():
    ap = argparse.ArgumentParser(description="Keyboard-only OOS acceptance tool from predictions JSON.")
    ap.add_argument("--pred_json", required=True, help=r"e.g., outputs\oos_vis_main\oos_regions.json")
    ap.add_argument("--out_csv", required=True, help=r"e.g., data\oos_gt_main.csv")
    args = ap.parse_args()

    data = load_pred(args.pred_json)  # {filename: [[x1,y1,x2,y2], ...], ...}
    files = sorted(data.keys())
    if not files:
        print("No predictions found in:", args.pred_json)
        sys.exit(1)

    print(HELP)
    idx = 0

    while 0 <= idx < len(files):
        fn = files[idx]
        boxes = data.get(fn, [])
        print(f"\n[{idx+1}/{len(files)}] {fn}  predicted boxes: {len(boxes)}")
        accepted = []
        i = 0
        while i < len(boxes):
            b = boxes[i]
            print(f"  Box {i+1}/{len(boxes)}: {b}   [y/n/a/r/s/ENTER/b/q/h]? ", end="", flush=True)
            try:
                ch = input().strip().lower()
            except EOFError:
                ch = "q"
            if ch == "y":
                accepted.append([fn, int(b[0]), int(b[1]), int(b[2]), int(b[3])])
                i += 1
            elif ch == "n":
                i += 1
            elif ch == "a":
                for j in range(i, len(boxes)):
                    bb = boxes[j]
                    accepted.append([fn, int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])])
                i = len(boxes)
            elif ch == "r":
                i = len(boxes)
            elif ch == "s":
                overwrite_for_image(args.out_csv, fn, accepted)
                print(f"  [SAVED partial] {fn} rows={len(accepted)} -> {args.out_csv}")
            elif ch == "":
                overwrite_for_image(args.out_csv, fn, accepted)
                print(f"  [NEXT] saved {fn} rows={len(accepted)} -> {args.out_csv}")
                break
            elif ch == "b":
                overwrite_for_image(args.out_csv, fn, accepted)
                print(f"  [BACK] saved {fn} rows={len(accepted)} -> {args.out_csv}")
                idx = max(0, idx-1)
                break
            elif ch == "q":
                overwrite_for_image(args.out_csv, fn, accepted)
                print(f"  [QUIT] saved {fn} rows={len(accepted)} -> {args.out_csv}")
                print("Bye.")
                return
            elif ch == "h":
                print(HELP)
            else:
                print("  (Unknown key. Use y/n/a/r/s/ENTER/b/q/h)")
        else:
            overwrite_for_image(args.out_csv, fn, accepted)
            print(f"  [AUTO-NEXT] saved {fn} rows={len(accepted)} -> {args.out_csv}")
        idx += 1

    print("\n[Done] Reached last image.")
    print("Convert CSV to JSON:")
    print(r"  python scripts\csv_to_oos_gt.py --csv_path data\oos_gt_main.csv --images_dir data\sku110k_subset_strat\images\test --out_json data\oos_gt_main.json")

if __name__ == "__main__":
    main()
