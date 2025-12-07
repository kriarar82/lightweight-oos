
import argparse, json, os, csv, cv2
from pathlib import Path

HELP = """
Full-Image Box Reviewer
--------------------------------------------
Shows the FULL image. Current box is thick & bright; other boxes are faint.
An inset (top-left) shows a zoomed crop of the current box.

Keys:
  y : accept current box
  n : reject current box
  LEFT/RIGHT : previous/next box
  ENTER/SPACE : next image (saves rows for this image)
  s : save partial CSV for this image
  b : save and go BACK one image
  q/ESC : save and quit
"""

def overwrite_for_image(csv_path, filename, rows):
    # Replace rows for 'filename' in CSV with 'rows' (list of [fn,x1,y1,x2,y2])
    existing = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for r in csv.reader(f):
                if r and len(r) >= 5:
                    existing.append(r)
    keep = [r for r in existing if r[0] != filename]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for r in keep: w.writerow(r)
        for r in rows: w.writerow(r)

def draw_full_view(img, boxes, idx):
    vis = img.copy()
    # draw others faint
    for i, (x1,y1,x2,y2) in enumerate(boxes):
        color = (0,255,0) if i != idx else (0,0,255)
        thickness = 1 if i != idx else 3
        alpha = 0.3 if i != idx else 1.0
        overlay = vis.copy()
        cv2.rectangle(overlay, (int(x1),int(y1)), (int(x2),int(y2)), color, thickness)
        cv2.addWeighted(overlay, alpha, vis, 1-alpha, 0, vis)
        if i == idx:
            cv2.putText(vis, f"#{i+1}", (int(x1)+4, int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
    # inset crop for current
    if boxes:
        x1,y1,x2,y2 = map(int, boxes[idx])
        X1, Y1 = max(0, x1-10), max(0, y1-10)
        X2, Y2 = min(vis.shape[1]-1, x2+10), min(vis.shape[0]-1, y2+10)
        crop = vis[Y1:Y2, X1:X2].copy()
        if crop.size:
            ch, cw = crop.shape[:2]
            scale = 250.0 / max(ch, cw)
            crop = cv2.resize(crop, (int(cw*scale), int(ch*scale)), interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(crop, (0,0), (crop.shape[1]-1, crop.shape[0]-1), (0,0,255), 2)
            vis[10:10+crop.shape[0], 10:10+crop.shape[1]] = crop
            cv2.rectangle(vis, (8,8), (12+crop.shape[1], 12+crop.shape[0]), (0,0,255), 2)
    return vis

def main():
    ap = argparse.ArgumentParser(description="Review predicted boxes on FULL image with highlight + inset zoom.")
    ap.add_argument("--pred_json", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    with open(args.pred_json, "r", encoding="utf-8") as f:
        preds = json.load(f)
    files = sorted(preds.keys())
    if not files:
        print("No predictions in", args.pred_json); return

    cv2.namedWindow("Full-Image Reviewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Full-Image Reviewer", 1200, 900)
    print(HELP)

    i_img = 0
    while 0 <= i_img < len(files):
        fn = files[i_img]
        img_path = os.path.join(args.images_dir, fn)
        img = cv2.imread(img_path)
        if img is None:
            alt = os.path.join(args.images_dir, Path(fn).stem + ".jpg")
            img = cv2.imread(alt)
            if img is None:
                print("[SKIP] cannot read:", fn)
                i_img += 1
                continue
        boxes = preds.get(fn, [])
        cur = 0
        accepted = []
        while boxes and 0 <= cur < len(boxes):
            vis = draw_full_view(img, boxes, cur)
            cv2.putText(vis, f"{fn}  box {cur+1}/{len(boxes)}  (y/n LEFT/RIGHT ENTER s b q)", (20, vis.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("Full-Image Reviewer", vis)
            k = cv2.waitKey(0) & 0xFF
            if k in (ord('y'), ord('Y')):
                x1,y1,x2,y2 = map(int, boxes[cur])
                accepted.append([fn, x1,y1,x2,y2]); cur += 1
            elif k in (ord('n'), ord('N')):
                cur += 1
            elif k == 81:  # LEFT
                cur = max(0, cur-1)
            elif k == 83:  # RIGHT
                cur = min(len(boxes)-1, cur+1)
            elif k in (13, 32):  # ENTER/SPACE -> next image
                break
            elif k in (ord('s'), ord('S')):
                overwrite_for_image(args.out_csv, fn, accepted)
                print(f"[SAVE] {fn}: rows={len(accepted)}")
            elif k in (ord('b'), ord('B')):
                overwrite_for_image(args.out_csv, fn, accepted)
                i_img = max(0, i_img-1)
                break
            elif k in (27, ord('q'), ord('Q')):
                overwrite_for_image(args.out_csv, fn, accepted)
                print(f"[QUIT] {fn}: rows={len(accepted)}")
                cv2.destroyAllWindows()
                return
        overwrite_for_image(args.out_csv, fn, accepted)
        print(f"[DONE IMG] {fn}: rows={len(accepted)}")
        i_img += 1

    cv2.destroyAllWindows()
    print("[ALL DONE] Convert CSV to JSON next.")
    print(r"python scripts\csv_to_oos_gt.py --csv_path data\oos_gt_main.csv --images_dir data\sku110k_subset_strat\images\test --out_json data\oos_gt_main.json")

if __name__ == "__main__":
    main()
