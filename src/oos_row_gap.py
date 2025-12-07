
import argparse, json, os, glob, math
from PIL import Image, ImageDraw

def group_rows(boxes, row_tol_px):
    # boxes: list of [x1,y1,x2,y2]
    # group by similar vertical center using tolerance
    centers = [( (b[1]+b[3])/2.0 , i) for i,b in enumerate(boxes)]
    centers.sort()
    rows = []
    for c,i in centers:
        placed=False
        for r in rows:
            # row represented by (min_c, max_c, idxs)
            if abs(c - r["mean"]) <= row_tol_px:
                r["idxs"].append(i)
                r["mean"] = sum((boxes[j][1]+boxes[j][3])/2.0 for j in r["idxs"]) / len(r["idxs"])
                placed=True
                break
        if not placed:
            rows.append({"mean": c, "idxs": [i]})
    return rows

def gaps_in_row(row_boxes, gap_factor=1.4, min_abs_gap=10):
    # row_boxes: list of [x1,y1,x2,y2] all from same shelf row
    if len(row_boxes) < 2:
        return []
    row_boxes = sorted(row_boxes, key=lambda b: (b[0]+b[2])/2.0)  # sort by center x
    # typical width
    widths = [b[2]-b[0] for b in row_boxes]
    med_w = sorted(widths)[len(widths)//2] if widths else 0
    if med_w <= 0:
        return []
    # scan gaps
    gaps = []
    for a,b in zip(row_boxes, row_boxes[1:]):
        gap = b[0] - a[2]  # space between a's right and b's left
        if gap >= max(min_abs_gap, gap_factor * med_w):
            # create a gap box spanning the vertical overlap of the two neighbors
            y1 = max(a[1], b[1]); y2 = min(a[3], b[3])
            if y2 > y1:
                gaps.append([a[2], y1, b[0], y2])
    return gaps

def draw_boxes(image_path, product_boxes, gap_boxes, out_path):
    with Image.open(image_path) as im:
        draw = ImageDraw.Draw(im, "RGBA")
        for x1,y1,x2,y2 in product_boxes:
            draw.rectangle([x1,y1,x2,y2], outline=(0,255,0,200), width=2)
        for x1,y1,x2,y2 in gap_boxes:
            draw.rectangle([x1,y1,x2,y2], outline=(255,0,0,255), width=3)
            draw.rectangle([x1,y1,x2,y2], fill=(255,0,0,60))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        im.save(out_path)

def main():
    ap = argparse.ArgumentParser(description="Compute OOS regions (gaps) from detection JSON and visualize.")
    ap.add_argument("--detections_json", required=True, help="detections JSON from infer_yolo.py")
    ap.add_argument("--images_dir", required=True, help="images root used for detections")
    ap.add_argument("--out_dir", required=True, help="where to write visualizations + oos_regions.json")
    ap.add_argument("--row_tol_px", type=float, default=30, help="vertical tolerance for row grouping (pixels)")
    ap.add_argument("--gap_factor", type=float, default=1.4, help="gap must be >= gap_factor * median box width")
    ap.add_argument("--min_abs_gap", type=float, default=10, help="absolute minimum gap in pixels")
    ap.add_argument("--max_vis", type=int, default=1000, help="limit number of images to visualize (0=off)")
    args = ap.parse_args()

    with open(args.detections_json, "r", encoding="utf-8") as f:
        det = json.load(f)

    oos = {}
    for fname, boxes in det.items():
        # group rows
        rows = group_rows(boxes, row_tol_px=args.row_tol_px)
        gap_boxes = []
        for r in rows:
            row_boxes = [boxes[i] for i in r["idxs"]]
            gap_boxes += gaps_in_row(row_boxes, gap_factor=args.gap_factor, min_abs_gap=args.min_abs_gap)
        oos[fname] = gap_boxes

    # write global oos json
    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "oos_regions.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(oos, f)
    print(f"[OK] wrote OOS JSON for {len(oos)} images to {out_json}")

    # visuals
    if args.max_vis != 0:
        count = 0
        for fname, boxes in det.items():
            img_path = os.path.join(args.images_dir, fname)
            vis_path = os.path.join(args.out_dir, fname)
            draw_boxes(img_path, boxes, oos.get(fname, []), vis_path)
            count += 1
            if args.max_vis > 0 and count >= args.max_vis:
                break
        print(f"[OK] wrote {count} visualizations to {args.out_dir}")

if __name__ == "__main__":
    main()
