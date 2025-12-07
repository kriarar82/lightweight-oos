
import os, csv, argparse, glob
from PIL import Image

# Heuristic header mapping for common CSV schemas
HEADER_ALIASES = {
    "image": ["image", "image_name", "filename", "file_name", "img", "img_name"],
    "x1":    ["x1", "xmin", "left", "x_min"],
    "y1":    ["y1", "ymin", "top", "y_min"],
    "x2":    ["x2", "xmax", "right", "x_max"],
    "y2":    ["y2", "ymax", "bottom", "y_max"],
    "class": ["class", "category", "label", "cls"]
}

def find_col(header, keys):
    h = [str(c).strip().lower() for c in header]
    for k in keys:
        if k in h:
            return h.index(k)
    return None

def map_headers(header):
    m = {}
    for canon, aliases in HEADER_ALIASES.items():
        idx = find_col(header, aliases)
        m[canon] = idx
    # class optional; image and all coords required if using header mapping
    if m["image"] is None or None in (m["x1"], m["y1"], m["x2"], m["y2"]):
        return None  # signal unmapped
    return m

def looks_like_headerless(first_row):
    if not first_row:
        return False
    name = str(first_row[0]).strip().lower()
    return name.endswith((".jpg",".jpeg",".png",".JPG",".PNG")) or name.endswith((".jpg",".jpeg",".png"))

def parse_row_headerless(row):
    # Expected order: image, x1, y1, x2, y2, class(optional), width(optional), height(optional)
    image = row[0].strip()
    x1 = float(row[1]); y1 = float(row[2]); x2 = float(row[3]); y2 = float(row[4])
    cls = 0  # single class
    W = H = None
    if len(row) >= 8:
        try:
            W = int(float(row[6])); H = int(float(row[7]))
        except Exception:
            W = H = None
    return image, x1, y1, x2, y2, cls, W, H

def load_image_size(images_root, rel_path):
    cand = [
        os.path.join(images_root, rel_path),
        os.path.join(images_root, os.path.basename(rel_path)),
    ]
    exts = ["", ".jpg", ".jpeg", ".png", ".JPG", ".PNG"]
    for c in cand:
        base, ext = os.path.splitext(c)
        if ext == "":
            for e in exts[1:]:
                p = base + e
                if os.path.exists(p):
                    c = p; break
        if os.path.exists(c):
            with Image.open(c) as im:
                return im.size  # (W,H)
    return None

def row_to_yolo_from_mapped(row, m, images_root):
    img_name = str(row[m["image"]]).strip()
    size = load_image_size(images_root, img_name)
    if not size:
        return img_name, None, None
    W,H = size
    x1 = float(row[m["x1"]]); y1 = float(row[m["y1"]])
    x2 = float(row[m["x2"]]); y2 = float(row[m["y2"]])
    # clamp and convert
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(W), x2), min(float(H), y2)
    w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
    if w <= 0 or h <= 0:
        return img_name, None, (W,H)
    cx = (x1 + w/2.0) / W
    cy = (y1 + h/2.0) / H
    nw = w / W
    nh = h / H
    return img_name, (0, cx, cy, nw, nh), (W,H)

def row_to_yolo_headerless(row, images_root):
    img_name, x1, y1, x2, y2, cls, W, H = parse_row_headerless(row)
    if W is None or H is None:
        size = load_image_size(images_root, img_name)
        if not size:
            return img_name, None, None
        W,H = size
    # clamp and convert
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(W), x2), min(float(H), y2)
    w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
    if w <= 0 or h <= 0:
        return img_name, None, (W,H)
    cx = (x1 + w/2.0) / W
    cy = (y1 + h/2.0) / H
    nw = w / W
    nh = h / H
    return img_name, (0, cx, cy, nw, nh), (W,H)

def convert_csv(csv_path, images_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    grouped = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        rows = list(rdr)
        if not rows:
            raise SystemExit("[ERROR] CSV is empty")
        first = rows[0]
        mapping = map_headers(first)
        headerless = False
        start_idx = 1
        if mapping is None:
            # maybe headerless; check
            if looks_like_headerless(first):
                headerless = True
                start_idx = 0
            else:
                raise SystemExit(f"[ERROR] Could not map columns and first row doesn't look like data. First row: {first}")

        for row in rows[start_idx:]:
            if not row or all(not str(c).strip() for c in row): 
                continue
            if headerless:
                img_name, yolo, _ = row_to_yolo_headerless(row, images_root)
            else:
                img_name, yolo, _ = row_to_yolo_from_mapped(row, mapping, images_root)
            if img_name is None:
                continue
            grouped.setdefault(os.path.basename(img_name), [])
            if yolo is not None:
                grouped[os.path.basename(img_name)].append(yolo)

    # write YOLO .txt per existing image in folder
    imgs = []
    for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG"):
        imgs += glob.glob(os.path.join(images_root, ext))
    img_set = {os.path.basename(p) for p in imgs}
    written = 0
    for img_base in img_set:
        stem = os.path.splitext(img_base)[0]
        outp = os.path.join(out_dir, stem + ".txt")
        lines = []
        for (cls, cx, cy, nw, nh) in grouped.get(img_base, []):
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        with open(outp, "w", encoding="utf-8") as w:
            w.write("\n".join(lines))
        written += 1
    print(f"[OK] Wrote {written} label files to {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Convert SKU-110K CSV (header or headerless) to YOLO labels.")
    ap.add_argument("--csv_path", required=True, help="Path to annotations_*.csv")
    ap.add_argument("--images_dir", required=True, help="Path to images/<split>")
    ap.add_argument("--out_dir", required=True, help="Output labels/<split>")
    args = ap.parse_args()
    convert_csv(args.csv_path, args.images_dir, args.out_dir)
