import json, argparse, os, pathlib

def coco_to_yolo(coco_json, images_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    img_map = {img["id"]: img["file_name"] for img in coco["images"]}
    size_map = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}
    anns_by_img = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0): 
            continue
        img_id = ann["image_id"]
        anns_by_img.setdefault(img_id, []).append(ann)

    for img_id, fname in img_map.items():
        stem = os.path.splitext(fname)[0]
        w, h = size_map[img_id]
        labels = []
        for ann in anns_by_img.get(img_id, []):
            x, y, bw, bh = ann["bbox"]
            xc = (x + bw/2) / w
            yc = (y + bh/2) / h
            nw = bw / w
            nh = bh / h
            labels.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        if labels:
            out_path = os.path.join(out_dir, f"{stem}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(labels))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_json", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    coco_to_yolo(args.coco_json, args.images_dir, args.out_dir)
    print("[OK] YOLO labels written to", args.out_dir)
