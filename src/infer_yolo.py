
import argparse, json, os, glob
from PIL import Image
from ultralytics import YOLO

def run(weights, images_dir, out_json, imgsz=640, conf=0.25, iou=0.45, device=None):
    model = YOLO(weights)
    paths = []
    for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG"):
        paths += glob.glob(os.path.join(images_dir, ext))
    paths = sorted(paths)
    out = {}
    for p in paths:
        im = Image.open(p); W,H = im.size; im.close()
        res = model.predict(p, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)[0]
        boxes_xyxy = []
        if res and res.boxes is not None:
            for b in res.boxes.xyxy.cpu().numpy().tolist():
                x1,y1,x2,y2 = b
                # clamp to image bounds
                x1 = max(0, min(float(x1), W)); x2 = max(0, min(float(x2), W))
                y1 = max(0, min(float(y1), H)); y2 = max(0, min(float(y2), H))
                if x2 > x1 and y2 > y1:
                    boxes_xyxy.append([x1,y1,x2,y2])
        out[os.path.basename(p)] = boxes_xyxy
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"[OK] wrote detections for {len(paths)} images to {out_json}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default=None, help="cuda:0 or cpu")
    args = ap.parse_args()
    run(args.weights, args.images_dir, args.out_json, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device)
