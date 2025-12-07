
import argparse, csv, glob, os, sys
import cv2

HELP = """
Quick Box Annotator (Lite)
--------------------------------------------
Mouse:
  - Click & drag  : draw a rectangle (OOS gap)
  - Release mouse : finalize rectangle
Keys:
  - ENTER/RETURN  : save CSV for this image and go to NEXT
  - n             : NEXT image (saves current boxes)
  - p             : PREVIOUS image (saves current boxes)
  - u             : UNDO last box
  - c             : CLEAR all boxes
  - 0             : Mark NO GAPS (saves empty row set)
  - s             : SAVE (write CSV) without moving
  - h             : HELP (print to console)
  - q/ESC         : QUIT (saves current, then exit)

Tip: Use --start_index and --pattern to jump within the folder.
CSV format: filename,x1,y1,x2,y2  (one row per box, pixel coords on ORIGINAL image)
"""

def overwrite_csv(csv_path, records):
    # records: list of (filename, x1,y1,x2,y2) or [] if no boxes
    keep = []
    existing = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            rdr = csv.reader(f)
            for row in rdr:
                if row and len(row) >= 5:
                    existing.append(row)
    targets = set(r[0] for r in records) if records else set()
    for row in existing:
        if row[0] not in targets:
            keep.append(row)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for row in keep:
            w.writerow(row)
        for r in records:
            w.writerow([r[0], int(r[1]), int(r[2]), int(r[3]), int(r[4])])

class AnnotatorLite:
    def __init__(self, images_dir, out_csv, pattern="*.jpg", start_index=0, max_side=1200):
        self.paths = sorted(glob.glob(os.path.join(images_dir, pattern)))
        if not self.paths:
            print("No images found:", images_dir, pattern)
            sys.exit(1)
        self.out_csv = out_csv
        self.idx = max(0, min(start_index, len(self.paths)-1))
        self.boxes = []           # boxes in ORIGINAL coords
        self.img = None           # original image
        self.disp = None          # display (resized) image
        self.scale = 1.0
        self.max_side = max(0, int(max_side))
        self.win = "OOS Annotator Lite"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, 1200, 800)
        cv2.setMouseCallback(self.win, self.on_mouse)
        self.drawing = False
        self.x0 = self.y0 = 0
        self.load_image(self.idx)

    def compute_scale(self, h, w):
        if self.max_side <= 0:
            return 1.0
        m = max(h, w)
        if m <= self.max_side:
            return 1.0
        return float(self.max_side) / float(m)

    def to_original(self, x, y):
        # map display coords to original coords
        ox = int(round(x / self.scale))
        oy = int(round(y / self.scale))
        return ox, oy

    def load_image(self, i):
        self.idx = i
        p = self.paths[i]
        print(f"[LOAD] {i+1}/{len(self.paths)} -> {os.path.basename(p)}")
        self.img = cv2.imread(p)
        if self.img is None:
            print("  [WARN] Failed to read:", p)
            self.img = 255 * np.ones((720,1280,3), dtype=np.uint8)
        h, w = self.img.shape[:2]
        self.scale = self.compute_scale(h, w)
        if self.scale != 1.0:
            self.disp = cv2.resize(self.img, (int(w*self.scale), int(h*self.scale)), interpolation=cv2.INTER_AREA)
        else:
            self.disp = self.img.copy()
        self.boxes = []
        self.refresh()

    def refresh(self):
        v = self.disp.copy()
        # legend
        txt = f"{os.path.basename(self.paths[self.idx])}  |  boxes={len(self.boxes)}  [u:undo c:clear 0:no-gaps s:save ENTER/n:next p:prev q:quit]"
        cv2.rectangle(v, (0,0), (v.shape[1], 28), (0,0,0), -1)
        cv2.putText(v, txt, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        # draw boxes (convert to display coords)
        for (x1,y1,x2,y2) in self.boxes:
            dx1, dy1 = int(round(x1*self.scale)), int(round(y1*self.scale))
            dx2, dy2 = int(round(x2*self.scale)), int(round(y2*self.scale))
            cv2.rectangle(v, (dx1,dy1), (dx2,dy2), (0,0,255), 2)
        cv2.imshow(self.win, v)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x0, self.y0 = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # show temporary rectangle
            self.refresh()
            tmp = self.disp.copy()
            cv2.rectangle(tmp, (self.x0, self.y0), (x, y), (0, 0, 255), 2)
            cv2.imshow(self.win, tmp)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            x1, y1 = min(self.x0, x), min(self.y0, y)
            x2, y2 = max(self.x0, x), max(self.y0, y)
            ox1, oy1 = self.to_original(x1, y1)
            ox2, oy2 = self.to_original(x2, y2)
            if ox2 > ox1 and oy2 > oy1:
                self.boxes.append((ox1, oy1, ox2, oy2))
                print(f"  [+] box {self.boxes[-1]} (orig px)")
            self.refresh()

    def save_current(self):
        fn = os.path.basename(self.paths[self.idx])
        rows = [(fn, x1,y1,x2,y2) for (x1,y1,x2,y2) in self.boxes]
        overwrite_csv(self.out_csv, rows)
        print(f"[SAVE] {fn} boxes={len(rows)} -> {self.out_csv}")
        return fn, len(rows)

    def mark_nogaps(self):
        fn = os.path.basename(self.paths[self.idx])
        overwrite_csv(self.out_csv, [(fn, 0,0,0,0)][:0])  # remove rows for this file
        print(f"[NO-GAPS] {fn} (cleared any existing rows)")

    def loop(self):
        print(HELP)
        while True:
            self.refresh()
            k = cv2.waitKey(30) & 0xFF
            if k == ord('h'):
                print(HELP)
            elif k in (ord('u'), 8):
                if self.boxes:
                    self.boxes.pop()
                    print("  [-] undo")
            elif k == ord('c'):
                self.boxes.clear()
                print("  [*] cleared")
            elif k == ord('0'):
                self.mark_nogaps()
            elif k == ord('s'):
                self.save_current()
            elif k in (13, ord('n')):
                self.save_current()
                self.load_image(min(len(self.paths)-1, self.idx+1))
            elif k == ord('p'):
                self.save_current()
                self.load_image(max(0, self.idx-1))
            elif k in (27, ord('q')):
                self.save_current()
                print("[QUIT] Bye!")
                break
        cv2.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser(description="Quick CSV OOS annotator (lite, resized display).")
    ap.add_argument("--images_dir", required=True, help=r"Folder with images, e.g., data\sku110k_subset_strat\images\test")
    ap.add_argument("--out_csv", required=True, help=r"Output CSV path, e.g., data\oos_gt_main.csv")
    ap.add_argument("--pattern", default="*.jpg", help="Glob pattern, default *.jpg")
    ap.add_argument("--start_index", type=int, default=0, help="Start from this index (in sorted file list)")
    ap.add_argument("--max_side", type=int, default=1200, help="Resize longest image side to this for speed (0=disable)")
    args = ap.parse_args()

    ann = AnnotatorLite(args.images_dir, args.out_csv, args.pattern, args.start_index, args.max_side)
    ann.loop()

if __name__ == "__main__":
    main()
