
import json, argparse, os, random
import numpy as np

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0: 
        return 0.0
    areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    areaB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    denom = areaA + areaB - inter
    return inter/denom if denom>0 else 0.0

def precision_recall(pred, gt, iou_thr=0.3):
    tp = 0; fp = 0; fn = 0
    for img, pred_boxes in pred.items():
        p = [list(map(float,b)) for b in pred_boxes]
        g = [list(map(float,b)) for b in gt.get(img, [])]
        matched = set()
        for pb in p:
            best = -1; best_iou = 0.0
            for j, gb in enumerate(g):
                if j in matched: 
                    continue
                iou_val = iou(pb, gb)
                if iou_val > best_iou:
                    best_iou = iou_val; best = j
            if best_iou >= iou_thr:
                tp += 1; matched.add(best)
            else:
                fp += 1
        fn += (len(g) - len(matched))
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    return prec, rec, tp, fp, fn

def bootstrap_ci(pred, gt, iou_thr=0.3, B=1000, seed=123):
    rng = np.random.default_rng(seed)
    images = sorted(set(list(pred.keys()) + list(gt.keys())))
    if not images:
        return (0,0,0,0,0), (0,0), (0,0)
    pvals = []; rvals = []
    for _ in range(B):
        sample = rng.choice(images, size=len(images), replace=True)
        pred_s = {img: pred.get(img, []) for img in sample}
        gt_s   = {img: gt.get(img, []) for img in sample}
        p, r, *_ = precision_recall(pred_s, gt_s, iou_thr=iou_thr)
        pvals.append(p); rvals.append(r)
    p_ci = (float(np.percentile(pvals, 2.5)), float(np.percentile(pvals, 97.5)))
    r_ci = (float(np.percentile(rvals, 2.5)), float(np.percentile(rvals, 97.5)))
    p_mean = float(np.mean(pvals)); r_mean = float(np.mean(rvals))
    return (p_mean, r_mean), p_ci, r_ci

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate OOS predictions with precision/recall + bootstrap CIs.")
    ap.add_argument("--pred_json", required=True, help="Predicted OOS JSON: {image: [[x1,y1,x2,y2], ...], ...}")
    ap.add_argument("--gt_json", required=True, help="Ground-truth OOS JSON (same format)")
    ap.add_argument("--iou_thr", type=float, default=0.3)
    ap.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations")
    args = ap.parse_args()

    with open(args.pred_json, "r", encoding="utf-8") as f:
        pred = json.load(f)
    with open(args.gt_json, "r", encoding="utf-8") as f:
        gt = json.load(f)

    prec, rec, tp, fp, fn = precision_recall(pred, gt, iou_thr=args.iou_thr)
    (p_mean, r_mean), p_ci, r_ci = bootstrap_ci(pred, gt, iou_thr=args.iou_thr, B=args.bootstrap)

    print(f"Base (all images): Precision={prec:.3f} Recall={rec:.3f}  TP={tp} FP={fp} FN={fn}  (IoU>={args.iou_thr})")
    print(f"Bootstrap means:   Precision={p_mean:.3f} Recall={r_mean:.3f}")
    print(f"95% CI (Precision): [{p_ci[0]:.3f}, {p_ci[1]:.3f}]")
    print(f"95% CI (Recall):    [{r_ci[0]:.3f}, {r_ci[1]:.3f}]")
