# Lightweight Out-of-Stock (OOS) Detection

A lightweight Python toolkit for detecting out-of-stock gaps in retail shelf images using YOLO object detection and gap analysis.

## Features

- Convert COCO format annotations to YOLO format
- Run YOLO inference on images
- Detect OOS gaps between product detections
- Interactive annotation tools for ground truth creation
- Evaluation tools with bootstrap confidence intervals
- Data quality control and subset management utilities

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd lightweight-OOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- Pillow >= 10.0.0
- ultralytics >= 8.0.0

## Project Structure

```
lightweight-OOS/
├── src/                          # Main scripts
│   ├── convert_to_yolo.py       # Convert COCO JSON to YOLO format
│   ├── csv_to_yolo_sku110k_v2.py # Convert CSV annotations to YOLO
│   ├── csv_to_oos_gt.py         # Convert CSV to OOS ground truth JSON
│   ├── infer_yolo.py            # Run YOLO inference on images
│   ├── oos_row_gap.py           # Detect OOS gaps from detections
│   ├── oos_eval_bootstrap.py    # Evaluate predictions with bootstrap CIs
│   ├── oos_label_from_predictions.py # Review predictions (no-GUI)
│   ├── quick_box_annotator_lite.py  # Interactive box annotator
│   ├── full_image_highlighter.py    # Full-image box reviewer
│   ├── subset_qc_tools.py       # Subset quality control utilities
│   └── remap_subset_split.py    # Remap subset train/val/test splits
├── requirements.txt
└── README.md
```

## Usage

### 1. Convert COCO to YOLO Format

Convert COCO format annotations to YOLO format:

```bash
python src/convert_to_yolo.py \
    --coco_json path/to/annotations.json \
    --images_dir path/to/images \
    --out_dir path/to/output/labels
```

### 2. Convert CSV to YOLO Format

Convert CSV annotations (SKU-110K format) to YOLO labels:

```bash
python src/csv_to_yolo_sku110k_v2.py \
    --csv_path path/to/annotations.csv \
    --images_dir path/to/images/train \
    --out_dir path/to/labels/train
```

### 3. Run YOLO Inference

Run YOLO model inference on images:

```bash
python src/infer_yolo.py \
    --weights path/to/model.pt \
    --images_dir path/to/images \
    --out_json path/to/detections.json \
    --imgsz 640 \
    --conf 0.25 \
    --iou 0.45 \
    --device cuda:0  # or cpu
```

### 4. Detect OOS Gaps

Detect out-of-stock gaps from product detections:

```bash
python src/oos_row_gap.py \
    --detections_json path/to/detections.json \
    --images_dir path/to/images \
    --out_dir path/to/output \
    --row_tol_px 30 \
    --gap_factor 1.4 \
    --min_abs_gap 10 \
    --max_vis 1000
```

### 5. Interactive Annotation Tools

#### Quick Box Annotator (Lite)
Interactive tool for drawing bounding boxes:

```bash
python src/quick_box_annotator_lite.py \
    --images_dir path/to/images \
    --out_csv path/to/annotations.csv \
    --pattern "*.jpg" \
    --start_index 0 \
    --max_side 1200
```

**Controls:**
- Click & drag: draw rectangle
- ENTER/n: next image
- p: previous image
- u: undo last box
- c: clear all boxes
- 0: mark no gaps
- s: save without moving
- q/ESC: quit

#### Full Image Highlighter
Review predicted boxes on full images:

```bash
python src/full_image_highlighter.py \
    --pred_json path/to/predictions.json \
    --images_dir path/to/images \
    --out_csv path/to/accepted.csv
```

**Controls:**
- y: accept current box
- n: reject current box
- LEFT/RIGHT: previous/next box
- ENTER/SPACE: next image
- s: save partial CSV
- b: save and go back
- q/ESC: quit

#### Prediction Reviewer (No-GUI)
Review predictions from command line:

```bash
python src/oos_label_from_predictions.py \
    --pred_json path/to/predictions.json \
    --out_csv path/to/accepted.csv
```

### 6. Evaluation

Evaluate OOS predictions with bootstrap confidence intervals:

```bash
python src/oos_eval_bootstrap.py \
    --pred_json path/to/predictions.json \
    --gt_json path/to/ground_truth.json \
    --iou_thr 0.3 \
    --bootstrap 1000
```

### 7. Data Quality Control

Check subset quality and generate manifests:

```bash
python src/subset_qc_tools.py \
    --root path/to/subset \
    --bins "0-10,11-30,31-80,81-150,151-9999" \
    --manifest_out outputs/subset_manifest.json
```

### 8. Remap Subset Splits

Copy images/labels between train/val/test splits:

```bash
python src/remap_subset_split.py \
    --root path/to/subset \
    --src_split train \
    --dst_split test \
    --overwrite
```

### 9. Convert CSV to OOS Ground Truth

Convert CSV annotations to OOS ground truth JSON:

```bash
python src/csv_to_oos_gt.py \
    --csv_path path/to/annotations.csv \
    --out_json path/to/ground_truth.json \
    --images_dir path/to/images  # optional
```

## Data Formats

### YOLO Format
Each label file (`.txt`) contains one line per detection:
```
class_id center_x center_y width height
```
All coordinates are normalized (0-1).

### Detection JSON Format
```json
{
  "image1.jpg": [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
  "image2.jpg": [[x1, y1, x2, y2], ...]
}
```

### CSV Format
CSV files for annotations use the format:
```csv
filename,x1,y1,x2,y2
image1.jpg,100,200,150,250
image1.jpg,300,400,350,450
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
