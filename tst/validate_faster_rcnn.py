"""
Validation of a Faster R-CNN model with a dataset in YOLO format
Metrics: mAP50-95, mAP50, Precision, Recall
"""
 
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
 
class YOLODataset(Dataset):
    """
    Expected structure of a YOLO dataset:
    """

    def __init__(self, images_dir: str, labels_dir: str, transforms=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        self.image_paths = sorted(
            p for p in self.images_dir.rglob("*") if p.suffix.lower() in exts
        )

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in: {images_dir}")

        print(f"  → {len(self.image_paths)} images found in '{images_dir}'")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        label_path = self.labels_dir / img_path.with_suffix(".txt").name
        boxes, labels = [], []

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])

                    x1 = (xc - bw / 2) * w
                    y1 = (yc - bh / 2) * h
                    x2 = (xc + bw / 2) * w
                    y2 = (yc + bh / 2) * h

                    x1, x2 = max(0, x1), min(w, x2)
                    y1, y2 = max(0, y1), min(h, y2)

                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls + 1)

        boxes  = torch.as_tensor(boxes,  dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([idx]),
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target
 
 
def collate_fn(batch):
    """
    Custom collate function used for object detection datasets.
    It groups images and targets separately because detection models
    expect a list of tensors instead of a single stacked tensor.
    """
    return tuple(zip(*batch))

def load_model(weights_path: str, num_classes: int, device: torch.device) -> FasterRCNN:
    """
    Loads a Faster R-CNN ResNet-50 FPN model and replaces the head with
    the correct number of classes saved in the checkpoint.
    """
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
 
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict",
                     checkpoint.get("state_dict",
                     checkpoint.get("model", checkpoint)))
    else:
        state_dict = checkpoint
 
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"  → Modelo carregado de '{weights_path}' ({num_classes} classes)")
    return model
 
def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """returns array IoU (N x M)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
 
    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
 
    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    union_area  = area1[:, None] + area2[None, :] - inter_area
    return inter_area / (union_area + 1e-6)

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Area under the P–R curve using 101-point interpolation (COCO style)."""
    recalls    = np.concatenate([[0.0], recalls,    [1.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])
 
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
 
    thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    for t in thresholds:
        p = precisions[recalls >= t]
        ap += (p.max() if p.size > 0 else 0.0)
    return ap / 101.0
 
 
def evaluate_class(preds_cls, gts_cls, iou_threshold: float):
    """
    preds_cls: list of (score, boxes_np[N,4]) per image  
    gts_cls:   list of boxes_np[M,4]           per image  
    Returns: AP, precision, recall
    """
    all_scores, all_boxes, all_img_ids = [], [], []
    for img_id, (scores, boxes) in enumerate(preds_cls):
        all_scores.extend(scores)
        all_boxes.extend(boxes)
        all_img_ids.extend([img_id] * len(scores))
 
    total_gt = sum(len(b) for b in gts_cls)
 
    if total_gt == 0 and len(all_scores) == 0:
        return 1.0, 1.0, 1.0 
    if total_gt == 0:
        return 0.0, 0.0, 0.0
    if len(all_scores) == 0:
        return 0.0, 0.0, 0.0
 
    order      = np.argsort(all_scores)[::-1]
    all_scores = np.array(all_scores)[order]
    all_boxes  = np.array(all_boxes)[order]
    all_img_ids = np.array(all_img_ids)[order]
 
    matched = [np.zeros(len(b), dtype=bool) for b in gts_cls]
    tp_list, fp_list = [], []
 
    for score, box, img_id in zip(all_scores, all_boxes, all_img_ids):
        gt_boxes = gts_cls[img_id]
        if len(gt_boxes) == 0:
            tp_list.append(0); fp_list.append(1)
            continue
 
        ious = box_iou(box[None], gt_boxes)[0]
        best_iou_idx = ious.argmax()
        best_iou     = ious[best_iou_idx]
 
        if best_iou >= iou_threshold and not matched[img_id][best_iou_idx]:
            matched[img_id][best_iou_idx] = True
            tp_list.append(1); fp_list.append(0)
        else:
            tp_list.append(0); fp_list.append(1)
 
    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
 
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
    recalls    = tp_cum / (total_gt + 1e-6)
 
    ap        = compute_ap(recalls, precisions)
    precision = precisions[-1] if len(precisions) > 0 else 0.0
    recall    = recalls[-1]    if len(recalls)    > 0 else 0.0
 
    return float(ap), float(precision), float(recall)
 
@torch.no_grad()
def collect_predictions(model, dataloader, device, conf_threshold=0.05):
    """
    Returns two structures indexed by class (1-indexed).
    """
    preds = defaultdict(list)  
    gts   = defaultdict(list)  
    n_images = 0
 
    for images, targets in tqdm(dataloader, desc="Inferência", unit="batch"):
        images = [img.to(device) for img in images]
        outputs = model(images)
 
        for output, target in zip(outputs, targets):
            n_images += 1
 
            gt_boxes  = target["boxes"].cpu().numpy()
            gt_labels = target["labels"].cpu().numpy()
            present_classes = set(gt_labels.tolist())

            all_gt_classes = set(gt_labels.tolist())
            for cls in all_gt_classes:
                mask = gt_labels == cls
                gts[cls].append(gt_boxes[mask])
 
            pred_boxes  = output["boxes"].cpu().numpy()
            pred_scores = output["scores"].cpu().numpy()
            pred_labels = output["labels"].cpu().numpy()
 
            mask_conf = pred_scores >= conf_threshold
            pred_boxes  = pred_boxes[mask_conf]
            pred_scores = pred_scores[mask_conf]
            pred_labels = pred_labels[mask_conf]
 
            pred_classes = set(pred_labels.tolist())
            all_classes  = all_gt_classes | pred_classes
 
            for cls in all_classes:
                m = pred_labels == cls
                scores = pred_scores[m]
                boxes  = pred_boxes[m]
                preds[cls].append((scores.tolist(), boxes))
 
                if cls not in all_gt_classes:
                    gts[cls].append(np.zeros((0, 4), dtype=np.float32))

            for cls in all_gt_classes:
                if cls not in pred_classes:
                    preds[cls].append(([], np.zeros((0, 4), dtype=np.float32)))
 
    print(f"  → {n_images} processed images")
    return dict(preds), dict(gts)
 
IOU_THRESHOLDS_5095 = np.arange(0.50, 1.00, 0.05) 
 
 
def compute_metrics(preds, gts, class_names=None):
    all_classes = sorted(set(preds) | set(gts))
 
    results = {}
    for cls in all_classes:
        cls_preds = preds.get(cls, [])
        cls_gts   = gts.get(cls, [])

        n = max(len(cls_preds), len(cls_gts))
        cls_preds = cls_preds + [([], np.zeros((0, 4)))] * (n - len(cls_preds))
        cls_gts   = cls_gts   + [np.zeros((0, 4))]      * (n - len(cls_gts))

        ap50, p50, r50 = evaluate_class(cls_preds, cls_gts, iou_threshold=0.50)

        aps = []
        for iou_t in IOU_THRESHOLDS_5095:
            ap, _, _ = evaluate_class(cls_preds, cls_gts, iou_threshold=round(iou_t, 2))
            aps.append(ap)
        ap5095 = float(np.mean(aps))
 
        name = class_names[cls - 1] if (class_names and 0 < cls <= len(class_names)) else f"class_{cls}"
        results[cls] = {
            "name":      name,
            "mAP50":     ap50,
            "mAP50-95":  ap5095,
            "precision": p50,
            "recall":    r50,
        }
 
    mean_map50    = float(np.mean([v["mAP50"]    for v in results.values()]))
    mean_map5095  = float(np.mean([v["mAP50-95"] for v in results.values()]))
    mean_precision= float(np.mean([v["precision"]for v in results.values()]))
    mean_recall   = float(np.mean([v["recall"]   for v in results.values()]))
 
    return results, {
        "mAP50":     mean_map50,
        "mAP50-95":  mean_map5095,
        "precision": mean_precision,
        "recall":    mean_recall,
    }
 
 

def print_report(per_class, summary, elapsed):
    sep = "─" * 68
    print(f"\n{sep}")
    print(f"{'VALIDATION RESULTS':^68}")
    print(sep)
    print(f"{'Class':<20} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>11} {'Recall':>9}")
    print(sep)
    for cls_id, v in sorted(per_class.items()):
        print(f"{v['name']:<20} {v['mAP50']:>8.4f} {v['mAP50-95']:>10.4f} "
              f"{v['precision']:>11.4f} {v['recall']:>9.4f}")
    print(sep)
    print(f"{'ALL (averages)':<20} {summary['mAP50']:>8.4f} {summary['mAP50-95']:>10.4f} "
          f"{summary['precision']:>11.4f} {summary['recall']:>9.4f}")
    print(sep)
    print(f"Total time: {elapsed:.1f}s\n")

def parse_args():
    p = argparse.ArgumentParser(description="Faster R-CNN validation with YOLO dataset")
    p.add_argument("--weights",     required=True,  help="Path to the .pth file")
    p.add_argument("--images-dir",  required=True,  help="Directory with validation images")
    p.add_argument("--labels-dir",  required=True,  help="Directory with YOLO labels (.txt)")
    p.add_argument("--num-classes", required=True,  type=int,
                   help="Number of classes (excluding background)")
    p.add_argument("--class-names", default=None,
                   help="Path to a .txt file with one class name per line (optional)")
    p.add_argument("--batch-size",  default=4,      type=int)
    p.add_argument("--conf-thresh", default=0.05,   type=float,
                   help="Minimum confidence threshold for predictions (default 0.05)")
    p.add_argument("--device",      default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--output-json", default=None,
                   help="Save results to JSON (optional)")
    return p.parse_args()
 
 
def main():
    args = parse_args()
 
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps"  if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n[Config] device={device}  batch={args.batch_size}  conf≥{args.conf_thresh}")
 
    class_names = None
    if args.class_names:
        with open(args.class_names) as f:
            class_names = [l.strip() for l in f if l.strip()]
        print(f"  → {len(class_names)} classes: {class_names}")

    print("\n[Dataset]")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = YOLODataset(args.images_dir, args.labels_dir, transforms=transform)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, collate_fn=collate_fn,
                         num_workers=min(4, os.cpu_count() or 1))
 
    print("\n[Model]")
    model = load_model(args.weights, num_classes=args.num_classes + 1, device=device)

    print("\n[Inferência]")
    t0 = time.time()
    preds, gts = collect_predictions(model, loader, device, conf_threshold=args.conf_thresh)

    print("\n[Calculating metrics...]")
    per_class, summary = compute_metrics(preds, gts, class_names)
    elapsed = time.time() - t0
 
    print_report(per_class, summary, elapsed)
 
    if args.output_json:
        out = {"summary": summary, "per_class": per_class}
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {args.output_json}")
 
 
if __name__ == "__main__":
    main()