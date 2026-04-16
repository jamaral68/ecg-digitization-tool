import torch
import cv2
import os

class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith((".png", ".jpg"))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        # imagem
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        boxes = []
        labels = []

        # ler YOLO
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f.readlines():
                    cls, xc, yc, bw, bh = map(float, line.strip().split())
                    
                    # converter para pixel
                    x1 = (xc - bw/2) * w
                    y1 = (yc - bh/2) * h
                    x2 = (xc + bw/2) * w
                    y2 = (yc + bh/2) * h
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cls) + 1)  # +1 por causa do background

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        img = torch.as_tensor(img / 255., dtype=torch.float32).permute(2, 0, 1)

        return img, target