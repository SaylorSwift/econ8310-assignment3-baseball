import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import math

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ==========================================
# 0. REQUIRED: CUSTOM COLLATE FUNCTION
# ==========================================
# Faster R-CNN expects a list of dictionaries. PyTorch's default collate
# tries to stack them into a single tensor and crashes. This prevents that.
def collate_fn(batch):
    return tuple(zip(*batch))

# ==========================================
# 1. THE DATASET CLASS (All-Frames Registry)
# ==========================================
class BaseballAllFrames(Dataset):
    def __init__(self, root, target_size=640):
        self.video_folder = os.path.join(root, "Raw Videos")
        self.anno_folder = os.path.join(root, "Annotations")
        self.target_size = target_size
        
        anno_files = sorted([f for f in os.listdir(self.anno_folder) if f.endswith('.xml')])
        self.registry = []

        print("Scanning videos and indexing ALL frames (positive and negative)...")
        for file in anno_files:
            file_name = os.path.splitext(file)[0]
            anno_path = os.path.join(self.anno_folder, file)
            video_path = os.path.join(self.video_folder, f"{file_name}.mov")

            if not os.path.exists(video_path):
                continue

            # 1. Get video dimensions and length
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()

            # 2. Parse XML for moving balls
            tree = ET.parse(anno_path)
            root_xml = tree.getroot()
            ball_map = {}
            
            for track in root_xml.findall('track'):
                if track.get('label') == 'baseball':
                    for box in track.findall('box'):
                        is_moving = any(attr.text == 'true' for attr in box.findall('attribute') if attr.get('name') == 'moving')
                        is_visible = box.get('outside') == '0'
                        
                        if is_moving and is_visible:
                            f_idx = int(box.get('frame'))
                            ball_map[f_idx] = [
                                float(box.get('xtl')), float(box.get('ytl')),
                                float(box.get('xbr')), float(box.get('ybr'))
                            ]

            # 3. Calculate the Safety Square scaling
            std_res = max(h, w)
            scale_ratio = self.target_size / std_res

            # 4. Register every single frame
            for f in range(total_frames):
                if f in ball_map:
                    raw_box = ball_map[f]
                    scaled_box = [
                        max(0.0, min(self.target_size - 1.0, raw_box[0] * scale_ratio)),
                        max(0.0, min(self.target_size - 1.0, raw_box[1] * scale_ratio)),
                        max(0.0, min(self.target_size - 1.0, raw_box[2] * scale_ratio)),
                        max(0.0, min(self.target_size - 1.0, raw_box[3] * scale_ratio))
                    ]
                    # Filter out boxes that are mathematically too small
                    if (scaled_box[2] - scaled_box[0]) >= 2.0 and (scaled_box[3] - scaled_box[1]) >= 2.0:
                        self.registry.append((video_path, f, scaled_box, 1, h, w, std_res))
                    else:
                        self.registry.append((video_path, f, None, 0, h, w, std_res))
                else:
                    # Negative frame (No ball)
                    self.registry.append((video_path, f, None, 0, h, w, std_res))
                    
        print(f"Registry Complete! {len(self.registry)} total frames queued.")

    def __len__(self):
        return len(self.registry)

    def __getitem__(self, idx):
        vid_path, f_idx, box, label, h, w, std_res = self.registry[idx]

        # Read Frame
        cap = cv2.VideoCapture(vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        cap.release()

        # Fallback for bad frame reads
        if not ret:
            frame = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Apply Safety Square Padding
            square = np.zeros((std_res, std_res, 3), dtype=np.uint8)
            square[0:h, 0:w] = frame
            frame = cv2.resize(square, (self.target_size, self.target_size))

        # Output format: [Channels, Height, Width] as uint8
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1)

        # Build Faster R-CNN Target Dictionary
        target = {}
        if label == 1 and box is not None:
            target["boxes"] = torch.tensor([box], dtype=torch.float32)
            target["labels"] = torch.tensor([1], dtype=torch.int64)
            target["area"] = torch.tensor([(box[2]-box[0]) * (box[3]-box[1])], dtype=torch.float32)
            target["iscrowd"] = torch.tensor([0], dtype=torch.int64)
        else:
            # Tell Faster R-CNN there are 0 objects here
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        return img_tensor, target

# ==========================================
# 2. THE MODEL DEFINITION
# ==========================================
def get_baseball_tracker_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    num_classes = 2 # 0: Background, 1: Baseball
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ==========================================
# 3. THE MAIN TRAINING SCRIPT
# ==========================================
if __name__ == "__main__":
    
    print("Initializing Dataset...")
    dataset = BaseballAllFrames(root=".", target_size=640) 
    
    total_size = len(dataset)
    if total_size == 0:
        print("Error: No data found.")
        exit()

    # Train/Val Split
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # IMPORTANT: We use our custom collate_fn here!
    # Because we are processing individual images now, we can safely bump batch_size to 8 on a 5090
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = get_baseball_tracker_model().to(device)

    save_file = "fasterrcnn_all_frames.pth"
    if os.path.exists(save_file):
        print(f"Loading existing brain from '{save_file}'...")
        model.load_state_dict(torch.load(save_file, map_location=device, weights_only=True))
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0005)
    
    # Required for safe mixed precision with complex models like Faster R-CNN
    scaler = torch.amp.GradScaler('cuda')

    num_epochs = 50
    print("\nStarting Training Loop...")

    for epoch in range(num_epochs):
        model.train() 
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_imgs, batch_targets in progress_bar:
            
            # Send uint8 images to GPU, then convert to float and scale
            images = list(img.to(device, dtype=torch.float32) / 255.0 for img in batch_imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch_targets]

            optimizer.zero_grad()
            
            # Hardware Acceleration
            with torch.amp.autocast('cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            if not math.isfinite(losses.item()):
                print(f"\n[WARNING] Loss exploded to {losses.item()}! Skipping.")
                continue
            
            # Scale gradients and step
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer) # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += losses.item()
            progress_bar.set_postfix(loss=f"{losses.item():.4f}")
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Summary | Avg Loss: {avg_loss:.4f} ---")

        torch.save(model.state_dict(), save_file)
        print("Checkpoint saved!")

    print("Training Complete!")