import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import os
from torch.utils.data import DataLoader, random_split

# import data loader
from data_loader import BaseballProj 

# cutom cnn
class CustomBaseballCNN(nn.Module):
    def __init__(self):
        super(CustomBaseballCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4), # Shrinks to 240x240
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4), # Shrinks to 60x60
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Shrinks to 30x30
        )
        
        self.flatten = nn.Flatten()
        
        self.classifier_head = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 1) # Outputs a single probability score
        )
        
        self.box_head = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        # Scan the image
        x = self.features(x)
        x = self.flatten(x)
        
        label_pred = self.classifier_head(x)
        box_pred = self.box_head(x)
        
        return label_pred, box_pred


# training loop
def train_loop(dataloader, model, optimizer, device):
    model.train()
    size = len(dataloader.dataset)
    epoch_loss = 0
    
    class_loss_fn = nn.BCEWithLogitsLoss()
    box_loss_fn = nn.SmoothL1Loss() 

    accumulation_steps = 2
    optimizer.zero_grad()  

    for batch_idx, (frames, boxes, labels) in enumerate(dataloader):
        frames = frames.squeeze(0).permute(1, 0, 2, 3).to(device)
        frames = frames.to(torch.float32) / 255.0
        
        boxes = boxes.squeeze(0).to(device)
        labels = labels.squeeze(0).unsqueeze(1).to(torch.float32).to(device)

        with torch.amp.autocast('cuda'):
            pred_labels, pred_boxes = model(frames)
            loss_class = class_loss_fn(pred_labels, labels)
            
            has_ball_mask = (labels == 1.0).squeeze()
            if has_ball_mask.any():
                loss_box = box_loss_fn(pred_boxes[has_ball_mask], boxes[has_ball_mask])
            else:
                loss_box = torch.tensor(0.0, device=device)
                
            total_loss = (loss_class + loss_box) / accumulation_steps
            
        if not math.isfinite(total_loss.item()):
            print(f"Loss exploded! Skipping.")
            continue
            
        total_loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss += total_loss.item() * accumulation_steps
        
        if batch_idx % 5 == 0:
            print(f"Video [{batch_idx+1}/{len(dataloader)}] | Loss: {total_loss.item()*accumulation_steps:.4f}")
            
    return epoch_loss / size

def test_loop(dataloader, model, device):
    model.eval()
    size = len(dataloader.dataset)
    test_loss = 0
    
    class_loss_fn = nn.BCEWithLogitsLoss()
    box_loss_fn = nn.SmoothL1Loss()

    with torch.no_grad():
        for frames, boxes, labels in dataloader:
            B, C, F, H, W = frames.shape
            frames = frames.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            frames = frames.to(device).to(torch.float32) / 255.0

            boxes = boxes.squeeze(0).to(device)
            labels = labels.squeeze(0).unsqueeze(1).to(torch.float32).to(device)
            
            pred_labels, pred_boxes = model(frames)
            
            loss_class = class_loss_fn(pred_labels, labels)
            
            has_ball_mask = (labels == 1.0).squeeze()
            if has_ball_mask.any():
                loss_box = box_loss_fn(pred_boxes[has_ball_mask], boxes[has_ball_mask])
            else:
                loss_box = torch.tensor(0.0, device=device)
                
            total_loss = loss_class + loss_box
            
            if math.isfinite(total_loss.item()):
                test_loss += total_loss.item()
                
    return test_loss / size


if __name__ == "__main__":    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    print("Loading Dataset...")
    full_dataset = BaseballProj(root=".")
    
    from torch.utils.data import DataLoader, random_split
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    
    train_data, test_data = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Total videos: {total_size} | Training on {train_size} | Testing on {test_size}")
    
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True) 
    
    model = CustomBaseballCNN().to(device)
    
    print("Initializing Lazy Layers...")
    dummy_input = torch.zeros(1, 3, 960, 960).to(device)
    model(dummy_input)
    
    save_file = "cnn_model.pth"
    if os.path.exists(save_file):
        print(f"Loading existing brain from '{save_file}'...")
        model.load_state_dict(torch.load(save_file, map_location=device, weights_only=True))
    else:
        print("No saved model found. Starting from scratch!")
        
    learning_rate = 1e-4
    epochs = 20
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("\nStarting Training Loop...")
    for t in range(epochs):
        print(f"\nEpoch {t+1}/{epochs}\n-------------------------------")
        
        # Run Training
        train_avg = train_loop(train_dataloader, model, optimizer, device)
        
        # Run Testing
        test_avg = test_loop(test_dataloader, model, device)
        
        print(f"--- Epoch {t+1} Summary ---")
        print(f"Train Loss: {train_avg:.4f} | Validation Loss: {test_avg:.4f}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), save_file)
        print("Checkpoint Saved!")
        
    print("Training Complete!")