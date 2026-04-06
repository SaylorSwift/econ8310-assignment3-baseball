""" 
#!pip3 install torch torchvision torchaudio
#!pip3 install plotly.express
#!pip install opencv-python
#!pip install "numpy<2"
#!pip install tqdm

import torch
print(torch.cuda.is_available())

# For reading data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np

# For visualizing
import plotly.express as px

# For model building
import torch
import torch.nn as nn
import torch.nn.functional as F

# Defining our data class

class BaseballProj(Dataset):
    def __init__(self, root, transform = None):
        # set folder location
        self.video_folder = os.path.join(root, "Raw Videos")
        self.anno_folder = os.path.join(root, "Annotations")
        self.transform = transform

        anno = sorted(os.listdir(self.anno_folder))

        #create paths for each existing annotation and corresponding video
        self.anno_paths = []
        self.video_paths = []
        for file in anno:
            file_name = os.path.splitext(file)[0]

            anno_path = os.path.join(self.anno_folder, file)
            video_path = os.path.join(self.video_folder, f"{file_name}.mov")

            self.anno_paths.append(anno_path)
            self.video_paths.append(video_path)

    def __len__(self):
        return len(self.anno_paths)

    # retrieve a single record based on index position `idx`
    def __getitem__(self, idx):
        vid_path = self.video_paths[idx]
        vid = cv2.VideoCapture(vid_path)

        std_size = 3840
        scale_size = 960
        std_frames = 48

        #load video data
        frames = []
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, channels = frame.shape
            
            square_frame = np.zeros((std_size, std_size, channels), dtype=np.uint8)
            
            # Paste the original frame into the top-left corner
            square_frame[0:h, 0:w] = frame
            
            # Resize down to 1024x1024 to save GPU memory
            resized_frame = cv2.resize(square_frame, (scale_size, scale_size))
            
            frames.append(resized_frame)
            
        vid.release()


        #load bonding boxes
        scale_ratio = scale_size / std_size

        xml_path = self.anno_paths[idx]
        tree = ET.parse(xml_path)
        root = tree.getroot()

        ball_dict = {}
        
        for track in root.findall('track'):
            if track.get('label') == 'baseball':
                for box in track.findall('box'):
                    
                    is_moving = False
                    for attr in box.findall('attribute'):
                        if attr.get('name') == 'moving' and attr.text == 'true':
                            is_moving = True
                            break
                    
                    is_visible = box.get('outside') == '0'
                    
                    if is_moving and is_visible:
                        frame_num = int(box.get('frame'))
                        xtl = float(box.get('xtl')) * scale_ratio
                        ytl = float(box.get('ytl')) * scale_ratio
                        xbr = float(box.get('xbr')) * scale_ratio
                        ybr = float(box.get('ybr')) * scale_ratio

                        ball_dict[frame_num] = [xtl, ytl, xbr, ybr]


        #adding filler frames
        target_boxes = []
        target_labels = []
        
        for i in range(len(frames)):
            if i in ball_dict:
                target_boxes.append(ball_dict[i])
                target_labels.append(1)
            else:
                target_boxes.append([0.0, 0.0, 0.0, 0.0])
                target_labels.append(0)

        if len(frames) > std_frames:
            frames = frames[:std_frames]
            target_boxes = target_boxes[:std_frames]
            target_labels = target_labels[:std_frames]

        while len(frames) < std_frames:
            # Note: Make sure the zeros array matches your 960 target size!
            black_frame = np.zeros((960, 960, 3), dtype=np.uint8)
            frames.append(black_frame)
            target_boxes.append([0.0, 0.0, 0.0, 0.0])
            target_labels.append(0)
        

        frames_tensor = torch.stack([torch.from_numpy(f) for f in frames])
        
        boxes_tensor = torch.tensor(target_boxes, dtype=torch.float32)
        
        labels_tensor = torch.tensor(target_labels, dtype=torch.int64)

        if self.transform:
            frames_tensor, boxes_tensor, labels_tensor = self.transform(
                frames_tensor, boxes_tensor, labels_tensor
            )
            
        return frames_tensor, boxes_tensor, labels_tensor


from torch.utils.data import random_split

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create separate dataloaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False) """