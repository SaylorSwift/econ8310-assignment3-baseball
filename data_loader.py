import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

class BaseballProj(Dataset):
    def __init__(self, root, transform=None):
        self.video_folder = os.path.join(root, "Raw Videos")
        self.anno_folder = os.path.join(root, "Annotations")
        self.transform = transform

        # standardize video
        self.std_frames = 60
        self.std_res = 3840 
        self.scale_res = 960 


        anno_files = sorted(os.listdir(self.anno_folder))
        self.dataset_meta = []

        for file in anno_files:
            if not file.endswith('.xml'):
                continue
                
            file_name = os.path.splitext(file)[0]
            anno_path = os.path.join(self.anno_folder, file)
            video_path = os.path.join(self.video_folder, f"{file_name}.mov")

            if not os.path.exists(video_path):
                continue

            # Find min and max moving ball frame length
            tree = ET.parse(anno_path)
            root = tree.getroot()
            
            min_frame = float('inf')
            max_frame = -1

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
                            f = int(box.get('frame'))
                            min_frame = min(min_frame, f)
                            max_frame = max(max_frame, f)

            if max_frame >= 0:
                # Start 1 frame before the ball moves
                extract_start = max(0, min_frame - 1)
                
                # Lock the end frame exactly to our hardcoded std_frames
                extract_end = extract_start + self.std_frames - 1
                
                self.dataset_meta.append({
                    'video_path': video_path,
                    'anno_path': anno_path,
                    'start_frame': extract_start,
                    'end_frame': extract_end
                })

        print(f"Dataset ready! Processed {len(self.dataset_meta)} valid videos.")
        print(f"Settings: {self.std_frames} frames @ {self.scale_res}x{self.scale_res}")

    def __len__(self):
        return len(self.dataset_meta)

    def __getitem__(self, idx):
        # load metadata
        meta = self.dataset_meta[idx]
        vid_path = meta['video_path']
        xml_path = meta['anno_path']
        extract_start = meta['start_frame']
        extract_end = meta['end_frame']

        # load video
        vid = cv2.VideoCapture(vid_path)
        vid.set(cv2.CAP_PROP_POS_FRAMES, extract_start)

        frames = []
        current_frame = extract_start
        
        while vid.isOpened() and current_frame <= extract_end:
            ret, frame = vid.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, channels = frame.shape

            square_frame = np.zeros((self.std_res, self.std_res, channels), dtype=np.uint8)
            square_frame[0:h, 0:w] = frame
            
            resized_frame = cv2.resize(square_frame, (self.scale_res, self.scale_res))
            frames.append(resized_frame)
            
            current_frame += 1
            
        vid.release()

        # load bonding box
        scale_ratio = self.scale_res / self.std_res

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
                        original_frame = int(box.get('frame'))
                        adjusted_idx = original_frame - extract_start
                        
                        # scale bounding box
                        if 0 <= adjusted_idx < len(frames):
                            xtl = float(box.get('xtl')) * scale_ratio
                            ytl = float(box.get('ytl')) * scale_ratio
                            xbr = float(box.get('xbr')) * scale_ratio
                            ybr = float(box.get('ybr')) * scale_ratio

                            # prevent out of bound error
                            max_bound = float(self.scale_res - 1.0)
                            xtl = max(0.0, min(max_bound, xtl))
                            ytl = max(0.0, min(max_bound, ytl))
                            xbr = max(0.0, min(max_bound, xbr))
                            ybr = max(0.0, min(max_bound, ybr))

                            if (xbr - xtl) < 2.0 or (ybr - ytl) < 2.0:
                                continue 

                            ball_dict[adjusted_idx] = [xtl, ytl, xbr, ybr]

        target_boxes = []
        target_labels = []
        
        for i in range(len(frames)):
            if i in ball_dict:
                target_boxes.append(ball_dict[i])
                target_labels.append(1)
            else:
                target_boxes.append([0.0, 0.0, 0.0, 0.0])
                target_labels.append(0)

        # cut video if frames too many
        if len(frames) > self.std_frames:
            frames = frames[:self.std_frames]
            target_boxes = target_boxes[:self.std_frames]
            target_labels = target_labels[:self.std_frames]

        # add frames if too few
        while len(frames) < self.std_frames:
            black_frame = np.zeros((self.scale_res, self.scale_res, 3), dtype=np.uint8)
            frames.append(black_frame)
            target_boxes.append([0.0, 0.0, 0.0, 0.0])
            target_labels.append(0)

        # Convert to pytorch form
        frames_tensor = torch.stack([torch.from_numpy(f) for f in frames])
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)
        
        boxes_tensor = torch.tensor(target_boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(target_labels, dtype=torch.int64)

        if self.transform:
            frames_tensor, boxes_tensor, labels_tensor = self.transform(
                frames_tensor, boxes_tensor, labels_tensor
            )
            
        return frames_tensor, boxes_tensor, labels_tensor
