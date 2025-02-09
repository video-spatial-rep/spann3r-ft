import os
import time
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Import your model.
from spann3r.model import Spann3R

# Allow safe globals for loading checkpoints
torch.serialization.add_safe_globals([argparse.Namespace])

###############################################################################
# Custom Dataset for Raw Videos
###############################################################################

class RawVideoDataset(Dataset):
    """
    A dataset that reads raw video files from a directory and, for each video,
    extracts a fixed number of frames (num_frames) evenly spaced across the video.
    
    Each video is returned as a list of frame dictionaries. Each frame dictionary
    contains:
      - 'img': a torch.Tensor of shape [1, 3, 224, 224] after resizing/cropping,
      - 'true_shape': a tensor of shape [1, 224, 224] indicating the image size,
      - 'label': the video file name.
    """
    def __init__(self, video_dir, num_frames=32, transform=None):
        self.video_dir = video_dir
        self.video_files = [osp.join(video_dir, f) for f in os.listdir(video_dir)
                            if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        self.num_frames = num_frames
        # Transformation pipeline: convert to PIL image, resize to 224x224, and convert to tensor.
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),   # Resize to 224x224
            transforms.ToTensor()            # Convert to tensor with shape [3, 224, 224]
        ])
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Compute indices for exactly num_frames evenly spaced frames.
        if total_frames < self.num_frames:
            # If the video has fewer frames, take all of them.
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / float(self.num_frames)
            frame_indices = [int(step * i) for i in range(self.num_frames)]
        
        frames_dict = []
        current_frame = 0
        ret = True
        while ret:
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame in frame_indices:
                # Convert BGR (OpenCV default) to RGB.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Apply the transformation pipeline.
                image_tensor = self.transform(frame)  # now shape: [3, 224, 224]
                # Add a batch dimension so that image_tensor becomes [1, 3, 224, 224].
                image_tensor = image_tensor.unsqueeze(0)
                # Set true_shape as a tensor with shape [1, 224, 224]. Since we resized, it is fixed.
                true_shape = torch.tensor([224, 224]).unsqueeze(0)  # you can also use [height, width]
                frames_dict.append({
                    'img': image_tensor,
                    'true_shape': true_shape,
                    'label': osp.basename(video_path)
                })
            current_frame += 1
        cap.release()
        return frames_dict

###############################################################################
# Argument Parser
###############################################################################

def get_args_parser():
    parser = argparse.ArgumentParser('Spann3R Raw Video Feature Extraction', add_help=False)
    parser.add_argument('--video_dir', type=str, default='/data_new/spatial/Training/videos/arkitscenes/arkitscenes_videos_128f',
                        help='Path to directory containing raw videos')
    parser.add_argument('--exp_path', type=str, default='/data_new/rilyn',
                        help='Path to experiment folder (results will be saved here)')
    parser.add_argument('--exp_name', type=str, default='ckpt_best',
                        help='Experiment name folder')
    parser.add_argument('--ckpt', type=str, default='/home/rilyn/project-files/02-pj-cambrians/cambrians-prep/LLaVA-NeXT/spann3r/spann3r.pth',
                        help='Full path to the checkpoint file')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use, e.g. "cuda:0" or "cpu"')
    parser.add_argument('--num_frames', type=int, default=32,
                        help='Number of frames to extract per video')
    return parser

###############################################################################
# Main Function
###############################################################################

def main(args):
    # Create experiment folder.
    exp_path = osp.join(args.exp_path, args.exp_name)
    os.makedirs(exp_path, exist_ok=True)
    
    # Create a subfolder to save f_H features.
    fH_save_path = osp.join(exp_path, 'fH_features')
    os.makedirs(fH_save_path, exist_ok=True)
    
    # Create dataset using raw videos.
    dataset = RawVideoDataset(video_dir=args.video_dir, num_frames=args.num_frames)
    datasets_all = {'RawVideos': dataset}
    
    # Load the model.
    model = Spann3R(
        dus3r_name='/home/rilyn/project-files/02-pj-cambrians/cambrians-prep/LLaVA-NeXT/spann3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth',
        use_feat=False
    ).to(args.device)
    
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device)['model'])
    model.eval()
    
    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            print(f"Dataset {name_data} has {len(dataset)} videos")
            
            # Create a folder to save results for this dataset.
            data_save_path = osp.join(exp_path, name_data)
            os.makedirs(data_save_path, exist_ok=True)
            print(f"Results for {name_data} will be saved in: {data_save_path}")
            
            # Use DataLoader with batch_size=1.
            # The collate_fn here simply returns the sample (a list of frames).
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])
            print(f"Starting evaluation loop for {name_data} with {len(dataloader)} videos")
            
            for i, video_frames in enumerate(dataloader):
                print(f"Processing video {i+1}/{len(dataloader)}")
                
                # Each video_frames is a list of frame dictionaries.
                # The 'img' and 'true_shape' fields already have a batch dimension.
                for view in video_frames:
                    view['img'] = view['img'].to(args.device, non_blocking=True)
                    # 'true_shape' is already a tensor with the correct shape.
                
                # Run the forward pass on the sequence of frames.
                # The forward method expects a list of frame dictionaries.
                preds, preds_all, f_H = model.forward(video_frames, return_memory=False)
                
                # Convert f_H to a CPU NumPy array and save.
                f_H_np = f_H.detach().cpu().numpy()
                # Use the video file name as an identifier.
                video_id = video_frames[0].get('label', f'video_{i}')
                if isinstance(video_id, str):
                    video_id = video_id.replace('/', '_')
                else:
                    video_id = f'video_{i}'
                fH_filename = osp.join(fH_save_path, f"fH_{video_id}.npy")
                np.save(fH_filename, f_H_np)
                print(f"Saved f_H for video {i+1} to {fH_filename}")
                
                # (Optional) Additional processing or evaluation metrics can be added here.
                
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
