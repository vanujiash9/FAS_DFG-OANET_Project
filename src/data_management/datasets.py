import os
import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.configs.config_loader import cfg

class OANetDataset(Dataset):
    def __init__(self, split_type, transform=None):
        if split_type not in ['train', 'val', 'test']:
            raise ValueError("split_type must be 'train', 'val', or 'test'")

        self.transform = transform
        self.samples = []
        
        self.cues_root_dir = Path(cfg.data.ANOMALOUS_CUES_DIR)
        splits_file_path = cfg.data.OANET_SPLITS_PATH 
        
        if not os.path.exists(splits_file_path):
            raise FileNotFoundError(f"OANet cue splits file not found: {splits_file_path}.")
        
        with open(splits_file_path, 'r') as f:
            splits_data = json.load(f)
        
        for relative_path, label in splits_data[split_type]:
            full_path = self.cues_root_dir / relative_path
            if os.path.exists(full_path):
                self.samples.append((str(full_path), label))
        
        print(f"OANetDataset: Loaded {len(self.samples)} CUE samples for {split_type} split.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cue_path, label = self.samples[idx]
        cue_tensor = torch.load(cue_path)
        return cue_tensor, label

def get_oanet_eval_transforms():
    return transforms.Compose([
        transforms.Resize((cfg.data.OANET_IMAGE_SIZE, cfg.data.OANET_IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

def get_oanet_dataloader(split_type, batch_size, shuffle=True, num_workers=None):
    if num_workers is None: num_workers = cfg.data.NUM_DATALOADER_WORKERS
    dataset = OANetDataset(split_type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)