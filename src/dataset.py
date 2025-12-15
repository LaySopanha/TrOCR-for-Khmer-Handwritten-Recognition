import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class KhmerOCRDataset(Dataset):
    def __init__(self, root_dir,  df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df 
        self.processor = processor
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        file_name = self.df.iloc[idx]["file_name"]
        text = self.df.iloc[idx]["text"]
        image_path = os.path.join(self.root_dir, file_name)
        
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
        ).input_ids
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels),
        }
