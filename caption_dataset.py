from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from typing import Type

# Dataset class for Phase 1
class CaptionDataset(Dataset):
    def __init__(self, captions, full_dataset, processor, smol_tokenizer):
        self.captions = captions
        self.full_dataset = full_dataset
        self.processor = processor
        self.smol_tokenizer = smol_tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption_data = self.captions[idx]
        image_idx = caption_data["image_idx"]
        caption = caption_data["caption"]

        pil_image = self.full_dataset[image_idx][0]
        pixel_values = self.processor(images=pil_image, return_tensors="pt")["pixel_values"].squeeze(0)
        caption_ids = self.smol_tokenizer(caption, return_tensors="pt")["input_ids"].squeeze(0)

        return pixel_values, caption_ids

    # Collate function to handle variable length captions
    def collate_fn(self, batch):
        pixel_values = torch.stack([item[0] for item in batch])
        caption_ids = torch.nn.utils.rnn.pad_sequence(
            [item[1] for item in batch], batch_first=True, padding_value=self.smol_tokenizer.pad_token_id or self.smol_tokenizer.eos_token_id
        )
        return pixel_values, caption_ids

def create_captions_dataset_loader(CaptionDataset: Type[Dataset], captions, full_train, processor, smol_tokenizer):

    # Create dataset and dataloader
    caption_dataset = CaptionDataset(captions, full_train, processor, smol_tokenizer)
    caption_loader = DataLoader(caption_dataset, batch_size=32, shuffle=True, collate_fn=caption_dataset.collate_fn)
    return caption_dataset, caption_loader


