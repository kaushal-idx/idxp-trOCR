"""
dataset fro recogntion

1. apply generate crops on batch of images
2. index the crops, per image, word_num
3. return a recog dataset
"""
import torch
from torch.utils.data import Dataset
from ..utils import inv_normalization
import numpy as np
from typing import List, Dict
from PIL import Image


class RecognitionDataset(Dataset):
    def __init__(self, bbox_image_idx_list, trocr_processor) -> None:
        super().__init__()
        self.items = bbox_image_idx_list
        self.processor = trocr_processor

    def __len__(self):
        return len(self.items)

    def _get_pixel_values(self, roi):
        image = Image.fromarray(roi).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        return pixel_values

    def _generate_crops(self, image: torch.Tensor, box: List, idx: int, i:int)->Dict:
        img = inv_normalization(image)
        left, right, top, bottom = int(box[0][0]), int(
            box[2][0]), int(box[0][1]), int(box[2][1])
        roi = np.array(img)[top:bottom, left:right]
        pixel_values = self._get_pixel_values(roi)
        return {
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "word_num": i,
            # "roi": roi,
            "pixel_values": pixel_values.squeeze(),
            "idxs": idx
        }

    def __getitem__(self, indx: int) -> Dict:
        image, bbox, idx , i = self.items[indx]
        return self._generate_crops(image, bbox, idx, i)