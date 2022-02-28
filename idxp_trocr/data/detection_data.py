import torch
from torch.utils.data import Dataset
from ..utils import read_image, pre_process

class DetectionDataset(Dataset):
    def __init__(self, file_paths, roi_flag) -> None:
        super().__init__()
        self.file_paths = file_paths
        self.roi_flag = roi_flag
    
    def __len__(self):
        return len(self.file_paths)
    
    def _preprocess(self, file_path):
        img = read_image(file_path)
        if self.roi_flag:
            target_h, target_w = 300, 1028
        else:
            target_h, target_w = 1028, 700
        
        # provide mean and variance from config
        return pre_process(img, size=(target_h, target_w))
    
    def __getitem__(self, idx) -> torch.Tensor:
        return {
            "images": self._preprocess(self.file_paths[idx]),
            "idxs": idx
        }