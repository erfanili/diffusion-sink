import pickle
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import json

class LatentNoiseDataset(Dataset):
    def __init__(self, pickle_file='laion_10000_noise_pred_all_initseed.pkl', caption_file='laion_400m_random10000.json', transform=None):
        """
        Args:
            pickle_file (str): Path to the pickle file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)
        
        with open(caption_file, 'r') as f:
            self.captions = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[str(idx)]
        # breakpoint()
        # if self.transform:
        #     sample = self.transform(sample)
        noise_pred = np.concatenate(sample[0], axis=0)
        seed = sample[1]
        text_prompt = self.captions[str(idx)]
        return noise_pred,seed, text_prompt

if __name__ == "__main__":
    dataset = LatentNoiseDataset()
    # breakpoint()
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, drop_last=True)
    for batch in dataloader:
        noise_pred,seed, text_prompt = batch 
        print(batch)
        breakpoint()# batch * 20 (timesteps) * 4 * 64 * 64
        