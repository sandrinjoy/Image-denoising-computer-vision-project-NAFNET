import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
from torchvision import transforms

class DenoisingDataset(Dataset):
    def __init__(self, root_dir='./data', train=True, transform=None, noise_level=0.1):
        """
        Args:
            root_dir (string): Directory with the CIFAR-10 dataset
            train (bool): If True, use training set, else use test set
            transform (callable, optional): Optional transform to be applied on images
            noise_level (float): Level of noise to add to images
        """
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.noise_level = noise_level
        
        # Load CIFAR-10 data
        if train:
            self.data = np.load(os.path.join(root_dir, 'cifar-10-batches-py', 'data_batch_1'))
        else:
            self.data = np.load(os.path.join(root_dir, 'cifar-10-batches-py', 'test_batch'))
            
        self.data = self.data['data'].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img)
        
        # Apply transforms
        img = self.transform(img)
        
        # Add noise
        noisy_img = img + torch.randn_like(img) * self.noise_level
        noisy_img = torch.clamp(noisy_img, -1, 1)
        
        return noisy_img, img

def get_dataloader(batch_size=32, num_workers=4, train=True, noise_level=0.1):
    """
    Create DataLoader for the denoising dataset
    """
    dataset = DenoisingDataset(train=train, noise_level=noise_level)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader 