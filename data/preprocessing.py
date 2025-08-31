import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os

class CornDiseaseDataset(Dataset):
    """Custom dataset for corn disease classification"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        # Convert label to LongTensor to avoid CUDA kernel error
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label

def get_transforms(augmentation_strength=2.0, is_training=True):  # Increased default strength
    """Get image transforms for training and validation with strong anti-overfitting augmentation"""
    
    if is_training:
        # Training transforms with STRONG augmentation to prevent overfitting
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(degrees=30 * augmentation_strength),  # Increased rotation
            transforms.RandomHorizontalFlip(p=0.6),  # Increased probability
            transforms.RandomVerticalFlip(p=0.3),  # Increased probability
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # NEW: Perspective transform
            transforms.ColorJitter(
                brightness=0.3 * augmentation_strength,  # Increased
                contrast=0.3 * augmentation_strength,    # Increased
                saturation=0.3 * augmentation_strength,  # Increased
                hue=0.15 * augmentation_strength         # Increased
            ),
            transforms.RandomResizedCrop(256, scale=(0.7, 1.0)),  # More aggressive crop
            transforms.RandomGrayscale(p=0.1),  # NEW: Random grayscale
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.2),  # NEW: Random blur
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),  # NEW: Random erasing
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def create_data_loaders(train_paths, train_labels, val_paths, val_labels, 
                       batch_size=32, augmentation_strength=1.0, num_workers=4):
    """Create PyTorch data loaders for training and validation"""
    
    # Get transforms
    train_transform = get_transforms(augmentation_strength=augmentation_strength, is_training=True)
    val_transform = get_transforms(is_training=False)
    
    # Create datasets
    train_dataset = CornDiseaseDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = CornDiseaseDataset(val_paths, val_labels, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

def preprocess_image(image_path, target_size=(256, 256)):
    """Preprocess a single image for inference"""
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor

def denormalize_image(tensor):
    """Denormalize image tensor for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    return tensor

def preprocess_data(train_paths, train_labels, val_paths, val_labels, 
                   batch_size=32, augmentation_strength=1.0):
    """Main preprocessing function - wrapper for backward compatibility"""
    return create_data_loaders(
        train_paths, train_labels, val_paths, val_labels,
        batch_size=batch_size, augmentation_strength=augmentation_strength
    )