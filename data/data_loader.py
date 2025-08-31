from sklearn.model_selection import train_test_split
import os
import numpy as np
from PIL import Image
import torch

def load_data(data_dir):
    """Load images and labels from the specified directory."""
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist")
    
    # Check if we have train subdirectory structure
    train_dir = os.path.join(data_dir, 'train')
    if os.path.exists(train_dir):
        # Use train directory for classes
        class_dir = train_dir
        print(f"Using train directory: {train_dir}")
    else:
        # Use data_dir directly if no train subdirectory
        class_dir = data_dir
        print(f"Using data directory directly: {data_dir}")
    
    class_names = sorted([d for d in os.listdir(class_dir) 
                         if os.path.isdir(os.path.join(class_dir, d))])
    all_paths = []
    all_labels = []

    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(class_dir, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            for img in images:
                img_path = os.path.join(class_path, img)
                # Verify image can be opened
                try:
                    with Image.open(img_path) as test_img:
                        test_img.verify()
                    all_paths.append(img_path)
                    all_labels.append(class_idx)
                except Exception as e:
                    print(f"Warning: Skipping corrupted image {img_path}: {e}")

    if len(all_paths) == 0:
        raise ValueError(f"No valid images found in {class_dir}")

    print(f"Found {len(all_paths)} images across {len(class_names)} classes")
    for idx, class_name in enumerate(class_names):
        count = all_labels.count(idx)
        print(f"  {class_name}: {count} images")

    return np.array(all_paths), np.array(all_labels), class_names

def split_data(paths, labels, test_size=0.2, random_state=42):
    """Split the dataset into training and validation sets."""
    return train_test_split(paths, labels, test_size=test_size, 
                          random_state=random_state, stratify=labels)

def split_data_with_model_seed(paths, labels, model_name, test_size=0.25):
    """Split data using model-specific seed to avoid identical validation sets"""
    from utils.config import MODEL_SPECIFIC_SEEDS
    
    # Utiliser le seed spécifique au modèle ou un seed par défaut
    seed = MODEL_SPECIFIC_SEEDS.get(model_name, 42)
    
    print(f"🎲 Using seed {seed} for {model_name} validation split")
    
    return train_test_split(paths, labels, test_size=test_size, 
                          random_state=seed, stratify=labels)

def get_data(data_dir):
    """Load and split the dataset."""
    paths, labels, class_names = load_data(data_dir)
    train_paths, val_paths, train_labels, val_labels = split_data(paths, labels)
    
    print(f"Training set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    
    return (train_paths, train_labels), (val_paths, val_labels), class_names

def get_class_weights(labels):
    """Calculate class weights for imbalanced datasets"""
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    n_classes = len(unique_labels)
    
    # Calculate weights inversely proportional to class frequency
    class_weights = {}
    for label, count in zip(unique_labels, counts):
        weight = total_samples / (n_classes * count)
        class_weights[int(label)] = weight
    
    print("Class weights:", class_weights)
    return class_weights

def create_pytorch_dataset_info(data_dir):
    """Create dataset information for PyTorch training"""
    (train_paths, train_labels), (val_paths, val_labels), class_names = get_data(data_dir)
    
    # Calculate class weights
    class_weights = get_class_weights(train_labels)
    
    # Convert to PyTorch tensor for loss weighting
    weight_tensor = torch.zeros(len(class_names))
    for label, weight in class_weights.items():
        weight_tensor[label] = weight
    
    dataset_info = {
        'train_paths': train_paths,
        'train_labels': train_labels,
        'val_paths': val_paths,
        'val_labels': val_labels,
        'class_names': class_names,
        'class_weights': class_weights,
        'weight_tensor': weight_tensor,
        'num_classes': len(class_names)
    }
    
    return dataset_info