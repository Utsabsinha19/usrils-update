import os
import urllib.request
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import datasets, transforms

# Paths for caching
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

def generate_synthetic_adult(n_samples=48842, input_dim=14):
    """Generates synthetic dataset mirroring the shape of UCI-Adult."""
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    # Simple linear decision boundary with some noise
    w = np.random.randn(input_dim)
    logits = np.dot(X, w) + 0.1 * np.random.randn(n_samples)
    y = (logits > 0).astype(np.int64)
    return torch.tensor(X), torch.tensor(y)

def get_uci_adult(train=True, use_synthetic=False):
    """
    Downloads, parses, and preprocesses the UCI-Adult dataset.
    Falls back to synthetic data if network/parsing fails.
    """
    if use_synthetic:
        n_samples = 32561 if train else 16281
        X, y = generate_synthetic_adult(n_samples=n_samples, input_dim=14)
        return TensorDataset(X, y)

    os.makedirs(DATA_DIR, exist_ok=True)
    filename = 'adult.data' if train else 'adult.test'
    filepath = os.path.join(DATA_DIR, filename)
    url = f"https://archive.ics.uci.edu/ml/machine-learning-databases/adult/{filename}"
    
    try:
        if not os.path.exists(filepath):
            print(f"Downloading UCI-Adult from {url}...")
            urllib.request.urlretrieve(url, filepath)
        
        # Parse data
        # adult.test has an extra first line that is a comment, we skip it
        skiprows = 1 if not train else 0
        names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country', 'label'
        ]
        df = pd.read_csv(filepath, names=names, sep=r'\s*,\s*', engine='python', skiprows=skiprows)
        
        # Clean labels (adult.test labels end with a dot '.')
        df['label'] = df['label'].astype(str).str.replace('.', '', regex=False)
        df = df[df['label'].isin(['<=50K', '>50K'])] # drop rows with bad labels
        
        # Convert label to binary
        y = (df['label'] == '>50K').astype(np.int64).values
        
        # Separate numeric and categorical
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
        numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        
        # Drop columns with too many missing values or represent high cardinality
        df_numeric = df[numeric_cols].copy()
        
        # Simple one-hot encoding for categorical variables
        df_categorical = pd.get_dummies(df[categorical_cols])
        
        # Combine
        df_processed = pd.concat([df_numeric, df_categorical], axis=1)
        
        # Convert to float32
        X = df_processed.values.astype(np.float32)
        
        # Truncate or pad to exactly 14 features for the default network, or use processed dimension.
        # To match the paper's default dimension of 14, we select the top 14 columns by variance
        if X.shape[1] > 14:
            variances = np.var(X, axis=0)
            top_indices = np.argsort(variances)[-14:]
            X = X[:, top_indices]
        elif X.shape[1] < 14:
            # Pad with zeros
            pad = np.zeros((X.shape[0], 14 - X.shape[1]), dtype=np.float32)
            X = np.hstack([X, pad])
            
        return TensorDataset(torch.tensor(X), torch.tensor(y))
        
    except Exception as e:
        print(f"Warning: Failed to fetch/parse UCI-Adult due to: {e}. Falling back to synthetic dataset.")
        n_samples = 32561 if train else 16281
        X, y = generate_synthetic_adult(n_samples=n_samples, input_dim=14)
        return TensorDataset(X, y)


class SyntheticImageDataset(Dataset):
    """Synthetic dataset for MNIST or CIFAR-10 replacement in offline mode."""
    def __init__(self, size, shape, num_classes):
        self.size = size
        self.shape = shape
        self.num_classes = num_classes
        
        # Pre-generate random tensors to speed up fetching
        self.data = torch.randn(size, *shape)
        self.labels = torch.randint(0, num_classes, (size,))
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_mnist(train=True, use_synthetic=False):
    """Loads MNIST dataset. Falls back to synthetic if offline."""
    if use_synthetic:
        return SyntheticImageDataset(60000 if train else 10000, (1, 28, 28), 10)
        
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    try:
        return datasets.MNIST(DATA_DIR, train=train, download=True, transform=transform)
    except Exception as e:
        print(f"Warning: Failed to load MNIST via torchvision ({e}). Falling back to synthetic.")
        return SyntheticImageDataset(60000 if train else 10000, (1, 28, 28), 10)


def get_cifar10(train=True, use_synthetic=False):
    """Loads CIFAR-10 dataset. Falls back to synthetic if offline."""
    if use_synthetic:
        return SyntheticImageDataset(50000 if train else 10000, (3, 32, 32), 10)
        
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    try:
        return datasets.CIFAR10(DATA_DIR, train=train, download=True, transform=transform)
    except Exception as e:
        print(f"Warning: Failed to load CIFAR-10 via torchvision ({e}). Falling back to synthetic.")
        return SyntheticImageDataset(50000 if train else 10000, (3, 32, 32), 10)


def get_dataset(name: str, train: bool = True, use_synthetic: bool = False):
    """Helper to load standard datasets by name."""
    name_lower = name.lower()
    if 'adult' in name_lower or 'uci' in name_lower:
        return get_uci_adult(train=train, use_synthetic=use_synthetic)
    elif 'mnist' in name_lower:
        return get_mnist(train=train, use_synthetic=use_synthetic)
    elif 'cifar' in name_lower:
        return get_cifar10(train=train, use_synthetic=use_synthetic)
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def get_dataloaders(name: str, batch_size: int = 128, val_ratio: float = 0.2, use_synthetic: bool = False):
    """
    Creates train, validation, and test dataloaders for the specified dataset.
    val_ratio splits the training dataset into train and validation sets.
    """
    full_train_dataset = get_dataset(name, train=True, use_synthetic=use_synthetic)
    test_dataset = get_dataset(name, train=False, use_synthetic=use_synthetic)
    
    # Split training dataset into train and validation
    total_len = len(full_train_dataset)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len
    
    # Fix the generator seed for reproducibility of the train/val split
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_len, val_len], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    return train_loader, val_loader, test_loader
