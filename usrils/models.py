import torch
import torch.nn as nn

def estimate_module_flops(module: nn.Module, input_size=None) -> int:
    """
    Estimates the number of FLOPs in a single forward pass of the given module.
    For Conv2d and Linear layers, we compute standard multiply-accumulate operations.
    """
    flops = 0
    if isinstance(module, nn.Linear):
        # Multiply-accumulate is 2 FLOPs per connection
        flops = 2 * module.in_features * module.out_features
    elif isinstance(module, nn.Conv2d):
        # We need the output dimensions to compute Conv2d FLOPs.
        # If input_size is not provided, we assume standard sizes based on the dataset.
        if input_size is None:
            # Fallbacks: assume standard sizes
            h_in, w_in = 32, 32
        else:
            h_in, w_in = input_size[-2:]
        
        # Calculate output dimensions
        kh, kw = module.kernel_size
        ph, pw = module.padding
        sh, sw = module.stride
        dh, dw = module.dilation
        
        h_out = ((h_in + 2 * ph - dh * (kh - 1) - 1) // sh) + 1
        w_out = ((w_in + 2 * pw - dw * (kw - 1) - 1) // sw) + 1
        
        # 2 * K_h * K_w * C_in * C_out * H_out * W_out
        flops = 2 * kh * kw * module.in_channels * module.out_channels * h_out * w_out
    return flops


class USRILSBaseModel(nn.Module):
    """Base class for USRILS models supporting FLOP estimation."""
    def flops(self) -> int:
        raise NotImplementedError("Each subclass must implement a flops() method.")


class AdultMLP(USRILSBaseModel):
    """2-layer MLP for binary classification on UCI-Adult dataset."""
    def __init__(self, input_dim: int = 14, hidden_dim: int = 64):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU()
        )
        self.classifier = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        # S1 normalisation and S2 representation extraction
        z = self.feature_extractor(x)
        # S3 prediction
        logits = self.classifier(z)
        return logits, z

    def flops(self) -> int:
        flops = 0
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                flops += estimate_module_flops(m)
        return flops


class MNISTMLP(USRILSBaseModel):
    """2-layer MLP for 10-class classification on MNIST."""
    def __init__(self, input_dim: int = 784, hidden_dim: int = 128):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU()
        )
        self.classifier = nn.Linear(hidden_dim, 10)
        
    def forward(self, x):
        # Flatten input
        x_flat = x.view(x.size(0), -1)
        z = self.feature_extractor(x_flat)
        logits = self.classifier(z)
        return logits, z

    def flops(self) -> int:
        flops = 0
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                flops += estimate_module_flops(m)
        return flops


class CIFARCNN(USRILSBaseModel):
    """CNN for 10-class classification on CIFAR-10."""
    def __init__(self, in_channels: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.SiLU()
        self.pool1 = nn.MaxPool2d(2, 2) # 16x16
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.SiLU()
        self.pool2 = nn.MaxPool2d(2, 2) # 8x8
        
        self.feature_extractor = nn.Sequential(
            self.conv1,
            self.relu1,
            self.pool1,
            self.conv2,
            self.relu2,
            self.pool2
        )
        
        self.fc1 = nn.Linear(32 * 8 * 8, hidden_dim)
        self.relu3 = nn.SiLU()
        
        self.classifier = nn.Linear(hidden_dim, 10)
        
    def forward(self, x):
        # Extract features (S2)
        features = self.feature_extractor(x)
        features_flat = features.view(features.size(0), -1)
        z = self.relu3(self.fc1(features_flat))
        # Predict (S3)
        logits = self.classifier(z)
        return logits, z

    def flops(self) -> int:
        # Estimate FLOPs stage-by-stage
        flops = 0
        # conv1 (input 3x32x32, kernel 3x3, out 16 channels, output shape 16x32x32)
        flops += estimate_module_flops(self.conv1, (3, 32, 32))
        # conv2 (input 16x16x16, kernel 3x3, out 32 channels, output shape 32x16x16)
        flops += estimate_module_flops(self.conv2, (16, 16, 16))
        # fc1
        flops += estimate_module_flops(self.fc1)
        # classifier
        flops += estimate_module_flops(self.classifier)
        return flops
