import torch
import torch.nn as nn
from typing import Dict, Any
from .optimizer import USRILSOptimizer
from .controller import RegulationController

class USRILSSystem:
    """
    Coordinates the six-stage USRILS Execution Pipeline (S1-S6)
    integrating normalization, extraction, prediction, evaluation, 
    regulation, and parameter updates.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: USRILSOptimizer,
        controller: RegulationController
    ):
        self.model = model
        self.optimizer = optimizer
        self.controller = controller
        self.mu = None
        self.sigma = None

    def stage_1_normalize(self, x: torch.Tensor, is_training: bool = True) -> torch.Tensor:
        """S1: Normalise Raw x -> x_hat. Complexity O(nd)."""
        # If input is already normalized or standard image data, we can optionally skip or normalize
        # For general tabular data, compute mean/std
        if x.dim() > 2:
            # For image data, they are already preprocessed using torchvision transforms.
            return x
            
        if is_training or self.mu is None or self.sigma is None:
            self.mu = x.mean(dim=0, keepdim=True)
            self.sigma = x.std(dim=0, keepdim=True) + 1e-8
            
        return (x - self.mu) / self.sigma

    def stage_2_extract(self, x_hat: torch.Tensor) -> torch.Tensor:
        """S2: Extract x_hat -> z in R^k. Complexity O(n*d*k)."""
        with torch.no_grad():
            _, z = self.model(x_hat)
        return z

    def stage_3_predict(self, z: torch.Tensor) -> torch.Tensor:
        """S3: Predict z -> y_hat. Complexity O(n*k*|Y|)."""
        with torch.no_grad():
            # classifier layer
            logits = self.model.classifier(z)
            y_hat = torch.softmax(logits, dim=-1)
        return y_hat

    def step(self, train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor) -> Dict[str, Any]:
        """
        Runs one step of the six-stage execution pipeline.
        S1: Normalise
        S2: Extract features (inside model)
        S3: Predict and compute losses (inside optimizer)
        S4: Evaluate metrics and CIM (inside optimizer)
        S5: Regulate learning rate (using controller)
        S6: Update parameters (inside optimizer)
        """
        # Stage 1: Normalise
        train_x_hat = self.stage_1_normalize(train_x, is_training=True)
        val_x_hat = self.stage_1_normalize(val_x, is_training=False)
        
        # Stages 2, 3, 4, 6: Extract, Predict, Evaluate, Update
        # Done in a unified way inside the custom optimizer to support double backward pass
        model_flops = self.model.flops()
        metrics = self.optimizer.step(train_x_hat, train_y, val_x_hat, val_y, model_flops)
        
        # Stage 5: Regulate
        old_lr = self.optimizer.lr
        new_lr, branch = self.controller.step(old_lr, metrics['cim'])
        self.optimizer.lr = new_lr
        
        # Record pipeline metadata
        metrics['old_lr'] = old_lr
        metrics['new_lr'] = new_lr
        metrics['regulation_branch'] = branch
        
        return metrics
