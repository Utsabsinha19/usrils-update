import torch
import torch.nn as nn
from typing import Dict, Any

class USRILSOptimizer:
    """
    Custom optimizer that updates parameters using the regulated step-size (lr)
    and the gradient of the composite loss (L = -Theta).
    Uses double autograd to handle gradients of the stability term.
    """
    def __init__(
        self,
        model: nn.Module,
        initial_lr: float = 0.05,
        alpha: float = 0.40,
        beta: float = 0.25,
        gamma: float = 0.25,
        lambda_: float = 0.10,
        rho: float = 0.50,
        delta_max: float = 0.20,
        flops_budget: float = 1.0,
        weight_decay: float = 0.0,
        loss_fn: nn.Module = nn.CrossEntropyLoss()
    ):
        self.model = model
        self.lr = initial_lr
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.rho = rho
        self.delta_max = delta_max
        self.flops_budget = flops_budget
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn

    def step(self, train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor, model_flops: float) -> Dict[str, Any]:
        """
        Performs a single regulated gradient descent update step.
        Returns a dictionary of metrics computed during this step.
        """
        # Ensure model gradients are clean
        self.model.zero_grad()
        
        # 1. Forward pass on training data to compute train loss
        train_logits, _ = self.model(train_x)
        train_loss = self.loss_fn(train_logits, train_y)
        
        # 2. Compute training gradients with graph creation for double backprop
        params = [p for p in self.model.parameters() if p.requires_grad]
        grad_train = torch.autograd.grad(
            train_loss,
            params,
            create_graph=True,
            retain_graph=True
        )
        
        # 3. Compute gradient norm and stability
        grad_norm = torch.sqrt(sum(g.pow(2).sum() for g in grad_train) + 1e-8)
        
        # Numerical stability safeguard: detach stability term if already highly stable
        # to prevent division-by-zero or underflow in double backward Hessian pass.
        if grad_norm.item() < 1e-3:
            stability = torch.exp(-grad_norm.detach() / self.rho)
        else:
            stability = torch.exp(-grad_norm / self.rho)
        
        # 4. Forward pass on validation data to compute val loss
        val_logits, _ = self.model(val_x)
        val_loss = self.loss_fn(val_logits, val_y)
        
        # 5. Compute accuracy, generalization, cost, and composite loss (L = -Theta)
        accuracy = 1.0 - val_loss
        generalization = 1.0 - (val_loss - train_loss) / self.delta_max
        cost = model_flops / self.flops_budget
        
        # Composite loss (we minimize L = -Theta)
        comp_loss = -(self.alpha * accuracy + self.beta * stability + self.gamma * generalization - self.lambda_ * cost)
        
        # 6. Explicitly compute gradients of composite loss with respect to parameters
        # to prevent PyTorch .backward() multiple-path graph conflicts under double-backward
        grads = torch.autograd.grad(comp_loss, params, retain_graph=False)
        
        # 7. Apply parameter update
        with torch.no_grad():
            for p, g in zip(params, grads):
                if g is not None:
                    # Apply weight decay (L2 regularization contraction)
                    grad_update = g
                    if self.weight_decay > 0:
                        grad_update = grad_update + self.weight_decay * p
                    
                    # Gradient clipping safeguard
                    grad_update = torch.clamp(grad_update, -1.0, 1.0)
                    
                    # Gradient descent on composite loss L
                    p.copy_(p - self.lr * grad_update)
        
        # Clean up graph gradients
        self.model.zero_grad()
        
        # Return detached metrics for evaluation/regulation
        return {
            'train_loss': train_loss.item(),
            'val_loss': val_loss.item(),
            'accuracy': accuracy.item(),
            'stability': stability.item(),
            'generalization': generalization.item(),
            'cost': cost,
            'cim': -comp_loss.item(), # Theta = -L
            'grad_norm': grad_norm.item()
        }
