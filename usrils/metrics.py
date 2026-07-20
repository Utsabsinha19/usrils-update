import torch
import numpy as np

class MetricEvaluator:
    """
    Computes accuracy, stability, generalization, cost, and the 
    Composite Intelligence Metric (CIM, Theta) for a USRILS model.
    """
    def __init__(
        self,
        alpha: float = 0.40,
        beta: float = 0.25,
        gamma: float = 0.25,
        lambda_: float = 0.10,
        rho: float = 0.50,
        delta_max: float = 0.20,
        flops_budget: float = 1.0
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.rho = rho
        self.delta_max = delta_max
        self.flops_budget = flops_budget

    def compute_accuracy(self, val_loss: torch.Tensor) -> torch.Tensor:
        """Accuracy A_t = 1 - L_val(theta_t)."""
        # Ensure it doesn't go below 0 or above 1 for normalization
        return 1.0 - val_loss

    def compute_stability(self, grad_norm: torch.Tensor) -> torch.Tensor:
        """Stability S_t = exp(-||grad L_t|| / rho)."""
        return torch.exp(-grad_norm / self.rho)

    def compute_generalization(self, val_loss: torch.Tensor, train_loss: torch.Tensor) -> torch.Tensor:
        """Generalization G_t = 1 - (L_val - L_train) / delta_max."""
        gap = val_loss - train_loss
        return 1.0 - gap / self.delta_max

    def compute_cost(self, flops: float) -> float:
        """Cost C_t = FLOPs_t / FLOPs_budget."""
        return flops / self.flops_budget

    def compute_cim(
        self,
        accuracy: torch.Tensor,
        stability: torch.Tensor,
        generalization: torch.Tensor,
        cost: float
    ) -> torch.Tensor:
        """
        CIM (Theta) = alpha * A + beta * S + gamma * G - lambda * C.
        Clamped to [-1, 1] as described in the paper to maintain normalization.
        """
        theta = self.alpha * accuracy + self.beta * stability + self.gamma * generalization - self.lambda_ * cost
        return torch.clamp(theta, min=-1.0, max=1.0)

    def evaluate(
        self,
        train_loss: torch.Tensor,
        val_loss: torch.Tensor,
        grad_norm: torch.Tensor,
        model_flops: float
    ):
        """
        Performs full metric evaluation at step t.
        Returns a dict of metrics.
        """
        acc = self.compute_accuracy(val_loss)
        stab = self.compute_stability(grad_norm)
        gen = self.compute_generalization(val_loss, train_loss)
        cost = self.compute_cost(model_flops)
        
        # We want to perform the CIM computation in a differentiable way if tensors are passed.
        theta = self.compute_cim(acc, stab, gen, cost)
        
        return {
            'accuracy': acc,
            'stability': stab,
            'generalization': gen,
            'cost': cost,
            'cim': theta
        }
