import pytest
import torch
import torch.nn as nn
from usrils.models import AdultMLP, MNISTMLP, CIFARCNN
from usrils.metrics import MetricEvaluator
from usrils.controller import RegulationController
from usrils.optimizer import USRILSOptimizer
from usrils.system import USRILSSystem
from usrils.datasets import get_dataloaders

def test_metric_evaluator():
    evaluator = MetricEvaluator(rho=0.5, delta_max=0.20, flops_budget=100.0)
    
    # Test accuracy: 1 - loss
    val_loss = torch.tensor(0.15)
    acc = evaluator.compute_accuracy(val_loss)
    assert pytest.approx(acc.item()) == 0.85
    
    # Test stability: exp(-norm / rho)
    grad_norm = torch.tensor(0.1)
    stab = evaluator.compute_stability(grad_norm)
    assert pytest.approx(stab.item()) == torch.exp(torch.tensor(-0.2)).item()
    
    # Test generalization: 1 - (val_loss - train_loss) / delta_max
    train_loss = torch.tensor(0.05)
    gen = evaluator.compute_generalization(val_loss, train_loss)
    assert pytest.approx(gen.item()) == 0.50
    
    # Test cost: flops / budget
    cost = evaluator.compute_cost(80.0)
    assert cost == 0.80
    
    # Test CIM
    cim = evaluator.compute_cim(acc, stab, gen, cost)
    expected_cim = 0.40 * 0.85 + 0.25 * stab.item() + 0.25 * 0.50 - 0.10 * 0.80
    assert pytest.approx(cim.item()) == expected_cim


def test_regulation_controller():
    # Target: 0.85, gain: 0.15, eps: 0.02
    controller = RegulationController(target_cim=0.85, regulation_gain=0.15, epsilon_ctrl=0.02)
    
    # EXPAND: delta > 0.02
    new_lr, branch = controller.step(current_lr=0.05, current_cim=0.90)
    assert branch == 'EXPAND'
    # delta = 0.05. factor = 1 + 0.15 * 0.05 = 1.0075. new_lr = 0.05 * 1.0075 = 0.050375
    assert pytest.approx(new_lr) == 0.050375
    
    # HOLD: |delta| <= 0.02
    new_lr, branch = controller.step(current_lr=0.05, current_cim=0.86)
    assert branch == 'HOLD'
    assert new_lr == 0.05
    
    # CONTRACT: delta < -0.02
    new_lr, branch = controller.step(current_lr=0.05, current_cim=0.80)
    assert branch == 'CONTRACT'
    # delta = -0.05. factor = 1 - 0.15 * 0.05 = 0.9925. new_lr = 0.05 * 0.9925 = 0.049625
    assert pytest.approx(new_lr) == 0.049625


def test_model_flops():
    model = AdultMLP(input_dim=14)
    flops = model.flops()
    assert isinstance(flops, int)
    assert flops > 0
    
    model2 = MNISTMLP()
    assert model2.flops() > 0
    
    model3 = CIFARCNN()
    assert model3.flops() > 0


def test_double_backward():
    """Verify that double backward pass executes without error on a dummy MLP."""
    model = AdultMLP(input_dim=4, hidden_dim=8)
    optimizer = USRILSOptimizer(
        model=model,
        initial_lr=0.05,
        flops_budget=10.0,
        rho=0.5,
        delta_max=0.20
    )
    
    # Create small dummy tensors
    tx = torch.randn(5, 4)
    ty = torch.randint(0, 2, (5,))
    vx = torch.randn(5, 4)
    vy = torch.randint(0, 2, (5,))
    
    # This step will run forward, backward (on train loss), double-backward (on CIM), and update parameters
    metrics = optimizer.step(tx, ty, vx, vy, model_flops=8.0)
    
    assert 'train_loss' in metrics
    assert 'val_loss' in metrics
    assert 'cim' in metrics
    assert 'grad_norm' in metrics


def test_dataloaders():
    train_l, val_l, test_l = get_dataloaders('adult', batch_size=10, use_synthetic=True)
    
    # Fetch a batch
    x, y = next(iter(train_l))
    assert x.shape == (10, 14)
    assert y.shape == (10,)
