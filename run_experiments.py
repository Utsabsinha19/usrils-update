import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ttest_rel

# Import USRILS components
from usrils.models import AdultMLP, MNISTMLP, CIFARCNN
from usrils.datasets import get_dataloaders
from usrils.metrics import MetricEvaluator
from usrils.controller import RegulationController
from usrils.optimizer import USRILSOptimizer
from usrils.system import USRILSSystem

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def train_sgd_adam_adagrad(method_name, model, train_loader, val_loader, epochs, eval_flops, flops_budget, device, lr=0.01):
    """Trains a model using standard SGD, Adam, or AdaGrad baselines."""
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    if method_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif method_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif method_name.lower() == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
    else:
        raise ValueError(f"Unknown optimizer {method_name}")
        
    evaluator = MetricEvaluator(flops_budget=flops_budget)
    
    # Store metrics per step
    step_metrics = []
    val_iter = iter(val_loader)
    
    for epoch in range(epochs):
        model.train()
        for train_x, train_y in train_loader:
            train_x, train_y = train_x.to(device), train_y.to(device)
            
            # Get val batch
            try:
                val_x, val_y = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_x, val_y = next(val_iter)
            val_x, val_y = val_x.to(device), val_y.to(device)
            
            # Step update on training loss
            optimizer.zero_grad()
            logits, _ = model(train_x)
            loss_train = loss_fn(logits, train_y)
            loss_train.backward()
            optimizer.step()
            
            # Evaluate metrics at current step (no regulation)
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(val_x)
                loss_val = loss_fn(val_logits, val_y)
                
                # Compute gradient norm for stability metric
                params = [p for p in model.parameters() if p.requires_grad]
                grad_norm = torch.sqrt(sum((p.grad.pow(2).sum() if p.grad is not None else torch.tensor(0.0, device=device)) for p in params) + 1e-8)
                
                metrics = evaluator.evaluate(loss_train, loss_val, grad_norm, eval_flops)
                
                # Detach items for storage
                step_metrics.append({
                    'accuracy': metrics['accuracy'].item(),
                    'stability': metrics['stability'].item(),
                    'generalization': metrics['generalization'].item(),
                    'cost': metrics['cost'],
                    'cim': metrics['cim'].item()
                })
            model.train()
            
    # Return average of the last epoch/steps
    last_steps = min(50, len(step_metrics))
    results = {k: np.mean([s[k] for s in step_metrics[-last_steps:]]) for k in ['accuracy', 'stability', 'generalization', 'cost', 'cim']}
    return results

def train_mtl(model, train_loader, val_loader, epochs, eval_flops, flops_budget, device):
    """Trains a model using Multi-Task Learning baseline (joint optimization, fixed LR)."""
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    evaluator = MetricEvaluator(flops_budget=flops_budget)
    
    step_metrics = []
    val_iter = iter(val_loader)
    
    for epoch in range(epochs):
        model.train()
        for train_x, train_y in train_loader:
            train_x, train_y = train_x.to(device), train_y.to(device)
            
            try:
                val_x, val_y = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_x, val_y = next(val_iter)
            val_x, val_y = val_x.to(device), val_y.to(device)
            
            # Double backward to optimize the joint composite loss
            optimizer.zero_grad()
            
            # Train loss
            train_logits, _ = model(train_x)
            loss_train = loss_fn(train_logits, train_y)
            params = [p for p in model.parameters() if p.requires_grad]
            grad_train = torch.autograd.grad(loss_train, params, create_graph=True, retain_graph=True)
            grad_norm = torch.sqrt(sum(g.pow(2).sum() for g in grad_train) + 1e-8)
            stability = torch.exp(-grad_norm / evaluator.rho)
            
            # Val loss
            val_logits, _ = model(val_x)
            loss_val = loss_fn(val_logits, val_y)
            
            acc = 1.0 - loss_val
            gen = 1.0 - (loss_val - loss_train) / evaluator.delta_max
            cost = eval_flops / flops_budget
            
            # MTL Loss (minimize negative CIM)
            mtl_loss = -(evaluator.alpha * acc + evaluator.beta * stability + evaluator.gamma * gen - evaluator.lambda_ * cost)
            mtl_loss.backward()
            optimizer.step()
            
            step_metrics.append({
                'accuracy': acc.item(),
                'stability': stability.item(),
                'generalization': gen.item(),
                'cost': cost,
                'cim': -mtl_loss.item()
            })
            
    last_steps = min(50, len(step_metrics))
    results = {k: np.mean([s[k] for s in step_metrics[-last_steps:]]) for k in ['accuracy', 'stability', 'generalization', 'cost', 'cim']}
    return results

def train_constrained_optimization(model, train_loader, val_loader, epochs, eval_flops, flops_budget, device):
    """Trains a model using Constrained Optimization baseline (penalty method)."""
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    evaluator = MetricEvaluator(flops_budget=flops_budget)
    
    step_metrics = []
    val_iter = iter(val_loader)
    
    for epoch in range(epochs):
        model.train()
        for train_x, train_y in train_loader:
            train_x, train_y = train_x.to(device), train_y.to(device)
            
            try:
                val_x, val_y = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_x, val_y = next(val_iter)
            val_x, val_y = val_x.to(device), val_y.to(device)
            
            optimizer.zero_grad()
            
            # Train loss
            train_logits, _ = model(train_x)
            loss_train = loss_fn(train_logits, train_y)
            
            params = [p for p in model.parameters() if p.requires_grad]
            grad_train = torch.autograd.grad(loss_train, params, create_graph=True, retain_graph=True)
            grad_norm = torch.sqrt(sum(g.pow(2).sum() for g in grad_train) + 1e-8)
            stability = torch.exp(-grad_norm / evaluator.rho)
            
            # Val loss
            val_logits, _ = model(val_x)
            loss_val = loss_fn(val_logits, val_y)
            
            acc = 1.0 - loss_val
            gen = 1.0 - (loss_val - loss_train) / evaluator.delta_max
            cost = eval_flops / flops_budget
            
            # Penalty terms for violating stability >= 0.85 and generalization >= 0.85
            penalty_stab = torch.pow(torch.clamp(0.85 - stability, min=0.0), 2)
            penalty_gen = torch.pow(torch.clamp(0.85 - gen, min=0.0), 2)
            
            # Lagrangian / Penalty loss
            co_loss = loss_train + 10.0 * penalty_stab + 10.0 * penalty_gen
            co_loss.backward()
            optimizer.step()
            
            # Compute CIM for logging
            cim = evaluator.compute_cim(acc, stability, gen, cost)
            
            step_metrics.append({
                'accuracy': acc.item(),
                'stability': stability.item(),
                'generalization': gen.item(),
                'cost': cost,
                'cim': cim.item()
            })
            
    last_steps = min(50, len(step_metrics))
    results = {k: np.mean([s[k] for s in step_metrics[-last_steps:]]) for k in ['accuracy', 'stability', 'generalization', 'cost', 'cim']}
    return results

def train_usrils(model, train_loader, val_loader, epochs, eval_flops, flops_budget, device, initial_lr=0.05):
    """Trains a model using the complete USRILS six-stage execution pipeline."""
    model.to(device)
    
    # Initialize components
    optimizer = USRILSOptimizer(
        model=model,
        initial_lr=initial_lr,
        flops_budget=flops_budget,
        loss_fn=nn.CrossEntropyLoss()
    )
    controller = RegulationController()
    system = USRILSSystem(model, optimizer, controller)
    
    step_metrics = []
    val_iter = iter(val_loader)
    
    for epoch in range(epochs):
        model.train()
        for train_x, train_y in train_loader:
            train_x, train_y = train_x.to(device), train_y.to(device)
            
            try:
                val_x, val_y = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_x, val_y = next(val_iter)
            val_x, val_y = val_x.to(device), val_y.to(device)
            
            # Run one USRILS pipeline step
            metrics = system.step(train_x, train_y, val_x, val_y)
            
            step_metrics.append({
                'accuracy': metrics['accuracy'],
                'stability': metrics['stability'],
                'generalization': metrics['generalization'],
                'cost': metrics['cost'],
                'cim': metrics['cim']
            })
            
    last_steps = min(50, len(step_metrics))
    results = {k: np.mean([s[k] for s in step_metrics[-last_steps:]]) for k in ['accuracy', 'stability', 'generalization', 'cost', 'cim']}
    return results

def run_experiment(dataset_name, epochs, use_synthetic, fast_mode, runs=3):
    """Runs USRILS vs Baselines experiment for a given dataset."""
    print(f"\n==========================================")
    print(f"Running experiments on: {dataset_name.upper()}")
    print(f"==========================================")
    
    # 1. Setup Dataloaders
    batch_size = 32 if fast_mode else 128
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name, batch_size=batch_size, use_synthetic=use_synthetic
    )
    
    # Get model info
    if dataset_name == 'adult':
        input_dim = 14
        model_class = AdultMLP
        eval_flops = 2000.0
        flops_budget = 2500.0
    elif dataset_name == 'mnist':
        model_class = MNISTMLP
        eval_flops = 200000.0
        flops_budget = 250000.0
    elif dataset_name == 'cifar':
        model_class = CIFARCNN
        eval_flops = 20000000.0
        flops_budget = 24000000.0
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    methods = ['SGD', 'Adam', 'AdaGrad', 'MTL', 'CO', 'USRILS']
    run_records = {m: [] for m in methods}
    
    # Repeat for statistical analysis
    for r in range(runs):
        print(f"\nRun {r+1}/{runs}...")
        torch.manual_seed(42 + r)
        np.random.seed(42 + r)
        
        # 1. SGD
        model = model_class()
        sgd_res = train_sgd_adam_adagrad('sgd', model, train_loader, val_loader, epochs, eval_flops, flops_budget, device)
        run_records['SGD'].append(sgd_res)
        print(f"  SGD: CIM = {sgd_res['cim']:.4f}")
        
        # 2. Adam
        model = model_class()
        adam_res = train_sgd_adam_adagrad('adam', model, train_loader, val_loader, epochs, eval_flops, flops_budget, device)
        run_records['Adam'].append(adam_res)
        print(f"  Adam: CIM = {adam_res['cim']:.4f}")
        
        # 3. AdaGrad
        model = model_class()
        adagrad_res = train_sgd_adam_adagrad('adagrad', model, train_loader, val_loader, epochs, eval_flops, flops_budget, device)
        run_records['AdaGrad'].append(adagrad_res)
        print(f"  AdaGrad: CIM = {adagrad_res['cim']:.4f}")
        
        # 4. MTL
        model = model_class()
        mtl_res = train_mtl(model, train_loader, val_loader, epochs, eval_flops, flops_budget, device)
        run_records['MTL'].append(mtl_res)
        print(f"  MTL: CIM = {mtl_res['cim']:.4f}")
        
        # 5. CO
        model = model_class()
        co_res = train_constrained_optimization(model, train_loader, val_loader, epochs, eval_flops, flops_budget, device)
        run_records['CO'].append(co_res)
        print(f"  CO: CIM = {co_res['cim']:.4f}")
        
        # 6. USRILS
        model = model_class()
        usrils_res = train_usrils(model, train_loader, val_loader, epochs, eval_flops, flops_budget, device)
        run_records['USRILS'].append(usrils_res)
        print(f"  USRILS: CIM = {usrils_res['cim']:.4f}")
        
    # Compile average and std dev
    results_summary = []
    for m in methods:
        cims = [r['cim'] for r in run_records[m]]
        accs = [r['accuracy'] for r in run_records[m]]
        stabs = [r['stability'] for r in run_records[m]]
        gens = [r['generalization'] for r in run_records[m]]
        costs = [r['cost'] for r in run_records[m]]
        
        results_summary.append({
            'Method': m,
            'Acc': np.mean(accs),
            'Stab': np.mean(stabs),
            'Gen': np.mean(gens),
            'Cost': np.mean(costs),
            'CIM': np.mean(cims),
            'CIM_std': np.std(cims)
        })
        
    df = pd.DataFrame(results_summary)
    print(f"\nFinal results table for {dataset_name.upper()}:")
    print(df.to_string(index=False, formatters={
        'Acc': '{:.3f}'.format, 'Stab': '{:.3f}'.format, 'Gen': '{:.3f}'.format,
        'Cost': '{:.3f}'.format, 'CIM': '{:.3f}'.format, 'CIM_std': '{:.3f}'.format
    }))
    
    # Statistical paired t-tests (against USRILS)
    usrils_cims = [r['cim'] for r in run_records['USRILS']]
    print("\nStatistical paired t-tests (CIM compared to USRILS):")
    for m in methods[:-1]:
        baseline_cims = [r['cim'] for r in run_records[m]]
        t_stat, p_val = ttest_rel(usrils_cims, baseline_cims)
        sig = "p < 0.001 (Significant)" if p_val < 0.001 else f"p = {p_val:.4f} (Not Significant)"
        print(f"  USRILS vs {m}: t-stat = {t_stat:.4f}, p-value = {p_val:.2e} -> {sig}")
        
    return df

def main():
    parser = argparse.ArgumentParser(description="Run USRILS vs Baselines Experiments.")
    parser.add_argument('--fast', action='store_true', default=True, help="Run quick verification with small subsets.")
    parser.add_argument('--full', action='store_true', help="Run full-scale experiments.")
    parser.add_argument('--datasets', type=str, default='adult,mnist,cifar', help="Datasets to run: comma-separated.")
    parser.add_argument('--runs', type=int, default=3, help="Number of repetitions.")
    parser.add_argument('--use_synthetic', action='store_true', default=True, help="Force use of synthetic data.")
    
    args = parser.parse_args()
    
    # If full is requested, turn off fast mode and synthetic fallback unless specified
    if args.full:
        args.fast = False
        args.use_synthetic = False
        args.runs = 15
        
    epochs = 2 if args.fast else 10
    datasets_to_run = args.datasets.split(',')
    
    results = {}
    for d in datasets_to_run:
        results[d] = run_experiment(d, epochs, args.use_synthetic, args.fast, args.runs)

if __name__ == "__main__":
    main()
