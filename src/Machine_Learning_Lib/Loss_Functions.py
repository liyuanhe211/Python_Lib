import sys
import pathlib

Python_Lib_path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(Python_Lib_path)

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

print("Loading Torch... ", end="")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def check_loss_calc_validity(Ys_pred: torch.Tensor, Ys_true: torch.Tensor) -> None:
    """
    Validates the input tensors for MSE Loss calculation to prevent common 
    runtime errors and logical pitfalls (e.g., silent broadcasting).

    Args:
        ys_pred (torch.Tensor): The prediction tensor.
        ys_true (torch.Tensor): The ground truth tensor.

    Raises:
        TypeError: If inputs are not PyTorch Tensors.
        ValueError: If shapes are mismatched, dtypes are incorrect, or devices differ.
    """
    
    # 1. 类型检查 (Type Consistency)
    if not isinstance(Ys_pred, torch.Tensor) or not isinstance(Ys_true, torch.Tensor):
        raise TypeError(
            f"Expected inputs to be torch.Tensor, but got "
            f"ys_pred: {type(Ys_pred)} and ys_true: {type(Ys_true)}."
        )

    # 2. 设备一致性检查 (Device Consistency)
    # 若张量分别位于 CPU 和 GPU，将导致 Runtime Error
    if Ys_pred.device != Ys_true.device:
        raise ValueError(
            f"Device mismatch detected. ys_pred is on {Ys_pred.device}, "
            f"but ys_true is on {Ys_true.device}."
        )

    # 3. 数据类型检查 (Data Type Precision)
    # MSE 需要浮点数进行梯度计算
    if not Ys_pred.is_floating_point() or not Ys_true.is_floating_point():
        raise ValueError(
            f"Expected floating point tensors for MSE computation. Got "
            f"ys_pred: {Ys_pred.dtype}, ys_true: {Ys_true.dtype}."
        )

    # 4. 形状严格同构检查 (Strict Shape Congruence)
    # 这是最关键的检查，旨在防止 (N, 1) 与 (N) 之间的静默广播导致计算错误
    if Ys_pred.shape != Ys_true.shape:
        raise ValueError(
            f"Shape mismatch detected. MSELoss expects strict shape congruence to avoid "
            f"ambiguous broadcasting.\n"
            f"ys_pred shape: {Ys_pred.shape}\n"
            f"ys_true shape: {Ys_true.shape}\n"
            f"Suggestion: Use .view(), .squeeze(), or .unsqueeze() to align dimensions."
        )

    # 5. 维度非空检查 (Non-empty Check)
    if Ys_pred.numel() == 0:
        raise ValueError("Input tensors are empty.")
    
    # 6. 批量维度检查 (Batch Dimension Check)
    if Ys_pred.dim() !=2 or Ys_true.dim() !=2:
        raise ValueError(f"Expected the tensor to be 2-dimensional (Batch, Features). ")

    if Ys_pred.size(1)>1:
        input("Warning: ys_pred has more than one feature dimension. This feature is not verified. Continue?")

    # (可选) 逻辑检查：通常预测值需要梯度，而标签不需要
    # 此处仅作记录，不抛出异常，因为在验证集(Validation)阶段可能都不需要梯度
    # if not ys_pred.requires_grad:
    #     print("Warning: ys_pred does not require grad. Check if this is intended.")

def loss_function_MSE(Ys_pred, Ys_true):
    check_loss_calc_validity(Ys_pred, Ys_true)
    return nn.MSELoss()(Ys_pred, Ys_true)

def loss_function_RMSE(Ys_pred, Ys_true):
    check_loss_calc_validity(Ys_pred, Ys_true)
    return torch.sqrt(nn.MSELoss()(Ys_pred, Ys_true))

def loss_function_R2(Ys_pred, Ys_true):
    check_loss_calc_validity(Ys_pred, Ys_true)
    target_mean = torch.mean(Ys_true)
    ss_tot = torch.sum((Ys_true - target_mean) ** 2)
    ss_res = torch.sum((Ys_true - Ys_pred) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return r2

def loss_function_group_pairwise_MSE(Ys_pred, Ys_true, Tags):
    """
    计算组内成对差值损失 (Group-wise Pairwise Difference MSE Loss)。
    
    Calculation Logic:
    1.  Divide samples into groups based on `Tags`.
    2.  For each group, considering samples i and j, calculate the difference in predictions (pred_i - pred_j)
        and the difference in ground truth (true_i - true_j).
    3.  Compute the MSE between these two sets of differences: Mean(( (pred_i - pred_j) - (true_i - true_j) )^2).
    4.  The final loss is the average of these MSEs across all groups.

    Explanation:
    This loss function focuses on the *consistency of relative differences* within each group.
    It penalizes the model if the predicted relationship (difference) between two samples in the same group 
    does not match the actual relationship, regardless of the absolute values. 
    This is useful for ranking or when absolute calibration varies by group but relative trends should be preserved.

    Args:
        Ys_pred: (Batch, 1) Predicted values.
        Ys_true: (Batch, 1) True values.
        Tags:    (Batch, D) Tags or Group IDs.
    
    Returns:
        torch.Tensor: Scalar loss value.
    """
    
    check_loss_calc_validity(Ys_pred, Ys_true)
    
    pairwise_loss_total = 0.0
    num_valid_groups = 0
    
    # 2. Get unique rows (groups)
    unique_groups = torch.unique(Tags, dim=0)
    
    for group in unique_groups:
        # Find indices for the current group
        # Tags is (N, D), group is (D,)
        # Check if all features match for each sample: (N, D) -> (N,)
        is_in_group = torch.all(Tags == group, dim=1)
        group_indexes = is_in_group.nonzero(as_tuple=True)[0]
        
        # Need at least 2 samples to compute pairwise differences
        if len(group_indexes) < 2:
            continue
            
        # Extract group data
        curr_pred = Ys_pred[group_indexes]
        curr_true = Ys_true[group_indexes]
        
        # Calculate all pairwise differences using broadcasting
        # (N, 1) - (1, N) -> (N, N) matrix where cell [i, j] is val[i] - val[j]
        diff_pred = curr_pred - curr_pred.t()
        diff_true = curr_true - curr_true.t()
        
        # Calculate MSE of the difference matrices
        # (diff_pred - diff_true) represents the error in the predicted difference
        # We take the mean of squared errors
        loss_group = torch.mean((diff_pred - diff_true) ** 2)
        
        pairwise_loss_total += loss_group
        num_valid_groups += 1
        
    if num_valid_groups > 0:
        return pairwise_loss_total / num_valid_groups
    else:
        # Return 0.0 with gradient tracking if possible, though no gradient from empty arithmetic
        return torch.tensor(0.0, device=Ys_pred.device, requires_grad=True)


def cap_output_range_loss(Ys_pred, Ys_true=None, Tags=None, min=None, max=None, penalty_weight=10.0):
    """
    A penalty for predictions outside the [min, max] range.
        
    Args:
        Ys_pred: (Batch, Features) or (Batch,) Predicted values. 
        min (float, list, or np.ndarray, optional): Lower bound. 
            If Ys_pred has multiple features, this can be a list/array with length matching the number of features.
            If scalar, it applies to all features.
        max (float, list, or np.ndarray, optional): Upper bound.
            If Ys_pred has multiple features, this can be a list/array with length matching the number of features.
            If scalar, it applies to all features.
        penalty_weight (float, optional): Weight for the penalty term. Default is 10.0.
        
    The penalty is calculated as:
        Loss_total = Loss_original + penalty_weight * (ReLU(min - pred)^2 + ReLU(pred - max)^2)
        
    Explanation of Ratio (Penalty vs Original):
        The 'penalty_weight' determines how strictly the boundary is enforced relative to the task loss.
        - The penalty is quadratic (squared error of the violation).
        - If penalty_weight is high (e.g., >10), the gradient from the boundary violation will dominate
        when the prediction is significantly out of bounds, pushing the model back into the valid range.
        - The exact ratio depends on the magnitude of the original loss. If the original loss (like MSE)
        is small (e.g., < 0.1), a weight of 10.0 provides a very strong constraint.
    """
    if min is None and max is None:
        raise ValueError("At least one of min or max must be provided.")
    
    # Get dimensions and device
    num_features = Ys_pred.shape[1] if Ys_pred.dim() > 1 else 1
    device = Ys_pred.device
    
    # Process bounds: convert to tensors with proper shape
    bounds = {}
    for bound_name, bound_val in [('min', min), ('max', max)]:
        if bound_val is None:
            bounds[bound_name] = None
            continue
        
        # Check if scalar, convert to list if so
        is_scalar = isinstance(bound_val, (int, float)) or (isinstance(bound_val, torch.Tensor) and bound_val.numel() == 1)
        if is_scalar:
            bound_val = bound_val.item() if isinstance(bound_val, torch.Tensor) else bound_val
            bound_val = [bound_val] * num_features
        
        # Convert to tensor
        bound_t = torch.tensor(bound_val, device=device, dtype=torch.float32) if not isinstance(bound_val, torch.Tensor) else bound_val.to(device)
        
        # Validate length
        if bound_t.numel() != num_features:
            raise ValueError(f"{bound_name} has {bound_t.numel()} elements but Ys_pred has {num_features} features.")
        
        # Reshape for broadcasting if Ys_pred is 2D
        if Ys_pred.dim() > 1:
            bound_t = bound_t.view(1, -1)
        
        bounds[bound_name] = bound_t
    
    # Calculate loss
    loss = torch.tensor(0.0, device=device)
    
    if bounds['min'] is not None:
        loss += torch.mean(torch.relu(bounds['min'] - Ys_pred) ** 2)
    
    if bounds['max'] is not None:
        loss += torch.mean(torch.relu(Ys_pred - bounds['max']) ** 2)
    
    return penalty_weight * loss


def cap_output_range_loss_decorator(func, min=None, max=None, penalty_weight=10.0):
    """
    Decorator version of cap_output_range_loss in case you are not using this in Train_NN_Network.
    """
    def wrapper(Ys_pred, Ys_true, *args, **kwargs):
        loss = func(Ys_pred, Ys_true, *args, **kwargs)
        return loss + cap_output_range_loss(Ys_pred, Ys_true, min=min, max=max, penalty_weight=penalty_weight)
        
    return wrapper


class Uncertainty_Weighted_Loss(nn.Module):
    """
    Automatically weighted multi-task loss based on uncertainty.

    Adapted from paper: 《Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (CVPR 2018)》
    
    The loss function is defined as:

    L[i] = exp(-eta[i]) * loss[i] + eta[i]
    L_total = sum(L)

    where eta[i] is a learnable parameter representing the log variance of task i.

    This encourages the model to reduce specific task weights if the task has high uncertainty (high loss),
    balanced by the regularization term log_var.
    """
    def __init__(self, number_of_loss_functions):
        assert number_of_loss_functions > 1, "Do not use only one loss function for AutomaticWeightedLoss."
        super(Uncertainty_Weighted_Loss, self).__init__()
        self.num_tasks = number_of_loss_functions
        self.eta_list = nn.Parameter(torch.zeros(number_of_loss_functions))
        self.loss_terms = []
        self.weights = []
        self.is_initialized = False

    def forward(self, *losses):
        """
        Args:
            losses (list[torch.Tensor]): List of scalar loss tensors for each task.
        """
        if len(losses) != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} losses, got {len(losses)}")

        # Automatic initialization on first run if weights are default (zeros)
        # This solves the "Magnitude Bias" problem where high-loss tasks dominate early training.
        if self.training and not self.is_initialized:
            # Check if weights seem to be already trained (loaded from checkpoint)
            # If any eta is non-zero, we assume it's a loaded model and we shouldn't overwrite it.
            if torch.any(self.eta_list != 0.0):
                 input("Inspect Automatic_Weighted_Loss code: Detected non-zero eta values, even if the self.is_initialized is never set True")
                 self.is_initialized = True
            else:
                 # Initialize eta based on initial loss magnitudes
                 print("Automatic_Weighted_Loss: Initializing weights based on first batch losses...")
                 with torch.no_grad():
                     for i, loss in enumerate(losses):
                         val = loss.item()
                         # Avoid potential math errors with 0 or negative inputs (though losses should be >=0)
                         if val > 1e-6:
                             # Target: exp(-eta) * Loss = 1  =>  -eta + ln(Loss) = 0  =>  eta = ln(Loss)
                             self.eta_list[i].fill_(np.log(val))
                             print(f"  Task {i}: Loss={val:.4g} -> Auto-set eta={self.eta_list[i].item():.4g}")
                         else:
                             print(f"  Task {i}: Loss={val:.4g} too small, keeping eta=0.")
                     
                     # Scale eta_list sum to 1
                     eta_sum = self.eta_list.sum()
                     if torch.abs(eta_sum) > 1e-6:
                         self.eta_list.div_(eta_sum)
                         print(f"  Sum of eta_list scaled from {eta_sum.item():.4g} to 1.0")
                         for i in range(self.num_tasks):
                             print(f"    Task {i}: Final auto-set eta={self.eta_list[i].item():.4g}")
                 
                 self.is_initialized = True

        total_loss = 0
        self.loss_terms = []
        self.weights = []
        for i, loss in enumerate(losses):
            weight = torch.exp(-self.eta_list[i])
            self.loss_terms.append(loss.item())
            self.weights.append(weight.item())
            total_loss += weight * loss + self.eta_list[i]
            
        return total_loss




