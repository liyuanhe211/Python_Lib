# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import math
import os
import subprocess
import inspect
import sys
import pathlib
import traceback


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

print("Torch loaded.")

print()
# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Check CUDA version
if torch.version.cuda is not None:
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("CUDA is not available in this PyTorch build.")
print()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from Python_Lib.My_Lib import *
from Python_Lib.My_Lib_Plot import *
from Python_Lib.My_Lib_Office import *
from Python_Lib.My_Lib_System import non_block_keyboard_interrupt, disable_keyboard_interrupt, enable_keyboard_interrupt
from Python_Lib.My_Lib_MachineLearning_Models_Public import *

try:
    from Python_Lib.My_Lib_MachineLearning_Models_Private import *
except ImportError:
    print("Using only public models in Python_Lib.")

from Python_Lib.My_Lib_MachineLearning_Loss_Functions import *

# Set matplotlib backend (should be done before importing pyplot)
matplotlib.use("QtAgg")

ML_RANDOM_SEED = 20211021
DEFAULT_INITIAL_EPOCH = 2000
DEFAULT_PLOT_SIZE = (3.6, 3.2)
DEFAULT_PLOT_FONT_SIZE = 9

# Set random seed to ensure reproducibility
def set_seed(seed=ML_RANDOM_SEED):
    global ML_RANDOM_SEED
    ML_RANDOM_SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(ML_RANDOM_SEED)

# def print_tensor(input_tensor: torch.tensor, precision=4, scientific_notation_limit=4):
#     smart_format_float()
    # TODO

# a = torch.tensor([1.,2.,3.,4.,5.1248120985602985694])
# print(a)

def is_same_dim(a: torch.Tensor, b: torch.Tensor):
    return a.shape == b.shape

def check_same_dim(a: torch.Tensor, b: torch.Tensor):
    if not is_same_dim(a, b):
        raise ValueError(f"Tensor dimension mismatch: {a.shape} vs {b.shape}")

# TODO
def load_and_preprocess_data(Xs,
                             Ys,
                             test_set_ratio,
                             forward_norm_functions=[]):
    
    input("load_and_preprocess_data() is not finished yet.")
    assert 0 < test_set_ratio < 1
    self.Xs_train, self.Xs_test, self.Ys_train, self.Ys_test = train_test_split(Xs, Ys, test_size=test_set_ratio)


def dataset_normalization(X, Y, *forward_norm_functions):
    """
    parameters:
        Given X,Y, both are n lines of tuple of input/output vectors
        e.g. [(1,2),
              (3,4),
              (5,6),
              ...]
        if one of X,Y is only n lines of float, it is converted to n lines of 1-tuple of input/output vectors

        [converting_functions]

        The length of the converting_functions should be the sum of the vector dimensions of the input/output vectors,
        e.g. for an input dimension of 5, and output dimension of 2, the converting_functions should have 7 objects.
        For each column (feature or target):
            1) [None] → standardization to (mean=0, std=1).
               If all values are identical, you get 0, and inverse returns by adding the mean back. 
            2) [False] → no transformation.
            3) (lower_bound, upper_bound) → linear scaling to [lower_bound, upper_bound].  
               If all values are identical, everything is mapped to the midpoint of [lower_bound, upper_bound],
               and the inverse returns the original value by adding the mean back.
            4) (forward_func, inverse_func) → custom column transform.  Both must be callable.  
               forward_func(col_data) yields transformed data,  
               inverse_func(t_data) yields original data.

    return:
        a 4-tuple of (normalized_X, normalized_Y, reverse_function_X, reverse_function_Y)
          normalized_X: Normalized list of x vectors
          normalized_Y: Normalized list of y vectors
          reverse_function_x: a function that, when called as reverse_function_x(one_normalized_x_vector),
                              returns the corresponding original (unnormalized) x-vector.
          reverse_function_y: similarly for one_normalized_y_vector.
    """

    # Convert X, Y to numpy arrays of shape (n, dX) and (n, dY).
    X_arr = np.array(X, dtype=float)
    Y_arr = np.array(Y, dtype=float)

    # If either is 1D, reshape to (n, 1).
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(-1, 1)

    n, dX = X_arr.shape
    n2, dY = Y_arr.shape
    if n != n2:
        raise ValueError("X and Y must have the same number of rows.")

    # Ensure we have enough converting functions for X + Y columns.
    total_dims = dX + dY
    if len(forward_norm_functions) != total_dims:
        raise ValueError(
            f"Expected {total_dims} converting functions for X (dim={dX}) + Y (dim={dY}), "
            f"got {len(forward_norm_functions)}."
        )

    # Combine X and Y horizontally for column-wise processing.
    combined = np.hstack([X_arr, Y_arr])

    # We will store the "inverse" transformations here.
    # reversing_functions[i] will be a function that inverts the transform on column i.
    reversing_functions = [None] * total_dims

    # Process each column according to the corresponding converting function.
    for i, conv in enumerate(forward_norm_functions):
        col = combined[:, i]

        # 1) Standardization (None)
        if conv is None:
            mean_ = col.mean()
            std_ = col.std()
            if np.isclose(std_, 0.0):
                # All values are identical: map them to 0 by subtracting the mean
                combined[:, i] = col - mean_  # this is all zeros

                def inv_func(transformed_value, _m=mean_):
                    return transformed_value + _m
            else:
                combined[:, i] = (col - mean_) / std_

                def inv_func(transformed_value, _m=mean_, _s=std_):
                    return transformed_value * _s + _m

            reversing_functions[i] = inv_func

        # 2) No change (False)
        elif conv is False:
            reversing_functions[i] = lambda c: c

        # 3) Possibly a tuple (for range-scaling or custom user function)
        elif isinstance(conv, tuple) and len(conv) == 2:
            # Distinguish (lower_bound, upper_bound) from (forward_func, inverse_func).
            # Check if both items are numeric => range-scaling
            # Otherwise, assume both are callable => custom transform
            if all(isinstance(x, (int, float)) for x in conv):
                # Linear scaling
                lb, ub = conv
                min_ = col.min()
                max_ = col.max()

                if np.isclose(max_, min_):
                    # All values are identical
                    midpoint = 0.5 * (lb + ub)
                    combined[:, i] = midpoint

                    def inv_func(transformed_value, _m=min_, shifted_to=midpoint):
                        return transformed_value + val - midpoint
                else:
                    range_ = max_ - min_
                    scale_ = (ub - lb) / range_
                    combined[:, i] = (col - min_) * scale_ + lb

                    def inv_func(transformed_value, _lb=lb, _ub=ub, _mn=min_, _mx=max_):
                        # Reverse of: col' = (col - mn)*scale_ + lb
                        # => col = (col' - lb)/scale_ + mn
                        return (transformed_value - _lb) * (_mx - _mn) / (_ub - _lb) + _mn

                reversing_functions[i] = inv_func

            elif all(callable(x) for x in conv):
                # Two callables => (forward_func, inverse_func)
                forward_func, inverse_func = conv
                transformed_col = forward_func(col)
                combined[:, i] = transformed_col

                def inv_func(transformed_value, inv_call=inverse_func):
                    return inv_call(transformed_value)

                reversing_functions[i] = inv_func

            else:
                raise ValueError(
                    f"Invalid converter for column {i}. If it's a tuple, it must be either "
                    "(lower_bound, upper_bound) with numeric bounds or "
                    "(forward_func, inverse_func) with callables."
                )
        else:
            raise ValueError(f"Invalid converter for column {i}: {conv}")

    # Split back into normalized X, Y.
    norm_X = combined[:, :dX]
    norm_Y = combined[:, dX:]

    # Define the reverse functions for X and Y (handle single-vector or batch).
    def reverse_function_x(x_norm):
        """
        Given a normalized x-vector or an array of them,
        returns the vector(s) in the original scale.
        """
        x_norm = np.array(x_norm, dtype=float, copy=True)
        if x_norm.ndim == 1:
            # Single row
            for idx in range(dX):
                x_norm[idx] = reversing_functions[idx](x_norm[idx])
        else:
            # Multiple rows
            for idx in range(dX):
                x_norm[:, idx] = reversing_functions[idx](x_norm[:, idx])
        return x_norm

    def reverse_function_y(y_norm):
        """
        Given a normalized y-vector or an array of them,
        returns the vector(s) in the original scale.
        """
        y_norm = np.array(y_norm, dtype=float, copy=True)
        if y_norm.ndim == 1:
            # Single row
            for idx in range(dY):
                y_norm[idx] = reversing_functions[dX + idx](y_norm[idx])
        else:
            # Multiple rows
            for idx in range(dY):
                y_norm[:, idx] = reversing_functions[dX + idx](y_norm[:, idx])
        return y_norm

    # Convert normalized arrays back to list-of-tuples for output consistency.
    norm_X_list = [tuple(row) for row in norm_X]
    norm_Y_list = [tuple(row) for row in norm_Y]

    return norm_X_list, norm_Y_list, reverse_function_x, reverse_function_y


def nice_numbers_around_sequence(current_number, last_number):
    """
    I wish to print stuff in an interval dictated by a geometric sequence,
    so that it prints more frequently at the begining, but not so much later.

    but the numbers in geometric sequences are ugly, for example:
        70 5912
        71 6504
        72 7155
        73 7871
        74 8659
        75 9525
        76 10478
        77 11526
    now I choose nice numbers that keep this trend and general gaps but are nicer:
        70 5912
        71 6500
        72 7200
        73 7900
        74 8700
        75 9500
        76 10500
        77 11500

    The output number has the minimum amount of digits so that
    it can be differentiated with the last number

    :param last_number:
    :param current_number:
    :return:
    """

    if current_number <= last_number:
        return current_number
    increase = current_number / last_number - 1
    order_of_magnitute = 10 ** (int(math.log10(current_number)))
    order_of_magnitute = max(order_of_magnitute, 1)
    if increase > 1:
        current_number = math.ceil(current_number / order_of_magnitute) * order_of_magnitute
    else:
        digit_needed = math.ceil(-math.log10(increase)) + 1
        divider = order_of_magnitute / 10 ** (digit_needed - 1)
        divider = max(divider, 1)
        current_number = math.ceil(current_number / divider) * divider
    return current_number


def next_output_epoch(last_number, every_n_epoch, by_geo_sequence):
    new_number = last_number * by_geo_sequence
    if every_n_epoch is not None:
        new_number = min(new_number, last_number + every_n_epoch)
    return nice_numbers_around_sequence(new_number, last_number)


# TODO: create functions for common conversions
def tensor_to_list(tensor: torch.Tensor):
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    # Convert to numpy array first, then to list
    return tensor.detach().numpy().tolist()


def to_independent_torch_tensor(x, dtype=None, device=None):
    """
    Convert input (torch.Tensor, numpy.ndarray, list, or tuple)
    into a new independent torch.Tensor.
    Supports arbitrarily nested lists/tuples of tensors/numbers.
    """
    if isinstance(x, torch.Tensor):
        return x.clone().detach().to(dtype=dtype, device=device)

    if isinstance(x, np.ndarray):
        return torch.tensor(x.copy(), dtype=dtype, device=device)

    if isinstance(x, (list, tuple)):
        # Recursively convert all elements
        converted = [to_independent_torch_tensor(el, dtype=dtype, device=device) for el in x]
        return torch.stack(converted)

    # Allow scalars (int/float)
    if isinstance(x, (int, float)):
        return torch.tensor(x, dtype=dtype, device=device)

    raise TypeError(f"Unsupported type: {type(x)}")



class Batch_Sampler_by_Tag(torch.utils.data.Sampler):
    '''
    Sampler that ensures all data with the same tag are in the same batch.
    It adds tags to a batch until the batch size exceeds the specified limit, then yields the batch.
    '''
    def __init__(self, tags, batch_size, shuffle):
        self.tags = tags.detach().cpu().numpy().tolist()
        self.batch_size = batch_size
        self.shuffle = shuffle
                
        # Convert tags from 2D tensor to list of tuples
        # tags is (N, tag_dim), so each row is a tag descriptor
        tags_list = [tuple(t) for t in self.tags]

        # Group indices by tag
        self.tag_to_indices = defaultdict(list)    
        for idx, tag in enumerate(tags_list):
            self.tag_to_indices[tag].append(idx)
            
        self.unique_tags = list(self.tag_to_indices.keys())
        
    def __iter__(self):
        batch = []
        tags_order = copy.copy(self.unique_tags)
        if self.shuffle:
            np.random.shuffle(tags_order)
            
        for tag in tags_order:
            indices = self.tag_to_indices[tag]
            batch.extend(indices)
            
            # Yield if batch size exceeded
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        
        # Yield remaining
        if len(batch) > 0:
            yield batch

    def __len__(self):
        # This is an estimation because actual number of batches depends on shuffle and tag sizes
        # But for progress bars it is usually okay
        return (len(self.tags) + self.batch_size - 1) // self.batch_size



class Multitask_Strategy:
    Uncertainty_Weighted_Loss = "Uncertainty_Weighted_Loss_2093hm8204gh974g029f7h398fm12089fh2907tgrg"
    GradNorm = "GradNorm_03q984gh0398hg98whmf028934fmxisfuvhn9340gh2937fh9w8df"


class GradNorm_Manager(nn.Module):
    """
    GradNorm: <Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks (ICML 2018)>
    利用反向传播的梯度幅度来动态调整权重。引入了损失函数L_grad更新任务权重。
    Computes the gradient norm $G_i$ for each task $i$'s loss with respect to the shared layer weights
    A metric $r_i(t) = \tilde{L}_i(t) / \tilde{L}_i(0)$ is defined, 
        where $\tilde{L}_i(t)$ represents the current loss,
        and $\tilde{L}_i(0)$ the initial loss. 
        This ratio quantifies the task's optimization progress relative to its initial state.
    The algorithm calculates the average gradient norm across all tasks, denoted as $\bar{G}$. 
        The target gradient norm for each task is established as 
        $\bar{G} \times [r_i(t)]^\alpha$ (where $\alpha$ is a hyperparameter). 
        Consequently, tasks with slower loss reduction (a higher $r_i$) are assigned a larger target gradient, 
        thereby necessitating a higher weight.
    Task weights $w_i$ are updated by minimizing the function $L_{grad} = \sum_i |G_i - G_{target}^{(i)}|_1$.
    """

    def __init__(self, n_tasks, model: nn.Module, device, alpha=3, lr=0.1):
        super().__init__()
        self.n_tasks = n_tasks
        self.model = model
        self.device = device
        self.alpha = alpha
        
        # weights_param: Weights for each task, initialized to 1.0 (learnable)
        self.weights_param = torch.ones(n_tasks, dtype=torch.float32, device=device, requires_grad=True)
        
        # Separate optimizer for weights
        self.weights_optim = torch.optim.Adam([self.weights_param], lr=lr) 
        
        self.initial_losses = None
        self.is_initialized = False # Initialize based on first batch
        self.shared_weights = None 
        
        # Public attributes for logging (matching Uncertainty_Weighted_Loss interface)
        self.loss_terms = [] 
        self.weights = [] 
        
        self.current_step_losses = [] 

    def forward(self, *losses):
        """
        Receives individual task losses.
        Returns the weighted sum for the main model update.
        """
        self.current_step_losses = list(losses)

        # Initialization: Scale weights so that w_i * L_i is roughly constant and sum(w_i) = 1
        if self.training and not self.is_initialized:
            with torch.no_grad():
                loss_vals = torch.tensor([l.item() for l in losses], device=self.device, dtype=torch.float32)
                # Avoid division by zero
                safe_loss_vals = loss_vals.clone()
                safe_loss_vals[safe_loss_vals < 1e-8] = 1e-8
                
                # We want w_i * L_i = k (constant)
                # w_i = k / L_i
                # sum(w_i) = 1  => sum(k/L_i) = 1 => k * sum(1/L_i) = 1 => k = 1 / sum(1/L_i)
                inv_losses = 1.0 / safe_loss_vals
                total_inv = inv_losses.sum()
                
                if total_inv > 0:
                    new_weights = inv_losses / total_inv
                    self.weights_param.data.copy_(new_weights)
                    print("GradNorm: Initialized weights to balance loss magnitudes (sum=1).")
                    print(f"  Losses: {loss_vals.cpu().numpy()}")
                    print(f"  Weights: {self.weights_param.data.cpu().numpy()}")
                    print(f"  Weighted Losses: {(loss_vals * self.weights_param.data).cpu().numpy()}")
                
                # Also set initial losses here as this is the "initial" state
                self.initial_losses = torch.tensor([l.item() for l in losses], device=self.device)
                
                self.is_initialized = True
        
        # Log values
        self.loss_terms = [l.item() for l in losses]
        self.weights = [w.item() for w in self.weights_param]
        
        # Calculate weighted loss
        # Use detached weights for the main loss so main model doesn't optimize w_i
        total_loss = sum(w * l for w, l in zip(self.weights_param.detach(), losses))
        
        return total_loss

    def backward_and_step(self, total_loss):
        """
        Performs backward pass on total_loss (updating main model grads)
        AND performs GradNorm update on self.weights.
        """
        # 1. Backward pass for the main model
        total_loss.backward(retain_graph=True)
        
        # 2. Identify shared layer (if not already found)
        if self.shared_weights is None:
            self.find_shared_weights()
            if self.shared_weights is None:
                return 

        # 3. GradNorm Update
        norms = []
        for loss in self.current_step_losses:
            # Calculate grad(L_i, W)
            g_tuple = torch.autograd.grad(loss, self.shared_weights, retain_graph=True, allow_unused=True)
            g = g_tuple[0]
            if g is None:
                norms.append(torch.tensor(0.0, device=self.device))
            else:
                norms.append(torch.norm(g))
        norms = torch.stack(norms)

        if self.initial_losses is None:
            self.initial_losses = torch.tensor([l.item() for l in self.current_step_losses], device=self.device)
            # Prevent division by zero
            self.initial_losses[self.initial_losses < 1e-8] = 1.0

        current_loss_vals = torch.tensor([l.item() for l in self.current_step_losses], device=self.device)
        loss_ratios = current_loss_vals / self.initial_losses
        
        avg_ratio = loss_ratios.mean()
        inverse_training_rates = loss_ratios / (avg_ratio + 1e-8)
        
        # Target norms
        weighted_norms = norms * self.weights_param.detach()
        mean_weighted_norm = weighted_norms.mean()
        
        target_norms = mean_weighted_norm * (inverse_training_rates ** self.alpha)
        target_norms = target_norms.detach()
        
        # L_grad = sum | w_i * ||grad L_i|| - target |
        l_grad = torch.nn.functional.l1_loss(self.weights_param * norms, target_norms)
        
        self.weights_optim.zero_grad()
        l_grad.backward()
        self.weights_optim.step()
        
        # Renormalize weights
        with torch.no_grad():
            normalize_coeff = 1.0 / (self.weights_param.sum() + 1e-8)
            self.weights_param.data = self.weights_param.data * normalize_coeff

    def find_shared_weights(self):
        candidate_params = list(self.model.parameters())
        common_ids = set(id(p) for p in candidate_params)
        
        for loss in self.current_step_losses:
            grads = torch.autograd.grad(loss, candidate_params, retain_graph=True, allow_unused=True)
            active_ids = {id(p) for p, g in zip(candidate_params, grads) if g is not None}
            common_ids &= active_ids
            
        if not common_ids:
            print("GradNorm Warning: No common parameters found for all tasks. GradNorm disabled.")
            return

        final_shared_param = None
        for p in reversed(candidate_params):
            if id(p) in common_ids:
                if p.dim() > 1: 
                     final_shared_param = p
                     break
                elif final_shared_param is None: 
                     final_shared_param = p
        
        if final_shared_param is not None:
             print(f"GradNorm: Auto-detected shared layer parameter with shape {final_shared_param.shape}")
             self.shared_weights = final_shared_param


# TODO: 处理输入的类型问题
class Train_NN_Network:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_functions: Union[Callable, List[Callable]],
                 # 如果只声明一个 loss_function：
                 #    loss_function should take 2 or 3 parameters, with (Ys_pred, Ys_known, [Tags])
                 #    Dimentions (batch_size, len(Ys), _), (batch_size, len(Ys), _), (batch_size, len(Ys), _)
                 #    3 parameters should be requested if the Tags and batch_by_tag is selected
                 # 如果声明多个 loss_function 作为一个 list of callable 传入
                 #    the requrement for each loss_function is same as above
                 #    Then Automatic Weighted Loss will be applied to balance the multiple objectives
                 #    With the additional parameter eta[i] added for each loss function and are optimized along with the model parameters
                 
                 multitask_strategy=None, 
                 # If multiple loss functions are given, choose the strategy to balance them, can be:
                 #   Multitask_Strategy.Uncertainty_Weighted_Loss
                 #   Multitask_Strategy.GradNorm

                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                 
                 *,
                 Xs_train: torch.Tensor,  # should be normalized
                 Ys_train: torch.Tensor,  # should be normalized
                 Xs_test: torch.Tensor,  # should be normalized
                 Ys_test: torch.Tensor,  # should be normalized
                 Tags_train: Optional[torch.Tensor] = None,  # (Num_Entries, N_Tags) Optional tags for each training data, in case it in needed for grouping dependent loss function
                 Tags_test: Optional[torch.Tensor] = None,   # (Num_Entries, N_Tags) Optional tags for each training data

                 batch_size=None,  # None for Everything as one batch
                 batch_by_tag = False,  # If True, each batch contains data with the same tag. Only works when Tags_train/test is given.
                 shuffle = True,

                 ########## Optimization helpers ##########
                 device=None,
                 max_gradient_norm=None,

                 ########## Stop conditions ##########
                 initial_max_epoch=None,
                 ask_after_max_epoch=True,
                 automatic_extension=True,
                 loss_weight_convergence_threshold=1e-3, # 在使用多个loss function需要weighting的情况下，loss weight必须在最后10%的epoch里的变化程度小于这个比例才叫收敛
                 early_stopping_strategy=None,
                 # Can choose from: do not stop, or stop when testing data hasen't dropped for n cycles, then load the lowest one

                 ########## Restart options ##########
                 load_from_save_path=None,
                 save_path_stem="",
                 save_every_n_epoch=None,
                 save_by_geo_sequence=1.02,  # save when it's more than 10% more epoch than last save
                 save_pngs=True,
                 save_script=True,

                 ########## Print and Plot options ##########
                 print_every_n_epoch=200,
                 print_by_geo_sequence=1.005,
                 plot_every_n_epoch=200,
                 plot_by_geo_sequence=1.005,

                 plot_current_linear_fit=True,
                 plot_loss_over_epoch=True,
                 function_for_numerical_evaluation=None,
                 # a callable to convert Ys output to a list of scalar, so that it can be plotted against the given output in a linear way

                 plot_cases=None,  # ((X1,Y1_known),(X2,Y2_known)...) # TODO: or just (X,Y)
                 plot_cases_tags=None,  # Optional tags for each plot case, for evaluating the loss for the test function
                 plot_cases_names=None,
                 function_for_plot_test_cases=None,
                 # a callable to convert one X and one Y to two list of scalars, so that it can be plotted against the predicted value
                 
                 callback_function=None,
                 non_opt_evaluation_functions=None,
                 non_opt_evaluation_function_names=None
                 ):
        """
        Docstring for __init__
        """

        ########## GPU/CPU selection ##########
        print()
        if device is not None:
            print("Using device:", device)
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")
            if torch.cuda.is_available():
                # Get the number of GPUs available
                num_gpus = torch.cuda.device_count()
                if num_gpus > 1:
                    input(f"Multiple GPUs ({num_gpus}) detected. The library did not account for this situation.")
                print(f"Using GPU {0}: {torch.cuda.get_device_name(0)}")
                self.device = torch.device('cuda:0')
            else:
                print("No GPUs available.")
        print()

        ########### Input checks ##########
        self.input_check(loss_functions, scheduler, Xs_train, Xs_test, Ys_train, Ys_test, Tags_train, Tags_test, batch_size, batch_by_tag, multitask_strategy)
        
        if not save_path_stem:
            caller_frame = inspect.stack()[1]
            caller_full_path = os.path.abspath(caller_frame.filename)
            caller_filename = filename_name(caller_full_path)
            save_folder = os.path.join(filename_parent(caller_full_path), "Checkpoints")
            save_folder = os.path.join(save_folder, f"{caller_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(save_folder)
            save_path_stem = os.path.join(save_folder, "Save")
        else:
            save_folder = os.path.dirname(save_path_stem)
        
        save_path_stem = get_unused_filename(save_path_stem)

        if save_script:
            # Save the caller script in the checkpoint folder
            caller_frame = inspect.stack()[1]
            caller_full_path = os.path.abspath(caller_frame.filename)
            dst_script_path = os.path.join(save_folder, filename_name(caller_full_path))
            os.makedirs(dst_script_path, exist_ok=True)
            try:
                shutil.copy2(caller_full_path, dst_script_path)
                print(f"Original script {caller_full_path} backed up to {dst_script_path}")
            except Exception as e:
                traceback.print_exc()
                print(e)
                print(f"Warning: could not save script {caller_full_path} -> {save_path_stem}: {e}")

        self.model = model.to(self.device)
        self.optimizer = optimizer

        ######### Loss function(s) ##########
        # 要么是weighted_multiple_loss_mode = True，并且存在self.loss_functions作为一个数组
        # 要么是weighted_multiple_loss_mode = False，并且存在self.loss_function作为单一函数

        self.multitask_strategy = multitask_strategy

        if isinstance(loss_functions, Sequence) and len(loss_functions)==1:
            self.loss_function = loss_functions[0]
            self.weighted_multiple_loss_mode = False
        elif isinstance(loss_functions, Callable):
            self.weighted_multiple_loss_mode = False
            self.loss_function = loss_functions
        elif isinstance(loss_functions, Sequence) and len(loss_functions)>1:
            self.weighted_multiple_loss_mode = True
            self.loss_functions = loss_functions
            
            if self.multitask_strategy == Multitask_Strategy.Uncertainty_Weighted_Loss:
                self.automatic_weighted_loss_model = Uncertainty_Weighted_Loss(len(loss_functions)).to(self.device)
                # usually a smaller lr for the loss weights to avoid the model learns to optimize weight instead of the main model parameters
                self.optimizer.add_param_group({'params': self.automatic_weighted_loss_model.parameters(),
                                                "lr": optimizer.defaults['lr']*10}) 
                print(f"Multi-task learning enabled with {len(loss_functions)} objectives (Uncertainty Weighted Loss).")
                
            elif self.multitask_strategy == Multitask_Strategy.GradNorm:
                print(f"Multi-task learning enabled with {len(loss_functions)} objectives (GradNorm).")
                # GradNorm Manager initialization
                self.automatic_weighted_loss_model = GradNorm_Manager(len(loss_functions), self.model, self.device, lr=optimizer.defaults['lr'])

        self.scheduler = scheduler

        self.save_path_stem = save_path_stem
        self.save_pngs = save_pngs
        self.save_every_n_epoch = save_every_n_epoch
        self.save_by_geo_sequence = save_by_geo_sequence
        self.print_every_n_epoch = print_every_n_epoch
        self.print_by_geo_sequence = print_by_geo_sequence
        self.plot_every_n_epoch = plot_every_n_epoch
        self.plot_by_geo_sequence = plot_by_geo_sequence
        self.plot_current_linear_fit = plot_current_linear_fit
        self.plot_loss_over_epoch = plot_loss_over_epoch
        self.function_for_numerical_evaluation = function_for_numerical_evaluation
        self.plot_cases = plot_cases or []
        if plot_cases_tags is None:
            self.plot_cases_tags = torch.empty((len(self.plot_cases), 0), dtype=torch.long)
        self.plot_cases_names = plot_cases_names or []
        self.function_for_plot_one_test = function_for_plot_test_cases
        self.callback_function = callback_function
        self.non_opt_evaluation_functions = non_opt_evaluation_functions or []
        self.non_opt_evaluation_function_names = non_opt_evaluation_function_names or []
        if len(self.non_opt_evaluation_functions) != len(self.non_opt_evaluation_function_names):
            self.non_opt_evaluation_function_names = [f"Eval {i + 1}" for i in range(len(self.non_opt_evaluation_functions))]

        self.linear_fit_window = Plot(fig_size=DEFAULT_PLOT_SIZE,
                                      font_size=10,
                                      x_axis_label="Expected",
                                      y_axis_label="Predicted")

        self.loss_over_epoch_window = Plot(fig_size=DEFAULT_PLOT_SIZE,
                                           font_size=DEFAULT_PLOT_FONT_SIZE,
                                           x_axis_label="Epoch",
                                           y_axis_label="Loss")

        if self.weighted_multiple_loss_mode:
            self.loss_terms_over_epoch_window = Plot(fig_size=DEFAULT_PLOT_SIZE,
                                                     font_size=DEFAULT_PLOT_FONT_SIZE,
                                                     x_axis_label="Epoch",
                                                     y_axis_label="Loss Terms")
            self.loss_weights_over_epoch_window = Plot(fig_size=DEFAULT_PLOT_SIZE,
                                                       font_size=DEFAULT_PLOT_FONT_SIZE,
                                                       x_axis_label="Epoch",
                                                       y_axis_label="Weights")
            self.loss_raw_over_epoch_window = Plot(fig_size=DEFAULT_PLOT_SIZE,
                                                       font_size=DEFAULT_PLOT_FONT_SIZE,
                                                       x_axis_label="Epoch",
                                                       y_axis_label="Raw Losses")

        self.evaluation_over_epoch_windows = [Plot(fig_size=DEFAULT_PLOT_SIZE,
                                                   font_size=DEFAULT_PLOT_FONT_SIZE,
                                                   x_axis_label="Epoch",
                                                   y_axis_label=name)
                                              for name in self.non_opt_evaluation_function_names]

        self.plot_cases_windows = [Plot(fig_size=DEFAULT_PLOT_SIZE,
                                        font_size=DEFAULT_PLOT_FONT_SIZE,
                                        x_axis_label="Input",
                                        y_axis_label="Output") for _ in range(len(self.plot_cases))]

        all_windows = [self.linear_fit_window, self.loss_over_epoch_window] + self.plot_cases_windows + self.evaluation_over_epoch_windows
        if self.weighted_multiple_loss_mode:
            all_windows.append(self.loss_terms_over_epoch_window)
            all_windows.append(self.loss_weights_over_epoch_window)
            all_windows.append(self.loss_raw_over_epoch_window)
        for window in all_windows:
            window.comrades = all_windows

        if Tags_train is None and Tags_test is None:
            Tags_train = torch.empty((len(Xs_train), 0), dtype=torch.long)
            Tags_test = torch.empty((len(Xs_test), 0), dtype=torch.long)

        self.Xs_train, self.Xs_test, self.Ys_train, self.Ys_test = Xs_train, Xs_test, Ys_train, Ys_test
        self.Tags_train:torch.Tensor = Tags_train
        self.Tags_test:torch.Tensor = Tags_test
        self.batch_by_tag = batch_by_tag
        self.shuffle = shuffle

        self.Xs_train = self.Xs_train.to(self.device)
        self.Ys_train = self.Ys_train.to(self.device)
        self.Xs_test = self.Xs_test.to(self.device)
        self.Ys_test = self.Ys_test.to(self.device)

        if batch_size:
            dataset_object = TensorDataset(self.Xs_train, self.Ys_train, self.Tags_train)
            if self.batch_by_tag:
                sampler = Batch_Sampler_by_Tag(self.Tags_train, batch_size=batch_size, shuffle=self.shuffle)
                batched_X_train_Y_train = DataLoader(dataset_object, batch_sampler=sampler)
            else:
                batched_X_train_Y_train = DataLoader(dataset_object, batch_size=batch_size, shuffle=self.shuffle)
        else:
            batched_X_train_Y_train = [[self.Xs_train, self.Ys_train, self.Tags_train]]


        self.epoch = 1
        if initial_max_epoch is None:
            initial_max_epoch = DEFAULT_INITIAL_EPOCH
        self.max_epoch = initial_max_epoch

        self.automatic_extension = automatic_extension
        self.ask_after_max_epoch=ask_after_max_epoch
        self.loss_weight_convergence_threshold=loss_weight_convergence_threshold

        self.training_losses = []
        self.test_losses = []
        if self.weighted_multiple_loss_mode:
            self.training_loss_terms_history = [] 
            self.test_loss_terms_history = []
            self.training_weights_history = []
            self.test_weights_history = []
        self.training_evaluations_history = [[] for _ in range(len(self.non_opt_evaluation_functions))]
        self.test_evaluations_history = [[] for _ in range(len(self.non_opt_evaluation_functions))]
        self.optimization_quality_history = []  # List of (epoch, is_good) tuples
        self.last_save_epoch = 0
        self.last_print_epoch = 0
        self.last_plot_epoch = 0
        self.user_interrupted = False  # Flag for manual interruption by pressing 'C'

        self.training_start_time = time.time()
        while self.epoch <= self.max_epoch:
            ############ Check for user interruption ############
            if non_block_keyboard_interrupt():
                print("\n[User requested interruption by pressing 'C'. Will stop after this epoch...]")
                self.user_interrupted = True
            
            ############ Training loop ############
            self.training_loss = 0.
            current_epoch_training_evaluations = [0.0] * len(self.non_opt_evaluation_functions)
            if self.weighted_multiple_loss_mode:
                current_epoch_loss_terms = [0.0] * len(self.loss_functions)
                current_epoch_weights = [0.0] * len(self.loss_functions)

            for batch_data in batched_X_train_Y_train:
                # Depending on how DataLoader unpacked it, it might be a list of tensors
                X_train_batch, Y_train_batch, Tags_train_batch = batch_data
                X_train_batch = X_train_batch.to(self.device)
                Y_train_batch = Y_train_batch.to(self.device)
                Tags_train_batch = Tags_train_batch.to(self.device)

                self.optimizer.zero_grad()
                Y_predict_train_batch = self.model(X_train_batch)
                
                batch_loss = self.evaluate_loss(Y_predict_train_batch, Y_train_batch, Tags_train_batch)

                if self.weighted_multiple_loss_mode:
                    for i, val in enumerate(self.automatic_weighted_loss_model.loss_terms):
                        current_epoch_loss_terms[i] += val
                    for i, val in enumerate(self.automatic_weighted_loss_model.weights):
                        current_epoch_weights[i] += val

                # Evaluation
                with torch.no_grad():
                    for i, func in enumerate(self.non_opt_evaluation_functions):
                        if func_param_count(func) == 3:
                            val = func(Y_predict_train_batch, Y_train_batch, Tags_train_batch)
                        elif func_param_count(func) == 2:
                            val = func(Y_predict_train_batch, Y_train_batch)
                        else:
                            raise ValueError(f"Evaluation function {i} has invalid number of parameters.")
                        if isinstance(val, torch.Tensor):
                            val = val.item()
                        current_epoch_training_evaluations[i] += val

                self.training_loss += batch_loss.item()
                if self.weighted_multiple_loss_mode and self.multitask_strategy == Multitask_Strategy.GradNorm:
                    self.automatic_weighted_loss_model.backward_and_step(batch_loss)
                else:
                    batch_loss.backward()

                if max_gradient_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_gradient_norm)
                self.optimizer.step()
            
            num_batches = len(batched_X_train_Y_train)

            self.training_loss /= num_batches
            for i in range(len(current_epoch_training_evaluations)):
                current_epoch_training_evaluations[i] /= len(batched_X_train_Y_train)
                self.training_evaluations_history[i].append(current_epoch_training_evaluations[i])

            if self.weighted_multiple_loss_mode:
                current_epoch_loss_terms = [x / num_batches for x in current_epoch_loss_terms]
                current_epoch_weights = [x / num_batches for x in current_epoch_weights]
                self.training_loss_terms_history.append(current_epoch_loss_terms)
                self.training_weights_history.append(current_epoch_weights)

            if self.scheduler is not None:
                self.scheduler.step()
            self.training_losses.append(self.training_loss)

            ############ Evaluation ############
            self.evaluate()

            ############ Termination control ############
            # Adjust max_epoch if still dropping; 如果只训练5个epoch，肯定是在做某种测试，不再extend
            if self.epoch == self.max_epoch:
                extended = self.optimization_extention_control()
                if not extended:
                    break

            self.epoch += 1

            if self.callback_function is not None:
                self.callback_function(self)

            ############ Check for user interruption ############
            if self.user_interrupted:
                print(f"Training interrupted by user at epoch {self.epoch}.")
                break

    def is_good_optimization_epochs(self, check_epoch):
        """
        Check if optimization is good at a specific epoch.
        """
        if check_epoch > len(self.test_losses) or check_epoch < 1:
            raise ValueError("check_epoch out of range in is_good_optimization_epochs().")
            
        # 1. If current loss is the lowest in history up to that point
        if self.test_losses[check_epoch - 1] <= min(self.test_losses[:check_epoch]):
            return True

        # 2. Check if loss is still dropping using trimmed mean comparison
        for ratio in [0.05, 0.1, 0.2]:
            window_size = math.ceil(check_epoch * ratio)
            group1 = self.test_losses[check_epoch - window_size:check_epoch]
            group2 = self.test_losses[check_epoch - 2 * window_size:check_epoch - window_size]
            if trimmed_mean(group1) < trimmed_mean(group2):
                return True
                
        return False
    
    def optimization_extention_control(self):
        """
        Control the extension of max_epoch based on the test loss trend.
        :return: True to continue, False to stop
        """
        time_check = time.time()
        if self.automatic_extension and self.epoch > 5:
            # Calculate all checkpoints with 5% geometric progression
            # 例如，self.optimization_quality_history 可能是
            # 
            # 1209, False  |  1270, False  |  1334, True  |  1401, True  |  1472, False  |  1546, False  
            # 1624, True  |  1706, True  |  1792, True  |  1882, True  |  1977, True  |  2076, True  |  
            # 2180, True  |  2289, True  |  2404, True  |  2525, True  |  2652, False  |  2785, False  |  
            # 2925, False  |  3072, False  |  3226, False  |  3388, False
            # 
            # 那么寻找最后一个连续是True的最大可能的阶段，此处是1547~2651，共1105个epoch
            # 那么新的max_epoch就是2651 + 3*1105 = 5966

            if not self.optimization_quality_history:
                check_epoch = 5
            else:
                last_check_epoch = self.optimization_quality_history[-1][0]
                check_epoch = math.ceil(last_check_epoch * 1.05)
            while check_epoch <= self.epoch:
                # Only calculate if not already in history
                is_good = self.is_good_optimization_epochs(check_epoch)
                self.optimization_quality_history.append((check_epoch, is_good))
                check_epoch = math.ceil(check_epoch * 1.05)

            # Find last True and start of its continuous block
            last_True_region_start = 0
            last_True_region_end = self.max_epoch

            if True in [x[1] for x in self.optimization_quality_history]:
                optimization_history = [(0, False)] + self.optimization_quality_history + [(self.epoch+1, False)]

                for i in range(1,len(optimization_history)-1):
                    before = optimization_history[i-1]
                    current = optimization_history[i]
                    after = optimization_history[i+1]

                    if before[1]==False and current[1]==True:
                        last_True_region_start = before[0]+1
                    if current[1]==True and after[1]==False:
                        last_True_region_end = after[0]-1
                
                new_max_epoch = last_True_region_end + 3 * (last_True_region_end - last_True_region_start + 1)
                if new_max_epoch > self.max_epoch:
                    print("Good optimization trend detection list:")
                    print(optimization_history)
                    print(f"Extension to {new_max_epoch} based on optimization quality history (3x length of last good area).")
                    self.max_epoch = new_max_epoch
                    print("Time taken for optimization extension control:", time.time() - time_check)
                    return True

        if self.ask_after_max_epoch:
            disable_keyboard_interrupt()  # Disable keyboard interrupt before input()
            while True:
                response = input(f'Max epoch {self.max_epoch} reached. To end the optimization, input "END"; otherwise input a new max_epoch number: ')
                if response == "END" or (is_int(response) and int(response) > self.max_epoch):
                    break
            enable_keyboard_interrupt()  # Re-enable keyboard interrupt after input()
            if response == "END":
                return False
            else:
                self.max_epoch = int(response)
                return True

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            Ys_predict_test: torch.Tensor = self.model(self.Xs_test)

            test_loss = self.evaluate_loss(Ys_predict_test, self.Ys_test, self.Tags_test)
            self.test_losses.append(test_loss)

            if self.weighted_multiple_loss_mode:
                self.test_loss_terms_history.append(self.automatic_weighted_loss_model.loss_terms)
                self.test_weights_history.append(self.automatic_weighted_loss_model.weights)

            # Evaluation
            current_test_evals = []
            for i, func in enumerate(self.non_opt_evaluation_functions):
                if len(inspect.signature(func).parameters) == 2:
                    val = func(Ys_predict_test, self.Ys_test)
                elif len(inspect.signature(func).parameters) == 3:
                    val = func(Ys_predict_test, self.Ys_test, self.Tags_test)
                else:
                    raise ValueError(f"Evaluation function {i} has invalid number of parameters.")    

                if isinstance(val, torch.Tensor):
                    val = val.item()
                self.test_evaluations_history[i].append(val)
                current_test_evals.append(val)

            info_text = f"[按C停训] Epoch {str(self.epoch).rjust(len(str(self.max_epoch)))}/{self.max_epoch} | "
            info_text += f"Loss{smart_format_float(self.training_losses[-1], 4):>7}/{smart_format_float(test_loss, 4):<7}"

            for name, train_val, test_val in zip(self.non_opt_evaluation_function_names,
                                                 [h[-1] for h in self.training_evaluations_history],
                                                 current_test_evals):
                info_text += f"| {name} {smart_format_float(train_val, 4):>7}/{smart_format_float(test_val, 4):<7}"

            if self.weighted_multiple_loss_mode:
                for i in range(len(self.loss_functions)):
                    train_term = self.training_loss_terms_history[-1][i]
                    test_term = self.test_loss_terms_history[-1][i]
                    train_weight = self.training_weights_history[-1][i]
                    test_weight = self.test_weights_history[-1][i]
                    
                    info_text += f"| T{i+1} {smart_format_float(train_term, 4)}/{smart_format_float(test_term, 4)}"
                    info_text += f" W{i+1} {smart_format_float(train_weight, 4)}/{smart_format_float(test_weight, 4)}"

            if len(info_text) > 60:
                indices = [i for i, c in enumerate(info_text) if c == '|']
                if indices:
                    split_idx = min(indices, key=lambda x: abs(x - len(info_text)//2))
                    info_text = info_text[:split_idx] + "\n    " + info_text[split_idx+1:].lstrip()

            filename_text = f"{self.epoch:0>5.0f}_{smart_format_float(self.training_losses[-1], 4)}_{smart_format_float(test_loss, 4)}"

            ############ Printing ############
            if self.should_print_this_epoch():
                print_elapsed_time = print_time_difference(time.time() - self.training_start_time, width=0)
                print_training_speed_epochs = 10 ** (int(math.log10(self.epoch / 10)))
                print_training_speed_epochs = max(print_training_speed_epochs, 1)
                print_training_speed = (time.time() - self.training_start_time) / self.epoch * print_training_speed_epochs
                print_training_speed = f"{print_time_difference(print_training_speed, width=0)}/{print_training_speed_epochs} Epoch{'s' if print_training_speed_epochs > 1 else ''}"

                print(f'{info_text} | Time: {print_elapsed_time} | {print_training_speed}')

            ############ Plotting ############
            active_windows: List[Plot] = []
            if self.should_plot_this_epoch():
                if self.plot_current_linear_fit:
                    self.linear_fit_window.set_window_title(f"Current Linear Fit @ Epoch {self.epoch}")
                    self.linear_fit_window.set_figure_title(info_text)

                    Ys_predict_train = self.model(self.Xs_train).clone().detach()
                    Ys_predict_test = Ys_predict_test.clone().detach()
                    Ys_test = self.Ys_test.clone().detach()
                    Ys_train = self.Ys_train.clone().detach()

                    plottable = False
                    if callable(self.function_for_numerical_evaluation):
                        test_curve_vertical = self.function_for_numerical_evaluation(Ys_predict_test)
                        test_curve_horizontal = self.function_for_numerical_evaluation(Ys_test)
                        train_curve_vertical = self.function_for_numerical_evaluation(Ys_predict_train)
                        train_curve_horizontal = self.function_for_numerical_evaluation(Ys_train)
                        plottable = True

                    elif Ys_predict_test.ndim == 2 and Ys_predict_test.shape[1] == 1:
                        test_curve_vertical = Ys_predict_test.squeeze(1)
                        test_curve_horizontal = Ys_test.squeeze(1)
                        train_curve_vertical = Ys_predict_train.squeeze(1)
                        train_curve_horizontal = Ys_train.squeeze(1)
                        plottable = True

                    if plottable:
                        linear_range_min = min(min(test_curve_horizontal), min(test_curve_vertical), min(train_curve_horizontal), min(train_curve_vertical))
                        linear_range_max = max(max(test_curve_horizontal), max(test_curve_vertical), max(train_curve_horizontal), max(train_curve_vertical))
                        linear_range_min -= (linear_range_max - linear_range_min) * 0.1
                        linear_range_max += (linear_range_max - linear_range_min) * 0.1

                        curves = [Curve(train_curve_horizontal, train_curve_vertical, plot_dot=True, dot_color=blue_color, Y_label="Training results")]
                        curves.append(Curve(test_curve_horizontal, test_curve_vertical, plot_dot=True, dot_color=red_color, Y_label="Test results"))
                        curves.append(Curve([linear_range_min, linear_range_max], [linear_range_min, linear_range_max], curve_color="#BBBBBB"))
                        self.linear_fit_window.Curve_objects = curves
                        active_windows.append(self.linear_fit_window)
                        if self.save_pngs and self.should_save_this_epoch():
                            self.linear_fit_window.save_png(self.save_path_stem + f"_Linear_Fit.{filename_text}.png", dpi=300, bbox_inches=None)

                if self.plot_loss_over_epoch:
                    self.loss_over_epoch_window.set_figure_title(info_text)
                    self.loss_over_epoch_window.set_window_title(f"Loss over Epoch @ Epoch {self.epoch}")
                    curve_X = list(range(self.epoch))
                    self.loss_over_epoch_window.Curve_objects = \
                        [Curve(curve_X, self.training_losses, curve_color=blue_color, Y_label="training loss"),
                         Curve(curve_X, self.test_losses, curve_color=red_color, Y_label="test loss")]
                    if self.save_pngs and self.should_save_this_epoch():
                        self.loss_over_epoch_window.save_png(self.save_path_stem + f"_0Losses.png", dpi=300, bbox_inches=None)
                    active_windows.append(self.loss_over_epoch_window)

                if self.weighted_multiple_loss_mode:
                    self.loss_terms_over_epoch_window.set_figure_title(info_text)
                    self.loss_terms_over_epoch_window.set_window_title(f"Loss Terms over Epoch @ Epoch {self.epoch}")
                    
                    curve_X = list(range(self.epoch))
                    curves = []
                    for i in range(len(self.loss_functions)):
                        train_losses_weighted = [epoch_data[i]*weight[i] for epoch_data,weight in zip(self.training_loss_terms_history,self.training_weights_history)]
                        test_losses_weighted = [epoch_data[i]*weight[i] for epoch_data,weight in zip(self.test_loss_terms_history,self.test_weights_history)]
                        color = matplotlib_colors[i + 2]
                        curves.append(Curve(curve_X, train_losses_weighted, curve_color=color, curve_format=dashed_line, Y_label=f"Weighted Loss {i+1} Train"))
                        curves.append(Curve(curve_X, test_losses_weighted, curve_color=color, curve_format=solid_line, Y_label=f"Weighted Loss {i+1} Test"))
                    
                    self.loss_terms_over_epoch_window.Curve_objects = curves
                    active_windows.append(self.loss_terms_over_epoch_window)
                    if self.save_pngs and self.should_save_this_epoch():
                        self.loss_terms_over_epoch_window.save_png(self.save_path_stem + f"_0LossTerms.png", dpi=300, bbox_inches=None)

                    self.loss_weights_over_epoch_window.set_figure_title(info_text)
                    self.loss_weights_over_epoch_window.set_window_title(f"Weights over Epoch @ Epoch {self.epoch}")
                    
                    curve_X = list(range(self.epoch))
                    curves = []
                    for i in range(len(self.loss_functions)):
                        train_losses_weighted = [epoch_data[i] for epoch_data in self.training_weights_history]
                        test_losses_weighted = [epoch_data[i] for epoch_data in self.test_weights_history]
                        color = matplotlib_colors[i + 2]
                        curves.append(Curve(curve_X, train_losses_weighted, curve_color=color, curve_format="--", Y_label=f"Weight {i+1}"))
                        # curves.append(Curve(curve_X, test_y, curve_color=color, curve_format="-", Y_label=f"Weight {i+1} Test"))
                    
                    self.loss_weights_over_epoch_window.Curve_objects = curves
                    self.loss_weights_over_epoch_window.y_lim = (0, None)
                    active_windows.append(self.loss_weights_over_epoch_window)
                    if self.save_pngs and self.should_save_this_epoch():
                        self.loss_weights_over_epoch_window.save_png(self.save_path_stem + f"_0Weights.png", dpi=300, bbox_inches=None)

                    self.loss_raw_over_epoch_window.set_figure_title(info_text)
                    self.loss_raw_over_epoch_window.set_window_title(f"Raw Losses over Epoch @ Epoch {self.epoch}")
                    
                    curve_X = list(range(self.epoch))
                    curves = []
                    for i in range(len(self.loss_functions)):
                        train_losses_raw = [epoch_data[i] for epoch_data in self.training_loss_terms_history]
                        test_losses_raw = [epoch_data[i] for epoch_data in self.test_loss_terms_history]
                        color = matplotlib_colors[i + 2]
                        curves.append(Curve(curve_X, train_losses_raw, curve_color=color, curve_format=dashed_line, Y_label=f"Raw Loss {i+1} Train"))
                        curves.append(Curve(curve_X, test_losses_raw, curve_color=color, curve_format=solid_line, Y_label=f"Raw Loss {i+1} Test"))
                    
                    self.loss_raw_over_epoch_window.Curve_objects = curves
                    active_windows.append(self.loss_raw_over_epoch_window)
                    if self.save_pngs and self.should_save_this_epoch():
                        self.loss_raw_over_epoch_window.save_png(self.save_path_stem + f"_0RawLosses.png", dpi=300, bbox_inches=None)

                # New Evaluation Windows
                curve_X = list(range(self.epoch))
                for i, window in enumerate(self.evaluation_over_epoch_windows):
                    window.set_figure_title(info_text)
                    window.set_window_title(f"{self.non_opt_evaluation_function_names[i]} over Epoch @ Epoch {self.epoch}")
                    window.Curve_objects = [
                        Curve(curve_X, self.training_evaluations_history[i], curve_color=blue_color, Y_label="training"),
                        Curve(curve_X, self.test_evaluations_history[i], curve_color=red_color, Y_label="test")
                    ]
                    if self.save_pngs and self.should_save_this_epoch():
                        window.save_png(self.save_path_stem + f"_0Eval_{self.non_opt_evaluation_function_names[i]}.png", dpi=300, bbox_inches=None)
                    active_windows.append(window)

                if self.plot_cases:
                    for window_count, (case_data, case_window, case_tag) in enumerate(zip(self.plot_cases, self.plot_cases_windows, self.plot_cases_tags)):
                        case_window.set_window_title(f"Example Curve {window_count + 1} @ Epoch {self.epoch}")

                        input_, output_known = case_data
                        input_ = to_independent_torch_tensor(input_)
                        output_known = to_independent_torch_tensor(output_known)
                        if callable(self.function_for_plot_one_test):
                            curve_X, curve_Y_known = self.function_for_plot_one_test(input_, output_known)
                        else:
                            curve_X, curve_Y_known = case_data
                        output_predicted = self.run_model_for_one_input(case_data[0])
                        if callable(self.function_for_plot_one_test):
                            curve_X, curve_Y_predicted = self.function_for_plot_one_test(input_, output_predicted)
                        else:
                            curve_Y_predicted = output_predicted

                        print(curve_X)
                        print(curve_Y_known)
                        print(curve_Y_predicted)

                        case_window.Curve_objects = \
                            [Curve(curve_X, curve_Y_known, curve_color=blue_color, Y_label="known curve"),
                             Curve(curve_X, curve_Y_predicted, curve_color=red_color, Y_label="predicted curve"), ]
                        active_windows.append(case_window)
                    
                        example_loss = self.evaluate_loss(output_predicted.unsqueeze(0), output_known.unsqueeze(0), case_tag.unsqueeze(0)).item()

                        case_window.set_figure_title(
                            info_text + f" | {self.plot_cases_names[window_count]} Case {window_count + 1} | Loss {smart_format_float(example_loss, 4)}")
                        if self.save_pngs and self.should_save_this_epoch():
                            case_window.save_png(self.save_path_stem + f"_Example.{filename_text}.png", dpi=300, bbox_inches=None)

            for window_count, window in enumerate(active_windows):
                window.shift_window = WINDOW_POSITIONS[len(active_windows)][window_count]

            if self.should_save_this_epoch():
                save_object = {'epoch': self.epoch,
                               'model_state_dict': self.model.state_dict(),
                               'optimizer_state_dict': self.optimizer.state_dict(),
                               'training_losses': self.training_losses,
                               'test_losses': self.test_losses}
                save_path = self.save_path_stem + f".Epoch_{self.epoch:0>5.0f}.pth"
                torch.save(save_object, save_path)

        self.model.train()

        return test_loss

    def evaluate_loss(self,Ys_pred, Ys_known, Tags=None):
        assert Ys_pred.shape == Ys_known.shape, "Ys_pred and Ys_known must have the same shape."
        if self.weighted_multiple_loss_mode:
            losses = []
            for func in self.loss_functions:
                if func_param_count(func)==3:
                    losses.append(func(Ys_pred, Ys_known, Tags))
                elif func_param_count(func)==2:
                    losses.append(func(Ys_pred, Ys_known))
                else:
                    raise ValueError("Each loss function should take 2 or 3 parameters.")
                
            loss = self.automatic_weighted_loss_model(*losses)
        else:
            if func_param_count(self.loss_function)==3:
                loss = self.loss_function(Ys_pred, Ys_known, Tags)
            else:
                loss = self.loss_function(Ys_pred, Ys_known)

        return loss

    def should_save_this_epoch(self):
        if self.epoch == self.max_epoch:
            self.last_save_epoch = self.epoch
            return True
        if not self.last_save_epoch and self.epoch <= 1:
            self.last_save_epoch = 1
            return True
        if self.epoch == next_output_epoch(self.last_save_epoch,
                                           self.save_every_n_epoch,
                                           self.save_by_geo_sequence):
            self.last_save_epoch = self.epoch
            return True
        return False

    def should_print_this_epoch(self):
        if self.epoch == self.max_epoch:
            self.last_print_epoch = self.epoch
            return True
        if not self.last_print_epoch and self.epoch <= 1:
            self.last_print_epoch = 1
            return True
        if self.epoch == next_output_epoch(self.last_print_epoch,
                                           self.print_every_n_epoch,
                                           self.print_by_geo_sequence):
            self.last_print_epoch = self.epoch
            return True
        return False

    def should_plot_this_epoch(self):
        if self.epoch == self.max_epoch:
            self.last_plot_epoch = self.epoch
            return True
        if not self.last_plot_epoch and self.epoch <= 1:
            self.last_plot_epoch = 1
            return True
        if self.epoch == next_output_epoch(self.last_plot_epoch,
                                           self.plot_every_n_epoch,
                                           self.plot_by_geo_sequence):
            self.last_plot_epoch = self.epoch
            return True
        return False

    def run_model_for_one_input(self, input_vector):
        if not isinstance(input_vector, torch.Tensor):
            input_vector = torch.tensor(input_vector, dtype=torch.float32)
        Xs = input_vector.unsqueeze(0)
        Xs = Xs.detach().clone()
        Xs = Xs.to(self.device)
        return self.model(Xs).squeeze(0).cpu()

    def input_check(self, loss_function, scheduler, Xs_train, Xs_test, Ys_train, Ys_test, Tags_train, Tags_test, batch_size,batch_by_tag, multitask_strategy):
        if isinstance(loss_function, Callable):
            loss_function = [loss_function]        
        param_counts = [func_param_count(f) for f in loss_function] # 必须要么是2要么是3
        if set(param_counts).difference({2,3}):
            raise ValueError("Each loss function should take 2 or 3 parameters.")

        if isinstance(loss_function, Sequence) and len(loss_function)>1:
            if multitask_strategy is None:
                raise ValueError("When multiple loss functions are provided, you must specify 'multitask_strategy'. "
                                    "Options: Multitask_Strategy.Uncertainty_Weighted_Loss, Multitask_Strategy.GradNorm")
            if multitask_strategy not in [Multitask_Strategy.Uncertainty_Weighted_Loss, Multitask_Strategy.GradNorm]:
                raise ValueError(f"Unknown multitask strategy: {multitask_strategy}")

        if batch_by_tag is True and batch_size is None:
            raise ValueError("batch_by_tag is True, but batch_size is None. Please provide a batch_size.")

        # 检查 Xs 和 Ys 都是 2D tensor
        if not Xs_train.dim()==Xs_test.dim()==Ys_train.dim()==Ys_test.dim()==2:
            raise ValueError("Xs and Ys must be 2D tensors. i.e. it should be like [[feature1, feature2, ...], [...], ...]. " \
            "If there is only one feature, it should be like [[feature1], [feature1], ...], instead of [feature1, feature1, ...].")
        
        # 检查 Xs 的行数和 Ys 的行数相等
        if not (len(Xs_train)==len(Ys_train) and len(Xs_test)==len(Ys_test)):
            raise ValueError("The number of samples in Xs and Ys must be equal.")   
        
        # 检查 Xs_train 和 Xs_test 的列数相等
        if not Xs_train.size(1)==Xs_test.size(1):
            raise ValueError("Xs_train and Xs_test must have the same number of columns.")
        
        # 检查 Ys_train 和 Ys_test 的列数相等
        if not Ys_train.size(1)==Ys_test.size(1):
            raise ValueError("Ys_train and Ys_test must have the same number of columns.")

        # 尚未确认 Ys 有多维特征输出的情况工作是正常的
        if Ys_train.size(1)>1:
            input("Warning: ys_pred has more than one feature dimension. This feature is not verified. Continue?")

        # 如果Tags_train和Tags_test不是None，检查它们的长度和Xs, Ys的长度相等
        if Tags_train is not None and Tags_test is not None:
            if not len(Tags_train)==len(Xs_train) or not len(Tags_test)==len(Xs_test):
                raise ValueError("The number of samples in Tags and Xs/Ys must be equal.")
            if not Tags_train.dim()==Tags_test.dim()==2:
                raise ValueError("Tags_train and Tags_test must be 2D tensors.")
            if Tags_train.size(1) != Tags_test.size(1):
                raise ValueError("Tags_train and Tags_test must have the same number of columns.")
            
        elif Tags_train is None and Tags_test is None:
            if batch_by_tag:
                raise ValueError("batch_by_tag is True, but both Tags_train and Tags_test are None.")

        # Tags_train, Tags_test 要么都是None, 要么都不是None
        else:
            raise ValueError("Tags_train and Tags_test must both be None or both be provided.")
