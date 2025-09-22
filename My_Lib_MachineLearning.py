# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import subprocess

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

print("Torch loaded.")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from Python_Lib.My_Lib import *

# Set matplotlib backend (should be done before importing pyplot)
matplotlib.use("QtAgg")

ML_RANDOM_SEED = 42


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


def print_tensor(input_tensor: torch.tensor, precision=4, scientific_notation_limit=4):
    smart_format_float()
    # TODO


# a = torch.tensor([1.,2.,3.,4.,5.1248120985602985694])
# print(a)


def dataset_normalization(X, Y, *converting_functions):
    """
    parameters:
        Given X,Y, both are n lines of tuple of input/output vectors
        e.g. [(1,2),
              (3,4),
              (5,6),
              ...]
        if one of X,Y is only n lines of float, it is converted to n lines of 1-tuple of input/output vectors
    
        The length of the converting_functions should be the sum of the vector dimensions of the input/output vectors,
        e.g. for an input dimension of 5, and output dimension of 2, the converting_functions should have 7 objects.
        For each column (feature or target):
            1) None → standardization to (mean=0, std=1).  
               If all values are identical, you get 0, and inverse returns by adding the mean back. 
            2) False → no transformation.
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
    if len(converting_functions) != total_dims:
        raise ValueError(
            f"Expected {total_dims} converting functions for X (dim={dX}) + Y (dim={dY}), "
            f"got {len(converting_functions)}."
        )

    # Combine X and Y horizontally for column-wise processing.
    combined = np.hstack([X_arr, Y_arr])

    # We will store the "inverse" transformations here.
    # reversing_functions[i] will be a function that inverts the transform on column i.
    reversing_functions = [None] * total_dims

    # Process each column according to the corresponding converting function.
    for i, conv in enumerate(converting_functions):
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


def train_NN_network(model,
                     optimizer,
                     loss_function,
                     X=None,
                     Y=None,
                     X_train=None,
                     Y_train=None,
                     X_test=None,
                     Y_test=None,
                     batch_size=None,  # Everything as one batch
                     max_epoch=None,
                     early_stopping_strategy=None,
                     # Can choose from: do not stop, or stop when testing data hasen't dropped for n cycles, then load the lowest one
                     load_torch_save_path=None,
                     save_by_n_epoch=None,
                     save_by_percent_of_epoch=0.1,
                     plot_current_linear_fit = True,
                     plot_loss_change_over_epoch = True,
                     plot_one_test_case = None,
                     save_path_stem=""):
    # TODO
    pass
