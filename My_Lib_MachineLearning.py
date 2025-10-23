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

# Set matplotlib backend (should be done before importing pyplot)
matplotlib.use("QtAgg")

ML_RANDOM_SEED = 20211021
DEFAULT_INITIAL_EPOCH = 2000


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

def load_and_preprocess_data(Xs,
                             Ys,
                             test_set_ratio,
                             forward_norm_functions=[]):
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


# DEFAULT_PLOT_SIZE = (5,4.8)
DEFAULT_PLOT_SIZE = (4, 3.8)
DEFAULT_PLOT_FONT_SIZE = 9


# TODO: 处理输入的类型问题
class train_NN_network:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_function,  # with [Tensor(batch_size, Y_predict_dim), Tensor(batch_size, Y_known_dim)], return a total batch loss
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,

                 Xs_train=None,  # should be normalized
                 Ys_train=None,  # should be normalized
                 Xs_test=None,  # should be normalized
                 Ys_test=None,  # should be normalized
                 batch_size=None,  # None for Everything as one batch

                 ########## Optimization helpers ##########
                 device=None,
                 max_gradient_norm=None,

                 ########## Stop conditions ##########
                 initial_max_epoch=None,
                 ask_after_max_epoch=True,
                 early_stopping_strategy=None,
                 # Can choose from: do not stop, or stop when testing data hasen't dropped for n cycles, then load the lowest one

                 ########## Restart options ##########
                 load_from_save_path=None,
                 save_path_stem="",
                 save_every_n_epoch=None,
                 save_by_geo_sequence=1.2,  # save when it's more than 10% more epoch than last save
                 save_pngs=True,
                 save_script=True,

                 ########## Print and Plot options ##########
                 print_every_n_epoch=200,
                 print_by_geo_sequence=1.01,
                 plot_every_n_epoch=200,
                 plot_by_geo_sequence=1.01,
                 plot_current_linear_fit=True,
                 plot_loss_over_epoch=True,
                 function_for_numerical_evaluation=None,
                 # a callable to convert Ys output to a list of scalar, so that it can be plotted against the given output in a linear way
                 plot_cases=None,  # ((X1,Y1_known),(X2,Y2_known)...) # TODO: or just (X,Y)
                 plot_cases_names=None,
                 function_for_plot_test_cases=None,
                 # a callable to convert one X and one Y to two list of scalars, so that it can be plotted against the predicted value
                 callback_function=None
                 ):

        print()
        if device is not None:
            print("Using device:", device)
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")
            if torch.cuda.is_available():
                # Get the number of GPUs available
                # TODO: for multiple GPUs
                num_gpus = torch.cuda.device_count()
                print(f"Using GPU {0}: {torch.cuda.get_device_name(0)}")
                self.device = torch.device('cuda:0')
            else:
                print("No GPUs available.")
        print()

        if not save_path_stem:
            caller_frame = inspect.stack()[1]
            caller_full_path = os.path.abspath(caller_frame.filename)
            caller_filename = filename_name(caller_full_path)
            save_folder = os.path.join(filename_parent(caller_full_path), "Checkpoints")
            save_folder = os.path.join(save_folder, f"{caller_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(save_folder)
            save_path_stem = os.path.join(save_folder, "Save")
        if save_script:
            # Save the caller script in the checkpoint folder
            caller_frame = inspect.stack()[1]
            caller_full_path = os.path.abspath(caller_frame.filename)
            try:
                dst_script_path = os.path.join(save_folder, filename_name(caller_full_path))
                os.makedirs(dst_script_path)
                shutil.copy2(caller_full_path, dst_script_path)
                print(f"Original script {caller_full_path} backed up to {dst_script_path}")
            except Exception as e:
                traceback.print_exc()
                print(e)
                print(f"Warning: could not save script {caller_full_path} -> {save_path_stem}: {e}")

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_function = loss_function
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
        self.plot_cases_names = plot_cases_names or []
        self.function_for_plot_one_test = function_for_plot_test_cases
        self.callback_function = callback_function

        self.linear_fit_window = Plot(fig_size=DEFAULT_PLOT_SIZE,
                                      font_size=10,
                                      x_axis_label="Expected",
                                      y_axis_label="Predicted")

        self.loss_over_epoch_window = Plot(fig_size=DEFAULT_PLOT_SIZE,
                                           font_size=DEFAULT_PLOT_FONT_SIZE,
                                           x_axis_label="Epoch",
                                           y_axis_label="Loss")

        self.plot_cases_windows = [Plot(fig_size=DEFAULT_PLOT_SIZE,
                                        font_size=DEFAULT_PLOT_FONT_SIZE,
                                        x_axis_label="Input",
                                        y_axis_label="Output") for _ in range(len(self.plot_cases))]

        all_windows = [self.linear_fit_window, self.loss_over_epoch_window] + self.plot_cases_windows
        for window in all_windows:
            window.comrades = all_windows

        self.Xs_train, self.Xs_test, self.Ys_train, self.Ys_test = Xs_train, Xs_test, Ys_train, Ys_test

        self.Xs_train = self.Xs_train.to(self.device)
        self.Ys_train = self.Ys_train.to(self.device)
        self.Xs_test = self.Xs_test.to(self.device)
        self.Ys_test: torch.Tensor = self.Ys_test.to(self.device)

        if batch_size:
            dataset_object = TensorDataset(self.Xs_train, self.Ys_train)
            batched_X_train_Y_train = DataLoader(dataset_object, batch_size=batch_size, shuffle=True)
        else:
            batched_X_train_Y_train = [[self.Xs_train, self.Ys_train]]

        self.epoch = 1
        if initial_max_epoch is None:
            initial_max_epoch = DEFAULT_INITIAL_EPOCH
        self.max_epoch = initial_max_epoch

        self.training_losses = []
        self.test_losses = []
        self.last_save_epoch = 0
        self.last_print_epoch = 0
        self.last_plot_epoch = 0

        self.training_start_time = time.time()
        while self.epoch <= self.max_epoch:
            ############ Training loop ############
            self.training_loss = 0.
            for X_train_batch, Y_train_batch in batched_X_train_Y_train:
                X_train_batch = X_train_batch.to(self.device)
                Y_train_batch = Y_train_batch.to(self.device)

                self.optimizer.zero_grad()
                Y_predict_train_batch = self.model(X_train_batch)
                batch_loss = self.loss_function(Y_predict_train_batch, Y_train_batch)
                self.training_loss += batch_loss.item()
                batch_loss.backward()
                if max_gradient_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_gradient_norm)
                self.optimizer.step()
            self.training_loss /= len(batched_X_train_Y_train)
            if self.scheduler is not None:
                self.scheduler.step()
            self.training_losses.append(self.training_loss)

            ############ Evaluation ############
            self.evaluate()

            ############ Termination control ############
            if self.epoch == self.max_epoch and ask_after_max_epoch:
                while True:
                    response = input(f'Max epoch {self.max_epoch} reached. To end the optimization, input "END"; otherwise input a new max_epoch number: ')
                    if response == "END" or (is_int(response) and int(response) > self.max_epoch):
                        break
                if response == "END":
                    break
                else:
                    self.max_epoch = int(response)
            self.epoch += 1

            if self.callback_function is not None:
                self.callback_function(self)

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            Ys_predict_test: torch.Tensor = self.model(self.Xs_test)
            test_loss = self.loss_function(Ys_predict_test, self.Ys_test).item()
            self.test_losses.append(test_loss)
            info_text = f"Epoch {str(self.epoch).rjust(len(str(self.max_epoch)))}/{self.max_epoch} | "
            info_text += f"Loss {smart_format_float(self.training_losses[-1], 4):>8}/{smart_format_float(test_loss, 4):<8}"
            filename_text = f"{self.epoch}_{smart_format_float(self.training_losses[-1], 4)}_{smart_format_float(test_loss, 4)}"

            ############ Printing ############
            if self.print_this_epoch():
                print_elapsed_time = print_time_difference(time.time() - self.training_start_time)
                print_training_speed_epochs = 10 ** (int(math.log10(self.epoch / 10)))
                print_training_speed_epochs = max(print_training_speed_epochs, 1)
                print_training_speed = (time.time() - self.training_start_time) / self.epoch * print_training_speed_epochs
                print_training_speed = f"{print_time_difference(print_training_speed)} for every {print_training_speed_epochs} Epoch"

                print(f"{info_text} | Time: {print_elapsed_time} | {print_training_speed}")

            ############ Plotting ############
            active_windows: List[Plot] = []
            if self.plot_this_epoch():
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
                        if self.save_pngs and self.save_this_epoch():
                            self.linear_fit_window.save_png(self.save_path_stem + f"_Linear_Fit.{filename_text}.png", dpi=300)

                if self.plot_loss_over_epoch:
                    self.loss_over_epoch_window.set_figure_title(info_text)
                    self.loss_over_epoch_window.set_window_title(f"Loss over Epoch @ Epoch {self.epoch}")
                    curve_X = list(range(self.epoch))
                    self.loss_over_epoch_window.Curve_objects = \
                        [Curve(curve_X, self.training_losses, curve_color=blue_color, Y_label="training loss"),
                         Curve(curve_X, self.test_losses, curve_color=red_color, Y_label="test loss")]
                    if self.save_pngs and self.save_this_epoch():
                        self.loss_over_epoch_window.save_png(self.save_path_stem + f"_Losses.{filename_text}.png", dpi=300)
                    active_windows.append(self.loss_over_epoch_window)

                if self.plot_cases:
                    for window_count, (case_data, case_window) in enumerate(zip(self.plot_cases, self.plot_cases_windows)):
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

                        example_loss = self.loss_function(output_predicted.unsqueeze(0), output_known.unsqueeze(0)).item()
                        case_window.set_figure_title(
                            info_text + f" | {self.plot_cases_names[window_count]} Case {window_count + 1} | Loss {smart_format_float(example_loss, 4)}")
                        if self.save_pngs and self.save_this_epoch():
                            case_window.save_png(self.save_path_stem + f"_Exfample.{filename_text}.png", dpi=300)

            for window_count, window in enumerate(active_windows):
                window.shift_window = WINDOW_POSITIONS[len(active_windows)][window_count]

            if self.save_this_epoch():
                save_object = {'epoch': self.epoch,
                               'model_state_dict': self.model.state_dict(),
                               'optimizer_state_dict': self.optimizer.state_dict(),
                               'training_losses': self.training_losses,
                               'test_losses': self.test_losses}
                save_path = self.save_path_stem + f".Epoch_{self.epoch}.pth"
                torch.save(save_object, save_path)

        self.model.train()

        return test_loss

    def save_this_epoch(self):
        if not self.last_save_epoch and self.epoch <= 1:
            self.last_save_epoch = 1
            return True
        if self.epoch == next_output_epoch(self.last_save_epoch,
                                           self.save_every_n_epoch,
                                           self.save_by_geo_sequence):
            self.last_save_epoch = self.epoch
            return True
        return False

    def print_this_epoch(self):
        if not self.last_print_epoch and self.epoch <= 1:
            self.last_print_epoch = 1
            return True
        if self.epoch == next_output_epoch(self.last_print_epoch,
                                           self.print_every_n_epoch,
                                           self.print_by_geo_sequence):
            self.last_print_epoch = self.epoch
            return True
        return False

    def plot_this_epoch(self):
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
