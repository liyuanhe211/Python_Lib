# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import pathlib
import sys
import time

parent_path = str(pathlib.Path(__file__).parent.resolve())
sys.path.insert(0, parent_path)

from My_Lib import *
from My_Lib_PyQt6 import *
from My_Lib_Science import *
import weakref
import csv
import threading
import numpy as np
import scipy.optimize
import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from scipy.interpolate import interp1d, griddata, Rbf, bisplrep, bisplev, RectBivariateSpline
from scipy.ndimage import gaussian_filter
from scipy import interpolate
import matplotlib.font_manager as font_manager
import operator
import colorsys
import matplotlib.colors as mcolors

NON_ASCII_FONT_FILE = os.path.join(filename_parent(__file__), "SourceHanSansSC-Medium.otf")

point_mkr = '.'
pixel_mkr = ','
circle_mkr = 'o'
hollow_circle_mkr = 'o_HOLLOW'
triangle_down_mkr = 'v'
triangle_up_mkr = '^'
triangle_left_mkr = '<'
triangle_right_mkr = '>'
tri_down_mkr = '1'
tri_up_mkr = '2'
tri_left_mkr = '3'
tri_right_mkr = '4'
square_mkr = 's'
hollow_square_mkr = 's_HOLLOW'
pentagon_mkr = 'p'
star_mkr = '*'
hexagon1_mkr = 'h'
hexagon2_mkr = 'H'
plus_mkr = '+'
x_mkr = 'x'
diamond_mkr = 'D'
thin_diamond_mkr = 'd'
vline_mkr = '|'
hline_mkr = '_'

solid_line = '-'
dashed_line = '--'
dash_dot_line = '-.'
dotted_line = ':'

blue_color = "tab:blue"
green_color = 'tab:green'
red_color = 'tab:red'
cyan_color = 'tab:cyan'
pink_color = 'tab:pink'
yellow_color = 'tab:olive'
purple_color = 'tab:purple'
orange_color = 'tab:orange'
brown_color = 'tab:brown'
grey_color = 'tab:grey'
black_color = 'k'
white_color = 'w'

matplotlib_dot_formats = [circle_mkr, square_mkr, diamond_mkr, triangle_down_mkr, triangle_up_mkr, triangle_left_mkr, triangle_right_mkr,
                          tri_down_mkr, tri_up_mkr, tri_left_mkr, tri_right_mkr, hollow_square_mkr, pentagon_mkr, star_mkr, hexagon1_mkr,
                          hexagon2_mkr, plus_mkr, x_mkr, thin_diamond_mkr, vline_mkr, hline_mkr]
matplotlib_colors = [blue_color, red_color, green_color, purple_color, orange_color, cyan_color, yellow_color, pink_color, brown_color, grey_color]
matplotlib_colors = matplotlib_colors * 10000

WINDOW_POSITIONS = {1: ((0, 0),),
                    2: ((-0.5, 0), (0.5, 0)),
                    3: ((-1, 0), (0, 0), (1, 0)),
                    4: ((-1.5, 0), (-0.5, 0), (0.5, 0), (1.5, 0)),
                    5: ((-0.5, -0.5), (0.5, -0.5),
                        (-1, 0.5), (0, 0.5), (1, 0.5)),
                    6: ((-1, -0.5), (0, -0.5), (1, -0.5),
                        (-1, 0.5), (0, 0.5), (1, 0.5)),
                    7: ((-1, -0.5), (0, -0.5), (1, -0.5),
                        (-1.5, 0.5), (-0.5, 0.5), (0.5, 0.5), (1.5, 0.5)),
                    8: ((-1.5, -0.5), (-0.5, -0.5), (0.5, -0.5), (1.5, -0.5),
                        (-1.5, 0.5), (-0.5, 0.5), (0.5, 0.5), (1.5, 0.5))}

# Define the wavelength ranges and their colors and labels
Solar_wavelengths = [(100, 290, 'black', 'Harmful'),
                     (290, 315, '#7603B2', 'UVB'),
                     (315, 335, '#8900D0', 'UVA2'),
                     (335, 380, '#7200E7', 'UVA1'),
                     (700, 1400, None, 'IRA'),
                     (1400, 3000, 'red', 'IRB'),
                     (3000, 1000000, 'darkred', 'IRC')]

# Define the wavelength ranges and their colors and labels
UV_wavelengths = [(100, 280, 'black', 'UVC'),
                  (280, 290, '#420361', '非天然\nUVB'),
                  (290, 295, '#7603B2', '罕见\nUVB'),
                  (295, 315, '#8900D0', '自然UVB'),
                  (315, 335, '#178471', 'UVA2')]

D3_wavelengths = [(100, 290, 'black', '会瞎'),
                  (290, 315, '#8900D0', '促进维生素D合成'),
                  (315, 335, '#178471', '抑制维生素D过量')]


class Global_QApplication:
    """
    Helper to ensure that only one QApplication is created/reused.
    """
    _app = None
    _timer = None

    @staticmethod
    def get_app():
        """
        Returns the global QApplication. If it doesn't exist yet, create it.
        Also starts a small QTimer so that the event loop keeps spinning
        in "interactive" mode, letting windows respond even if we never call exec().
        """
        if Global_QApplication._app is None:
            Global_QApplication._app = QtWidgets.QApplication(sys.argv or [""])
            # A small recurring timer that keeps the event loop alive
            Global_QApplication._timer = QTimer()
            Global_QApplication._timer.timeout.connect(lambda: None)  # dummy callback
            Global_QApplication._timer.start(50)  # run every 50 ms
        return Global_QApplication._app


def convert_torch_tensor(input_):
    if type(input_).__name__ == "Tensor" and type(input_).__module__ == "torch":
        from My_Lib_MachineLearning import tensor_to_list
        return tensor_to_list(input_)
    elif isinstance(input_, (list, tuple)):
        # recursively handle lists/tuples
        return [convert_torch_tensor(x) for x in input_]
    else:
        # not a tensor or list/tuple, return as-is
        return input_


class Curve:
    def __init__(self,
                 X=None,
                 Y=None,
                 Y_errorbar=None,  # error bar lengths for one side of the bar. A n-list for symmetric error bar, or two rows of n-list for asymmetric error bar
                 Y_sampling_data=None,  # a (n,m) dimention list where n is the number of plotted points, m is the sample size of each point.
                 # If this is selected, the average and error bar will be calculated, and the error bar ploted
                 X_label="",
                 Y_label="",
                 X_and_Y=None,  # can be [X_list, Y_list] or [X_list, Y_list, X_Label, Y_Label] Generated by remove_invalid_data
                 XY_pairs=None,  # [(x1,y1),(x2,y2)...]
                 interp1d_object=None,  # input an interp1d_object, the xs and ys are automatically set to the interp1d objects x and y
                 plot_dot=None,
                 dot_format=point_mkr,  # given as string, or tuple of strings with the sequence of marker-line
                 dot_color=None,
                 dot_width=5,
                 plot_curve=None,
                 curve_format="",  # given as string, or tuple of strings with the sequence of marker-line
                 curve_color=None,
                 curve_width=0.8,
                 do_interpolation=True,
                 interpolation_kind="linear",  # 'zero', 'slinear', 'quadratic', 'cubic', 'spline', 'linear', 'nearest', 'nearest-up', 'previous','next'
                 interpolation_smoothing=0,
                 interpolation_number=5000,
                 fitted_function=None,
                 curve_legend_color="",
                 curve_legend_format="",
                 normalize_to: Union[None, Tuple[float, float], float] = None,
                 scale_factor: Optional[float] = None,
                 XYs=None  # Same as X_and_Y, For backward compatibility
                 ):

        """

        :param X:
        :param Y:
        :param X_label:
        :param Y_label:
        :param XYs:
        :param interp1d_object:
        :param plot_dot:
        :param dot_format:
        :param dot_color:
        :param dot_width:
        :param plot_curve:
        :param curve_format:
        :param curve_color:
        :param curve_width:
        :param interpolation_kind:
        :param interpolation_number:
        :param fitted_function: Used for plotting a fitted function, the range is defined for 10% extended on either side of the X data.
                                If this is defined, X,Y are not needed
        """

        # Initialization to avoid mutable initial value
        X = [] if X is None else X
        Y = [] if Y is None else Y
        Y_errorbar = [] if Y_errorbar is None else Y_errorbar
        Y_sampling_data = [] if Y_sampling_data is None else Y_sampling_data
        X_and_Y = [] if X_and_Y is None else X_and_Y
        XY_pairs = [] if XY_pairs is None else XY_pairs
        XYs = [] if XYs is None else XYs

        self.interpolation_kind = interpolation_kind
        self.interpolation_number = interpolation_number
        self.interpolation_smoothing = interpolation_smoothing
        self.moving_average_processed = False

        if XYs:
            X_and_Y = XYs
        if XY_pairs:
            X_and_Y = transpose_2d_list(XY_pairs)

        assert not (interp1d_object is not None and fitted_function is not None)
        assert not (interp1d_object is not None and Y)
        assert not (fitted_function is not None and Y)
        assert not (Y_sampling_data and X_and_Y)
        assert not (Y_errorbar and Y_sampling_data)
        if Y_sampling_data:
            assert X
            assert len(Y_sampling_data) == len(X)

        if normalize_to is not None and scale_factor is not None:
            raise Exception("You can specify normalization and scaling at the same time.")

        if interp1d_object:
            X = interp1d_object.x
            Y = interp1d_object(X)

        if fitted_function is not None:
            assert X, "X needs to be given to display fitted function, for determining the range of Xs"
            X_range = max(X) - min(X)
            X_temp = np.linspace(min(X) - X_range / 10, max(X) + X_range / 10, interpolation_number)
            try:  # 有时方程里带interp1d，超限报错
                Y = fitted_function(X_temp)
                X = X_temp
            except ValueError as _:
                Y = fitted_function(X)

        self.Y_error_bar = Y_errorbar
        if Y_sampling_data:
            Y = np.mean(Y_sampling_data, axis=1)
            self.Y_error_bar = np.apply_along_axis(error_bar, axis=1, arr=Y_sampling_data)

        assert (len(X) and len(Y)) or len(X_and_Y), "No data is given."
        assert not ((len(X) or len(Y)) and len(X_and_Y)), "Duplicated data is given."

        Y = [float(y) if (isinstance(y, str) and is_float(y)) else y for y in Y]

        self.Xs = copy.copy(X)
        self.Ys = copy.copy(Y)
        self.X_label = X_label
        self.Y_label = Y_label

        if X_and_Y:
            assert len(X_and_Y) in [2, 4], "Wrong X_and_Y data format is given"
            if len(X_and_Y) == 4:
                self.Xs, self.Ys, self.X_label, self.Y_label = X_and_Y
            else:
                self.Xs, self.Ys = X_and_Y

        if normalize_to is not None:
            self.normalize_to(normalize_to)
        elif scale_factor is not None:
            self.Ys = [y * scale_factor for y in self.Ys]

        self.Xs = convert_torch_tensor(self.Xs)
        self.Ys = convert_torch_tensor(self.Ys)

        self.Xs = np.array(self.Xs)
        self.Ys = np.array(self.Ys)

        if dot_color and not curve_color:
            curve_color = dot_color
        if curve_color and not dot_color:
            dot_color = curve_color
        if plot_curve is None and plot_dot is None:
            plot_curve = True
        if plot_curve and not curve_format:
            curve_format = '-'
        self.plot_dot = plot_dot
        self.dot_format = dot_format
        self.dot_color = dot_color
        self.dot_width = dot_width
        self.plot_curve = plot_curve
        self.curve_format = curve_format
        self.curve_color = curve_color
        self.curve_width = curve_width
        self.do_interpolation = do_interpolation
        self.curve_legend_color = curve_legend_color
        self.curve_legend_format = curve_legend_format
        self._interpolation_xs = None
        self._interp1d = None

        self.X_and_Y = list(zip(self.Xs, self.Ys))  # for debugging

    @property
    def interpolation_xs(self):
        if self._interpolation_xs is None:
            self.update_interp1d()
        return self._interpolation_xs

    @property
    def interp1d(self):
        if self._interp1d is None:
            self.update_interp1d()
        return self._interp1d

    def normalize_to(self, normalize_to: Union[int, float, Sequence]):
        self.normalization_target_value = normalize_to
        max_Y = max(self.Ys)
        min_Y = min(self.Ys)
        if isinstance(normalize_to, int) or isinstance(normalize_to, float):
            normalize_min = 0
            normalize_max = normalize_to
        else:
            normalize_min, normalize_max = normalize_to

        current_range = max_Y - min_Y
        normalized_range = normalize_max - normalize_min

        self.Ys = [(y + normalize_min - min_Y) * normalized_range / current_range for y in self.Ys]
        self.update_interp1d()

    def update_interp1d(self):
        # print(min(self.Xs), max(self.Xs),self.interpolation_number)
        self._interpolation_xs = np.linspace(min(self.Xs), max(self.Xs), num=self.interpolation_number)
        self._interp1d = interpolation_with_grouping(self.Xs, self.Ys, kind=self.interpolation_kind, smoothing=self.interpolation_smoothing)

    def minus_constant(self, constant):
        self.Xs = [x - constant for x in self.Xs]
        self.update_interp1d()

    def shift_to_non_negative(self):
        min_Y = min(self.Ys)
        if min_Y < 0:
            self.Ys -= min_Y
        self.update_interp1d()

    def normalize(self):
        """
        linearly rescale so that the largest value is 1
        :return:
        """

        max_y = max(self.Ys)
        self.Ys = self.Ys / max_y
        self.update_interp1d()

    def baseline_correction(self, λ=5, ratio=3, show_plot=False):
        """
        Use arPLS method to perform baseline correction
        :param λ:
        :param ratio:
        :param show_plot:
        :return:
        """

        self.before_bkg_correction = self.backup(tag="Original")

        self.non_corrected_Ys = copy.deepcopy(self.Ys)
        import rampy
        baseline_area = np.array([(min(self.Xs)), max(self.Xs)])  # this is not needed for arPLS method, but still required to generate one

        scaling = max(self.Ys)
        self.Ys, self.background = rampy.baseline(self.Xs, self.Ys / scaling, baseline_area, 'arPLS', lam=10 ** λ, ratio=10 ** (-ratio))
        self.Ys = self.Ys * scaling
        self.background = self.background * scaling
        self.Ys = self.Ys.flatten()
        self.background = self.background.flatten()

        self.update_interp1d()

        self.background_object = Curve(X=self.Xs,
                                       Y=self.background,
                                       X_label=self.X_label,
                                       Y_label=self.Y_label + " Bkg",
                                       plot_dot=self.plot_dot,
                                       dot_format=self.dot_format,
                                       dot_color=self.dot_color,
                                       dot_width=self.dot_width,
                                       plot_curve=self.plot_curve,
                                       curve_format=self.curve_format,
                                       curve_color=self.curve_color,
                                       curve_width=self.curve_width,
                                       interpolation_kind=self.interpolation_kind,
                                       interpolation_number=self.interpolation_number)

        if show_plot:
            original_color = self.curve_color
            self.before_bkg_correction.curve_format = dashed_line
            self.before_bkg_correction.curve_color = black_color
            self.background_object.curve_color = red_color
            self.curve_color = blue_color
            Plot([self, self.before_bkg_correction, self.background_object],
                 non_blocking=True)

            self.curve_color = original_color

    def moving_average(self, number_for_average):
        assert not self.moving_average_processed, "You cannot do moving average more than once."
        assert (number_for_average - 1) % 2 == 0
        extend = (number_for_average - 1) / 2
        assert is_int(extend)
        extend = int(extend)
        current_Y = [self.Ys[0]] * extend + self.Ys + [self.Ys[-1]] * extend
        new_Y = []
        for i in range(extend, len(self.Ys) + extend):
            to_average = current_Y[i - extend:i + extend + 1]
            assert len(to_average) == number_for_average
            new_Y.append(average(to_average))
        self.Ys = new_Y
        self.update_interp1d()
        self.moving_average_processed = True

    def export_xlsx(self, path="", filename=""):
        filename = os.path.join(path, filename)
        if path and not os.path.isdir(path):
            os.makedirs(path)
        if not filename:
            path = self.Y_label + '.xlsx'
            path = os.path.realpath(path)
            path = get_unused_filename(path, use_proper_filename=False)

        export_list = [[self.X_label] + list(self.Xs), [self.Y_label] + list(self.Ys)]
        write_xlsx(path, export_list, transpose=True)

    def backup(self, tag=""):
        """
            :return: an object which has the same content of this object, but are independent to internal changes.
        """

        raise Exception("The backup function is outdated and needs to be updated with the new parameters.")
        # TODO: This need to be updated

        if tag:
            tag = " " + tag

        return Curve(X=copy.deepcopy(self.Xs),
                     Y=copy.deepcopy(self.Ys),
                     X_label=self.X_label,
                     Y_label=self.Y_label + " " + tag,
                     plot_dot=self.plot_dot,
                     dot_format=self.dot_format,
                     dot_color=self.dot_color,
                     dot_width=self.dot_width,
                     plot_curve=self.plot_curve,
                     curve_format=self.curve_format,
                     curve_color=self.curve_color,
                     curve_width=self.curve_width,
                     interpolation_kind=self.interpolation_kind,
                     interpolation_number=self.interpolation_number)

    def manipulate_with(self, other, computation_function):
        if isinstance(other, Curve):
            interp1 = self.interp1d
            interp2 = other.interp1d
            ret_interp = interp_manipulation(interp1, interp2, self.Xs, other.Xs, computation_function)
            ret = Curve(interp1d_object=ret_interp)
        elif is_float(other):
            ret = self.backup()
            ret.Ys = [computation_function(y, other) for y in ret.Ys]
        else:
            raise Exception("Manipulation object type error.")

        return ret

    def __add__(self, other):
        return self.manipulate_with(other, operator.add)

    def __sub__(self, other):
        return self.manipulate_with(other, operator.sub)

    def __mul__(self, other):
        return self.manipulate_with(other, operator.mul)

    def __truediv__(self, other):
        return self.manipulate_with(other, operator.truediv)

    def __pow__(self, other):
        return self.manipulate_with(other, operator.pow)


class Grid:
    def __init__(self,
                 XYZ_triples=None,
                 Xs=None,
                 Ys=None,
                 do_interpolation=True,
                 interpolation_type="linear",  # linear, cubic, nearest, multiquadric, inverse, gaussian, thin_plate
                 must_pass_through_points=True,
                 smoothing=0,
                 interpolation_density=200,

                 Z_axis_colors=None,  # list of (z, color)
                 Z_linear_or_log="linear",  # linear or log

                 show_contour=False,
                 contour_levels=None,
                 contour_values=None,
                 show_contour_labels=False, # True, False or list of strings
                 contour_style=dashed_line,
                 contour_width=0.5,
                 contour_color='k',

                 contour_do_interpolation=None,
                 contour_interpolation_type: Optional[Literal['gaussian_filter', 
                                                              'b_spline', 
                                                              'spline', 
                                                              'multiquadric', 
                                                              'inverse', 
                                                              'gaussian', 
                                                              'linear', 
                                                              'cubic', 
                                                              'quintic', 
                                                              'thin_plate']]=None,
                 contour_must_pass_through_points=None,
                 contour_smoothing=None,

                 normalize_to=None,
                 scale_factor=None,
                 
                 grid_line_X=None, # True, False, list of strings or float (spacing)
                 grid_line_Y=None, # True, False, list of strings or float (spacing)
                 show_colorbar=False
                 ):
        """
        Initialize a Grid object for plotting 3D data (heatmap/contour).

        :param XYZ_triples: List of tuples (x, y, z) representing the data points.
                            If provided, Xs, Ys, Zs will be extracted from this.
        :param Xs: Optional. 1D array of X coordinates for the grid or input data X points if XYZ_triples is not used (not fully supported independently).
                   If used with Ys to define a grid, they define the mesh.
        :param Ys: Optional. 1D array of Y coordinates.
        :param do_interpolation: Boolean. If True, interpolation is performed to generate a regular grid from scattered data.
                                 If False, nearest neighbor interpolation is used to fill the grid (pixelated view).
        :param interpolation_type: String. algorithm for interpolation.
                                   Options for griddata (must_pass_through_points=True): 'linear', 'cubic', 'nearest'.
                                   Options for Rbf (must_pass_through_points=False): 'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'.
                                   Options for smoothing: 'gaussian_filter', 'b_spline'.
        :param must_pass_through_points: Boolean. If True, uses scipy.interpolate.griddata (exact interpolation).
                                         If False, uses scipy.interpolate.Rbf (Radial Basis Function, can smooth data).
        :param smoothing: Float. Smoothing factor for Rbf or gaussian_filter interpolation.
        :param interpolation_density: Int. Number of points along each axis for the generated grid (density * density points total).
        :param color_map_data: List of (value, color) tuples to define a custom colormap.
                               The colormap will interpolate between these colors based on the Z values.
        :param color_map_scale: String. 'linear' or 'log'. Scale for the colormap.
        :param show_contour: Boolean. Whether to draw contour lines on top of the heatmap.
        :param contour_levels: Int or None. Number of contour levels to draw automatically.
        :param contour_values: List of floats or None. Specific Z values at which to draw contour lines.
        :param show_contour_values: Boolean or List. If True, shows numerical values on the contour lines.
                                    If a list, must be same length as contour_values or contour_levels, specifying custom labels.
        :param contour_style: String. Line style for contour lines (e.g., '-', '--').
        :param contour_width: Float. Width of contour lines.
        :param contour_color: Color for contour lines.
        :param contour_do_interpolation: Boolean. If None, uses do_interpolation.
        :param contour_interpolation_type: String. If None, uses interpolation_type.

        'gaussian_filter'
        'b_spline'
        'spline'
        'multiquadric'
        'inverse'
        'gaussian'
        'linear'
        'cubic'
        'quintic'
        'thin_plate'
        'multiquadric'

        :param contour_must_pass_through_points: Boolean. If None, uses must_pass_through_points.
        :param contour_smoothing: Float. If None, uses smoothing.
        :param normalize_to: Float or Tuple(min, max). Normalize Z values to this range (0, val) or (min, max).
        :param scale_factor: Float. Multiply all Z values by this factor.
        """

        self.XYZs = XYZ_triples if XYZ_triples else []
        self.Xs_input = Xs
        self.Ys_input = Ys
        self.do_interpolation = do_interpolation
        self.interpolation_type = interpolation_type
        self.must_pass_through_points = must_pass_through_points
        self.smoothing = smoothing
        self.grid_density = interpolation_density
        self.Z_axis_colors = Z_axis_colors
        self.Z_linear_or_log = Z_linear_or_log
        self.show_contour = show_contour
        self.contour_levels = contour_levels
        self.contour_values = contour_values
        self.show_contour_values = show_contour_labels
        self.contour_style = contour_style
        self.contour_width = contour_width
        self.contour_color = contour_color
        
        self.contour_do_interpolation = contour_do_interpolation if contour_do_interpolation is not None else do_interpolation
        self.contour_interpolation_type = contour_interpolation_type if contour_interpolation_type is not None else interpolation_type
        self.contour_must_pass_through_points = contour_must_pass_through_points if contour_must_pass_through_points is not None else must_pass_through_points
        self.contour_smoothing = contour_smoothing if contour_smoothing is not None else smoothing
        
        self.normalize_to = normalize_to
        self.scale_factor = scale_factor
        
        self.grid_line_X = grid_line_X
        self.grid_line_Y = grid_line_Y
        self.show_colorbar = show_colorbar

        self.Xs = None
        self.Ys = None
        self.Zs = None

        if self.XYZs:
            # Assuming list of tuples/lists
            raw_data = np.array([list(pt) for pt in self.XYZs])
            if raw_data.ndim == 2 and raw_data.shape[1] == 3:
                self.Xs = raw_data[:, 0]
                self.Ys = raw_data[:, 1]
                self.Zs = raw_data[:, 2]

                if self.normalize_to is not None:
                    if isinstance(self.normalize_to, (int, float)):
                        norm_min, norm_max = 0, self.normalize_to
                    else:
                        norm_min, norm_max = self.normalize_to

                    z_min, z_max = np.min(self.Zs), np.max(self.Zs)
                    if z_max != z_min:
                        self.Zs = (self.Zs - z_min) / (z_max - z_min) * (norm_max - norm_min) + norm_min

                if self.scale_factor is not None:
                    self.Zs *= self.scale_factor

    def _calculate_grid_z(self, grid_x, grid_y, do_interp, interp_type, must_pass, smoothing_val):
        grid_z = None
        if do_interp:
            if interp_type == 'gaussian_filter':
                try:
                    grid_z_base = griddata((self.Xs, self.Ys), self.Zs, (grid_x, grid_y), method='linear')
                    
                    if np.isnan(grid_z_base).any():
                         grid_z_nearest = griddata((self.Xs, self.Ys), self.Zs, (grid_x, grid_y), method='nearest')
                         grid_z_base[np.isnan(grid_z_base)] = grid_z_nearest[np.isnan(grid_z_base)]

                    grid_z = gaussian_filter(grid_z_base, sigma=smoothing_val)
                except Exception as e:
                    print(f"gaussian_filter failed: {e}")
            
            elif interp_type in ['b_spline', 'spline']:
                 try:
                      grid_z_base = griddata((self.Xs, self.Ys), self.Zs, (grid_x, grid_y), method='linear')

                      if np.isnan(grid_z_base).any():
                           grid_z_nearest = griddata((self.Xs, self.Ys), self.Zs, (grid_x, grid_y), method='nearest')
                           grid_z_base[np.isnan(grid_z_base)] = grid_z_nearest[np.isnan(grid_z_base)]
                      
                      # RectBivariateSpline expects s as sum of squared errors. 
                      # For a dense grid (e.g. 40000 points), s needs to be very large to have an effect.
                      # We scale smoothing_val by the number of points so that smoothing_val acts like "Mean Squared Error Allowed".
                      total_points = grid_z_base.size
                      sv = smoothing_val if smoothing_val is not None else 0
                      s_val = (sv * total_points) if sv > 0 else 0
                      
                      x_1d = grid_x[0, :]
                      y_1d = grid_y[:, 0]
                      
                      # RectBivariateSpline expects Z shape (nx, ny) where x is first dim
                      # grid_z_base is (ny, nx) from meshgrid
                      rb_spline = RectBivariateSpline(x_1d, y_1d, grid_z_base.T, s=s_val)
                      
                      # Evaluate
                      grid_z = rb_spline(x_1d, y_1d, grid=True).T

                 except Exception as e:
                      print(f"b_spline failed: {e}")
            
            elif must_pass:
                # griddata
                try:
                    grid_z = griddata((self.Xs, self.Ys), self.Zs, (grid_x, grid_y), method=interp_type)
                except Exception as e:
                    print(f"griddata failed: {e}")
            else:
                # Rbf
                func_type = interp_type if interp_type in ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'] else 'multiquadric'
                try:
                    rbf = Rbf(self.Xs, self.Ys, self.Zs, function=func_type, smooth=smoothing_val)
                    grid_z = rbf(grid_x, grid_y)
                except Exception as e:
                    print(f"Rbf failed: {e}, falling back to linear griddata")
                    grid_z = griddata((self.Xs, self.Ys), self.Zs, (grid_x, grid_y), method='linear')
        else:
            # If not interpolation, use nearest neighbor to fill the pixels, creating a blocky effect
            try:
                grid_z = griddata((self.Xs, self.Ys), self.Zs, (grid_x, grid_y), method='nearest')
            except Exception as e:
                print(f"griddata (nearest) failed: {e}")
        return grid_z

    def get_grid_data(self):
        if self.Xs is None:
            return None, None, None, None

        if self.Xs_input is not None and self.Ys_input is not None:
             grid_x_1d = np.array(self.Xs_input)
             grid_y_1d = np.array(self.Ys_input)
        else:
             x_min, x_max = np.min(self.Xs), np.max(self.Xs)
             y_min, y_max = np.min(self.Ys), np.max(self.Ys)

             grid_x_1d = np.linspace(x_min, x_max, self.grid_density)
             grid_y_1d = np.linspace(y_min, y_max, self.grid_density)
        
        grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)

        grid_z_mesh = self._calculate_grid_z(grid_x, grid_y, self.do_interpolation, self.interpolation_type, self.must_pass_through_points, self.smoothing)
        
        if self.show_contour:
             grid_z_contour = self._calculate_grid_z(grid_x, grid_y, self.contour_do_interpolation, self.contour_interpolation_type, self.contour_must_pass_through_points, self.contour_smoothing)
        else:
             grid_z_contour = None
             
        return grid_x, grid_y, grid_z_mesh, grid_z_contour

    def get_colormap(self):
        if not self.Z_axis_colors:
            return None

        z_color_pairs = list(self.Z_axis_colors)
        z_color_pairs.sort(key=lambda x: x[0])

        zs = [float(x[0]) for x in z_color_pairs]
        colors_in = [x[1] for x in z_color_pairs]
        
        vmin = min(zs)
        vmax = max(zs)

        if self.Z_linear_or_log == 'log':
            if any(z <= 0 for z in zs):
                print("Warning: Log scale requested for colormap but z values non-positive. using linear.")
            else:
                zs = np.log10(zs)

        min_z, max_z = min(zs), max(zs)
        if max_z == min_z:
            norm_zs = [0.0] * len(zs)
        else:
            norm_zs = [(z - min_z) / (max_z - min_z) for z in zs]

        def to_rgb_arr(c):
            if isinstance(c, str):
                return mcolors.to_rgb(c)
            else:
                return c

        rgbs_in = [to_rgb_arr(c) for c in colors_in]

        N = 256
        res_colors = []
        vals = np.linspace(0, 1, N)

        for v in vals:
            idx = 0
            for i in range(len(norm_zs) - 1):
                if norm_zs[i] <= v <= norm_zs[i + 1]:
                    idx = i
                    break
            else:
                idx = len(norm_zs) - 2

            if idx < 0: idx = 0

            lower, upper = norm_zs[idx], norm_zs[idx + 1]
            if upper == lower:
                t = 0
            else:
                t = (v - lower) / (upper - lower)

            r1, g1, b1 = rgbs_in[idx]
            r2, g2, b2 = rgbs_in[idx + 1]

            r = r1 + (r2 - r1) * t
            g = g1 + (g2 - g1) * t
            b = b1 + (b2 - b1) * t

            res_colors.append((r, g, b))

        return mcolors.ListedColormap(res_colors, name='custom_hsv'), vmin, vmax


class Plot(QtWidgets.QWidget, Qt_Widget_Common_Functions):
    def __init__(
            self,

            # You can leave both to be None to create an empty plot window, and update them later with update_plot
            Curve_objects: Union[Curve, Sequence[Curve], None] = None,
            Grid_objects: Union[Grid, Sequence[Grid], None] = None,
            
            x_axis_label="X",
            y_axis_label="Y",
            fig_size=(4, 3),
            font_size=8,
            plot_legend=True,
            legend_font_size=None,
            x_lim=None,
            # None: Choose automatically, (None, 400): Choose left automatically, choose right as 400, (100,400): choose 100-400
            y_lim=None,
            x_log=False,
            y_log=False,
            show_grid=False,
            x_tick_positions=None,
            x_tick_texts=None,
            y_tick_positions=None,
            y_tick_texts=None,
            y2_tick_positions=None,
            y2_tick_texts=None,
            y2_axis_label="",
            non_blocking=False,
            auto_color=None,
            # if True, the color of the lines will be overrided with the default matplotlib color list.
            # If auto_color = None and there is no color specified in any of the curves, auto_color = True
            save_img_filepath=None,
            # if is a string, the plot will be saved to the filepath as png file, without showing
            save_img_dpi=3000,
            use_chinese_font=False,
            parent=None,
            shift_window=(0, 0),
            # shift the window position like in a grid layout for x grid to the right and y grid down. The grid size is the current window size
            multiple_plot_arrangement=None,  # e.g. (3,8) means that the window should be positioned as if it is the 3rd window in a 8 windows grid.
            figure_title="",
            window_title="Plot Points Window"
    ):
        # Make sure a QApplication exists
        self._app = Global_QApplication.get_app()

        super(self.__class__, self).__init__()

        self.current_title = figure_title
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.fig_size = fig_size
        self.font_size = font_size
        self.plot_legend = plot_legend
        self.legend_font_size = legend_font_size
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.x_log = x_log
        self.y_log = y_log
        self.show_grid = show_grid
        self.x_tick_positions = x_tick_positions or []
        self.x_tick_texts = x_tick_texts or []
        self.y_tick_positions = y_tick_positions or []
        self.y_tick_texts = y_tick_texts or []
        self.y2_tick_positions = y2_tick_positions or []
        self.y2_tick_texts = y2_tick_texts or []
        self.y2_tick_texts = [str(x) for x in self.y2_tick_texts]
        self.y2_axis_label = y2_axis_label
        self.auto_color = auto_color
        self.chinese_font = use_chinese_font
        self.save_img_filepath = save_img_filepath
        self.save_img_dpi = save_img_dpi
        self.current_location = (0, 0)
        self._additional_offset_px = (0, 0)
        self.comrades: Sequence[QWidget] = []
        if isinstance(shift_window, (int, float)):
            shift_window = (shift_window, 0)
        if multiple_plot_arrangement is not None:
            current_window_number, total_windows = multiple_plot_arrangement
            shift_window = WINDOW_POSITIONS[total_windows][current_window_number]
        self._shift_window = shift_window

        matplotlib.rcParams['savefig.dpi'] = self.save_img_dpi
        plt.ion()

        self.setStyleSheet("background-color: white;")

        layout = QtWidgets.QVBoxLayout(self)
        
        # Calculate figure size to ensure the axes area matches self.fig_size
        # Using user provided relative ratios: top 1, bottom 0.076, left 0.130, right 0.964
        user_left = 0.130
        user_right = 0.964
        user_bottom = 0.076
        user_top = 0.99

        ax_width, ax_height = self.fig_size
        
        # Determine total figure dimensions such that the axis fraction equals the requested size
        width_fraction = user_right - user_left
        height_fraction = user_top - user_bottom
        
        if width_fraction <= 0 or height_fraction <= 0:
             # Fallback if ratios are invalid
             fig_width = ax_width + 1.0
             fig_height = ax_height + 1.0
             self._layout_params = {'left': 0.15, 'right': 0.95, 'top': 0.95, 'bottom': 0.15}
        else:
            fig_width = ax_width / width_fraction
            fig_height = ax_height / height_fraction
            
            self._layout_params = {
                'left': user_left,
                'bottom': user_bottom,
                'right': user_right,
                'top': user_top
            }

        self._fig = Figure(figsize=(fig_width, fig_height), dpi=100, constrained_layout=False)
        # self._fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.08, hspace=0.08, wspace=0.08)
        self._canvas = FigureCanvas(self._fig)
        self._ax = self._fig.add_subplot(111)

        # Navigation toolbar setup
        self._toolbar = NavigationToolbar(self._canvas, self)
        self._toolbar.setStyleSheet("background-color: #f0f0f0;")
        self._toolbar.setIconSize(QtCore.QSize(16, 16)) # Small icons
        
        # ALLOW TOOLBAR TO SHRINK: Ignore horizontal size hint and set min width to 0
        self._toolbar.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self._toolbar.setMinimumWidth(0)

        # Create the “Pause” button
        self.pause_button = QtWidgets.QPushButton("Pause", self)
        self.pause_button.setStyleSheet("background-color: #f0f0f0;")
        self.pause_button.setCheckable(True)
        pause_font = QtGui.QFont("Arial", 10)
        self.pause_button.setFont(pause_font)
        # Match toolbar button height
        toolbar_height = self._toolbar.sizeHint().height()
        self.pause_button.setFixedSize(60, toolbar_height)

        # Create the “Bring to front” button
        self.raise_button = QtWidgets.QPushButton("Raise All", self)
        self.raise_button.setStyleSheet("background-color: #f0f0f0;")
        raise_font = QtGui.QFont("Arial", 10)
        self.raise_button.setFont(raise_font)
        self.raise_button.setFixedSize(60, toolbar_height)
        connect_once(self.raise_button, self.bring_to_front)

        # Container layout for toolbar + custom buttons
        toolbar_container_layout = QtWidgets.QHBoxLayout()
        toolbar_container_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_container_layout.setSpacing(0)
        
        toolbar_container_layout.addWidget(self._toolbar) # Toolbar is ignored policy, so it shrinks
        toolbar_container_layout.addWidget(self.raise_button) # Fixed size
        toolbar_container_layout.addWidget(self.pause_button) # Fixed size

        # Add horizontal layout for the resizable label with left spacer
        label_layout = QtWidgets.QHBoxLayout()
        label_layout.addSpacing(50)  # horizontal left spacer
        self.resizable_label = ResizableLabel(figure_title, self, font_size + 1)

        # self.resizable_label.setStyleSheet("background-color: #f00000;")
        label_layout.addWidget(self.resizable_label)
        label_layout.addSpacing(10)  # horizontal left spacer
        # label_layout.addStretch()  # optional: push label to the left if needed

        # Now add the toolbar container as the first row
        layout.addLayout(toolbar_container_layout)
        layout.addSpacing(10)
        layout.addLayout(label_layout)
        layout.addWidget(self._canvas)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # self._ax.setCursor(QCrossCursor) # 设置到toolbar去了

        # Optionally enable the Chinese font, or other special fonts
        if self.chinese_font:
            prop = font_manager.FontProperties(fname=NON_ASCII_FONT_FILE, weight='bold')
            plt.rcParams['font.family'] = prop.get_name()

        self._plot_objects = []
        
        if Grid_objects is not None:
            if isinstance(Grid_objects, (Grid, Curve)):
                Grid_objects = [Grid_objects]
            self.plot_objects = Grid_objects
        else:
            if isinstance(Curve_objects, (Grid, Curve)):
                Curve_objects = [Curve_objects]
            self.plot_objects = Curve_objects if Curve_objects else []

        self.setWindowTitle(window_title)
        if self.plot_objects:
            self.show()
            self.center_the_widget()
            self.move_window()

    #     # 关键：检测是否无人引用
    #     self._auto_pause_check()
    #
    # def _auto_pause_check(self):
    #     """
    #     检查Plot对象是否被外部引用，如果没有，就自动pause。
    #     同时保持一个强引用，防止在pause期间被销毁。
    #     """
    #     import weakref, sys, threading
    #
    #     ref = weakref.ref(self)
    #
    #     def check_ref():
    #         obj = ref()
    #         if obj is not None:
    #             ref_count = sys.getrefcount(obj)
    #             # print("refcount", ref_count)
    #             if ref_count <= 3:  # 没有外部变量引用
    #                 # 临时保存强引用，防止垃圾回收
    #                 Plot._temporary_ref = obj
    #                 # 启动pause
    #                 obj.pause()
    #                 # pause退出后清除引用
    #                 Plot._temporary_ref = None
    #
    #     # 延迟一点检查，等__init__栈释放
    #     threading.Timer(0.05, check_ref).start()

    def bring_to_front(self):
        self.activateWindow()
        for window in self.comrades:
            window.activateWindow()

    @property
    def shift_window(self):
        return self._shift_window

    @shift_window.setter
    def shift_window(self, new_tuple):
        self._shift_window = new_tuple
        self.move_window()

    @property
    def additional_offset_px(self):
        return self._additional_offset_px

    @additional_offset_px.setter
    def additional_offset_px(self, offset):
        self._additional_offset_px = offset
        self.move_window()

    def move_window(self):
        app = Global_QApplication.get_app()
        screens = app.screens()
        if len(screens) > 1:
            target_screen = screens[-1]
        else:
            target_screen = app.primaryScreen()

        screen_geometry = target_screen.availableGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        screen_x = screen_geometry.x()
        screen_y = screen_geometry.y()

        # Calculate top-left point if centered
        x_centered = screen_x + (screen_width - self.width()) // 2
        y_centered = screen_y + (screen_height - self.height()) // 2

        new_x_position = x_centered + self.shift_window[0] * (self.width() + 3) + self.additional_offset_px[0]
        new_y_position = y_centered + self.shift_window[1] * (self.height() + 30) + self.additional_offset_px[1]
        if (new_x_position, new_y_position) != self.current_location:
            self.move(round(new_x_position), round(new_y_position))
            self.current_location = (new_x_position, new_y_position)

    @property
    def plot_objects(self):
        return self._plot_objects

    @plot_objects.setter
    def plot_objects(self, new_objects):
        if isinstance(new_objects, (Curve, Grid)):
            self._plot_objects = [new_objects]
        else:
            self._plot_objects = list(new_objects)
        self._update_plot()
        self.move_window()

    def set_figure_title(self, title):
        self.current_title = str(title)
        self.resizable_label.setText(self.current_title)
        # if not title.strip():
        #     self.resizable_label.hide()
        # else:
        #     self.resizable_label.show()

    def set_window_title(self, title):
        self.setWindowTitle(title)

    def update_data(self, new_Curves_or_Grids, pause=0):
        self.plot_objects = new_Curves_or_Grids
        self.pause(pause)

    def pause(self, seconds=0):
        """
        Allows the plot to remain on screen and remain responsive for 'seconds' duration,
        then returns control to the calling script.  
        """
        # Ensure the latest drawn state is visible
        self._canvas.draw()
        QtWidgets.QApplication.processEvents()

        if seconds:
            end_time = time.time() + seconds
            while time.time() < end_time:
                QtWidgets.QApplication.processEvents()
                time.sleep(0.001)
        else:
            app = Global_QApplication.get_app()
            app.exec()

    def _update_plot(self):
        """
        Clear the canvas, re-plot everything, and update the display.
        If the “Pause” button is checked, skip redrawing new data so
        user can pan/zoom the current plot without updates.
        """
        # If pause button is checked, do not update the data portion

        if not self._plot_objects:
            self.hide()
            return
        else:
            if not self.isVisible():
                self.show()

        while self.pause_button.isChecked():
            self.pause(0.05)

        plt.rcParams.update({'font.size': self.font_size, 'font.family': 'arial', 'mathtext.default': 'regular'})

        self._ax.clear()

        # Check mode based on the first object
        first_obj = self._plot_objects[0]
        if isinstance(first_obj, Grid):
            self._update_plot_grid()
        else:
            self._update_plot_curve()

    def _update_plot_grid(self):
        for curve in self._plot_objects:
            if not isinstance(curve, Grid): continue

            grid_x, grid_y, grid_z_mesh, grid_z_contour = curve.get_grid_data()
            mesh_plot = None 
            if grid_z_mesh is not None:
                cmap_info = curve.get_colormap()
                
                if cmap_info is not None:
                     cmap, vmin, vmax = cmap_info
                     mesh_plot = self._ax.pcolormesh(grid_x, grid_y, grid_z_mesh, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto', zorder=1)
                else:
                     mesh_plot = self._ax.pcolormesh(grid_x, grid_y, grid_z_mesh, shading='auto', zorder=1)
                
                if curve.show_colorbar and mesh_plot:
                    self._fig.colorbar(mesh_plot, ax=self._ax)

            if curve.show_contour and grid_z_contour is not None:
                levels = curve.contour_values if curve.contour_values is not None else curve.contour_levels
                contour_set = self._ax.contour(grid_x, grid_y, grid_z_contour, levels=levels,
                                 colors=curve.contour_color, linewidths=curve.contour_width,
                                 linestyles=curve.contour_style, zorder=3) # Higher zorder than grid
                
                if curve.show_contour_values:
                    if isinstance(curve.show_contour_values, list):
                        # Ensure we have specific levels to match against used_levels might vary if levels was int
                        used_levels = contour_set.levels
                        if len(curve.show_contour_values) != len(used_levels):
                             print(f"Warning: show_contour_values length ({len(curve.show_contour_values)}) does not match contour levels ({len(used_levels)}).")
                        
                        fmt = {}
                        # Create a dictionary mapping the actual level value to the string label
                        for l, s in zip(used_levels, curve.show_contour_values):
                             fmt[l] = s
                        self._ax.clabel(contour_set, inline=True, fontsize=10, fmt=fmt)
                    else:
                        self._ax.clabel(contour_set, inline=True, fontsize=10)
                        
            # Grid lines logic
            def process_grid_line(option, min_val, max_val):
                lines = []
                if option is True:
                     return True, None
                elif isinstance(option, (int, float)) and not isinstance(option, bool):
                     # Spacing
                     start = math.ceil(min_val / option) * option
                     lines = np.arange(start, max_val + option * 0.001, option) 
                elif isinstance(option, (list, tuple, np.ndarray)):
                     lines = option
                
                return False, lines

            if grid_x is not None and grid_y is not None:
                x_min, x_max = np.min(grid_x), np.max(grid_x)
                y_min, y_max = np.min(grid_y), np.max(grid_y)
                
                use_default_x_grid, x_lines = process_grid_line(curve.grid_line_X, x_min, x_max)
                use_default_y_grid, y_lines = process_grid_line(curve.grid_line_Y, y_min, y_max)
                
                if use_default_x_grid:
                     self._ax.xaxis.grid(True, linestyle=curve.contour_style, linewidth=0.5, color='gray', zorder=2) # zorder between mesh and contour
                elif x_lines is not None and len(x_lines) > 0:
                     self._ax.set_xticks(x_lines)
                     self._ax.xaxis.grid(True, linestyle=curve.contour_style, linewidth=0.5, color='gray', zorder=2)
                
                if use_default_y_grid:
                     self._ax.yaxis.grid(True, linestyle=curve.contour_style, linewidth=0.5, color='gray', zorder=2)
                elif y_lines is not None and len(y_lines) > 0:
                     self._ax.set_yticks(y_lines)
                     self._ax.yaxis.grid(True, linestyle=curve.contour_style, linewidth=0.5, color='gray', zorder=2)

        self._finalize_plot_settings()

    def _update_plot_curve(self):
        # Automatic color logic
        if self.auto_color is None:
            # If no color is specified in any curve, auto_color = True
            any_colors = any((obj.curve_color or obj.curve_legend_color or obj.dot_color)
                             for obj in self._plot_objects)
            auto_color_active = not any_colors
        else:
            auto_color_active = self.auto_color

        if auto_color_active:
            for i, obj in enumerate(self._plot_objects):
                if isinstance(obj, Curve):
                    color = matplotlib_colors[i]
                    obj.curve_color = color
                    obj.curve_legend_color = color
                    obj.dot_color = color

        legend_handles = []

        # Plot each Curve
        for curve in self._plot_objects:
            if not isinstance(curve, Curve): continue # Should not happen if filtered, but good for safety

            X = curve.Xs
            Y = curve.Ys

            # Plot lines
            if curve.plot_curve:
                if curve.do_interpolation:
                    if self.x_log:
                        assert min(X) > 0, "You are asking to log a negative number."
                        min_log_x_position = math.log10(min(X) * 1.00001)
                        max_log_x_position = math.log10(max(X) * 0.99999)
                        fitted_X = 10 ** (np.linspace(min_log_x_position, max_log_x_position, num=curve.interpolation_number))
                        fitted_Y = interpolation_with_grouping(X, Y, kind=curve.interpolation_kind, smoothing=curve.interpolation_smoothing)(fitted_X)
                    else:
                        # If user stored interpolation_xs:
                        if curve.interpolation_xs is not None:
                            fitted_X = curve.interpolation_xs
                        else:
                            # fallback
                            fitted_X = np.linspace(min(X), max(X), curve.interpolation_number)
                        fitted_Y = curve.interp1d(fitted_X)

                    self._ax.plot(fitted_X,
                                  fitted_Y,
                                  curve.curve_format,
                                  color=curve.curve_color,
                                  linewidth=curve.curve_width)
                else:
                    # no interpolation
                    self._ax.plot(X,
                                  Y,
                                  curve.curve_format,
                                  color=curve.curve_color,
                                  linewidth=curve.curve_width)

            # Plot dots
            if curve.plot_dot:
                # Check if it's "HOLLOW"
                if "_HOLLOW" in curve.dot_format:
                    marker_face_color = 'none'
                else:
                    marker_face_color = curve.dot_color
                marker_edge_color = curve.dot_color
                self._ax.plot(
                    X, Y,
                    curve.dot_format.replace('_HOLLOW', ""),
                    markerfacecolor=marker_face_color,
                    markeredgecolor=marker_edge_color,
                    markersize=curve.dot_width
                )

            # Optional legend entry
            if curve.Y_label:
                if "_HOLLOW" in curve.dot_format:
                    marker_face_color = 'none'
                else:
                    marker_face_color = curve.dot_color
                marker_edge_color = curve.dot_color

                # Create an "empty" plot just for the legend handle
                legend_line = self._ax.plot(
                    [],
                    [],
                    linestyle=curve.curve_legend_format if curve.curve_legend_format else curve.curve_format,
                    marker=curve.dot_format.replace('_HOLLOW', ""),
                    markersize=curve.dot_width,
                    label=curve.Y_label,
                    markerfacecolor=marker_face_color,
                    markeredgecolor=marker_edge_color,
                    color=curve.curve_legend_color if curve.curve_legend_color else curve.curve_color,
                    linewidth=curve.curve_width
                )
                legend_handles.append(legend_line)

            # Error bars
            if curve.Y_error_bar:
                self._ax.errorbar(
                    X, Y, curve.Y_error_bar,
                    fmt='o',
                    ecolor=curve.dot_color,
                    markersize=0,
                    elinewidth=curve.curve_width,
                    capthick=curve.curve_width,
                    capsize=2
                )
        
        # Legend
        if self.plot_legend and legend_handles:
            if not self.legend_font_size:
                self.legend_font_size = self.font_size
            self._ax.legend(fontsize=self.legend_font_size)

        self._finalize_plot_settings()

    def _finalize_plot_settings(self):
        # Apply log scales
        if self.x_log:
            self._ax.set_xscale('log')
        if self.y_log:
            self._ax.set_yscale('log')

        # Axis labels
        self._ax.set_xlabel(self.x_axis_label)
        self._ax.set_ylabel(self.y_axis_label)

        # Grid
        if self.show_grid:
            self._ax.grid(True, linewidth=0.2)

        # Ticks
        if self.x_tick_positions:
            self._ax.set_xticks(self.x_tick_positions)
            if self.x_tick_texts:
                self._ax.set_xticklabels(self.x_tick_texts)
        if self.y_tick_positions:
            self._ax.set_yticks(self.y_tick_positions)
            if self.y_tick_texts:
                self._ax.set_yticklabels(self.y_tick_texts)

        # Secondary Y axis if needed
        if self.y2_tick_positions:
            ax2 = self._ax.twinx()
            ax2.set_yticks(self.y2_tick_positions)
            if self.y2_tick_texts:
                ax2.set_yticklabels(self.y2_tick_texts)
            ax2.set_ylabel(self.y2_axis_label)

        # Limits
        if self.x_lim:
            left, right = self.x_lim
            if left is not None:
                self._ax.set_xlim(left=left)
            if right is not None:
                self._ax.set_xlim(right=right)
        if self.y_lim:
            bottom, top = self.y_lim
            if bottom is not None:
                self._ax.set_ylim(bottom=bottom)
            if top is not None:
                self._ax.set_ylim(top=top)

        # Connect events (e.g. mouse click)
        self._fig.canvas.mpl_connect('button_press_event', on_mouse_click)

        # Apply the pre-calculated layout to ensure fixed axis size
        self._fig.subplots_adjust(
            left=self._layout_params['left'],
            right=self._layout_params['right'],
            top=self._layout_params['top'],
            bottom=self._layout_params['bottom']
        )
        # self._fig.tight_layout()
        self._canvas.draw()

        if self.save_img_filepath:
            self._fig.savefig(self.save_img_filepath, dpi=self.save_img_dpi)

        update_UI()

    def save_csv(self, output_filename):
        """
        Legacy for save_plot_history
        """
        self.save_plot_history(output_filename)
        
    def save_plot_history(self, output_filename):
        plot_history_folder = os.path.join(filename_parent(output_filename), "Plot_History")
        os.makedirs(plot_history_folder, exist_ok=True)
        csv_path = os.path.join(plot_history_folder, filename_replace_last_append(filename_name(output_filename),".csv"))
        csv_path = get_unused_filename(csv_path)

        data_columns = []
        headers = []
        max_len = 0

        for i, curve in enumerate(self.plot_objects):
            # Use original data
            xs = list(curve.Xs)
            ys = list(curve.Ys)

            max_len = max(len(xs), max_len)

            data_columns.append(xs)
            data_columns.append(ys)

            # Try to get a label
            label = curve.Y_label if curve.Y_label else f"Curve_{i + 1}"
            headers.append(f"{label}_X")
            headers.append(f"{label}_Y")

        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for i in range(max_len):
                row = []
                for col in data_columns:
                    if i < len(col):
                        row.append(col[i])
                    else:
                        row.append("")
                writer.writerow(row)

        # Log title and filename if title exists
        if hasattr(self, 'current_title') and self.current_title:
            if not hasattr(self, 'log_filenames'):
                self.log_filenames = {}
            
            if plot_history_folder not in self.log_filenames:
                os.makedirs(plot_history_folder, exist_ok=True)
                base_log_filename = filename_name(output_filename)
                base_log_filename = f"0_Plot_Infos_{replace_last_append(base_log_filename, "txt")}"
                base_log = get_unused_filename(os.path.join(plot_history_folder, base_log_filename))
                self.log_filenames[plot_history_folder] = base_log
            
            log_file = self.log_filenames[plot_history_folder]

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{log_file}\tPlot Title\t{self.current_title}\n")


    def save_png(self, output_filename, dpi=None, bbox_inches:Optional[str]="tight"):
        """
        Save the current plot as a PNG file.
        Also saves the numerical data of the curves to a CSV file in a 'Numeric_Plot_Data' subfolder.
        If the plot has a title, it logs the title and filename to '0_Plot_Infos.txt'.

        :param output_filename: The path to save the PNG file.
        :param dpi: The resolution in dots per inch. If None, uses self.save_img_dpi.
        :param bbox_inches: Bounding box in inches: 'tight' or None.
        """
        dpi = dpi or self.save_img_dpi
        output_filename = filename_replace_last_append(output_filename, ".png")
        self._fig.savefig(output_filename, dpi=dpi, bbox_inches=bbox_inches)

        self.save_plot_history(output_filename)


    def save_svg(self, output_filename):
        """
        Save the current plot as a SVG file.
        Also saves the numerical data of the curves to a CSV file in a 'Numeric_Plot_Data' subfolder.

        :param output_filename: The path to save the SVG file.
        """
        output_filename = filename_replace_last_append(output_filename, ".svg")
        self._fig.savefig(output_filename, bbox_inches="tight")
        self.save_plot_history(output_filename)


    def percentage_zoom_Y(self, percentile=0.8, ensure_last_points=0.2, lower_limit_only = False, upper_limit_only = False):
        """
        Automatically adjust y_lim based on data.
        :param percentile: ratio of data to cover (0 < p <= 1). Default 0.8.
                            We calculate the range [50 - 50*p, 50 + 50*p] percentile for each curve,
                            then take the unions. 
                            If 0.8, we take [10%, 90%] of the data range.
        :param ensure_last_points: 
                            if <= 1, ratio of last points to keep (e.g. 0.2 means last 20%).
                            if >= 2, count of last points.
                            These points are GUARANTEED to be in the view range.
                            
        The final range is expanded by 5% on both ends.
        """
        all_min = []
        all_max = []
        import numpy as np
        
        for curve in self.plot_objects:
            if not isinstance(curve, Curve): continue
            Ys = np.array(curve.Ys, dtype=float)
            if len(Ys) == 0: continue
            
            # 1. Percentile range (centered)
            # e.g. 0.8 -> 10% to 90%
            p_low = (1.0 - percentile) / 2.0 * 100
            p_high = (1.0 + percentile) / 2.0 * 100
            
            c_min = np.percentile(Ys, p_low)
            c_max = np.percentile(Ys, p_high)
            all_min.append(c_min)
            all_max.append(c_max)
            
            # 2. Last points
            if ensure_last_points > 0:
                count = 0
                if ensure_last_points <= 1:
                    count = int(len(Ys) * ensure_last_points)
                    count = max(count, 1) # At least 1 if ratio provided?
                else:
                    count = int(ensure_last_points)
                
                # Careful with count > len(Ys)
                last_Ys = Ys[-count:] if count < len(Ys) else Ys
                if len(last_Ys) > 0:
                    all_min.append(np.min(last_Ys))
                    all_max.append(np.max(last_Ys))
        
        if not all_min: return # No data
        
        final_min = min(all_min)
        final_max = max(all_max)
        
        # Expansion
        span = final_max - final_min
        if span == 0:
                span = abs(final_max) * 0.1 if final_max != 0 else 1.0
                
        final_min -= span * 0.05
        final_max += span * 0.05
        
        if upper_limit_only:
            final_min = self.y_lim[0] if self.y_lim else None
        if lower_limit_only:
            final_max = self.y_lim[1] if self.y_lim else None
        self.y_lim = (final_min, final_max)
        self._update_plot()



def integrate_discrete_points(x_values, y_values, start_range, end_range):
    sort = sorted(list(zip(x_values, y_values)), key=lambda x: x[0])
    x_values, y_values = transpose_2d_list(sort)

    if start_range < x_values[0] or end_range > x_values[-1]:
        print("----------------WARNING--------------\n"
              "Integration range exceeds data range.\n"
              "----------------WARNING--------------")
    start_range = max(x_values[0], start_range)
    end_range = min(x_values[-1], end_range)
    # raise ValueError("Integration range exceeds data range.")

    # Initialize integration result
    integration_result = 0

    for i in range(1, len(x_values) - 1):
        # Determine the start and end of the region
        region_start = (x_values[i] + x_values[i - 1]) / 2
        region_end = (x_values[i] + x_values[i + 1]) / 2

        # Check if the region is within the integration range
        if region_end < start_range or region_start > end_range:
            continue

        # Adjust region start and end to the integration range
        region_start = max(region_start, start_range)
        region_end = min(region_end, end_range)

        # Calculate area (trapezoidal rule)
        base = region_end - region_start
        height = y_values[i]
        area = base * height

        integration_result += area

    return integration_result


def integrate_reptile_light_zones(x_values, y_values, title):
    print(title)

    def fix_range_l(wavelength):
        return max(min(x_values), wavelength)

    def fix_range_r(wavelength):
        return min(max(x_values), wavelength)

    ret = [integrate_discrete_points(x_values, y_values, fix_range_l(250), fix_range_r(290)),
           integrate_discrete_points(x_values, y_values, fix_range_l(290), fix_range_r(295)),
           integrate_discrete_points(x_values, y_values, fix_range_l(295), fix_range_r(315)),
           integrate_discrete_points(x_values, y_values, fix_range_l(315), fix_range_r(335)),
           integrate_discrete_points(x_values, y_values, fix_range_l(335), fix_range_r(380)),
           integrate_discrete_points(x_values, y_values, fix_range_l(380), fix_range_r(780))]

    percent_ret = [x / sum(ret) * 100 for x in ret]

    print("UVC+UVBX", ret[0], "{:.2f}%".format(percent_ret[0]), sep='\t')
    print("Rare UVB", ret[1], "{:.2f}%".format(percent_ret[1]), sep='\t')
    print("Good UVB", ret[2], "{:.2f}%".format(percent_ret[2]), sep='\t')
    print("UVA2", ret[3], "{:.2f}%".format(percent_ret[3]), sep='\t')
    print("UVA1", ret[4], "{:.2f}%".format(percent_ret[4]), sep='\t')
    print("Vis", ret[5], "{:.2f}%".format(percent_ret[5]), sep='\t')
    print()


def error_bar(samples, sigma_count=1):
    import scipy.stats
    return scipy.stats.sem(samples) * sigma_count * 2


def simulate_data_for_sigma(input_mean, sigma):
    """
    For a input value and sigma, generate three data points (value-x, value, value+x) so that the scipy.stats.sem of the data point is the given sigma
    :param input_mean:
    :param sigma:
    :return:
    """
    x = math.sqrt(sigma ** 2 * 3)
    return (input_mean - x, input_mean, input_mean + x)


def on_mouse_click(event):
    # print('Event received:',int(event.xdata),int(event.xdata%1*60),event.ydata)
    print('Event received:', event.xdata, event.ydata)


def interp_manipulation(interp1, interp2,
                        Xs_1: list[float], Xs_2: list[float],
                        computation_function,
                        number_of_points=5000,
                        interpolation_kind="cubic") -> Callable:
    """
    do interpolation manipulation by evaluate value of points, do point-by-point calculation, then re-interpolate
    :param computation_function:
    :param interpolation_kind:
    :param Xs_1:
    :param Xs_2:
    :param interp1:
    :param interp2:
    :param number_of_points:
    :return:
    """

    min_x = max(min(Xs_1), min(Xs_2))
    max_x = min(max(Xs_1), max(Xs_2))

    xvals = np.linspace(min_x, max_x, num=number_of_points)

    if isinstance(interp1, Curve):
        interp1 = interp1.interp1d
    if isinstance(interp2, Curve):
        interp2 = interp2.interp1d

    Y1 = interp1(xvals)
    Y2 = interp2(xvals)

    Y = computation_function(Y1, Y2)

    return interp1d(xvals, Y, kind=interpolation_kind)


def substract_component(curve, bkg,
                        effective_x_range=None,
                        interpolation_count=2000,
                        negative_penalty_power=2,
                        bound=(0, 2),
                        intensity_weight_parameter=2,
                        scaling_negative_penalty=1):
    """
    remove_background, so that the average value in effective X becomes close to 0
    按照背景的最大值归一化为100处理，为了保证惩罚函数的相似性

    :param curve: curve as a interp1d function
    :param bkg: bkg as a interp1d function
    :param effective_x_range:
    :param negative_penalty_power: default 2, must be an even number, the larger it gets,
                                   the larger effort is taken to guarantee that no negative values are generated.
    :param interpolation_count:
    :param bound: the limit for the resulted factor, default (0,2)
    :param intensity_weight_parameter:
            if set to 0, it is irrelevant to background strength.
                         If background spectrum have negative value larger than the maximum background strength, this is the only allowed method
            if set to -1, the allowed negative value is weighted to sqrt(I), i.e. the punishment is (I + 1)^(-1)
                          with 1% of the maximum value of the background added to the background spectrum, to avoid devided by zero problem
            if set to positive, beyond this number, the detector is considered saturated,
                                and it is not punished for having negative value here, e.g. absorbance >2 for IR
                                the transition from have punishment and have not is smooth
                                thus the penalty for generating a negative signal should not be as high as normal-range data
    :param scaling_negative_penalty:scaling the penalty when the result point is negative, a less-than one value encourages the program to subtract more

    :return:
    """

    # 归一化到100，保证函数相似性
    normalization = 100

    from scipy import optimize

    if not effective_x_range:
        effective_x_range = common_range_of_2_interp1d(curve, bkg)

    averaging_Xs = np.linspace(min(effective_x_range), max(effective_x_range), interpolation_count)
    curve_points = curve(averaging_Xs)
    bkg_points = bkg(averaging_Xs)

    bkg_max = max(bkg_points)
    assert bkg_max > 0
    scaling = normalization / bkg_max
    curve_points = curve_points * scaling
    bkg_points = bkg_points * scaling
    bkg_min = min(bkg_points)
    bkg_max = max(bkg_points)

    import bezier
    def evaluate_bezier_at_x(bezier_curve, x, bound, out_of_bound_values):
        """
        Evaluate a bezier curve at x
        :param bezier_curve:
        :param x:
        :param bound: 曲线的定义域，2-tuple
        :param out_of_bound_values: 2-tuple （小于曲线定义域下限时的返回值，大于曲线定义域上限时的返回值）
        :return:
        """
        if bound[0] < x < bound[1]:
            # print(x)
            line_nodes = np.asfortranarray([[x, x], [0, 1]])
            line = bezier.Curve(line_nodes, degree=1)
            intersections = bezier_curve.intersect(line)
            s_vals = np.asfortranarray(intersections[0, :])
            # print(s_vals)
            # print(bezier_curve.evaluate_multi(s_vals))
            return bezier_curve.evaluate_multi(s_vals)[1][0]
        if x < bound[0]:
            return out_of_bound_values[0]
        if x > bound[1]:
            return out_of_bound_values[1]

    def intensity_weight_function(bkg_intensity):
        # weight is only applied to negative post-processed spectrum points
        if intensity_weight_parameter == 0:
            return 1
        if intensity_weight_parameter == -1:
            return (abs(bkg_intensity) + 1) ** (-1 / 2) - (normalization + 1) ** (-1 / 2) + 0.1  # 使最小值是0.1
        if intensity_weight_parameter > 0:
            dead_zone = scaling * intensity_weight_parameter
            # 定义为三点构成的Bézier Curve, (0,1) (dead_zone/2, 0) (dead_zone,0)
            # print("Dead zone:",dead_zone)
            nodes = np.asfortranarray([[0.0, dead_zone / 3, dead_zone],
                                       [1.0, 0.1, 0.1]])
            curve = bezier.Curve(nodes, degree=2)
            return evaluate_bezier_at_x(curve, bkg_intensity, (0, dead_zone), (1, 0))

    intensity_weight_function = np.vectorize(intensity_weight_function)

    def penalty(ratio):
        monitor_by_xlsx = []
        ret_points = curve_points - bkg_points * ratio
        negative_punish_weights = intensity_weight_function(bkg_points)
        punish = np.array([(x + 1 if x > 0 else scaling_negative_penalty * abs((x - 1) ** negative_penalty_power)) for x in ret_points])
        ret = np.multiply(negative_punish_weights, punish, where=ret_points < 0, out=np.zeros_like(ret_points)) + \
              np.multiply(punish, 1, where=ret_points >= 0, out=np.zeros_like(ret_points))

        # monitor_by_xlsx.append(["No"]+list(range(len(curve_points))))
        # monitor_by_xlsx.append(["Curve"] + list(curve_points))
        # monitor_by_xlsx.append(["Bkg"] + list(bkg_points))
        # monitor_by_xlsx.append(["Curve at {:.3f}".format(ratio)] + list(ret_points/max(ret_points)*100))
        # monitor_by_xlsx.append(['Intensity_weight']+list(negative_punish_weights/max(negative_punish_weights)*100))
        # monitor_by_xlsx.append(['Punish'] + list(punish))
        # monitor_by_xlsx.append(['Weighted Punish'] + list(ret))
        # write_xlsx(get_unused_filename('ReactIR/Penalty Check {:.3f}.xlsx'.format(ratio),use_proper_filename=False),monitor_by_xlsx,transpose=True)

        return sum(ret)

    ratio = optimize.minimize_scalar(penalty, bounds=bound, method='Bounded')
    ratio = ratio.x
    print("Background Correction Ratio:", ratio)

    return interp_manipulation(curve, bkg, lambda x, y: x - y * ratio, interpolation_count)


def draw_xlsx(xlsx_files):
    plot_objects = []
    for xlsx_file in xlsx_files:
        lists = read_xlsx_to_horizontal_lists(xlsx_file)
        plot_object = Curve(X_and_Y=remove_invalid_data(lists[0], lists[1], first_as_label=True))
        plot_objects.append(plot_object)

    Plot(plot_objects)


def nonlinear_model_fit(model, X, Y, initial_guesses, variable_names=None, Y_uncertainties=None,
                        show_plot=True, x_axis_label=None, y_axis_label=None, lower_bounds=None, upper_bounds=None,
                        print_covariances=True, method=None, print_result=True):
    """
    Do a NonlinearModelFit like Mathematica
    :param x_axis_label:
    :param y_axis_label:
    :param print_covariances:
    :param method:
    :param model: two_Gaussian should be function that accept n+1 parameters,
                  the first parameter should be the variable
                  the second to last parameter is those that needs to be fitted.
                  two_Gaussian should NOT be vectorized
    :param X:
    :param Y:
    :param initial_guesses:
    :param variable_names:
    :param Y_uncertainties: a list of sigma for Y data, if None, it is seen as it all has zero uncertainty
    :param show_plot:
    :param lower_bounds: a list of lower bounds for each parameter, -inf if not designated.
    :param upper_bounds:
    :return:optimal_values
    """
    model = np.vectorize(model)

    if lower_bounds is None:
        lower_bounds = [-float('inf') for _ in range(len(initial_guesses))]
    if upper_bounds is None:
        upper_bounds = [float('inf') for _ in range(len(initial_guesses))]

    if Y_uncertainties is None:
        opt_values, covariances = scipy.optimize.curve_fit(model, X, Y, initial_guesses, maxfev=10000, bounds=(lower_bounds, upper_bounds), method=method)
    else:
        # TODO: 似乎不对
        raise Exception("Y_uncertainties Not Implemented Yet")
        # new_X = []
        # new_Y = []
        # for x,y,y_sigma in zip(X,Y,Y_uncertainties):
        #     new_X+=[x,x,x]
        #     new_Y+=list(simulate_data_for_sigma(y,y_sigma))
        # opt_values, covariances = scipy.optimize.curve_fit(two_Gaussian, new_X, new_Y, initial_guesses, maxfev=10000)
    if print_covariances and print_result:
        print(covariances)
    std_errs = np.sqrt(np.diag(covariances))

    if variable_names is None:
        variable_names = [f"Var_{x}" for x in range(1, len(initial_guesses) + 1)]

    if print_result:
        for count, variable in enumerate(variable_names):
            print(variable, ':', print_float_and_stderr(opt_values[count], std_errs[count], sig_digits=3))

    # print("IC50: ","{:.2f}, 95% CI ({:.2f}-{:.2f})".format(10**opt_values[0], 10**(opt_values[0]-std_errs[0]*2), 10**(opt_values[0]+std_errs[0]*2)))
    ret_plot = None
    if show_plot:
        dot = Curve(X=X, Y=Y, plot_dot=True, dot_format=point_mkr, dot_width=5, plot_curve=False, dot_color=blue_color)
        # 去掉第一个和最后一点，防止interp1d超限
        curve = Curve(X=X, curve_color=red_color, fitted_function=lambda x: model(x, *opt_values))

        ret_plot = Plot([dot, curve],
                        plot_legend=False,
                        x_axis_label=x_axis_label,
                        y_axis_label=y_axis_label,
                        font_size=10
                        # non_blocking=True
                        # x_lim=(0, None),
                        # y_lim=(curve.interp1d(0), None),
                        # x_log=True
                        # y_log=True
                        )
        ret_plot.pause()

    return opt_values, ret_plot


def numerical_derivative(model: Callable, params: Sequence, param_perturbations: Sequence, x: float, x_perturbation=0):
    """

    :param model: two_Gaussian should be function that accept n+1 parameters,
                  the first parameter should be the variable
                  the second to last parameter is those that needs to be fitted.
                  two_Gaussian should NOT be vectorized
    :param params: length n list of fitted parameter value
    :param param_perturbations: length n list of perturbation, for partial derivitive, should be [0,0..0,epsilon,0..0]
    :param x:
    :param x_perturbation: to calculate the perturbation of x
    :return: the derivitive as a number
    """
    perturbations = param_perturbations + [x_perturbation]
    assert any(x != 0 for x in perturbations), "No perturbation given."
    pos_shift = model(x + x_perturbation, *list(np.array(params) + np.array(param_perturbations)))
    neg_shift = model(x - x_perturbation, *list(np.array(params) - np.array(param_perturbations)))
    derivative = (pos_shift - neg_shift) / np.linalg.norm(perturbations) / 2
    return derivative


def function_value_with_uncertainty(model: Callable, parameters: Sequence, param_uncertainties: Sequence, x: float, x_uncertainty=0):
    """

    :param model: two_Gaussian should be function that accept n+1 parameters,
                  the first parameter should be the variable
                  the second to last parameter is those that needs to be fitted.
                  two_Gaussian should NOT be vectorized
    :param parameters: the fitted optimal parameter value
    :param param_uncertainties: the uncertainties given by nonlinear two_Gaussian fit
    :param x: where to estimate
    :param x_uncertainty: the uncertainty of the x parameter
    :return: (estimated_value, uncertainties of estimated value)
    """
    estimated_value = model(x, *parameters)

    # calculate partial derivative to each parameter
    for i in range(-3, -10, -1):  # test convergence relative to epsilon
        partial_derivitives = [None for _ in range(len(parameters) + 1)]
        for j in range(len(parameters)):
            perturbations = [0 for _ in range(len(parameters))]
            # 如果parameter超过1E-7，用parameter乘以上述比例作为epsilon；否则假设为1E-7
            if abs(parameters[j]) > 1E-7:
                epsilon = parameters[j] * (10 ** i)
            else:
                epsilon = 1E-7 * (10 ** i)
            perturbations[j] = epsilon
            partial_derivitives[j] = numerical_derivative(model, parameters, perturbations, x)

        if abs(x) > 1E-7:
            epsilon = x * (10 ** i)
        else:
            epsilon = 1E-7 * (10 ** i)
        partial_derivitives[-1] = numerical_derivative(model, parameters, [0 for _ in range(len(parameters))], x, epsilon)

        Δf_square = 0
        for partial_derivitive, uncertainty in zip(partial_derivitives, param_uncertainties + [x_uncertainty]):
            Δf_square += (partial_derivitive * uncertainty) ** 2
        Δf = math.sqrt(Δf_square)
        print(i, Δf)

    return estimated_value, Δf


def mass_spec(_2D_list):
    """
    :param _2D_list: a list of intensities for each mass e.g [(10,80),(11,3)]
    :return: data to plot a centroid vertical-line plot like mass spec [(10,0),(10,80),(10,0),(11,0),(11,3),(11,0)]
    """
    ret = []
    for i in _2D_list:
        ret.append([i[0], 0])
        ret.append([i[0], 0])
        ret.append(i)
        ret.append(i)
        ret.append([i[0], 0])
        ret.append([i[0], 0])
    return ret


def plot_spectrum_sections(x_range,
                           wavelengths,
                           fig_size=(3, 3),
                           font_size=10):
    plt.figure(figsize=fig_size)
    plt.rcParams.update({'font.size': font_size, 'font.family': '思源黑体', 'font.weight': 'bold', 'mathtext.default': 'regular'})
    ax = plt.gca()
    plt.rcParams['font.family'] = prop.get_name()

    # Plot each wavelength range as a horizontal bar
    for start, end, color, label in wavelengths:
        ax.add_patch(patches.Rectangle((start, 0), end - start, 1, color=color))
        ax.add_patch(patches.Rectangle((start, 0), start + 2, 1, color=color))
        ax.add_patch(patches.Rectangle((end - 2, 0), end, 1, color=color))
        # Add label in the center of the bar
        if start > x_range[1]:
            continue
        if end < x_range[0]:
            continue
        start = max(start, x_range[0])
        end = min(end, x_range[1])
        ax.text((start + end) / 2, 0.5, label, ha='center', va='center', color='white', fontproperties=prop)

    # Create a smooth transition for the visible spectrum (380-780 nm)
    vis_start, vis_end = 380, 780
    vis_colors = plt.cm.rainbow(np.linspace(0, 1, vis_end - vis_start))
    for i, color in enumerate(vis_colors, start=vis_start):
        ax.add_patch(patches.Rectangle((i, 0), 1, 1, color=color))
    ax.text((vis_start + vis_end) / 2, 0.5, 'Vis', ha='center', va='center', color='white', fontproperties=prop)

    # Set the x and y limits
    ax.set_xlim(x_range)
    ax.set_ylim(0, 1)

    # Remove y-axis and ticks
    ax.get_yaxis().set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)

    # Set the x-axis label
    ax.set_xlabel('Wavelength (nm)')

    # Show the plot
    plt.show()


def regression_on_UVI_over_distance(input_X, input_Y):
    def model(x, x0, a, k):
        return k * (x + x0) ** (-a)

    opt_values = nonlinear_model_fit(model, input_X, input_Y, (0, 2, 10000), ("x0", "a", "k"), show_plot=False)

    def fitted_function(x):
        return model(x, *opt_values)

    return fitted_function


def sum_of_multiple_Gaussians(x, *parameters):
    assert len(parameters) % 3 == 0
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    ret = 0
    for parameter_index in range(0, len(parameters), 3):
        A, x0, sigma = parameters[parameter_index:parameter_index + 3]
        ret += A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    return ret


def gaussian_peaks_fitting(X, Y, max_gaussians=10):
    """
    Iteratively Fit a curve as the summation of multiple gaussian functions.
    :param X:
    :param Y:
    :param max_gaussians:
    :return:
        [ret]
            a list of list, (one_gaussian_fit_parameters (3-tuple of A, x0, sigma),
                             two_gaussian_fit_parameters (6-tuple of  A, x0, sigma,  A, x0, sigma),
                             three_gaussian_fit_parameters (9-tuple),
                             ...
                             )
            The returned tuple will truncate at max_gaussians, or the first case where the fitting failed.
        [residue]
            the excess and deficient percentage of the curve - fitted_curve
    """

    X_lower_bound = min(X)
    X_upper_bound = max(X)
    Y_upper_bound = max(Y)
    wavelength_bound_relax = 50
    intensity_bound = 5
    min_sigma = 10
    max_sigma = 50

    AUC = integrate_discrete_points(X, Y, X_lower_bound, X_upper_bound)

    ret = []
    residue = []
    current_residue = Y
    for num_Gaussian in range(1, max_gaussians + 1):
        points = list(zip(X, current_residue))
        X_at_max, Y_at_max = list(sorted(points, key=lambda point: point[1], reverse=True))[0]
        single_gaussian = nonlinear_model_fit(sum_of_multiple_Gaussians,
                                              X, current_residue,
                                              (Y_at_max, X_at_max, 20),
                                              lower_bounds=(0, X_lower_bound - wavelength_bound_relax, min_sigma),
                                              upper_bounds=(Y_upper_bound * intensity_bound,
                                                            X_upper_bound + wavelength_bound_relax,
                                                            max_sigma),
                                              show_plot=False, print_result=False)

        try:
            all_gaussian = nonlinear_model_fit(sum_of_multiple_Gaussians,
                                               X, Y,
                                               (ret[-1] if ret else []) + list(single_gaussian),
                                               lower_bounds=[0, X_lower_bound - wavelength_bound_relax, min_sigma] * num_Gaussian,
                                               upper_bounds=[Y_upper_bound * intensity_bound,
                                                             X_upper_bound + wavelength_bound_relax,
                                                             max_sigma] * num_Gaussian,
                                               print_result=False, show_plot=False)
        except RuntimeError as _:  # 没有拟合出来
            print("Stopped after fitting", num_Gaussian, "Gaussians.")
            break

        ret.append(list(all_gaussian.tolist()))
        current_residue = ([y - sum_of_multiple_Gaussians(x, *ret[-1]) for x, y in zip(X, Y)])

        positive_residue_Y = [max(y, 0) for y in current_residue]
        negative_residue_Y = [min(y, 0) for y in current_residue]

        positive_residue_percentage = integrate_discrete_points(X, positive_residue_Y, X_lower_bound, X_upper_bound) / AUC
        negative_residue_percentage = integrate_discrete_points(X, negative_residue_Y, X_lower_bound, X_upper_bound) / AUC

        residue.append((positive_residue_percentage, negative_residue_percentage))

        ret_print = ["{:.2f}".format(x) for x in ret[-1]]

        print(f"Δ +{positive_residue_percentage * 100:.1f}%, {negative_residue_percentage * 100:.1f}%     |     {', '.join(ret_print)}")

    return ret, residue


def xyz_triples_to_2D_list(xyz_triples, default_value="", print_table=True, output_file = None):
    """
    Convert a list of [x, y, z] to X_headers, Y_headers, and Z_matrix[x_idx][y_idx].
    X_headers will be the row headers (index of outer list of Z_matrix).
    Y_headers will be the column headers (index of inner list of Z_matrix).
    :param xyz_triples: List of [x, y, z]
    :param default_value: Value to fill if (x, y) is missing
    :param print_table: Boolean, whether to print the table to stdout
    :return: Xs, Ys, Z_matrix
    """
    print(output_file)
    Xs = sorted(list(set([x[0] for x in xyz_triples])))
    Ys = sorted(list(set([x[1] for x in xyz_triples])))

    Z_matrix = [[default_value for _ in range(len(Ys))] for _ in range(len(Xs))]

    # Map coordinates to indices
    x_map = {x: i for i, x in enumerate(Xs)}
    y_map = {y: i for i, y in enumerate(Ys)}

    for item in xyz_triples:
        x, y = item[0], item[1]
        z = item[2] if len(item) > 2 else default_value
        if x in x_map and y in y_map:
             Z_matrix[x_map[x]][y_map[y]] = z

    if print_table:
        # Print Header
        if output_file is not None:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write("\t" + "\t".join([str(y) for y in Ys]) + "\n")
        print("\t" + "\t".join([str(y) for y in Ys]))
        for i, x in enumerate(Xs):
            row_str = str(x) + "\t" + "\t".join([str(z) for z in Z_matrix[i]])
            print(row_str)
            if output_file is not None:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(row_str)

    if output_file:
        print("File saved to:", output_file)

    return Xs, Ys, Z_matrix


def plot_2D_scatter_surface_matplotlib(_X, _Y, _Z, X_Label="X axis", Y_Label="Y axis", color_scale_percentage=80, traces_list=[], color_map_shift=0,
                                       override_color_scale=None, without_colorfill=False):
    '''

    :param _X:
    :param _Y:
    :param _Z:
    :param X_Label:
    :param Y_Label:
    :param color_scale_percentage:
    :param traces_list: a list of list, [[trace1_x,trace1_y],[trace2_x,trace2_y]]
    :param color_map_shift: a number, if it is 0.1, then the color map will be shifted up for 10% of max(Z)-min(Z)
    :return:
    '''
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    import matplotlib.tri as tri
    from scipy import interpolate
    import matplotlib
    import numpy as np

    min_X = min(_X)
    max_X = max(_X)
    dist_X = max_X - min_X
    X_range = [min_X, max_X]

    print("Min, max, dist of X:", min_X, max_X, dist_X, X_range)

    min_Y = min(_Y)
    max_Y = max(_Y)
    dist_Y = max_Y - min_Y
    Y_range = [min_Y, max_Y]

    print("Min, max, dist of Y:", min_Y, max_Y, dist_Y, Y_range)

    ranges_for_imshow = X_range + Y_range
    ranges_for_contours = X_range + list(reversed(Y_range))

    figure(figsize=(10, 3.5), facecolor='w', edgecolor='k')
    matplotlib.rcParams.update({'font.size': 8})
    matplotlib.rcParams.update({'font.family': "Arial"})

    if dist_Y > dist_X:
        grid_x = 200
        grid_y = int(grid_x / dist_X * dist_Y)
        # 限制最多10:1的长宽比
        grid_y = min(grid_y, 10 * grid_x)
    else:
        grid_y = 200
        grid_x = int(grid_y / dist_Y * dist_X)
        # 限制最多10:1的长宽比
        grid_x = min(grid_x, 10 * grid_y)

    new_x_values = np.linspace(min_X, max_X, grid_x)
    new_y_values = np.linspace(min_Y, max_Y, grid_y)

    print('')
    print("Num of New X/Y Values:", len(new_x_values), len(new_y_values))
    # no idea how this works, just generate a list of 2-tuples, with all combination of new_x_values and new_y_values, x in front
    # new_Xs_meshgrid, new_Ys_meshgrid = np.meshgrid(new_x_values, new_y_values)
    meshgrid = np.array(np.meshgrid(new_x_values, new_y_values)).T.reshape(-1, 2)
    new_X_grid, new_Y_grid = meshgrid[:, 0], meshgrid[:, 1]  # 各自为一个一维向量，二者并在一起是X和Y所有点的组合
    print("New X,Y Grid size:", new_X_grid.shape, new_Y_grid.shape)  # should all be grid_x*grid_y

    from scipy.interpolate import griddata
    new_Zs_grid_matplotlib = griddata((_X, _Y), _Z, (new_x_values[None, :], new_y_values[:, None]), method='cubic')
    print(new_Zs_grid_matplotlib.shape)

    new_Zs_list = []
    for x in range(grid_x):
        for y in range(grid_y):
            new_Zs_list.append(new_Zs_grid_matplotlib[y][x])
    new_Zs_list = np.array(new_Zs_list)

    print(new_Zs_list.shape)

    if not override_color_scale:
        color_scale_range = find_range_for_certain_percentage(_Z, color_scale_percentage, color_map_shift)
    else:
        color_scale_range = override_color_scale
    # color_map = plt.cm.coolwarm
    color_map = plt.cm.Spectral_r

    color_normalize = matplotlib.cm.colors.Normalize(vmin=color_scale_range[0], vmax=color_scale_range[1])
    planar_subplot = plt.subplot2grid((1, 2), (0, 0), colspan=1)
    if not without_colorfill:
        planar_subplot.contour(new_x_values, new_y_values, new_Zs_grid_matplotlib, levels=30,
                               colors='k', linewidths=0.3, linestyles='dashed')
    else:
        planar_subplot.contour(new_x_values, new_y_values, new_Zs_grid_matplotlib, levels=30,
                               colors='k', linewidths=3)

    x_lim = planar_subplot.get_xlim()
    y_lim = planar_subplot.get_ylim()

    discard_first_n_points = 20  # to hide the starting point

    for trace in traces_list:
        if trace[0][-1] > trace[1][-1]:
            planar_subplot.plot(trace[0][discard_first_n_points:], trace[1][discard_first_n_points:], '-', color='#C818FD', linewidth=0.3)
        else:
            planar_subplot.plot(trace[0][discard_first_n_points:], trace[1][discard_first_n_points:], '-', color='#FC4242', linewidth=0.3)
    planar_subplot.set_xlim(x_lim)
    planar_subplot.set_ylim(y_lim)

    if not without_colorfill:
        contourf_object = planar_subplot.contourf(new_x_values, new_y_values, new_Zs_grid_matplotlib, levels=500,
                                                  cmap=color_map,
                                                  extent=ranges_for_imshow,
                                                  norm=color_normalize)
        fig = plt.gcf()
        fig.colorbar(contourf_object, ax=planar_subplot)
        planar_subplot.set_xlabel(X_Label)
        planar_subplot.set_ylabel(Y_Label)

    planar_subplot = plt.subplot2grid((1, 2), (0, 1), colspan=1)
    planar_subplot.tricontour(_X, _Y, _Z, levels=20,
                              colors='k', linewidths=0.3)

    contourf_object = planar_subplot.tricontourf(_X, _Y, _Z, levels=255,
                                                 cmap=color_map,
                                                 extent=ranges_for_imshow,
                                                 norm=color_normalize)
    fig = plt.gcf()
    fig.colorbar(contourf_object, ax=planar_subplot)
    planar_subplot.set_xlabel(X_Label)
    planar_subplot.set_ylabel(Y_Label)

    plt.show()


def plot_2D_scatter_surface_mayavi(_X, _Y, _Z, X_Label="X axis", Y_Label="Y axis", color_scale_percentage=80, traces_list=[]):
    """
    Plot a color mapped f(x,y)=z surface that can be rotated
    :param _X: 
    :param _Y: 
    :param _Z: 
    :param X_Label: 
    :param Y_Label:
    :param color_scale_percentage: 
    :param traces_list: 
    :return: 
    """

    from mayavi import mlab

    min_X = min(_X)
    max_X = max(_X)
    dist_X = max_X - min_X
    X_range = [min_X, max_X]

    min_Y = min(_Y)
    max_Y = max(_Y)
    dist_Y = max_Y - min_Y
    Y_range = [min_Y, max_Y]

    grid = 300

    if dist_Y > dist_X:
        grid_x = grid
        grid_y = int(grid_x / dist_X * dist_Y)
        # 限制最多10:1的长宽比
        grid_y = min(grid_y, 10 * grid_x)
    else:
        grid_y = grid
        grid_x = int(grid_y / dist_Y * dist_X)
        # 限制最多10:1的长宽比
        grid_x = min(grid_x, 10 * grid_y)

    new_x_list = np.linspace(min_X, max_X, grid_x)
    new_y_list = np.linspace(min_Y, max_Y, grid_y)
    new_x_matrix, new_y_matrix = np.mgrid[min_X:max_X:grid_x * 1j, min_Y:max_Y:grid_y * 1j]
    new_Zs_grid = interpolate.griddata((_X, _Y), _Z, (new_x_list[None, :], new_y_list[:, None]), method='cubic').T

    def interpolation_function(x, y, x_list, y_list, Zs_grid):

        x_pos = len(x_list) - 1
        y_pos = len(y_list) - 1
        for x_try in range(len(x_list)):
            if x_list[x_try] > x:
                x_pos = x_try
                break
        for y_try in range(len(y_list)):
            if y_list[y_try] > y:
                y_pos = y_try
                break

        return Zs_grid[x_pos][y_pos]

    print(new_x_matrix.shape, new_y_matrix.shape)
    print(new_Zs_grid.shape)

    color_range = find_range_for_certain_percentage(_Z, color_scale_percentage)
    # generate a matrix for 2D-color-mapped plot only, as the color map range is not correctly mapped in mayavi for 2D imshow
    new_Zs_grid_fix_color_range = np.copy(new_Zs_grid)
    new_Zs_grid_fix_color_range[new_Zs_grid_fix_color_range > color_range[1]] = color_range[1]
    new_Zs_grid_fix_color_range[new_Zs_grid_fix_color_range < color_range[0]] = color_range[0]

    extent_XY = [min_X, max_X, min_Y, max_Y]
    print(extent_XY)
    z_extent = [np.nanmin(new_Zs_grid) / 100, np.nanmax(new_Zs_grid) / 100]
    z_extent_factor = 0.01
    z_extent = [np.nanmin(new_Zs_grid) * z_extent_factor, np.nanmax(new_Zs_grid) * z_extent_factor]
    z_label = "Electronic Energy (×" + str(1 / z_extent_factor) + 'kJ/mol)'
    print(z_label)

    # 可能需要调整X、Y的范围（横向缩放），否则画出来是条纵向的线
    mlab.clf()
    mlab.figure(bgcolor=(1, 1, 1), size=(1500, 1500), fgcolor=(0, 0, 0))

    # draw_a_line_in_mayavi((0,0,0),(0,0,3))
    # draw_a_line_in_mayavi((0, 0, 0), (0, 3, 0))
    # draw_a_line_in_mayavi((0, 0, 0), (3, 0, 0))

    mlab.imshow(new_Zs_grid_fix_color_range, extent=extent_XY + [0, 0], vmin=color_range[0] / 10, vmax=color_range[1] / 10)
    s = mlab.mesh(new_x_matrix, new_y_matrix, new_Zs_grid,
                  vmin=color_range[0], vmax=color_range[1],
                  extent=extent_XY + z_extent)

    # plot the trajectory
    for traceXY in traces_list:
        tube_radius = 0.003
        traceX, traceY = traceXY
        discard_first_n_points = 20  # to hide the starting point
        traceX = traceX[discard_first_n_points:]
        traceY = traceY[discard_first_n_points:]
        traceZ = [interpolation_function(x, traceY[count], new_x_list, new_y_list, new_Zs_grid) * z_extent_factor + tube_radius * 10 for count, x in
                  enumerate(traceX)]

        temp_X = []
        temp_Y = []
        temp_Z = []

        for trace_count, trace_pointX in enumerate(traceX):
            trace_pointY = traceY[trace_count]
            trace_pointZ = traceZ[trace_count]
            if not math.isnan(trace_pointZ):
                temp_X.append(trace_pointX)
                temp_Y.append(trace_pointY)
                temp_Z.append(trace_pointZ)

        traceX, traceY, traceZ = temp_X, temp_Y, temp_Z
        if traceX:
            if traceX[-1] < traceY[-1]:
                mlab.plot3d(traceX, traceY, traceZ, color=(0xfc / 0xff, 0x42 / 0xff, 0x42 / 0xff), representation='surface', tube_radius=tube_radius)
            else:
                mlab.plot3d(traceX, traceY, traceZ, color=(0xc8 / 0xff, 0x18 / 0xff, 0xfd / 0xff), representation='surface', tube_radius=tube_radius)

    mlab.colorbar(s, nb_colors=256, nb_labels=10, label_fmt='%.0f', orientation="vertical")

    x_ticks, y_ticks, z_ticks = get_appropriate_ticks(X_range), get_appropriate_ticks(Y_range), get_appropriate_ticks(z_extent)
    print(x_ticks, y_ticks, z_ticks)
    tick_counts = [int((a[1] - a[0]) / a[2]) for a in (x_ticks, y_ticks, z_ticks)]
    axis_object = mlab.axes(s, color=(0, 0, 0), extent=extent_XY + z_extent, nb_labels=5, ranges=x_ticks[0:2] + y_ticks[0:2] + z_ticks[0:2],
                            xlabel=X_Label, ylabel=Y_Label, zlabel=z_label)
    axis_object.label_text_property.font_family = 'arial'
    axis_object.label_text_property.font_size = 3
    axis_object.title_text_property.font_family = 'arial'
    axis_object.title_text_property.font_size = 3

    mlab.show()


if __name__ == "__main__":
    a = read_xlsx(r"E:\My_Program\Voldy_Lamp_Tests\Self_Lamp_Tests\F110\0 Selected UVI Table.xlsx", all_sheets=True)
    table = a['5.0 (4)']
    if "X↓" in table[0][0] and "Y→" in table[0][0]:
        X_along_row = False
    elif "X→" in table[0][0] and "Y↓" in table[0][0]:
        X_along_row = True
    XYZs = list_2D_with_header_to_xyz_triples(table, X_along_row=not X_along_row)
    my_grid = Grid(
        XYZ_triples=XYZs,
        do_interpolation=False,
        # interpolation_type='linear',
        # smoothing=50,
        Z_axis_colors=[(0, blue_color), (0.5, green_color), (5, orange_color), (15, red_color)],
        show_contour=True,
        contour_values=[0.5,1,5],
        show_contour_labels=True,
        contour_style=solid_line,
        contour_do_interpolation=True,
        contour_interpolation_type='b_spline',
        contour_smoothing=0.5,
        grid_line_X=True,
        grid_line_Y=True
    )
    
    Plot(my_grid, figure_title="Grid Plot Test", fig_size=(10,6)).pause()
