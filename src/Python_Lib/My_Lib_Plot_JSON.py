# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union, Tuple, List
import json
import numpy as np
import os

# Constants

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

def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if type(obj).__name__ == 'Tensor':
        try:
            return obj.tolist()
        except:
            pass

    # Handle torch.Tensor if torch is present or just by checking class name to avoid hard dependency if possible
    # But safer to check if it has .tolist() and is a tensor-like thing
    if hasattr(obj, 'tolist') and hasattr(obj, 'detach'): # Heuristic for torch.Tensor
        # If it's on GPU, we need detach().cpu() first usually, but .tolist() often handles it on newer torch?
        # Actually .tolist() on a cuda tensor works directly in modern pytorch.
        return obj.tolist()
        
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

@dataclass
class Curve_DataClass:
    """
    Data class representing a Curve object for JSON serialization and Plotting.
    
    This class mirrors the parameters available in `Curve`.
    
    """
    X: Optional[Sequence[float]] = None
    Y: Optional[Sequence[float]] = None
    Y_errorbar: Optional[Sequence[float]] = None
    Y_sampling_data: Optional[Sequence[Sequence[float]]] = None
    X_label: str = ""
    Y_label: str = ""
    plot_dot: Optional[bool] = None
    dot_format: str = point_mkr
    dot_color: Optional[str] = None
    dot_width: int = 5
    plot_curve: Optional[bool] = None
    curve_format: str = ""
    curve_color: Optional[str] = None
    curve_width: float = 0.8
    do_interpolation: bool = True
    interpolation_kind: str = "linear"
    interpolation_smoothing: float = 0
    interpolation_number: int = 5000
    curve_legend_color: str = ""
    curve_legend_format: str = ""
    fill_color: Optional[str] = None
    normalize_to: Union[None, Tuple[float, float], float] = None
    scale_factor: Optional[float] = None
    
    # Extra fields for flexibility allowing X_and_Y etc if user wants to use them like Curve()
    X_and_Y: Optional[Sequence] = None
    XY_pairs: Optional[Sequence] = None
    XYs: Optional[Sequence] = None
    interp1d_object: Optional[object] = None  # Not serializable, excluded from JSON
    fitted_function: Optional[object] = None  # Not serializable, excluded from JSON
    
    def to_dict(self):
        return dataclasses.asdict(self)

    def dump_to_JSON(self, filename, automatic_extension = True):
        if automatic_extension and not filename.lower().endswith('.curve'):
            filename = filename + '.Curve'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, default=_json_default, indent=4)
            

@dataclass
class Grid_DataClass:
    """
    Data class representing a Grid object for 3D/Heatmap plotting.
    
    """
    XYZ_triples: Optional[Sequence[Tuple[float, float, float]]] = None
    Xs: Optional[Sequence[float]] = None
    Ys: Optional[Sequence[float]] = None
    do_interpolation: bool = True
    interpolation_type: str = "linear"
    must_pass_through_points: bool = True
    smoothing: float = 0
    interpolation_density: int = 200
    Z_axis_colors: Optional[Sequence[Tuple[float, str]]] = None
    Z_linear_or_log: str = "linear"
    show_contour: bool = False
    contour_levels: Optional[int] = None
    contour_values: Optional[Sequence[float]] = None
    show_contour_labels: bool = False
    contour_style: str = dashed_line
    contour_width: float = 0.5
    contour_color: str = 'k'
    contour_do_interpolation: Optional[bool] = None
    contour_interpolation_type: Optional[str] = None
    contour_must_pass_through_points: Optional[bool] = None
    contour_smoothing: Optional[float] = None
    normalize_to: Union[float, Tuple[float, float], None] = None
    scale_factor: Optional[float] = None
    grid_line_X: Union[bool, Sequence[str], float, None] = None
    grid_line_Y: Union[bool, Sequence[str], float, None] = None
    show_colorbar: bool = False

    def to_dict(self):
        return dataclasses.asdict(self)

    def dump_to_JSON(self, filename, automatic_extension = True):
        if automatic_extension and not filename.lower().endswith('.grid'):
            filename = filename + '.Grid'

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, default=_json_default, indent=4)


@dataclass
class Plot_DataClass:
    """
    Data class representing a Plot window.
    """
    Curve_objects: Union[Curve_DataClass, Sequence[Curve_DataClass], None] = None
    Grid_objects: Union[Grid_DataClass, Sequence[Grid_DataClass], None] = None
    x_axis_label: str = "X"
    y_axis_label: str = "Y"
    fig_size_inch: Optional[Tuple[float, float]] = None
    fig_size_pixel: Optional[Tuple[int, int]] = None  # (width_px, height_px) overrides fig_size_inch
    window_size_pixel: Optional[Tuple[int, int]] = None  # (width_px, height_px) window size, overrides fig_size_pixel and fig_size_inch
    font_size: int = 8
    plot_legend: bool = True
    legend_font_size: Optional[int] = None
    x_lim: Optional[Tuple[Optional[float], Optional[float]]] = None
    y_lim: Optional[Tuple[Optional[float], Optional[float]]] = None
    x_log: bool = False
    y_log: bool = False
    show_grid: bool = False
    x_tick_positions: Optional[Sequence[float]] = None
    x_tick_texts: Optional[Sequence[str]] = None
    y_tick_positions: Optional[Sequence[float]] = None
    y_tick_texts: Optional[Sequence[str]] = None
    y2_tick_positions: Optional[Sequence[float]] = None
    y2_tick_texts: Optional[Sequence[str]] = None
    y2_axis_label: str = ""
    auto_color: Optional[bool] = None
    save_img_filepath: Optional[str] = None
    save_img_dpi: int = 3000
    use_chinese_font: bool = False
    shift_window: Union[Tuple[float, float], int, float] = (0, 0)
    multiple_plot_arrangement: Optional[Tuple[int, int]] = None
    figure_title: str = ""
    window_title: str = "Plot Points Window"
    keep_front: bool = False
    parent: Optional[object] = None  # Not serializable, excluded from JSON

    def to_dict(self):
        return dataclasses.asdict(self)

    def dump_to_JSON(self, filename, automatic_extension = True):
        if automatic_extension and not filename.lower().endswith('.plot'):
            filename = filename + '.Plot'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, default=_json_default, indent=4)


def check_consistency(Curve_Class, Grid_Class, Plot_Class):
    """
    Checks if the JSON data classes (Curve_JSON, Grid_JSON, Plot_JSON) 
    are consistent with the implementation classes (Curve, Grid, Plot).
    
    This function should be called with the actual classes from My_Lib_Plot.py
    """
    import inspect
    
    def check_one_pair(JSON_Class, Impl_Class, Class_Name):
        json_fields = {f.name for f in dataclasses.fields(JSON_Class)}
        
        # Get __init__ parameters of the implementation class
        sig = inspect.signature(Impl_Class.__init__)
        impl_params = set(sig.parameters.keys())
        impl_params.discard('self')
        impl_params.discard('kwargs') 
        impl_params.discard('args')
        
        # Check for missing fields in JSON class
        missing_in_json = impl_params - json_fields
        if missing_in_json:
            print(f"Warning: {Class_Name} implementation has parameters not in {Class_Name}_JSON: {missing_in_json}")
            
        # Check for extra fields in JSON class (not necessarily an error if JSON class has helpers, but good to know)
        extra_in_json = json_fields - impl_params
        # Filter commonly used extra fields or internal helpers if any
        # For now, just print usually
        if extra_in_json:
             pass # extra fields in JSON are fine usually (like X_and_Y aliases)

    check_one_pair(Curve_DataClass, Curve_Class, "Curve")
    check_one_pair(Grid_DataClass, Grid_Class, "Grid")
    check_one_pair(Plot_DataClass, Plot_Class, "Plot")


def Plot_json_to_image(json_file, output_format='png', dpi=600, output_path=None):
    """
    Convert a Plot JSON file to an image (PNG or SVG).
    
    Parameters
    ----------
    json_file : str
        Path to the JSON file
    output_format : str
        'png' or 'svg'
    dpi : int
        DPI for the output image (mainly affects PNG)
    output_path : str, optional
        Output path for the image. If None, uses the same location as json_file
        
    Returns
    -------
    str or None
        Path to the generated image, or None if conversion failed
    """
    # Validate input file exists
    if not os.path.isfile(json_file):
        print(f"Error: File not found: {json_file}")
        return None
    
    # Try to load and validate as Plot JSON
    try:
        # First try to load it as a dict to check structure
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if it looks like a Plot JSON (has expected keys)
        # A valid Plot JSON should have at least one of: Curve_objects, Grid_objects
        if not isinstance(data, dict):
            print(f"Error: Invalid JSON structure in {json_file}")
            return None
        
        has_curves = 'Curve_objects' in data
        has_grids = 'Grid_objects' in data
        
        if not (has_curves or has_grids):
            print(f"Warning: JSON file doesn't appear to be a Plot JSON (no Curve_objects or Grid_objects)")
            # Still try to proceed in case it's valid
            
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file {json_file}: {e}")
        return None
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return None
    
    # Determine output path
    if output_path is None:
        base_path = os.path.splitext(json_file)[0]
        output_path = f"{base_path}.{output_format}"
    
    # Try to create Plot and save
    try:
        # Import here to avoid circular import issues
        from My_Lib_Plot import Plot_from_JSON
        
        # Modify data to include save settings
        data['save_img_filepath'] = output_path
        data['save_img_dpi'] = dpi
        
        # Create Plot (this should trigger the save)
        plot = Plot_from_JSON(data)
        
        # Check if file was created
        if os.path.isfile(output_path):
            print(f"✓ Saved: {output_path}")
            return output_path
        else:
            print(f"Error: Failed to create image file {output_path}")
            return None
            
    except Exception as e:
        print(f"Error converting {json_file} to image: {e}")
        import traceback
        traceback.print_exc()
        return None


def batch_json_to_image():
    """
    Interactive batch conversion of JSON files to images.
    Prompts user for output format and file list, then converts each file.
    """
    # Import here to avoid issues if My_Lib is not available
    try:
        from My_Lib import get_input_with_while_cycle
    except ImportError:
        from My_Lib_Stock import get_input_with_while_cycle
    
    print("=== Batch JSON to Image Converter ===")
    print()
    
    # Ask for output format
    print("Select output format:")
    print("1. PNG")
    print("2. SVG")
    print("3. Both PNG and SVG")
    print()
    
    format_choice = input("Enter choice (1/2/3): ").strip()
    
    formats = []
    if format_choice == '1':
        formats = ['png']
    elif format_choice == '2':
        formats = ['svg']
    elif format_choice == '3':
        formats = ['png', 'svg']
    else:
        print("Invalid choice. Defaulting to PNG.")
        formats = ['png']
    
    print(f"Output format(s): {', '.join(formats).upper()}")
    print()
    
    # Ask for DPI (mainly for PNG)
    if 'png' in formats:
        dpi_input = input("Enter DPI for PNG (default 3000): ").strip()
        if dpi_input and dpi_input.isdigit():
            dpi = int(dpi_input)
        else:
            dpi = 3000
    else:
        dpi = 3000
    
    print()
    print("Enter JSON file paths (one per line, empty line to finish):")
    
    # Get file list using get_input_with_while_cycle
    file_list = get_input_with_while_cycle(
        break_condition=lambda x: not x.strip(),
        strip_quote=True
    )
    
    if not file_list:
        print("No files provided. Exiting.")
        return
    
    print()
    print(f"Processing {len(file_list)} file(s)...")
    print()
    
    # Process each file
    success_count = 0
    fail_count = 0
    
    for file_path in file_list:
        file_path = file_path.strip()
        if not file_path:
            continue
        
        print(f"\nProcessing: {file_path}")
        
        # Convert to absolute path if relative
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        
        # Try to convert to each format
        file_success = False
        for fmt in formats:
            result = Plot_json_to_image(file_path, output_format=fmt, dpi=dpi)
            if result:
                file_success = True
        
        if file_success:
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print()
    print("=== Conversion Complete ===")
    print(f"Successfully converted: {success_count}")
    print(f"Failed: {fail_count}")


if __name__ == '__main__':
    batch_json_to_image()
    
