import sys
import os
import pathlib
import json
import copy
import datetime
import math
import hashlib
import pandas as pd
import numpy as np
import re
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QComboBox,
                             QFileDialog, QScrollArea, QFrame, QGroupBox,
                             QFormLayout, QMessageBox, QSizePolicy,
                             QStackedWidget, QInputDialog, QSplitter,
                             QGridLayout, QDialog, QDialogButtonBox)
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

from Python_Lib.My_Lib_Plot import (Plot, Curve, Grid, DEFAULT_FIG_DPI,
                         Plot_from_JSON, Curve_from_JSON, Grid_from_JSON,
                         MARKER_DISPLAY, MARKER_DISPLAY_REV,
                         LINE_FORMAT_DISPLAY, LINE_FORMAT_DISPLAY_REV,
                         COLOR_DISPLAY, COLOR_DISPLAY_REV,
                         GRADIENT_COLORMAP_GROUPS, GRADIENT_COLORMAPS,
                         FILL_SCHEMES, format_value_at_pixel_resolution)
from Python_Lib.My_Lib_PyQt6 import ColumnEditTextEdit

# ─────────────────────── Constants ───────────────────────
MARKER_OPTIONS = [
    '', '.', ',', 'o', 'o_HOLLOW', 'v', '^', '<', '>',
    '1', '2', '3', '4', 's', 's_HOLLOW', 'p', '*',
    'h', 'H', '+', 'x', 'D', 'd', '|', '_',
]
LINE_FORMAT_OPTIONS = ['', '-', '--', '-.', ':']
COLOR_OPTIONS = [
    '', 'tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange',
    'tab:cyan', 'tab:olive', 'tab:pink', 'tab:brown', 'tab:grey', 'k', 'w',
]

# ─────────────────────── Horizontal Line Constants ───────────────────────
HORIZONTAL_LINE_COLOR = '#999999'
HORIZONTAL_LINE_WIDTH = 1.0
HORIZONTAL_LINE_FORMAT = '-'
HORIZONTAL_LINE_MARKER = ''
HORIZONTAL_LINE_LABEL_PREFIX = 'Baseline Y='
PLOT_LOAD_REPLACE = 'replace'
PLOT_LOAD_APPEND = 'append'
PLOT_LOAD_CANCEL = 'cancel'

# ─── Inline property editing in curve list ───
INLINE_PROPERTY_OPTIONS = [
    'None',
    'Label', 'Curve Color', 'Curve Width', 'Curve Format', 'Marker Format',
    'Dot Color', 'Dot Width', 'Dot Alpha', 'Dot Edge Color', 'Dot Edge Width',
    'Error Bar', 'EB Cap Size',
    'Interpolation', 'Interp Kind', 'Scale Factor', 'Offset', 'Normalize To', 'X -> -X',
    'Method', 'Contours', 'Density', 'Colorbar',
]

# Maps property name → (entry_attr, widget_type, combo_options_or_None)
# widget_type: 'line_edit', 'editable_combo', 'combo', 'checkable'
CURVE_INLINE_PROPS = {
    'Label':              ('inp_label',             'line_edit',      None),
    'Curve Color':        ('inp_color',             'editable_combo', list(COLOR_DISPLAY.keys())),
    'Curve Width':        ('inp_width',             'line_edit',      None),
    'Curve Format':       ('inp_fmt',               'editable_combo', list(LINE_FORMAT_DISPLAY.keys())),
    'Marker Format':      ('inp_marker',            'editable_combo', list(MARKER_DISPLAY.keys())),
    'Dot Color':          ('inp_dot_color',         'editable_combo', list(COLOR_DISPLAY.keys())),
    'Dot Width':          ('inp_dot_width',         'line_edit',      None),
    'Dot Alpha':          ('inp_dot_alpha',         'line_edit',      None),
    'Dot Edge Color':     ('inp_dot_edge_color',    'editable_combo', list(COLOR_DISPLAY.keys())),
    'Dot Edge Width':     ('inp_dot_edge_width',    'line_edit',      None),
    'Interpolation':      ('chk_interp',            'checkable',      None),
    'Interp Kind':        ('inp_interp_kind',       'combo',          ["linear", "cubic", "quadratic", "nearest", "zero", "slinear", "spline", "nearest-up", "previous", "next"]),
    'Interp Smoothing':   ('inp_interp_smoothing',  'line_edit',      None),
    'Interp Number':      ('inp_interp_number',     'line_edit',      None),
    'Scale Factor':       ('inp_scale',             'line_edit',      None),
    'Offset':             ('inp_offset',            'line_edit',      None),
    'Normalize To':       ('inp_normalize',         'line_edit',      None),
    'Legend Color':       ('inp_legend_color',      'editable_combo', list(COLOR_DISPLAY.keys())),
    'Legend Format':      ('inp_legend_format',     'editable_combo', list(LINE_FORMAT_DISPLAY.keys())),
    'Error Bar':           ('chk_errorbar',          'checkable',      None),
    'EB Cap Size':         ('inp_errorbar_capsize',  'line_edit',      None),
    'X -> -X':             ('chk_neg_x',             'checkable',      None),
}

GRID_INLINE_PROPS = {
    'Method':   ('inp_interp_type', 'combo',     ["linear", "cubic", "nearest", "multiquadric", "gaussian"]),
    'Contours': ('chk_contour',     'checkable', None),
    'Density':  ('inp_density',     'line_edit', None),
    'Colorbar': ('chk_colorbar',    'checkable', None),
}

# Properties that can be copied/pasted between curves, in UI layout order.
# Each entry: (display_name, attr_name, widget_type, section_name_or_None)
# section_name is used to insert section headers in the paste dialog.
CURVE_COPY_PROPS = [
    ('Format',           'inp_fmt',              'editable_combo', 'Curve'),
    ('Color',            'inp_color',            'editable_combo', None),
    ('Width',            'inp_width',            'line_edit',      None),
    ('Fill Color',       'inp_fill_color',       'editable_combo', None),
    ('Marker Format',    'inp_marker',           'editable_combo', 'Dot'),
    ('Dot Color',        'inp_dot_color',        'editable_combo', None),
    ('Dot Size',         'inp_dot_width',        'line_edit',      None),
    ('Dot Alpha',        'inp_dot_alpha',        'line_edit',      None),
    ('Dot Edge Color',   'inp_dot_edge_color',   'editable_combo', None),
    ('Dot Edge Width',   'inp_dot_edge_width',   'line_edit',      None),
    ('Error Bar',        'chk_errorbar',         'checkable',      'Error Bar'),
    ('EB Cap Size',      'inp_errorbar_capsize', 'line_edit',      None),
    ('Interpolation',    'chk_interp',           'checkable',      'Interpolation'),
    ('Interp Kind',      'inp_interp_kind',      'combo',          None),
    ('Interp Smoothing', 'inp_interp_smoothing', 'line_edit',      None),
    ('Interp Number',    'inp_interp_number',    'line_edit',      None),
    ('Scale Factor',     'inp_scale',            'line_edit',      'Other'),
    ('Offset',           'inp_offset',           'line_edit',      None),
    ('Normalize To',     'inp_normalize',        'line_edit',      None),
    ('Align Ref',        'inp_align_ref',        'line_edit',      None),
    ('Align Range',      'inp_align_range',      'line_edit',      None),
    ('Align Scale',      'chk_align_scale',      'checkable',      None),
    ('Align Offset',     'chk_align_offset',     'checkable',      None),
    ('Legend Color',     'inp_legend_color',     'editable_combo', None),
    ('Legend Format',    'inp_legend_format',    'editable_combo', None),
    ('X -> -X',          'chk_neg_x',            'checkable',      None),
    ('Label',            'inp_label',            'line_edit',      None),
]


def safe_eval_number(text, default=None):
    """Evaluate a numeric expression string safely (supports e.g. '1/5').
    Returns float or *default* on failure."""
    text = text.strip()
    if not text:
        return default
    try:
        return float(text)
    except (ValueError, TypeError):
        pass
    try:
        # Allow basic arithmetic: +, -, *, /, **, (), digits, dots
        if re.fullmatch(r'[\d.+\-*/() eE]+', text):
            val = float(eval(text, {"__builtins__": {}}, {}))
            return val
    except Exception:
        pass
    return default


def resolve_display_value(text, display_map):
    """Resolve a human-readable display name to its program value.
    Falls back to the raw text for custom values (e.g. hex colours)."""
    return display_map.get(text, text)


# ────────────── Curve-formula (inter-curve calculation) ──────────────
# A curve whose Data Source field contains {{code}} references — and which has
# no pasted data of its own (the data box is empty) — is a *computed curve*: its
# Y values are evaluated from the referenced curves' data.  Codes are the
# alphanumeric identifiers (any length) entered in the small box to the right of
# each curve in the list, e.g. the Data Source "({{A1}} - 2*{{B2}})/2" subtracts
# twice curve B2 from curve A1, /2.  The Label is kept free for the legend.
CURVE_CODE_REF_RE = re.compile(r'\{\{\s*([A-Za-z0-9]+)\s*\}\}')

# Upper bound on the point count a computed curve is interpolated to when the
# referenced curves do not already share an identical X grid.
FORMULA_MAX_POINTS = 100_000

# Math names exposed both unprefixed (sqrt(...)) and via math. inside a formula.
_FORMULA_MATH_NAMES = [
    'sqrt', 'exp', 'log', 'log10', 'log2', 'sin', 'cos', 'tan',
    'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh',
    'floor', 'ceil', 'fabs', 'degrees', 'radians', 'hypot',
]

# numpy equivalents (same names as the math module) for vectorised evaluation.
_FORMULA_VECTOR_MAP = {
    'sqrt': np.sqrt, 'exp': np.exp, 'log': np.log, 'log10': np.log10,
    'log2': np.log2, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
    'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
    'atan2': np.arctan2, 'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
    'floor': np.floor, 'ceil': np.ceil, 'fabs': np.fabs,
    'degrees': np.degrees, 'radians': np.radians, 'hypot': np.hypot,
    'pi': np.pi, 'e': np.e,
}


class _VectorMath:
    """Stand-in for the ``math`` module mapping ``math.<fn>`` to its numpy
    equivalent, so ``math.sqrt({{A1}})`` evaluates over the whole array."""

    def __getattr__(self, name):
        try:
            return _FORMULA_VECTOR_MAP[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _formula_vector_globals():
    g = {'__builtins__': {}, 'math': _VectorMath(), 'np': np, 'numpy': np,
         'abs': np.abs, 'min': np.minimum, 'max': np.maximum, 'pow': np.power}
    g.update(_FORMULA_VECTOR_MAP)
    return g


def _formula_scalar_globals():
    g = {'__builtins__': {}, 'math': math,
         'abs': abs, 'min': min, 'max': max, 'pow': pow}
    g.update({n: getattr(math, n) for n in _FORMULA_MATH_NAMES if hasattr(math, n)})
    g['pi'] = math.pi
    g['e'] = math.e
    return g


_FORMULA_VECTOR_GLOBALS = _formula_vector_globals()
_FORMULA_SCALAR_GLOBALS = _formula_scalar_globals()


def formula_to_expr(formula):
    """Replace each ``{{code}}`` in *formula* with a safe variable name.
    Returns ``(expr, var_map)`` where var_map maps UPPER-cased code → varname."""
    var_map = {}

    def repl(match):
        key = match.group(1).upper()
        if key not in var_map:
            var_map[key] = f"__v{len(var_map)}__"
        return var_map[key]

    return CURVE_CODE_REF_RE.sub(repl, formula), var_map


def eval_formula_pointwise(expr, var_map, values, n):
    """Evaluate *expr* (``{{code}}`` already replaced by var names) over *n*
    points.  ``values`` maps UPPER code → 1-D float array of length *n*.

    A single vectorised numpy eval is tried first; on any failure it falls back
    to a safe per-point scalar eval — exactly matching the documented behaviour
    of replacing each ``{{code}}`` with that point's value, e.g.
    ``eval('math.sqrt(1.5 - 2*2.3)')``."""
    local_vec = {var_map[k]: np.asarray(values[k], dtype=float) for k in var_map}
    try:
        result = eval(expr, _FORMULA_VECTOR_GLOBALS, local_vec)
        arr = np.asarray(result, dtype=float)
        if arr.ndim == 0:
            arr = np.full(n, float(arr))
        if arr.shape == (n,):
            return arr
    except Exception:
        pass
    # Per-point fallback (uses the real math module for full coverage).
    code_obj = compile(expr, '<curve-formula>', 'eval')
    cols = {k: np.asarray(values[k], dtype=float) for k in var_map}
    out = np.empty(n, dtype=float)
    for i in range(n):
        ns = {var_map[k]: float(cols[k][i]) for k in var_map}
        out[i] = float(eval(code_obj, _FORMULA_SCALAR_GLOBALS, ns))
    return out


# Profile directory: 3 levels up from this script → project root
PROFILE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src_Saves", "Plot_Editor_Profiles",
)
LAST_SETTINGS_FILE = os.path.join(PROFILE_DIR, "_last_settings.json")
WEIGHTS_HISTORY_FILE = os.path.join(PROFILE_DIR, "_weights_history.json")
MAX_WEIGHTS_HISTORY = 20


# ─────────────────────── Utility ─────────────────────────
def ensure_profile_dir():
    os.makedirs(PROFILE_DIR, exist_ok=True)


def get_preset_names():
    ensure_profile_dir()
    presets = []
    for f in sorted(os.listdir(PROFILE_DIR)):
        if f.startswith("preset_") and f.endswith(".json"):
            presets.append(f[7:-5])
    return presets


def load_weights_history():
    """Load list of recent weights file paths."""
    try:
        if os.path.exists(WEIGHTS_HISTORY_FILE):
            with open(WEIGHTS_HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data[:MAX_WEIGHTS_HISTORY]
    except Exception:
        pass
    return []


def save_weights_history(paths):
    """Save list of recent weights file paths."""
    try:
        ensure_profile_dir()
        with open(WEIGHTS_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(paths[:MAX_WEIGHTS_HISTORY], f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def add_to_weights_history(filepath):
    """Add a weights file to history (most recent first), dedup, limit 20."""
    filepath = os.path.normpath(filepath)
    history = load_weights_history()
    # Remove if already present
    history = [p for p in history if os.path.normpath(p) != filepath]
    history.insert(0, filepath)
    history = history[:MAX_WEIGHTS_HISTORY]
    save_weights_history(history)
    return history


def format_float_repr(val):
    """Format float using repr() to preserve precision, or int if integer."""
    try:
        f = float(val)
        if f != f:  # NaN
            return "NaN"
        if f == int(f) and abs(f) < 1e15:
            return str(int(f))
        return repr(f)
    except (ValueError, TypeError):
        return str(val)


def _is_plain_float(s):
    """Check if a string is a plain decimal float (not scientific notation)."""
    return '.' in s and 'e' not in s.lower() and 'nan' not in s.lower() and 'inf' not in s.lower()


def _all_plain_floats_in_column(str_rows, col):
    """Check if all values in a column are plain decimal floats."""
    for row in str_rows:
        if col < len(row):
            s = row[col]
            if s and not _is_plain_float(s) and '.' in s:
                return False  # has sci notation
            # integers without dots are fine, they align at the "ones" position
    return True


def format_data_to_aligned_text(data_rows):
    """Format row-wise data into aligned text with decimal-point alignment
    when all values in a column are plain floats (not scientific notation)."""
    if not data_rows:
        return ""

    # Convert all to string first using format_float_repr
    str_rows = [[format_float_repr(v) for v in row] for row in data_rows]

    if not str_rows:
        return ""
    num_cols = max(len(r) for r in str_rows)

    # Determine per-column whether to do decimal alignment
    # A column uses decimal alignment if all non-empty values are plain floats
    # (no scientific notation like 1e-5, no NaN/Inf)
    use_decimal_align = [False] * num_cols
    for col in range(num_cols):
        col_vals = [row[col] for row in str_rows if col < len(row) and row[col]]
        if not col_vals:
            continue
        # Check: all values are either integers (no dot) or plain floats (dot, no 'e')
        all_plain = all(
            ('.' not in s or _is_plain_float(s))
            for s in col_vals
        )
        # Only align if at least one value has a dot
        has_dot = any('.' in s for s in col_vals)
        use_decimal_align[col] = all_plain and has_dot

    # For decimal-aligned columns, compute max digits before and after the dot
    max_before = [0] * num_cols
    max_after = [0] * num_cols
    col_widths = [0] * num_cols

    for col in range(num_cols):
        if use_decimal_align[col]:
            for row in str_rows:
                if col < len(row):
                    s = row[col]
                    if '.' in s:
                        before, after = s.split('.', 1)
                        # Handle negative sign: before includes '-'
                        max_before[col] = max(max_before[col], len(before))
                        max_after[col] = max(max_after[col], len(after))
                    else:
                        # Integer: all digits are "before" the dot
                        max_before[col] = max(max_before[col], len(s))
        else:
            for row in str_rows:
                if col < len(row) and len(row[col]) > col_widths[col]:
                    col_widths[col] = len(row[col])

    # Format rows
    lines = []
    for row in str_rows:
        parts = []
        for i, s in enumerate(row):
            if i >= num_cols:
                break
            if use_decimal_align[i]:
                if '.' in s:
                    before, after = s.split('.', 1)
                    formatted = before.rjust(max_before[i]) + '.' + after.ljust(max_after[i])
                else:
                    # Integer: align at the dot position, pad right with spaces for the "after" part
                    formatted = s.rjust(max_before[i]) + ' ' + ' ' * max_after[i]
                parts.append(formatted)
            else:
                parts.append(s.rjust(col_widths[i]))
        # Use 4 spaces separation for clear column distinction
        lines.append("    ".join(parts))

    return "\n".join(lines)


# Matches a floating-point number (including scientific notation, optional sign).
_FLOAT_TOKEN_RE = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')


def extract_numeric_rows(text):
    """Extract rows of floats from free-form text.

    Any non-numeric characters between numbers are treated as separators, so
    the same routine handles X,Y / X Y / X\\tY / "X | Y" / aligned whitespace
    and mixed layouts. Returns a list of lists of floats, or None if any
    non-empty line yields fewer than 2 numbers (i.e. cannot be interpreted as
    tabular X/Y data)."""
    rows = []
    for raw_line in text.split('\n'):
        line = raw_line.strip()
        if not line:
            continue
        tokens = _FLOAT_TOKEN_RE.findall(line)
        if len(tokens) < 2:
            return None
        try:
            nums = [float(t) for t in tokens]
        except ValueError:
            return None
        rows.append(nums)
    return rows or None


def normalize_data_text(text):
    """Parse data in arbitrary X/Y layouts → aligned text.
    Returns normalised text or None on parse failure."""
    rows = extract_numeric_rows(text)
    if not rows:
        return None
    # Use the smallest column count across all rows so the table stays
    # rectangular even if some lines contain extra tokens.
    min_cols = min(len(r) for r in rows)
    if min_cols < 2:
        return None
    rows = [r[:min_cols] for r in rows]
    return format_data_to_aligned_text(rows)


def get_gradient_colors(n, cmap_name='Tab colors', inverse=False):
    """Return *n* evenly-spaced colour strings from *cmap_name*."""
    if n <= 0:
        return []
    tab = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange',
           'tab:cyan', 'tab:olive', 'tab:pink', 'tab:brown', 'tab:grey']
    if cmap_name == 'Tab colors':
        colors = [tab[i % len(tab)] for i in range(n)]
    else:
        try:
            import matplotlib.cm as mcm
            cmap = mcm.get_cmap(cmap_name)
            colors = []
            for i in range(n):
                rgba = cmap(i / max(n - 1, 1))
                colors.append('#{:02x}{:02x}{:02x}'.format(
                    int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)))
        except Exception:
            colors = [tab[i % len(tab)] for i in range(n)]
    if inverse:
        colors = colors[::-1]
    return colors


def _generate_colormap_pixmap(cmap_name, width=120, height=16):
    """Generate a QPixmap showing a horizontal color bar for the given colormap."""
    from PyQt6.QtGui import QPixmap, QColor, QPainter
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(255, 255, 255))
    painter = QPainter(pixmap)
    try:
        if cmap_name == 'Tab colors':
            tab = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange',
                   'tab:cyan', 'tab:olive', 'tab:pink', 'tab:brown', 'tab:grey']
            import matplotlib.colors as mcolors
            seg_w = max(width // len(tab), 1)
            for i, c in enumerate(tab):
                rgba = mcolors.to_rgba(c)
                painter.fillRect(i * seg_w, 0, seg_w, height,
                                 QColor(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)))
        else:
            import matplotlib.cm as mcm
            cmap = mcm.get_cmap(cmap_name)
            for x in range(width):
                rgba = cmap(x / max(width - 1, 1))
                painter.fillRect(x, 0, 1, height,
                                 QColor(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)))
    except Exception:
        pass
    finally:
        painter.end()
    return pixmap


# ─────────────────────── Styles ──────────────────────────
class GlobalStyles:
    STYLE_SHEET = """
    QLineEdit, QPushButton, QComboBox, QLabel {
        font-family: Arial;
        font-size: 10pt;
    }
    QLineEdit, QComboBox {
        min-height: 30px;
        max-height: 30px;
    }
    QPushButton {
        min-height: 30px;
    }
    """

    # Normal single-click button: clean default look
    NORMAL_BTN = """
        QPushButton {
            min-height: 30px; max-height: 30px;
        }
    """

    CHECKABLE_BTN = """
        QPushButton {
            background-color: #DDDDDD;
            border: 1.5px solid #888888;
            border-top: 1.5px solid #FFFFFF;
            border-left: 1.5px solid #FFFFFF;
            min-height: 28px; max-height: 28px;
        }
        QPushButton:checked {
            background-color: #90CAF9;
            border: 1.5px solid #FFFFFF;
            border-top: 1.5px solid #888888;
            border-left: 1.5px solid #888888;
        }
    """

    ENTRY_BTN = """
        QPushButton {
            text-align: left;
            border: 1px solid #CCCCCC;
            min-height: 25px;
            max-height: 25px;
        }
        QPushButton:checked {
            background-color: #BBDEFB;
            border: 2px solid #1976D2;
            font-weight: bold;
        }
        QPushButton:hover { background-color: #E3F2FD; }
    """

    DELETE_BTN = "background-color: #FFCDD2;"


# ──────────────────── DroppableTextEdit ──────────────────
class DroppableTextEdit(QtWidgets.QTextEdit):
    """QTextEdit that accepts CSV/XLSX file drops and forces plain-text paste."""
    file_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def insertFromMimeData(self, source):
        """Override to always paste as plain text."""
        if source.hasUrls():
            for url in source.urls():
                f = url.toLocalFile()
                if os.path.isfile(f):
                    self.file_dropped.emit(f)
        elif source.hasText():
            self.insertPlainText(source.text())

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                f = url.toLocalFile()
                if os.path.isfile(f):
                    self.file_dropped.emit(f)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


# ──────────────── DroppableComboBox ──────────────────────
class DroppableComboBox(QComboBox):
    """Editable QComboBox that accepts file drops and strips quotes from pasted paths."""
    file_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setAcceptDrops(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                f = url.toLocalFile()
                if os.path.isfile(f):
                    self.setCurrentText(f)
                    self.file_dropped.emit(f)
                    break
            event.acceptProposedAction()
        elif event.mimeData().hasText():
            text = event.mimeData().text().strip().strip('"').strip("'")
            self.setCurrentText(text)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def focusOutEvent(self, event):
        """Strip surrounding quotes from pasted paths on focus-out."""
        text = self.currentText().strip().strip('"').strip("'")
        if text != self.currentText():
            self.setCurrentText(text)
        super().focusOutEvent(event)


class AutoHeightTextEdit(QtWidgets.QTextEdit):
    """Read-only text edit that wraps anywhere and grows to fit its content."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setAcceptRichText(False)
        self.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.WidgetWidth)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        self.setMinimumWidth(0)

        text_option = self.document().defaultTextOption()
        text_option.setWrapMode(QtGui.QTextOption.WrapMode.WrapAnywhere)
        self.document().setDefaultTextOption(text_option)
        self.document().documentLayout().documentSizeChanged.connect(
            self._update_height
        )
        self._update_height()

    def _update_height(self, *_args):
        doc_margin = self.document().documentMargin()
        frame = self.frameWidth()
        height = self.document().size().height() + doc_margin * 2 + frame * 2
        self.setFixedHeight(max(32, int(height)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_height()


class BulkLabelEditorDialog(QDialog):
    """Non-modal, always-on-top window that edits every curve label at once —
    one label per line. Stays in front of the main editor; the caller locks
    label/order editing in the main window while it is open."""

    def __init__(self, labels, curve_count, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Curve Labels")
        # Float on top of the main window but do NOT block it (non-modal).
        self.setModal(False)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.resize(460, 520)
        layout = QVBoxLayout(self)

        info = QLabel(
            "One label per line — line N maps to curve N.\n"
            "Grey line numbers are past the last curve (ignored on OK).\n"
            "Hold Alt and drag for column (vertical) editing.")
        info.setStyleSheet("color: #555555; font-size: 9pt;")
        layout.addWidget(info)

        self.editor = ColumnEditTextEdit()
        self.editor.set_curve_count(curve_count)
        self.editor.setPlainText("\n".join(labels))
        layout.addWidget(self.editor, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_labels(self):
        return self.editor.toPlainText().split("\n")


class EntryButton(QPushButton):
    """Curve/Grid list button that can be dragged vertically to reorder layers.

    Plain clicks still select the entry; a click only fires when the press and
    release stay within the drag threshold. Dragging past the threshold
    suppresses the click and emits reorder signals instead."""

    drag_move = pyqtSignal(object, QtCore.QPoint)   # (button, global_pos)
    drag_drop = pyqtSignal(object, QtCore.QPoint)   # (button, global_pos)

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._press_pos = None
        self._dragging = False
        self._sync_min_width()

    def _sync_min_width(self):
        """Keep a minimum width that fits the (bold, when selected) label text
        so the button never clips it. When a label is wider than the list pane,
        this forces the scroll area to grow and reveal a horizontal scrollbar
        instead of truncating the text."""
        f = QtGui.QFont(self.font())
        f.setBold(True)
        text_w = QtGui.QFontMetrics(f).horizontalAdvance(self.text())
        self.setMinimumWidth(text_w + 20)  # + left padding / borders

    def setText(self, text):
        super().setText(text)
        self._sync_min_width()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
            self._dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (self._press_pos is not None
                and event.buttons() & Qt.MouseButton.LeftButton):
            moved = (event.position().toPoint() - self._press_pos).manhattanLength()
            if not self._dragging and moved >= QApplication.startDragDistance():
                self._dragging = True
            if self._dragging:
                self.drag_move.emit(self, event.globalPosition().toPoint())
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._press_pos = None
            self.setDown(False)
            self.drag_drop.emit(self, event.globalPosition().toPoint())
            event.accept()
            return   # swallow the release → suppress click/selection
        self._press_pos = None
        super().mouseReleaseEvent(event)


# ──────────────────── DataEntryWidget ────────────────────
class DataEntryWidget(QWidget):
    """Base class for a single Curve or Grid entry shown in the middle pane."""
    data_changed_signal = pyqtSignal()
    delete_requested_signal = pyqtSignal()
    label_changed_signal = pyqtSignal(str)

    def __init__(self, entry_type_name, parent=None):
        super().__init__(parent)
        self.entry_type_name = entry_type_name   # "Curve" / "Grid"
        self._is_enabled = True
        self.loaded_data = None
        self._file_path = None
        self._is_normalizing = False
        # True while the data box shows a computed (formula) curve's points:
        # read-only display, never parsed back into loaded_data.
        self._formula_display_active = False

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # ── Top row: Data Source info ──
        self.data_label = QLabel("<b>Data Source</b>")
        self.data_label.setWordWrap(True)
        root.addWidget(self.data_label)

        # User-editable annotation / source name (initialized from filename)
        self.inp_data_source = QLineEdit()
        self.inp_data_source.setPlaceholderText(
            "Data source / annotation — or, with the data box empty, a formula "
            "like ({{A1}}-2*{{B2}})/2")
        self.inp_data_source.editingFinished.connect(self.emit_change)
        root.addWidget(self.inp_data_source)
        root.addSpacing(8)

        # ── Horizontal: (left) data + paste | (right) properties ──
        # Use QSplitter so user can drag to resize text edit vs properties
        self.h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left column: label + text paste area
        left_w = QWidget()
        left = QVBoxLayout(left_w)
        left.setContentsMargins(0, 0, 6, 0)
        self.left_col_layout = left  # subclasses insert extra rows here

        # Label input above text paste (populated by subclass for CurveEntry)
        self.label_row_widget = QWidget()
        label_row = QHBoxLayout(self.label_row_widget)
        label_row.setContentsMargins(0, 0, 0, 4)
        label_row.setSpacing(4)
        self.label_row_widget.setVisible(False)  # hidden by default, subclass shows it
        left.addWidget(self.label_row_widget)

        self.text_paste = DroppableTextEdit()
        self.text_paste.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
        paste_font = QFont("Consolas", 9)
        self.text_paste.setFont(paste_font)
        self._paste_font_metrics = QtGui.QFontMetrics(paste_font)
        self.text_paste.setTabStopDistance(self._paste_font_metrics.averageCharWidth() * 16)
        self.text_paste.setPlaceholderText(
            "Paste data here (X,Y  or  X Y  or  X\\tY)\n"
            "or drop CSV / XLSX files …")
        # Don't connect textChanged for immediate updates
        self.text_paste.installEventFilter(self)
        self.text_paste.file_dropped.connect(self.load_file)
        left.addWidget(self.text_paste, 1)

        # "Clean up data" button: runs the same normalize + parse that
        # focus-out triggers, so the user can force a cleanup without
        # losing focus in the editor.
        self.btn_cleanup_data = QPushButton("Clean up data")
        self.btn_cleanup_data.setStyleSheet(GlobalStyles.NORMAL_BTN)
        self.btn_cleanup_data.clicked.connect(self._cleanup_data)
        left.addWidget(self.btn_cleanup_data)

        # Slot below the data editor where the controller docks the shared
        # Result log (only the active entry hosts it at any moment).
        self.bottom_slot = QWidget()
        self.bottom_slot_layout = QVBoxLayout(self.bottom_slot)
        self.bottom_slot_layout.setContentsMargins(0, 4, 6, 0)
        self.bottom_slot_layout.setSpacing(0)

        # Vertical splitter: data editor (+ button) on top, Result log
        # below. The ratio is user-adjustable; the controller persists it.
        self.left_v_splitter = QSplitter(Qt.Orientation.Vertical)
        self.left_v_splitter.addWidget(left_w)
        self.left_v_splitter.addWidget(self.bottom_slot)
        self.left_v_splitter.setStretchFactor(0, 1)
        self.left_v_splitter.setStretchFactor(1, 1)
        self.left_v_splitter.setMinimumWidth(100)

        # Right column: header label + property form (populated by subclass)
        right_container = QWidget()
        right_vbox = QVBoxLayout(right_container)
        right_vbox.setContentsMargins(0, 0, 0, 0)
        right_vbox.setSpacing(0)

        # Header label for section name (e.g. "Curve"), aligned with label_row_widget
        self.right_header_label = QLabel("")
        self.right_header_label.setFixedHeight(25)
        self.right_header_label.setVisible(False)
        right_vbox.addWidget(self.right_header_label)
        # Spacing to match label_row_widget bottom margin
        self.right_header_spacer = QWidget()
        self.right_header_spacer.setFixedHeight(4)
        self.right_header_spacer.setVisible(False)
        right_vbox.addWidget(self.right_header_spacer)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_w = QWidget()
        self.form_layout = QFormLayout(right_w)
        self.form_layout.setContentsMargins(0, 0, 12, 0)
        right_scroll.setWidget(right_w)
        right_scroll.setMinimumWidth(200)
        right_vbox.addWidget(right_scroll, 1)

        self.h_splitter.addWidget(self.left_v_splitter)
        self.h_splitter.addWidget(right_container)
        self.h_splitter.setStretchFactor(0, 0)  # text pane: don't stretch by default
        self.h_splitter.setStretchFactor(1, 1)  # property pane: stretch
        self.h_splitter.setChildrenCollapsible(False)
        root.addWidget(self.h_splitter, 1)

    # ── helpers for subclass setup ──
    def add_line_edit(self, label, default=""):
        le = QLineEdit(str(default))
        le.editingFinished.connect(self.emit_change)
        self.form_layout.addRow(label, le)
        return le

    def add_editable_combo(self, label, options, default=""):
        cb = QComboBox()
        cb.setEditable(True)
        cb.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        cb.addItems(options)
        cb.setMaxVisibleItems(min(len(options), 20))
        if default:
            cb.setCurrentText(default)
        cb.lineEdit().editingFinished.connect(self.emit_change)
        cb.activated.connect(self.emit_change)
        self.form_layout.addRow(label, cb)
        return cb

    def add_combo(self, label, options, default=None):
        cb = QComboBox()
        cb.addItems(options)
        cb.setMaxVisibleItems(min(len(options), 20))
        if default and default in options:
            cb.setCurrentText(default)
        cb.activated.connect(self.emit_change)
        self.form_layout.addRow(label, cb)
        return cb

    def add_checkable_button(self, label, checked=False):
        btn = QPushButton(label)
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setStyleSheet(GlobalStyles.CHECKABLE_BTN)
        btn.toggled.connect(self.emit_change)
        self.form_layout.addRow(btn)
        return btn

    # ── signals / data ──
    def emit_change(self):
        self.data_changed_signal.emit()

    def eventFilter(self, obj, event):
        if obj is self.text_paste and event.type() == QtCore.QEvent.Type.FocusOut:
            # Skip when the box is a read-only display (formula result or
            # scale/offset-processed data) — its content is derived output
            # and must never be parsed back into loaded_data.
            if not self._formula_display_active and not self.text_paste.isReadOnly():
                self._try_normalize_text()
                self.parse_text_data()  # Parse and emit change when focus leaves
        return super().eventFilter(obj, event)

    def _try_normalize_text(self):
        if self._is_normalizing:
            return
        text = self.text_paste.toPlainText()
        if not text.strip():
            return
        normalized = normalize_data_text(text)
        if normalized is not None and normalized != text:
            self._is_normalizing = True
            self.text_paste.blockSignals(True)
            self.text_paste.setPlainText(normalized)
            self.text_paste.blockSignals(False)
            self._is_normalizing = False

    def _cleanup_data(self):
        """Manually trigger the same normalize + parse that focus-out runs."""
        self._try_normalize_text()
        self.parse_text_data()

    def parse_text_data(self):
        text = self.text_paste.toPlainText()
        if not text.strip():
            self.loaded_data = None
            self.data_label.setText("<b>Data Source</b>")
            self.emit_change()
            return
        rows = extract_numeric_rows(text)
        if not rows:
            self.data_label.setText("<b>Data Source</b> (parse error)")
            return
        min_cols = min(len(r) for r in rows)
        if min_cols < 2:
            self.data_label.setText("<b>Data Source</b> (parse error)")
            return
        try:
            df = pd.DataFrame([r[:min_cols] for r in rows])
            self.loaded_data = df
            self.data_label.setText(
                f"<b>Data Source</b> ({df.shape[0]} × {df.shape[1]})")
            self.emit_change()
        except Exception:
            self.data_label.setText("<b>Data Source</b> (parse error)")

    def load_file(self, filepath):
        try:
            self._file_path = filepath
            ext = os.path.splitext(filepath)[1].lower()
            if ext in ('.xlsx', '.xls'):
                self.loaded_data = pd.read_excel(filepath, header=None)
            else:
                self.loaded_data = pd.read_csv(
                    filepath, sep=None, engine='python', header=None)
            self.data_label.setText(
                f"<b>Data Source</b> ({self.loaded_data.shape[0]} × "
                f"{self.loaded_data.shape[1]})")
            if not self.inp_data_source.text().strip():
                self.inp_data_source.blockSignals(True)
                self.inp_data_source.setText(os.path.basename(filepath))
                self.inp_data_source.blockSignals(False)
            self._display_data_as_text()
            self.emit_change()
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    def _display_data_as_text(self):
        """Show current loaded_data in normalised aligned format inside text_paste."""
        if self.loaded_data is None:
            return

        # Prepare data rows for formatter
        data_rows = []
        for _, row in self.loaded_data.iterrows():
            # Get valid values from the row
            data_rows.append([v for v in row.values if pd.notna(v)])

        formatted_text = format_data_to_aligned_text(data_rows)

        self._is_normalizing = True
        self.text_paste.blockSignals(True)
        self.text_paste.setPlainText(formatted_text)
        self.text_paste.blockSignals(False)
        self._is_normalizing = False

        QTimer.singleShot(0, lambda: self._adjust_text_paste_width(formatted_text))

    def _adjust_text_paste_width(self, text):
        """Set the text paste pane width based on the longest line, with some margin."""
        if not text:
            return
        lines = text.split('\n')
        max_line_len = max(len(line) for line in lines) if lines else 0
        if max_line_len <= 0:
            return
        # Calculate pixel width: character width * max_chars + margins
        char_width = self._paste_font_metrics.averageCharWidth()
        # Add margin: scrollbar (~20px) + document margins + some padding
        needed_width = int(char_width * (max_line_len + 4)) + 40
        needed_width = max(needed_width, 100)  # minimum
        needed_width = min(needed_width, 600)  # cap at a reasonable size

        # Set the splitter sizes: text pane gets needed_width, property pane gets the rest
        total = self.h_splitter.width()
        if total > 0:
            prop_width = max(200, total - needed_width)
            self.h_splitter.setSizes([needed_width, prop_width])

    def get_data_arrays(self):
        if self.loaded_data is not None:
            df = self.loaded_data.dropna()
            ncols = df.shape[1]
            if ncols == 1:
                return np.arange(len(df)), df.iloc[:, 0].values, None
            elif ncols == 2:
                return df.iloc[:, 0].values, df.iloc[:, 1].values, None
            elif ncols >= 3:
                return (df.iloc[:, 0].values,
                        df.iloc[:, 1].values,
                        df.iloc[:, 2].values)
        return None, None, None


# ──────────────────────── CurveEntry ─────────────────────
class CurveEntry(DataEntryWidget):
    def __init__(self, parent=None):
        super().__init__("Curve", parent)
        self.curve_code = ""        # short alias (e.g. A1) used in formulas
        self._compute_hook = None   # set by controller: entry -> (X, Y) | None
        self._align_hook = None     # set by controller: (entry, X, Y) -> (scale, offset) | None
        self._formula_cache = None  # (cache_key, (X, Y)) memo for computed curves
        self._computed_xy = None    # (X, Y) currently shown in the data box (for Save CSV)
        self._computed_text = None  # exact text of the computed display (detects user overwrite)
        self._displayed_xy = None   # identity memo: skip re-formatting an unchanged result
        self._normal_placeholder = ""  # data-box placeholder to restore on exit
        # ── Original / Processed data views ──
        self._view_mode = 'original'      # which view the data box shows
        self._processed_xy = None         # (X, Y) after scale/offset, or None
        self._had_processed = False       # previous-render flag (drives auto-switch)
        self._processed_display_active = False  # data box currently shows processed data
        self._displayed_processed_xy = None     # identity memo for the processed text
        self._last_is_formula = False     # last render: entry was a computed curve
        self._last_computed_xy = None     # last computed (X, Y) for view switching
        self._scale_user_backup = None    # manual Scale Factor text while align owns the box
        self._offset_user_backup = None   # manual Offset text while align owns the box
        self._setup_ui()
        self._update_errorbar_status()  # grey out Error Bar until data has EB columns

    def _setup_ui(self):
        # Replace the QFormLayout with a QGridLayout for precise control
        form_widget = self.form_layout.parentWidget()
        # Remove the old form layout
        QtWidgets.QWidget().setLayout(self.form_layout)
        grid = QGridLayout(form_widget)
        grid.setContentsMargins(0, 0, 12, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        self.form_layout = grid  # keep attribute for compatibility

        row = 0

        # ── Label (in the left column, above text paste) ──
        lbl_label = QLabel("Label")
        self.label_row_widget.layout().addWidget(lbl_label)
        self.inp_label = self._mk_line_edit("")
        self.inp_label.editingFinished.connect(
            lambda: self.label_changed_signal.emit(self.inp_label.text()))
        self.inp_label.setPlaceholderText("Curve label (shown in the legend)")
        self.label_row_widget.layout().addWidget(self.inp_label, 1)
        self.label_row_widget.setVisible(True)

        # ── Original / Processed view toggle (below Label, above data box) ──
        # Shown only when the curve has two data sets: the original input
        # (user data or formula result) and the scale/offset-processed data.
        self.view_toggle_widget = QWidget()
        vt_row = QHBoxLayout(self.view_toggle_widget)
        vt_row.setContentsMargins(0, 0, 0, 4)
        vt_row.setSpacing(4)
        self.btn_view_original = QPushButton("Original")
        self.btn_view_original.setCheckable(True)
        self.btn_view_original.setStyleSheet(GlobalStyles.CHECKABLE_BTN)
        self.btn_view_original.clicked.connect(
            lambda: self._set_view_mode('original'))
        self.btn_view_processed = QPushButton("Processed")
        self.btn_view_processed.setCheckable(True)
        self.btn_view_processed.setStyleSheet(GlobalStyles.CHECKABLE_BTN)
        self.btn_view_processed.clicked.connect(
            lambda: self._set_view_mode('processed'))
        vt_row.addWidget(self.btn_view_original)
        vt_row.addWidget(self.btn_view_processed)
        self.left_col_layout.insertWidget(
            self.left_col_layout.indexOf(self.label_row_widget) + 1,
            self.view_toggle_widget)
        self.view_toggle_widget.setVisible(False)

        # Hint: an empty curve becomes a *computed curve* when the Data Source
        # field holds a formula referencing other curves by their codes (the
        # small box to the right of each curve in the list).  The Label is kept
        # free for the legend.
        self.text_paste.setPlaceholderText(
            "Paste data here (X,Y  or  X Y  or  X\\tY) or drop CSV / XLSX files …\n\n"
            "Computed curve: leave this data box empty and type a formula in the\n"
            "Data Source field above, e.g.  ({{A1}} - 2*{{B2}})/2  or  math.sqrt({{A1}}) .\n"
            "{{A1}} means the curve whose code (the small box to the right of\n"
            "each curve in the list) is A1. Codes are case-insensitive.\n"
            "Supports parentheses, +-*/**, and math functions (sqrt, sin, …).")

        # ── Curve section header (above scroll, aligned with Label row) ──
        self.right_header_label.setText("<b>Curve</b>")
        self.right_header_label.setVisible(True)
        self.right_header_spacer.setVisible(True)

        # Format [____]
        grid.addWidget(QLabel("Format"), row, 0)
        self.inp_fmt = self._mk_editable_combo(list(LINE_FORMAT_DISPLAY.keys()), "Solid (-)")
        grid.addWidget(self.inp_fmt, row, 1, 1, 3)
        row += 1

        # Color [____]  Width [____]
        grid.addWidget(QLabel("Color"), row, 0)
        self.inp_color = self._mk_editable_combo(list(COLOR_DISPLAY.keys()), "Blue")
        grid.addWidget(self.inp_color, row, 1)
        grid.addWidget(QLabel("Width"), row, 2)
        self.inp_width = self._mk_line_edit("0.8")
        grid.addWidget(self.inp_width, row, 3)
        row += 1

        # Fill Color [____]
        grid.addWidget(QLabel("Fill Color"), row, 0)
        fill_options = [''] + list(COLOR_DISPLAY.keys())[1:] + list(FILL_SCHEMES.keys())
        self.inp_fill_color = self._mk_editable_combo(fill_options, "")
        grid.addWidget(self.inp_fill_color, row, 1, 1, 3)
        row += 1

        # ── Dot section separator ──
        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine); sep2.setFrameShadow(QFrame.Shadow.Sunken)
        grid.addWidget(sep2, row, 0, 1, 4); row += 1
        dot_lbl = QLabel("<b>Dot</b>")
        dot_lbl.setFixedHeight(25)
        grid.addWidget(dot_lbl, row, 0, 1, 4); row += 1

        # Format [____]  (was "Marker Format")
        grid.addWidget(QLabel("Format"), row, 0)
        self.inp_marker = self._mk_editable_combo(list(MARKER_DISPLAY.keys()), "")
        grid.addWidget(self.inp_marker, row, 1, 1, 3)
        row += 1

        # Color [____]  Size [____]
        grid.addWidget(QLabel("Color"), row, 0)
        self.inp_dot_color = self._mk_editable_combo(list(COLOR_DISPLAY.keys()), "")
        grid.addWidget(self.inp_dot_color, row, 1)
        grid.addWidget(QLabel("Size"), row, 2)
        self.inp_dot_width = self._mk_line_edit("5")
        grid.addWidget(self.inp_dot_width, row, 3)
        row += 1

        # Alpha [____]
        grid.addWidget(QLabel("Alpha"), row, 0)
        self.inp_dot_alpha = self._mk_line_edit("")
        grid.addWidget(self.inp_dot_alpha, row, 1, 1, 3)
        row += 1

        # Edge | Color [____]  Width [____]
        grid.addWidget(QLabel("Edge Color"), row, 0)
        self.inp_dot_edge_color = self._mk_editable_combo(list(COLOR_DISPLAY.keys()), "")
        grid.addWidget(self.inp_dot_edge_color, row, 1)
        grid.addWidget(QLabel("Width"), row, 2)
        self.inp_dot_edge_width = self._mk_line_edit("")
        grid.addWidget(self.inp_dot_edge_width, row, 3)
        row += 1

        # Legend | Color [___]  Format [___]
        grid.addWidget(QLabel("Legend Color"), row, 0)
        self.inp_legend_color = self._mk_editable_combo(list(COLOR_DISPLAY.keys()), "")
        grid.addWidget(self.inp_legend_color, row, 1)
        grid.addWidget(QLabel("Format"), row, 2)
        self.inp_legend_format = self._mk_editable_combo(list(LINE_FORMAT_DISPLAY.keys()), "")
        grid.addWidget(self.inp_legend_format, row, 3)
        row += 1

        # ── Error Bar section separator ──
        sep_ebar = QFrame(); sep_ebar.setFrameShape(QFrame.Shape.HLine); sep_ebar.setFrameShadow(QFrame.Shadow.Sunken)
        grid.addWidget(sep_ebar, row, 0, 1, 4); row += 1

        # [Error Bar]  Cap Size [___]  — button spans cols 0-1, label col 2, input col 3
        self.chk_errorbar = QPushButton("Error Bar")
        self.chk_errorbar.setCheckable(True)
        self.chk_errorbar.setChecked(True)
        self.chk_errorbar.setStyleSheet(GlobalStyles.CHECKABLE_BTN)
        self.chk_errorbar.toggled.connect(self.emit_change)
        grid.addWidget(self.chk_errorbar, row, 0, 1, 2)
        grid.addWidget(QLabel("Cap Size"), row, 2)
        self.inp_errorbar_capsize = self._mk_line_edit("2")
        grid.addWidget(self.inp_errorbar_capsize, row, 3)
        row += 1

        # ── Interpolation section separator ──
        sep3 = QFrame(); sep3.setFrameShape(QFrame.Shape.HLine); sep3.setFrameShadow(QFrame.Shadow.Sunken)
        grid.addWidget(sep3, row, 0, 1, 4); row += 1

        # [Interpolation] checkable button
        self.chk_interp = QPushButton("Interpolation")
        self.chk_interp.setCheckable(True)
        self.chk_interp.setChecked(True)
        self.chk_interp.setStyleSheet(GlobalStyles.CHECKABLE_BTN)
        self.chk_interp.toggled.connect(self.emit_change)
        grid.addWidget(self.chk_interp, row, 0, 1, 4)
        row += 1

        # Interp Kind [___]  Smoothing [___]
        grid.addWidget(QLabel("Interp Kind"), row, 0)
        self.inp_interp_kind = self._mk_combo(
            ["linear", "cubic", "quadratic", "nearest", "zero", "slinear", "spline", "nearest-up", "previous", "next"], "linear")
        grid.addWidget(self.inp_interp_kind, row, 1)
        grid.addWidget(QLabel("Smoothing"), row, 2)
        self.inp_interp_smoothing = self._mk_line_edit("0")
        grid.addWidget(self.inp_interp_smoothing, row, 3)
        row += 1

        # Interp Number [___]  Offset [___]
        grid.addWidget(QLabel("Interp Number"), row, 0)
        self.inp_interp_number = self._mk_line_edit("5000")
        grid.addWidget(self.inp_interp_number, row, 1)
        grid.addWidget(QLabel("Offset"), row, 2)
        self.inp_offset = self._mk_line_edit("")
        self.inp_offset.setToolTip(
            "Vertical offset added to the Y data (applied after Scale Factor).")
        grid.addWidget(self.inp_offset, row, 3)
        row += 1

        # Scale Factor [___]  Normalize to [___]
        grid.addWidget(QLabel("Scale Factor"), row, 0)
        self.inp_scale = self._mk_line_edit("")
        grid.addWidget(self.inp_scale, row, 1)
        grid.addWidget(QLabel("Normalize to"), row, 2)
        self.inp_normalize = self._mk_line_edit("")
        grid.addWidget(self.inp_normalize, row, 3)
        row += 1

        # Align With | ref [Line Variable {{a}}]  range [(0,100)]  [Scale][Offset]
        # Solve the best scale factor and/or vertical offset so this curve's data
        # within the range best overlaps the referenced ("Line Variable") curve.
        grid.addWidget(QLabel("Align With"), row, 0)
        self.inp_align_ref = self._mk_line_edit("")
        self.inp_align_ref.setPlaceholderText("Line Variable {{a}}")
        self.inp_align_ref.setToolTip(
            "Reference another curve by its code, e.g. {{a}}.  The best scale\n"
            "and/or vertical offset is solved (linear least squares) so this\n"
            "curve best overlaps it over the range to the right.")
        grid.addWidget(self.inp_align_ref, row, 1)
        self.inp_align_range = self._mk_line_edit("")
        self.inp_align_range.setPlaceholderText("Range: 0,100")
        self.inp_align_range.setToolTip(
            "X range over which to match, e.g. 0,100.  Leave blank to use the\n"
            "full overlapping X range of the two curves.")
        # Don't let the line edit's wide default sizeHint stretch column 2 (a
        # narrow label column shared with "Normalize to", "Cap Size", …).
        self.inp_align_range.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            self.inp_align_range.sizePolicy().verticalPolicy())
        grid.addWidget(self.inp_align_range, row, 2)
        align_btns = QHBoxLayout()
        align_btns.setContentsMargins(0, 0, 0, 0)
        align_btns.setSpacing(2)
        self.chk_align_scale = QPushButton("Scale")
        self.chk_align_scale.setCheckable(True)
        self.chk_align_scale.setStyleSheet(GlobalStyles.CHECKABLE_BTN)
        self.chk_align_scale.setToolTip("Solve for the best scale factor.")
        self.chk_align_scale.toggled.connect(self.emit_change)
        self.chk_align_offset = QPushButton("Offset")
        self.chk_align_offset.setCheckable(True)
        self.chk_align_offset.setStyleSheet(GlobalStyles.CHECKABLE_BTN)
        self.chk_align_offset.setToolTip("Solve for the best vertical offset.")
        self.chk_align_offset.toggled.connect(self.emit_change)
        align_btns.addWidget(self.chk_align_scale)
        align_btns.addWidget(self.chk_align_offset)
        align_btns_container = QWidget()
        align_btns_container.setLayout(align_btns)
        grid.addWidget(align_btns_container, row, 3)
        row += 1

        # X → -X
        self.chk_neg_x = QPushButton("X → -X")
        self.chk_neg_x.setCheckable(True)
        self.chk_neg_x.setChecked(False)
        self.chk_neg_x.setStyleSheet(GlobalStyles.CHECKABLE_BTN)
        self.chk_neg_x.toggled.connect(self.emit_change)
        grid.addWidget(self.chk_neg_x, row, 0, 1, 4)
        row += 1

        # ── Integration section separator ──
        sep4 = QFrame(); sep4.setFrameShape(QFrame.Shape.HLine); sep4.setFrameShadow(QFrame.Shadow.Sunken)
        grid.addWidget(sep4, row, 0, 1, 4); row += 1
        grid.addWidget(QLabel("<b>Integration</b>"), row, 0, 1, 4); row += 1

        # Start [___]  Stop [___]
        grid.addWidget(QLabel("Start"), row, 0)
        self.inp_integ_start = self._mk_line_edit("")
        grid.addWidget(self.inp_integ_start, row, 1)
        grid.addWidget(QLabel("Stop"), row, 2)
        self.inp_integ_stop = self._mk_line_edit("")
        grid.addWidget(self.inp_integ_stop, row, 3)
        row += 1

        # Weights file with browse
        grid.addWidget(QLabel("Weights"), row, 0)
        weights_row = QHBoxLayout()
        self.inp_integ_weights = DroppableComboBox()
        self.inp_integ_weights.setMaxVisibleItems(20)
        for path in load_weights_history():
            self.inp_integ_weights.addItem(path)
        self.inp_integ_weights.file_dropped.connect(
            lambda f: (add_to_weights_history(f),
                        self._refresh_weights_combo(f)))
        weights_row.addWidget(self.inp_integ_weights, stretch=1)
        btn_browse_weights = QPushButton("...")
        btn_browse_weights.setFixedWidth(30)
        btn_browse_weights.clicked.connect(self._browse_weights_file)
        weights_row.addWidget(btn_browse_weights)
        weights_container = QWidget()
        weights_container.setLayout(weights_row)
        weights_row.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(weights_container, row, 1, 1, 3)
        row += 1

        # Normalize Integration to
        grid.addWidget(QLabel("Normalize to"), row, 0)
        self.inp_integ_normalize = self._mk_line_edit("")
        grid.addWidget(self.inp_integ_normalize, row, 1, 1, 3)
        row += 1

        # Compute Integration button
        self.btn_integrate = QPushButton("Compute Integration")
        self.btn_integrate.setStyleSheet(GlobalStyles.NORMAL_BTN)
        grid.addWidget(self.btn_integrate, row, 0, 1, 4)
        row += 1

        # Vertical spacer to push content up
        grid.setRowStretch(row, 1)
        row += 1

        # Column stretch: label columns fixed, input columns stretch
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)
        grid.setColumnStretch(3, 1)

    # ── Private widget creators (don't add to layout) ──
    def _mk_line_edit(self, default=""):
        le = QLineEdit(str(default))
        le.editingFinished.connect(self.emit_change)
        return le

    def _mk_editable_combo(self, options, default=""):
        cb = QComboBox()
        cb.setEditable(True)
        cb.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        cb.addItems(options)
        cb.setMaxVisibleItems(min(len(options), 20))
        if default:
            cb.setCurrentText(default)
        cb.lineEdit().editingFinished.connect(self.emit_change)
        cb.activated.connect(self.emit_change)
        return cb

    def _mk_combo(self, options, default=None):
        cb = QComboBox()
        cb.addItems(options)
        cb.setMaxVisibleItems(min(len(options), 20))
        if default and default in options:
            cb.setCurrentText(default)
        cb.activated.connect(self.emit_change)
        return cb

    def _browse_weights_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Weights File", "",
            "Data Files (*.csv *.xlsx);;All Files (*.*)")
        if path:
            self.inp_integ_weights.setCurrentText(path)
            add_to_weights_history(path)
            self._refresh_weights_combo(path)

    def _refresh_weights_combo(self, current_path=""):
        """Reload weights history into dropdown, keeping current_path selected."""
        self.inp_integ_weights.blockSignals(True)
        self.inp_integ_weights.clear()
        for p in load_weights_history():
            self.inp_integ_weights.addItem(p)
        if current_path:
            self.inp_integ_weights.setCurrentText(current_path)
        self.inp_integ_weights.blockSignals(False)

    def get_errorbar_data(self):
        """Extract error bar data from loaded data columns 3 and 4.

        Returns
        -------
        errorbar : list or None
            - None if no error bar data or error bar is disabled
            - 1D list for symmetric error bar (3 columns: col3 = SE)
            - [lower, upper] for asymmetric error bar (4 columns: col3 = lower, col4 = upper)
        """
        if not self.chk_errorbar.isChecked():
            return None
        if self.loaded_data is None:
            return None
        df = self.loaded_data.dropna()
        ncols = df.shape[1]
        if ncols == 3:
            return df.iloc[:, 2].values.tolist()
        elif ncols >= 4:
            return [df.iloc[:, 2].values.tolist(), df.iloc[:, 3].values.tolist()]
        return None

    def _update_errorbar_status(self):
        """Enable the Error Bar button only when the data actually carries
        error-bar columns (≥3 columns); otherwise grey it out so it cannot be
        checked.  The kind of error bar is shown via the button's tooltip."""
        status = ""
        if self.loaded_data is not None:
            ncols = self.loaded_data.dropna().shape[1]
            if ncols == 3:
                status = "symmetric"
            elif ncols >= 4:
                status = "asymmetric"
        has_eb = bool(status)
        self.chk_errorbar.setEnabled(has_eb)
        self.chk_errorbar.setToolTip(
            f"Error bars: {status}" if has_eb else "No error-bar data (needs ≥3 data columns)")

    def emit_change(self):
        self._update_errorbar_status()
        super().emit_change()

    def get_formula(self):
        """If this entry is a *computed curve* — the data window is completely
        empty and the Data Source field holds a {{code}} formula — return that
        formula text; else None.  The Label is never used as a formula so it
        stays free for the legend."""
        if self.loaded_data is not None:
            return None
        # In formula-display mode the data box holds the computed result, not
        # user-pasted data, so its content must not disqualify the formula.
        if not self._formula_display_active and self.text_paste.toPlainText().strip():
            return None
        formula = self.inp_data_source.text()
        if formula and CURVE_CODE_REF_RE.search(formula):
            return formula
        return None

    # ── computed-curve display (read-only data box + Save CSV) ──
    def _set_formula_display(self, active):
        """Enter / leave the computed-curve display mode: the data box becomes
        a read-only (grey) view of the computed points — still selectable and
        copyable — and the "Clean up data" button turns into "Save CSV"."""
        if active == self._formula_display_active:
            return
        self._formula_display_active = active
        self.text_paste.setReadOnly(active)
        if active:
            self._normal_placeholder = self.text_paste.placeholderText()
            self.text_paste.setPlaceholderText(
                "Computed curve: the formula's data points will appear here.")
            self.text_paste.setStyleSheet(
                "QTextEdit { background-color: #EBEBEB; }")
            self.btn_cleanup_data.setText("Save CSV")
        else:
            self.text_paste.setStyleSheet("")
            self.text_paste.setPlaceholderText(self._normal_placeholder)
            self.btn_cleanup_data.setText("Clean up data")
            # Clear the displayed points only if they are still ours — a file
            # load may already have replaced them with real data.
            if (self._computed_text is not None
                    and self.text_paste.toPlainText() == self._computed_text):
                self._is_normalizing = True
                self.text_paste.blockSignals(True)
                self.text_paste.clear()
                self.text_paste.blockSignals(False)
                self._is_normalizing = False
                self.data_label.setText("<b>Data Source</b>")
            self._computed_xy = None
            self._computed_text = None
            self._displayed_xy = None

    def _show_computed_data(self, xy):
        """Fill the read-only data box with the computed (X, Y) points, or
        clear it when the formula currently fails to evaluate (xy is None)."""
        self._set_formula_display(True)
        if xy is None:
            if self._computed_xy is not None:
                self._is_normalizing = True
                self.text_paste.blockSignals(True)
                self.text_paste.clear()
                self.text_paste.blockSignals(False)
                self._is_normalizing = False
                self._computed_xy = None
                self._computed_text = None
                self._displayed_xy = None
                self.data_label.setText("<b>Data Source</b>")
            return
        if xy is self._displayed_xy:
            return  # unchanged result (formula cache hit) — keep the display
        X, Y = xy
        text = format_data_to_aligned_text(np.column_stack([X, Y]).tolist())
        self._is_normalizing = True
        self.text_paste.blockSignals(True)
        self.text_paste.setPlainText(text)
        self.text_paste.blockSignals(False)
        self._is_normalizing = False
        self._computed_xy = (np.asarray(X, dtype=float),
                             np.asarray(Y, dtype=float))
        self._computed_text = text
        self._displayed_xy = xy
        self.data_label.setText(
            f"<b>Data Source</b> ({len(X)} × 2, computed)")
        QTimer.singleShot(0, lambda: self._adjust_text_paste_width(text))

    # ── Original / Processed data views ──
    def _set_view_mode(self, mode):
        """Switch the data box between the original and the processed data."""
        if mode == self._view_mode:
            self._sync_view_buttons()
            return
        self._view_mode = mode
        self._refresh_data_views()

    def _sync_view_buttons(self):
        for btn, mode in ((self.btn_view_original, 'original'),
                          (self.btn_view_processed, 'processed')):
            btn.blockSignals(True)
            btn.setChecked(self._view_mode == mode)
            btn.blockSignals(False)

    def _set_processed_display(self, active):
        """Enter / leave the processed-data display mode: like the formula
        display, the box becomes a read-only (grey) view and the button turns
        into "Save CSV" — but the underlying original data stays untouched.
        Idempotent: always re-asserts the appearance of the current mode."""
        self._processed_display_active = active
        if active:
            self.text_paste.setReadOnly(True)
            self.text_paste.setStyleSheet(
                "QTextEdit { background-color: #EBEBEB; }")
            self.btn_cleanup_data.setText("Save CSV")
        elif self._formula_display_active:
            # Fall back to the formula display's own read-only look
            self.text_paste.setReadOnly(True)
            self.text_paste.setStyleSheet(
                "QTextEdit { background-color: #EBEBEB; }")
            self.btn_cleanup_data.setText("Save CSV")
        else:
            self.text_paste.setReadOnly(False)
            self.text_paste.setStyleSheet("")
            self.btn_cleanup_data.setText("Clean up data")

    def _show_processed_data(self):
        """Fill the read-only data box with the scale/offset-processed points."""
        self._set_processed_display(True)
        xy = self._processed_xy
        if xy is None:
            return
        if xy is self._displayed_processed_xy:
            return  # unchanged — keep the display
        X, Y = xy
        text = format_data_to_aligned_text(np.column_stack([X, Y]).tolist())
        self._is_normalizing = True
        self.text_paste.blockSignals(True)
        self.text_paste.setPlainText(text)
        self.text_paste.blockSignals(False)
        self._is_normalizing = False
        self._displayed_processed_xy = xy
        # Invalidate the computed-display memo so switching back to the
        # Original view re-renders the original points.
        self._displayed_xy = None
        self.data_label.setText(
            f"<b>Data Source</b> ({len(X)} × 2, processed)")
        QTimer.singleShot(0, lambda: self._adjust_text_paste_width(text))

    def _refresh_data_views(self):
        """Sync the Original/Processed buttons and the data box content with
        the current render state (called at the end of every get_object)."""
        has_processed = self._processed_xy is not None
        # Auto-switch: entering scale/offset shows the processed data,
        # losing it falls back to the original data.
        if has_processed and not self._had_processed:
            self._view_mode = 'processed'
        if not has_processed:
            self._view_mode = 'original'
        self._had_processed = has_processed
        self.view_toggle_widget.setVisible(has_processed)
        self._sync_view_buttons()

        if self._view_mode == 'processed':
            self._show_processed_data()
            return
        # Original view
        was_processed = self._processed_display_active
        self._set_processed_display(False)
        self._displayed_processed_xy = None
        if self._last_is_formula:
            self._show_computed_data(self._last_computed_xy)
        elif was_processed:
            # Restore the user's own data text that the processed display
            # had overwritten.
            if self.loaded_data is not None:
                self._display_data_as_text()
                df = self.loaded_data
                self.data_label.setText(
                    f"<b>Data Source</b> ({df.shape[0]} × {df.shape[1]})")
            else:
                self._is_normalizing = True
                self.text_paste.blockSignals(True)
                self.text_paste.clear()
                self.text_paste.blockSignals(False)
                self._is_normalizing = False
                self.data_label.setText("<b>Data Source</b>")

    # ── Align-owned Scale Factor / Offset boxes ──
    def user_scale_text(self):
        """The user's own Scale Factor text (not the align-solved display value)."""
        if self.inp_scale.isReadOnly() and self._scale_user_backup is not None:
            return self._scale_user_backup
        return self.inp_scale.text()

    def user_offset_text(self):
        """The user's own Offset text (not the align-solved display value)."""
        if self.inp_offset.isReadOnly() and self._offset_user_backup is not None:
            return self._offset_user_backup
        return self.inp_offset.text()

    def _manual_value(self, box, backup):
        """Numeric value the USER put in a box. While align owns the box its
        text is the solved value, so read the user's value from the backup."""
        text = backup if box.isReadOnly() and backup is not None else box.text()
        text = (text or '').strip()
        return safe_eval_number(text, None) if text else None

    def _sync_align_owned_box(self, box, owns, solved_value, backup_attr):
        """Fill a Scale/Offset box with the align-solved value (read-only) or
        return it to the user (restoring their own text)."""
        if owns:
            if not box.isReadOnly():
                setattr(self, backup_attr, box.text())
                box.setReadOnly(True)
                box.setStyleSheet("QLineEdit { background-color: #EBEBEB; }")
            text = f"{solved_value:.6g}"
            if box.text() != text:
                box.blockSignals(True)
                box.setText(text)
                box.blockSignals(False)
        else:
            if box.isReadOnly():
                box.setReadOnly(False)
                box.setStyleSheet("")
                backup = getattr(self, backup_attr)
                box.blockSignals(True)
                box.setText(backup if backup is not None else "")
                box.blockSignals(False)
                setattr(self, backup_attr, None)

    def _cleanup_data(self):
        if self._processed_display_active or self._formula_display_active:
            self._save_computed_csv()
        else:
            super()._cleanup_data()

    def _save_computed_csv(self):
        """Save the currently displayed derived data (computed or processed)
        as a two-column CSV file."""
        xy = self._processed_xy if self._processed_display_active else self._computed_xy
        if xy is None or len(xy[0]) == 0:
            QMessageBox.information(
                self, "Save CSV",
                "No computed data to save — the formula has not produced "
                "any points yet.")
            return
        default_name = (self.inp_label.text().strip()
                        or self._entry_code_or_default()) + ".csv"
        default_name = re.sub(r'[\\/:*?"<>|]', '_', default_name)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save computed data as CSV", default_name,
            "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            np.savetxt(path, np.column_stack(xy), delimiter=",", fmt="%.10g")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))

    def _entry_code_or_default(self):
        code = (getattr(self, 'curve_code', '') or '').strip()
        return code if code else "computed_curve"

    def get_object(self):
        if not self._is_enabled:
            return None
        formula = self.get_formula()
        self._last_is_formula = formula is not None
        if formula is not None:
            xy = self._compute_hook(self) if self._compute_hook is not None else None
            self._last_computed_xy = xy
            self._set_formula_display(True)
            if xy is None:
                self._processed_xy = None
                self._refresh_data_views()
                return None
            X, Y = xy
            if X is None or Y is None or len(Y) == 0:
                self._processed_xy = None
                self._refresh_data_views()
                return None
        else:
            self._last_computed_xy = None
            self._set_formula_display(False)
            X, Y, _Z = self.get_data_arrays()
            if Y is None or len(Y) == 0:
                self._processed_xy = None
                self._refresh_data_views()
                return None
        width = safe_eval_number(self.inp_width.text(), 0.8)
        dot_width = safe_eval_number(self.inp_dot_width.text(), 5)
        dot_alpha = safe_eval_number(self.inp_dot_alpha.text(), None) if self.inp_dot_alpha.text().strip() else None
        dot_edge_width = safe_eval_number(self.inp_dot_edge_width.text(), None) if self.inp_dot_edge_width.text().strip() else None
        # Manual Scale Factor / Offset (align-solved values never count as manual)
        scale = self._manual_value(self.inp_scale, self._scale_user_backup)
        offset = self._manual_value(self.inp_offset, self._offset_user_backup)
        interp_smoothing = safe_eval_number(self.inp_interp_smoothing.text(), 0)
        interp_number = int(safe_eval_number(self.inp_interp_number.text(), 5000))

        normalize_to = None
        nt = self.inp_normalize.text().strip()
        if "," in nt:
            try:
                normalize_to = tuple(
                    safe_eval_number(x.strip(), 0) for x in nt.split(",") if x.strip())
            except (ValueError, TypeError):
                pass
        elif nt:
            val = safe_eval_number(nt)
            if val is not None:
                normalize_to = val

        # Align With: solve the optimal scale/offset to overlay this curve onto a
        # reference curve over the chosen range.  When active, the solved values
        # take over the Scale Factor / Offset boxes (shown read-only) and the
        # manual Normalize To field is ignored.
        align_hook = getattr(self, '_align_hook', None)
        align_xform = align_hook(self, X, Y) if align_hook is not None else None
        owns_scale = owns_offset = False
        if align_xform is not None:
            a, b = align_xform
            if self.chk_align_scale.isChecked():
                scale, owns_scale = a, True
            if self.chk_align_offset.isChecked():
                offset, owns_offset = b, True
            normalize_to = None
        self._sync_align_owned_box(self.inp_scale, owns_scale, scale,
                                   '_scale_user_backup')
        self._sync_align_owned_box(self.inp_offset, owns_offset, offset,
                                   '_offset_user_backup')

        # Apply the vertical transform to the data itself (rather than via the
        # Curve's scale_factor) so the Processed view can show the result.
        if scale is not None or offset is not None:
            Y_processed = ((scale if scale is not None else 1.0)
                           * np.asarray(Y, dtype=float)
                           + (offset if offset is not None else 0.0))
            self._processed_xy = (np.asarray(X, dtype=float), Y_processed)
            Y = Y_processed
        else:
            self._processed_xy = None
        self._refresh_data_views()

        if self.chk_neg_x.isChecked():
            X = -X
        color = resolve_display_value(
            self.inp_color.currentText().strip(), COLOR_DISPLAY)
        fmt = resolve_display_value(
            self.inp_fmt.currentText().strip(), LINE_FORMAT_DISPLAY)
        marker = resolve_display_value(
            self.inp_marker.currentText().strip(), MARKER_DISPLAY)
        dot_color = resolve_display_value(
            self.inp_dot_color.currentText().strip(), COLOR_DISPLAY)
        dot_edge_color = resolve_display_value(
            self.inp_dot_edge_color.currentText().strip(), COLOR_DISPLAY)
        legend_color = resolve_display_value(
            self.inp_legend_color.currentText().strip(), COLOR_DISPLAY)
        legend_format = resolve_display_value(
            self.inp_legend_format.currentText().strip(), LINE_FORMAT_DISPLAY)
        # Resolve fill_color: display name → color value, or scheme name, or raw color
        fill_raw = self.inp_fill_color.currentText().strip()
        fill_color = None
        if fill_raw:
            if fill_raw in FILL_SCHEMES:
                fill_color = fill_raw
            else:
                resolved = resolve_display_value(fill_raw, COLOR_DISPLAY)
                fill_color = resolved if resolved else fill_raw
        
        errorbar_data = self.get_errorbar_data()
        errorbar_capsize = safe_eval_number(self.inp_errorbar_capsize.text(), 2)

        curve = Curve(
            X=X, Y=Y,
            Y_errorbar=errorbar_data,
            X_label="",
            Y_label=self.inp_label.text(),
            curve_color=color if color else None,
            curve_width=width,
            plot_curve=bool(fmt),
            curve_format=fmt,
            plot_dot=bool(marker),
            dot_format=marker if marker else '.',
            dot_color=dot_color if dot_color else None,
            dot_alpha=dot_alpha,
            dot_edge_color=dot_edge_color if dot_edge_color else None,
            dot_edge_width=dot_edge_width,
            dot_width=dot_width,
            do_interpolation=self.chk_interp.isChecked(),
            interpolation_kind=self.inp_interp_kind.currentText(),
            interpolation_smoothing=interp_smoothing,
            interpolation_number=interp_number,
            curve_legend_color=legend_color if legend_color else "",
            curve_legend_format=legend_format if legend_format else "",
            fill_color=fill_color,
            # scale/offset already applied to Y above (Processed view shows it)
            normalize_to=normalize_to,
        )
        curve.errorbar_capsize = errorbar_capsize
        return curve


# ──────────────────────── GridEntry ──────────────────────
class GridEntry(DataEntryWidget):
    def __init__(self, parent=None):
        super().__init__("Grid", parent)
        self._setup_ui()

    def _setup_ui(self):
        self.inp_interp_type = self.add_combo(
            "Method",
            ["linear", "cubic", "nearest", "multiquadric", "gaussian"],
            "linear")
        self.chk_contour = self.add_checkable_button("Contours", False)
        self.inp_density = self.add_line_edit("Density", "100")
        self.chk_colorbar = self.add_checkable_button("Colorbar", True)

    def get_object(self):
        if not self._is_enabled:
            return None
        X, Y, Z = self.get_data_arrays()
        if Z is None or len(Z) == 0:
            return None
        xyz = list(zip(X, Y, Z))
        dens = int(safe_eval_number(self.inp_density.text(), 100))
        return Grid(
            XYZ_triples=xyz,
            interpolation_type=self.inp_interp_type.currentText(),
            show_contour=self.chk_contour.isChecked(),
            interpolation_density=dens,
            show_colorbar=self.chk_colorbar.isChecked(),
        )


# ───────────────── File-drop column parsing ─────────────────
# Delimiter options exposed in the column-picker dialog. The order here drives
# the on-screen button order. Each tuple is (settings key, button label, char).
DROP_DELIM_OPTIONS = [
    ('tab',   r'\t',   '\t'),
    ('space', 'Space', ' '),
    ('comma', ',',     ','),
    ('pipe',  '|',     '|'),
]
DEFAULT_DROP_DELIMS = ('tab', 'space', 'comma')


def _delim_regex_from_keys(delim_keys):
    chars = [ch for k, _label, ch in DROP_DELIM_OPTIONS if k in delim_keys]
    if not chars:
        return None
    return '[' + ''.join(re.escape(c) for c in chars) + ']+'


def _read_text_lines(filepath):
    """Read a text file, sniffing the encoding so Chinese / latin-1 files
    don't come back as garbage. UTF-8 first (incl. BOM), then GB18030 for
    Chinese Windows files (charset_normalizer often misreads GB as Shift_JIS),
    then charset_normalizer as a smarter fallback, then a small cascade."""
    with open(filepath, 'rb') as fh:
        raw = fh.read()
    for enc in ('utf-8-sig', 'utf-8', 'gb18030'):
        try:
            return raw.decode(enc).splitlines()
        except UnicodeDecodeError:
            continue
    try:
        import charset_normalizer
        result = charset_normalizer.from_bytes(raw).best()
        if result is not None:
            return str(result).splitlines()
    except Exception:
        pass
    for enc in ('cp1252', 'latin-1'):
        try:
            return raw.decode(enc).splitlines()
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', errors='replace').splitlines()


def _is_numeric_cell(v):
    """Return True if v parses to a finite number."""
    if v is None:
        return False
    if isinstance(v, (bool,)):
        return False
    if isinstance(v, (int, np.integer)):
        return True
    if isinstance(v, (float, np.floating)):
        return not np.isnan(v)
    try:
        x = float(str(v).strip())
        return not np.isnan(x)
    except (ValueError, TypeError):
        return False


def _split_text_lines_to_rows(lines, delim_keys):
    """Turn an iterable of text lines into raw rows, applying the delimiter set."""
    pat = _delim_regex_from_keys(delim_keys)
    raw_rows = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if pat is None:
            parts = [stripped]
        else:
            parts = re.split(pat, stripped)
        raw_rows.append(parts)
    return raw_rows


def _rows_to_columns(raw_rows):
    """Turn padded raw rows into (headers, columns): sniff a header row and
    drop any column that contains no numeric cells."""
    if not raw_rows:
        return [], []

    max_cols = max(len(r) for r in raw_rows)
    padded = [list(r) + [None] * (max_cols - len(r)) for r in raw_rows]

    has_header = False
    if len(padded) >= 2:
        first_row_full = all(
            v is not None and str(v).strip() != ''
            for v in padded[0])
        first_row_all_text = first_row_full and not any(
            _is_numeric_cell(v) for v in padded[0])
        rest_any_num = any(
            _is_numeric_cell(v) for r in padded[1:] for v in r)
        if first_row_all_text and rest_any_num:
            has_header = True

    if has_header:
        headers_all = [
            (str(padded[0][c]).strip() if padded[0][c] is not None
             and str(padded[0][c]).strip() != '' else f"Col {c + 1}")
            for c in range(max_cols)
        ]
        data_rows = padded[1:]
    else:
        headers_all = [f"Col {c + 1}" for c in range(max_cols)]
        data_rows = padded

    cols_all = [
        [data_rows[r][c] if c < len(data_rows[r]) else None
         for r in range(len(data_rows))]
        for c in range(max_cols)
    ]

    kept_headers, kept_cols = [], []
    for h, c in zip(headers_all, cols_all):
        if any(_is_numeric_cell(v) for v in c):
            kept_headers.append(h)
            kept_cols.append(c)
    return kept_headers, kept_cols


def parse_data_file_columns(filepath, delim_keys=None):
    """Read a txt/csv/xlsx file and return (headers, columns).

    - txt and csv files are split by any character in ``delim_keys``
      (defaults to tab, space, comma).
    - xlsx ignores ``delim_keys`` and uses the workbook cell grid.
    - Drops any column that contains no numeric cells.
    - If the first row is fully non-numeric and the rest contains numbers,
      treats it as a header row.

    Returns
    -------
    headers : list[str]   one label per kept column
    columns : list[list]  raw values per kept column
    """
    if delim_keys is None:
        delim_keys = DEFAULT_DROP_DELIMS

    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.xlsx', '.xls'):
        df = pd.read_excel(filepath, header=None)
        raw_rows = [list(r) for r in df.values.tolist()]
    else:
        raw_rows = _split_text_lines_to_rows(
            _read_text_lines(filepath), delim_keys)

    return _rows_to_columns(raw_rows)


def parse_data_text_columns(text, delim_keys=None):
    """Parse a raw text blob (e.g. clipboard from Excel copy-paste) into
    ``(headers, columns)`` using the same header sniffing and column filter
    as :func:`parse_data_file_columns`."""
    if delim_keys is None:
        delim_keys = DEFAULT_DROP_DELIMS
    raw_rows = _split_text_lines_to_rows(text.splitlines(), delim_keys)
    return _rows_to_columns(raw_rows)


def columns_to_numeric_df(cols):
    """Build a numeric DataFrame from a list of column-value lists.

    Non-numeric cells become NaN; rows where any selected column is NaN are
    dropped. Columns are renamed 0..N-1 to match CurveEntry conventions.
    """
    if not cols:
        return pd.DataFrame()
    n_rows = max(len(c) for c in cols)
    padded = [list(c) + [None] * (n_rows - len(c)) for c in cols]
    data = {i: pd.to_numeric(pd.Series(padded[i]), errors='coerce')
            for i in range(len(padded))}
    df = pd.DataFrame(data)
    df = df.dropna().reset_index(drop=True)
    return df


class ColumnPickerDialog(QDialog):
    """Modeless dialog: pick X/Y/Err columns from a multi-column data file
    and emit `curve_requested(df, label, filepath)` for each Add Curve click.

    Delimiter buttons under "Add Curve" let the user re-parse the file with
    a different set of column separators. Changes are pushed to the parent
    controller, which echoes them to every other open picker so multi-file
    drops stay in sync.
    """

    curve_requested = pyqtSignal(object, str, str)

    ROLES = [
        ('x',    'As X'),
        ('y',    'As Y'),
        ('err',  'As Error Bar Col'),
        ('err2', 'As 2nd Error Bar Col'),
    ]

    def __init__(self, filepath, controller, parent=None, *, raw_text=None):
        super().__init__(parent)
        display_name = (filepath if raw_text is not None
                        else os.path.basename(filepath))
        self.setWindowTitle(f"Select Columns - {display_name}")
        self.setModal(False)
        self.filepath = filepath
        self.raw_text = raw_text
        self.controller = controller
        self.headers: list[str] = []
        self.columns: list[list] = []
        self.role_assign: dict[str, int | None] = {r: None for r, _ in self.ROLES}
        self._role_btns: dict[str, list[QPushButton]] = {r: [] for r, _ in self.ROLES}
        self._delim_btns: dict[str, QPushButton] = {}
        self.table: QtWidgets.QTableWidget | None = None
        self._user_modified = False

        layout = QVBoxLayout(self)

        top_row = QHBoxLayout()
        self.btn_add_and_close = QPushButton("Add Curve and Close")
        self.btn_add_and_close.setStyleSheet(GlobalStyles.NORMAL_BTN)
        self.btn_add_and_close.clicked.connect(self._on_add_curve_and_close)
        self.btn_add_and_close.setEnabled(False)
        top_row.addWidget(self.btn_add_and_close)
        self.btn_add = QPushButton("Add Curve")
        self.btn_add.setStyleSheet(GlobalStyles.NORMAL_BTN)
        self.btn_add.clicked.connect(self._on_add_curve)
        self.btn_add.setEnabled(False)
        top_row.addWidget(self.btn_add)
        self.lbl_status = QLabel(
            "Pick a column for X and Y, then click Add Curve. "
            "You can keep adding curves until you close this window.")
        self.lbl_status.setStyleSheet("color: #666;")
        self.lbl_status.setWordWrap(True)
        top_row.addWidget(self.lbl_status, stretch=1)
        layout.addLayout(top_row)

        delim_row = QHBoxLayout()
        delim_row.addWidget(QLabel("Delimiters:"))
        for key, label, _ch in DROP_DELIM_OPTIONS:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setStyleSheet(GlobalStyles.CHECKABLE_BTN)
            btn.setFixedWidth(60)
            btn.toggled.connect(
                lambda checked, k=key: self._on_delim_toggled(k, checked))
            self._delim_btns[key] = btn
            delim_row.addWidget(btn)
        delim_row.addStretch()
        layout.addLayout(delim_row)

        self._table_holder = QWidget()
        self._table_holder_layout = QVBoxLayout(self._table_holder)
        self._table_holder_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._table_holder, stretch=1)

        controller.drop_delims_changed.connect(self._on_external_delim_change)
        if hasattr(controller, 'register_picker'):
            controller.register_picker(self)
        self._reload_from_controller()
        self.resize(900, 600)

    def closeEvent(self, event):
        try:
            self.controller.drop_delims_changed.disconnect(
                self._on_external_delim_change)
        except (TypeError, RuntimeError):
            pass
        if hasattr(self.controller, 'unregister_picker'):
            self.controller.unregister_picker(self)
        super().closeEvent(event)

    # ── delimiter handling ──
    def _on_delim_toggled(self, key, checked):
        cur = list(self.controller.get_drop_delim_keys())
        if checked and key not in cur:
            cur.append(key)
        elif not checked and key in cur:
            cur.remove(key)
        self.controller.set_drop_delim_keys(cur)

    def _on_external_delim_change(self, _keys):
        self._reload_from_controller()

    def _reload_from_controller(self):
        keys = self.controller.get_drop_delim_keys()
        for key, btn in self._delim_btns.items():
            btn.blockSignals(True)
            btn.setChecked(key in keys)
            btn.blockSignals(False)
        try:
            if self.raw_text is not None:
                self.headers, self.columns = parse_data_text_columns(
                    self.raw_text, delim_keys=keys)
            else:
                self.headers, self.columns = parse_data_file_columns(
                    self.filepath, delim_keys=keys)
        except Exception as e:
            QMessageBox.warning(self, "Parse Error", str(e))
            self.headers, self.columns = [], []
        for role, _ in self.ROLES:
            self.role_assign[role] = None
        self._user_modified = False
        self._rebuild_table()
        self._update_status()

    def _rebuild_table(self):
        if self.table is not None:
            self._table_holder_layout.removeWidget(self.table)
            self.table.deleteLater()
            self.table = None
        for role, _ in self.ROLES:
            self._role_btns[role] = []

        n_cols = len(self.columns)
        n_role_rows = len(self.ROLES)
        n_data_rows = max((len(c) for c in self.columns), default=0)

        table = QtWidgets.QTableWidget(n_role_rows + n_data_rows, n_cols)
        table.setHorizontalHeaderLabels(self.headers)
        table.setVerticalHeaderLabels(
            [name for _, name in self.ROLES]
            + [str(i + 1) for i in range(n_data_rows)])
        vh = table.verticalHeader()
        if vh is not None:
            vh.setDefaultSectionSize(22)

        role_row_height = 0
        for r_idx, (role_key, role_name) in enumerate(self.ROLES):
            for c_idx in range(n_cols):
                btn = QPushButton(role_name)
                btn.setCheckable(True)
                btn.setStyleSheet(GlobalStyles.CHECKABLE_BTN)
                btn.clicked.connect(
                    lambda _checked, r=role_key, c=c_idx, b=btn:
                    self._on_role_clicked(r, c, b))
                table.setCellWidget(r_idx, c_idx, btn)
                self._role_btns[role_key].append(btn)
                role_row_height = max(role_row_height, btn.sizeHint().height())
        role_row_height = max(role_row_height + 4, 30)
        for r_idx in range(n_role_rows):
            table.setRowHeight(r_idx, role_row_height)

        for c_idx, col in enumerate(self.columns):
            for r_idx in range(n_data_rows):
                v = col[r_idx] if r_idx < len(col) else None
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    text = ''
                else:
                    text = str(v)
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(n_role_rows + r_idx, c_idx, item)

        table.resizeColumnsToContents()
        self._table_holder_layout.addWidget(table)
        self.table = table

    def _on_role_clicked(self, role, col_idx, btn):
        checked = btn.isChecked()
        self._user_modified = True
        if checked:
            prev = self.role_assign.get(role)
            if prev is not None and prev != col_idx:
                other = self._role_btns[role][prev]
                other.blockSignals(True)
                other.setChecked(False)
                other.blockSignals(False)
            for other_role, other_col in list(self.role_assign.items()):
                if other_role != role and other_col == col_idx:
                    obtn = self._role_btns[other_role][col_idx]
                    obtn.blockSignals(True)
                    obtn.setChecked(False)
                    obtn.blockSignals(False)
                    self.role_assign[other_role] = None
            self.role_assign[role] = col_idx
        else:
            if self.role_assign.get(role) == col_idx:
                self.role_assign[role] = None
        self._update_status()
        if hasattr(self.controller, 'broadcast_picker_role'):
            header = self.headers[col_idx] if 0 <= col_idx < len(self.headers) else None
            self.controller.broadcast_picker_role(self, role, header, checked)

    def apply_external_role(self, role, header, set_checked):
        """Mirror a role assignment broadcast from another picker.

        Does not flip ``_user_modified`` — auto-sync should not block future
        broadcasts to this picker.
        """
        if not set_checked:
            prev = self.role_assign.get(role)
            if prev is not None:
                btn = self._role_btns[role][prev]
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
                self.role_assign[role] = None
                self._update_status()
            return
        if header is None:
            return
        try:
            col_idx = self.headers.index(header)
        except ValueError:
            return
        prev = self.role_assign.get(role)
        if prev == col_idx:
            return
        if prev is not None:
            other = self._role_btns[role][prev]
            other.blockSignals(True)
            other.setChecked(False)
            other.blockSignals(False)
        for other_role, other_col in list(self.role_assign.items()):
            if other_role != role and other_col == col_idx:
                obtn = self._role_btns[other_role][col_idx]
                obtn.blockSignals(True)
                obtn.setChecked(False)
                obtn.blockSignals(False)
                self.role_assign[other_role] = None
        btn = self._role_btns[role][col_idx]
        btn.blockSignals(True)
        btn.setChecked(True)
        btn.blockSignals(False)
        self.role_assign[role] = col_idx
        self._update_status()

    def _update_status(self):
        x_idx = self.role_assign['x']
        y_idx = self.role_assign['y']
        enabled = x_idx is not None and y_idx is not None
        self.btn_add.setEnabled(enabled)
        self.btn_add_and_close.setEnabled(enabled)
        if x_idx is None or y_idx is None:
            need = []
            if x_idx is None:
                need.append("X")
            if y_idx is None:
                need.append("Y")
            self.lbl_status.setText(
                "Pick a column for " + " and ".join(need)
                + ", then click Add Curve.")
            return
        parts = [f"X={self.headers[x_idx]}", f"Y={self.headers[y_idx]}"]
        err_idx = self.role_assign['err']
        if err_idx is not None:
            parts.append(f"Err={self.headers[err_idx]}")
        err2_idx = self.role_assign['err2']
        if err2_idx is not None:
            parts.append(f"Err2={self.headers[err2_idx]}")
        self.lbl_status.setText("Ready: " + ", ".join(parts))

    def _build_label(self, y_idx: int) -> str:
        if self.raw_text is not None:
            base = self.filepath or "Clipboard"
        else:
            base = os.path.splitext(os.path.basename(self.filepath))[0]
        h = self.headers[y_idx] if 0 <= y_idx < len(self.headers) else ''
        if h and not h.startswith("Col "):
            return f"{base} - {h}"
        return base

    def _on_add_curve(self):
        x_idx = self.role_assign['x']
        y_idx = self.role_assign['y']
        if x_idx is None or y_idx is None:
            return
        chosen = [self.columns[x_idx], self.columns[y_idx]]
        err_idx = self.role_assign['err']
        err2_idx = self.role_assign['err2']
        if err_idx is not None:
            chosen.append(self.columns[err_idx])
            if err2_idx is not None:
                chosen.append(self.columns[err2_idx])
        df = columns_to_numeric_df(chosen)
        if df.empty:
            QMessageBox.warning(
                self, "No Data",
                "The selected columns have no overlapping numeric rows.")
            return
        label = self._build_label(y_idx)
        self.curve_requested.emit(df, label, self.filepath)

    def _on_add_curve_and_close(self):
        if not self.btn_add.isEnabled():
            return
        self._on_add_curve()
        self.close()


class CurvePasteDialog(QDialog):
    """Dialog shown on Ctrl+V to select which curve properties to paste."""

    # Layout spec mirroring CurveEntry._setup_ui's 4-column grid.
    # Columns: [btn/0] [preview/1] [btn/2] [preview/3]
    # 'header'  → section separator + bold header label
    # 'full'    → btn at col 0, preview spans cols 1-3
    # 'pair'    → (btn+preview) at cols 0+1 and (btn+preview) at cols 2+3
    # 'check'   → checkable btn spans cols 0-2, bool preview at col 3
    _LAYOUT = [
        ('header', 'Curve'),
        ('full',  'Format',           'inp_fmt'),
        ('pair',  ('Color',           'inp_color'),          ('Width',        'inp_width')),
        ('full',  'Fill Color',       'inp_fill_color'),
        ('header', 'Dot'),
        ('full',  'Marker Format',    'inp_marker'),
        ('pair',  ('Color',           'inp_dot_color'),      ('Size',         'inp_dot_width')),
        ('full',  'Alpha',            'inp_dot_alpha'),
        ('pair',  ('Edge Color',      'inp_dot_edge_color'), ('Edge Width',   'inp_dot_edge_width')),
        ('header', 'Interpolation'),
        ('check', 'Interpolation',    'chk_interp'),
        ('pair',  ('Interp Kind',     'inp_interp_kind'),    ('Smoothing',    'inp_interp_smoothing')),
        ('full',  'Interp Number',    'inp_interp_number'),
        ('pair',  ('Scale Factor',    'inp_scale'),          ('Offset',       'inp_offset')),
        ('full',  'Normalize To',     'inp_normalize'),
        ('pair',  ('Legend Color',    'inp_legend_color'),   ('Legend Format','inp_legend_format')),
        ('check', 'X → -X',          'chk_neg_x'),
        ('full',  'Label',            'inp_label'),
    ]

    # Display text when the copied value is empty string (shows real effective value)
    _EMPTY_DISPLAY = {
        'inp_fmt':             '(no line)',
        'inp_color':           '(auto)',
        'inp_fill_color':      '(none)',
        'inp_marker':          '(none)',
        'inp_dot_color':       '(auto)',
        'inp_dot_alpha':       '(none)',
        'inp_dot_edge_color':  '(none)',
        'inp_dot_edge_width':  '(none)',
        'inp_width':           '0.8',
        'inp_dot_width':       '5',
        'inp_interp_smoothing':'0',
        'inp_interp_number':   '5000',
        'inp_scale':           '(none)',
        'inp_normalize':       '(none)',
        'inp_legend_color':    '(auto)',
        'inp_legend_format':   '(auto)',
    }

    def __init__(self, copied_props: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Paste Curve Properties")
        self.setMinimumWidth(420)
        self.checkboxes: dict[str, QPushButton] = {}  # attr_name → checkable QPushButton

        layout = QVBoxLayout(self)

        # Select All / Deselect All buttons
        btn_row = QHBoxLayout()
        btn_select_all = QPushButton("Select All")
        btn_select_all.setStyleSheet(GlobalStyles.NORMAL_BTN)
        btn_select_all.clicked.connect(lambda: self._set_all(True))
        btn_deselect_all = QPushButton("Deselect All")
        btn_deselect_all.setStyleSheet(GlobalStyles.NORMAL_BTN)
        btn_deselect_all.clicked.connect(lambda: self._set_all(False))
        btn_row.addWidget(btn_select_all)
        btn_row.addWidget(btn_deselect_all)
        layout.addLayout(btn_row)

        # 4-column grid matching CurveEntry layout
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        row = 0
        first_section = True

        for entry in self._LAYOUT:
            kind = entry[0]

            if kind == 'header':
                title = entry[1]
                if not first_section:
                    sep = QFrame()
                    sep.setFrameShape(QFrame.Shape.HLine)
                    sep.setFrameShadow(QFrame.Shadow.Sunken)
                    grid.addWidget(sep, row, 0, 1, 4)
                    row += 1
                first_section = False
                header = QLabel(f"<b>{title}</b>")
                header.setFixedHeight(25)
                grid.addWidget(header, row, 0, 1, 4)
                row += 1

            elif kind == 'full':
                _, display_name, attr_name = entry
                if attr_name not in copied_props:
                    continue
                btn = self._make_prop_btn(display_name, attr_name)
                prev = self._make_preview(attr_name, copied_props[attr_name])
                grid.addWidget(btn, row, 0)
                grid.addWidget(prev, row, 1, 1, 3)
                row += 1

            elif kind == 'pair':
                _, (dn1, a1), (dn2, a2) = entry
                has1 = a1 in copied_props
                has2 = a2 in copied_props
                if not has1 and not has2:
                    continue
                if has1:
                    btn1 = self._make_prop_btn(dn1, a1)
                    prev1 = self._make_preview(a1, copied_props[a1])
                    grid.addWidget(btn1, row, 0)
                    grid.addWidget(prev1, row, 1)
                if has2:
                    btn2 = self._make_prop_btn(dn2, a2)
                    prev2 = self._make_preview(a2, copied_props[a2])
                    grid.addWidget(btn2, row, 2)
                    grid.addWidget(prev2, row, 3)
                row += 1

            elif kind == 'check':
                _, display_name, attr_name = entry
                if attr_name not in copied_props:
                    continue
                btn = self._make_prop_btn(display_name, attr_name)
                prev = self._make_preview(attr_name, copied_props[attr_name])
                grid.addWidget(btn, row, 0, 1, 3)
                grid.addWidget(prev, row, 3)
                row += 1

        layout.addLayout(grid)

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _make_prop_btn(self, display_name: str, attr_name: str) -> QPushButton:
        btn = QPushButton(display_name)
        btn.setCheckable(True)
        btn.setChecked(attr_name != 'inp_label')
        btn.setStyleSheet(GlobalStyles.CHECKABLE_BTN)
        self.checkboxes[attr_name] = btn
        return btn

    def _make_preview(self, attr_name: str, value) -> QLabel:
        if isinstance(value, bool):
            text = '✓' if value else '✗'
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #666; font-weight: bold;")
        else:
            text = str(value)
            if not text and attr_name in self._EMPTY_DISPLAY:
                text = self._EMPTY_DISPLAY[attr_name]
                lbl = QLabel(text)
                lbl.setStyleSheet("color: #999; font-style: italic;")
            else:
                lbl = QLabel(text)
                lbl.setStyleSheet("color: #444;")
        return lbl

    def _set_all(self, checked: bool):
        for btn in self.checkboxes.values():
            btn.setChecked(checked)

    def get_selected_attrs(self) -> set[str]:
        return {attr for attr, btn in self.checkboxes.items() if btn.isChecked()}


class StreamRedirector(QtCore.QObject):
    text_written = pyqtSignal(str)
    
    def write(self, text):
        self.text_written.emit(str(text))
        
    def flush(self):
        pass


# ────────────────────── PlotController ───────────────────
class PlotController(QMainWindow):
    drop_delims_changed = pyqtSignal(object)

    # Keep strong references to every open editor window so that windows
    # spawned via "New Window" are not garbage-collected the moment the
    # local variable goes out of scope.
    _open_windows: list["PlotController"] = []

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LYH Plot Editor")
        self.resize(1500, 900)
        self.setStyleSheet(GlobalStyles.STYLE_SHEET)
        self.setAcceptDrops(True)

        self._drop_delim_keys: tuple[str, ...] = tuple(DEFAULT_DROP_DELIMS)
        self._open_pickers: list[ColumnPickerDialog] = []
        self.plot_window = None
        # Embedded copy of the floating figure, docked at the bottom of the
        # right panel (below the curve list)
        self.preview_plot = None
        # Fraction of the left column's height given to the data editor
        # (the docked Result log gets the rest); persisted in last settings.
        self._editor_log_split_ratio = 0.7
        # Fraction of the right panel's height given to the curve list
        # (the figure preview gets the rest); persisted in last settings.
        self._right_split_ratio = 0.5
        self._applying_splitter_ratio = False
        self.entries: list[DataEntryWidget] = []
        self.entry_buttons: list[QPushButton] = []
        self.entry_checkboxes: list[QtWidgets.QCheckBox] = []
        self.entry_prop_containers: list[QWidget] = []
        self._associated_plot_json_path = None
        # Path of the .Plot_Editor session file this editor is bound to
        # (set when a session is loaded or saved). Quick Save targets it.
        self._session_save_path = None
        # JSON snapshot of the last saved/loaded state, used to detect
        # unsaved changes when the window is closed.
        self._saved_snapshot = None
        self._copied_curve_props = None  # dict of {attr_name: value} for Ctrl+C/V
        self._selected_entries: set[DataEntryWidget] = set()
        self._active_entry = None
        self._selection_anchor = None
        # Drag-to-reorder + bulk-label-dialog state
        self._drop_indicator = None
        self._reorder_locked = False
        self._label_dialog = None
        self._label_dialog_curves = None
        self._suppress_update = False
        self._suppress_prop_sync = False
        self._last_applied_fig_size = None   # (width_px, height_px) last sent to plot window
        self._last_applied_font_size = None  # font_size last sent to plot window
        self._last_applied_legend_font_size = None
        self._init_done = False
        # Multi-frame state: stored when loading a multi-frame Plot JSON.
        # Cleared when user edits entries manually.
        self._stored_curve_frames = None   # list[list[Curve]] or None
        self._stored_grid_frames = None    # list[list[Grid]] or None
        self._stored_frame_labels = None   # list[str] or None
        self._stored_current_frame_index = 0
        # Toolbar-zoom preservation
        self._user_xlim = None
        self._user_ylim = None
        self._x_zoomed_by_user = False
        self._y_zoomed_by_user = False
        self._suppress_lim_callback = False
        self._lim_callbacks_connected = False

        PlotController._open_windows.append(self)

        self._center_window()
        self._init_ui()
        self._load_last_settings()
        self._update_hline_button_state()  # Initialize button state
        self._init_done = True
        # Baseline for the unsaved-changes check: a freshly opened editor is
        # considered "clean" until the user actually changes something.
        self._mark_session_saved()
        self.show()

    # ────── window helpers ──────
    def _center_window(self):
        qr = self.frameGeometry()
        cp = QtGui.QGuiApplication.primaryScreen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):
        if self._has_unsaved_changes():
            resp = QMessageBox.question(
                self, "Unsaved Changes",
                "This session has unsaved changes. Save before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save)
            if resp == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if resp == QMessageBox.StandardButton.Save and not self.quick_save():
                # Save was cancelled or failed — keep the window open.
                event.ignore()
                return
        self._auto_save_settings()
        if self.plot_window is not None:
            self.plot_window.close()
            self.plot_window = None
        try:
            PlotController._open_windows.remove(self)
        except ValueError:
            pass
        super().closeEvent(event)

    def _refresh_editor_title(self):
        title = "LYH Plot Editor"
        bound_path = self._session_save_path or self._associated_plot_json_path
        if bound_path:
            title = f"{title} - {bound_path}"
        self.setWindowTitle(title)

    def _refresh_associated_plot_json_ui(self):
        self._refresh_editor_title()

    def _set_associated_plot_json_path(self, path):
        self._associated_plot_json_path = os.path.abspath(path) if path else None
        self._refresh_associated_plot_json_ui()

    def _on_plot_json_saved(self, path):
        self._set_associated_plot_json_path(path)

    def _on_curve_clicked(self, curve_index):
        """Map a plot_object index back to the corresponding entry and select it."""
        # Build the same mapping used in _update_and_show_plot: entries whose
        # get_object() returns non-None form the plot_objects list in order.
        obj_idx = 0
        for entry in self.entries:
            try:
                obj = entry.get_object()
            except Exception:
                obj = None
            if obj is None:
                continue
            if obj_idx == curve_index:
                self.select_entry(entry)
                return
            obj_idx += 1

    @staticmethod
    def _format_click_coord(value, per_pixel):
        """Format a click coordinate to ~0.2 px resolution (shared with hover)."""
        s = format_value_at_pixel_resolution(value, per_pixel)
        return s if s is not None else f"{value:.6g}"

    def _on_canvas_clicked(self, x, y, x_per_pixel=None, y_per_pixel=None):
        """Append the data coordinates of a plot click to the Result pane."""
        x_str = self._format_click_coord(x, x_per_pixel)
        y_str = self._format_click_coord(y, y_per_pixel)
        self.ui_program_output.append(f"Clicked: x = {x_str}, y = {y_str}")
        # QTextEdit.append() leaves the cursor where it was, so move it to the
        # end and scroll there — otherwise newly appended lines stay hidden
        # below the fixed-height pane.
        self.ui_program_output.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self.ui_program_output.ensureCursorVisible()

    def _on_canvas_resized(self, width_px, height_px):
        """Reflect user-driven plot-window resize into the W/H fields without re-resizing the window on the next replot."""
        w_int = int(round(float(width_px)))
        h_int = int(round(float(height_px)))
        for widget, value in ((self.ui_fig_w, w_int), (self.ui_fig_h, h_int)):
            widget.blockSignals(True)
            widget.setText(str(value))
            widget.blockSignals(False)
        # Keep _last_applied_fig_size in sync so do_update_plot's layout-change
        # check doesn't see this as a settings change and force another resize.
        self._last_applied_fig_size = (w_int, h_int)
        self._auto_save_settings()

    def _clear_associated_plot_json_path(self):
        self._set_associated_plot_json_path(None)

    # ────── drag & drop on main window ──────
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            f = url.toLocalFile()
            if not os.path.isfile(f):
                continue
            lower_name = os.path.basename(f).lower()
            if lower_name.endswith(('.plot_editor', '.plot', '.json.plot', '.json')):
                loaded = self._try_load_session_or_plot_file(f, show_error=False)
                if loaded is not False:
                    continue
            if lower_name.endswith(('.txt', '.csv', '.xlsx', '.xls')):
                self._drop_data_file(f)
            else:
                self.add_entry('curve', f)

    def get_drop_delim_keys(self):
        """Return the current set of delimiter keys for file-drop parsing."""
        return tuple(getattr(self, '_drop_delim_keys', DEFAULT_DROP_DELIMS))

    def set_drop_delim_keys(self, keys):
        """Update the file-drop delimiter set, persist, and broadcast to
        every open column-picker dialog so they re-parse in sync."""
        normalized = tuple(
            k for k, _, _ in DROP_DELIM_OPTIONS
            if k in keys)
        if normalized == self.get_drop_delim_keys():
            return
        self._drop_delim_keys = normalized
        self._auto_save_settings()
        self.drop_delims_changed.emit(normalized)

    def register_picker(self, picker):
        if picker not in self._open_pickers:
            self._open_pickers.append(picker)

    def unregister_picker(self, picker):
        if picker in self._open_pickers:
            self._open_pickers.remove(picker)

    def broadcast_picker_role(self, source, role, header, set_checked):
        """Mirror a user role click to every other open column-picker that
        has the same column header and that the user has not yet touched."""
        for picker in list(self._open_pickers):
            if picker is source:
                continue
            if getattr(picker, '_user_modified', False):
                continue
            picker.apply_external_role(role, header, set_checked)

    def _drop_data_file(self, filepath):
        """Drop handler for txt/csv/xlsx files. Two numeric columns import
        directly; more open the column-picker dialog."""
        delim_keys = self.get_drop_delim_keys()
        try:
            headers, columns = parse_data_file_columns(
                filepath, delim_keys=delim_keys)
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))
            return
        if len(columns) < 2:
            QMessageBox.warning(
                self, "Load Error",
                f"'{os.path.basename(filepath)}' has fewer than 2 numeric "
                "columns to plot.")
            return
        base_label = os.path.splitext(os.path.basename(filepath))[0]
        if len(columns) == 2:
            df = columns_to_numeric_df([columns[0], columns[1]])
            if df.empty:
                QMessageBox.warning(
                    self, "Load Error",
                    f"'{os.path.basename(filepath)}' has no rows where both "
                    "columns are numeric.")
                return
            self._add_curve_from_dataframe(df, base_label, filepath)
            return
        dlg = ColumnPickerDialog(filepath, self, parent=self)
        dlg.curve_requested.connect(self._add_curve_from_dataframe)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.show()

    def _add_curve_from_dataframe(self, df, label, filepath):
        """Add a CurveEntry pre-populated from a pandas DataFrame.
        Uses `label` as the legend label and stores `filepath` as data source.
        """
        widget = self.add_entry('curve')
        if not isinstance(widget, CurveEntry):
            return widget
        widget._file_path = filepath
        widget.loaded_data = df.reset_index(drop=True)
        widget.data_label.setText(
            f"<b>Data Source</b> ({widget.loaded_data.shape[0]} × "
            f"{widget.loaded_data.shape[1]})")
        if filepath and not widget.inp_data_source.text().strip():
            widget.inp_data_source.blockSignals(True)
            widget.inp_data_source.setText(os.path.basename(filepath))
            widget.inp_data_source.blockSignals(False)
        if label:
            widget.inp_label.setText(label)
            widget.label_changed_signal.emit(label)
        widget._display_data_as_text()
        widget.emit_change()
        if self._maybe_apply_neg_x_to_entry(widget):
            self.request_update(from_entry_edit=True)
        self._auto_color_if_enabled()
        return widget

    def _auto_color_if_enabled(self):
        """Reassign colors across all curves if the Auto-Color toggle is on."""
        if self._suppress_update:
            return
        if not getattr(self, 'ui_auto_color', None) or not self.ui_auto_color.isChecked():
            return
        any_real_curve = any(
            isinstance(e, CurveEntry) and not self._is_horizontal_line(e)
            for e in self.entries)
        if not any_real_curve:
            return
        self.auto_assign_colors()

    # ════════════════════ UI SETUP ════════════════════════
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_h = QHBoxLayout(central)
        main_h.setContentsMargins(5, 5, 5, 5)

        # ═══════ LEFT PANEL ═══════
        left_scroll = QScrollArea()
        # Ensure scroll area automatically resizes to fit content width, no horizontal scroll
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Avoid fixed width, use minimum expanding policy
        left_scroll.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        left_scroll.setMinimumWidth(320)  # Base width sufficient for standard labels
        # Optionally allow automatic resize if contents grow (uncomment if dynamic sizing needed)
        # left_scroll.setSizeAdjustPolicy(QScrollArea.SizeAdjustPolicy.AdjustToContents)
        left_w = QWidget()
        sl = QVBoxLayout(left_w)
        sl.setContentsMargins(5, 5, 5, 5)

        self._refresh_associated_plot_json_ui()

        # -- New Window / Load Session / Save Session / Save Session As --
        nw_ls_row = QHBoxLayout()
        btn_new_window = QPushButton("New Window")
        btn_new_window.setStyleSheet(GlobalStyles.NORMAL_BTN)
        btn_new_window.clicked.connect(self.new_window)
        btn_ls = QPushButton("Load Session")
        btn_ls.setStyleSheet(GlobalStyles.NORMAL_BTN)
        btn_ls.clicked.connect(self.load_session)
        nw_ls_row.addWidget(btn_new_window)
        nw_ls_row.addWidget(btn_ls)
        sl.addLayout(nw_ls_row)

        ss_row = QHBoxLayout()
        btn_quick_save = QPushButton("Save Session")
        btn_quick_save.setStyleSheet(GlobalStyles.NORMAL_BTN)
        btn_quick_save.clicked.connect(self.quick_save)
        btn_ss = QPushButton("Save Session As...")
        btn_ss.setStyleSheet(GlobalStyles.NORMAL_BTN)
        btn_ss.clicked.connect(self.save_session)
        ss_row.addWidget(btn_quick_save)
        ss_row.addWidget(btn_ss)
        sl.addLayout(ss_row)

        # Ctrl+S → Quick Save
        save_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Save, self)
        save_shortcut.activated.connect(self.quick_save)

        sl.addSpacing(8)

        # -- Load Preset --
        sl.addWidget(QLabel("<b>Load Preset</b>"))
        self.ui_preset_combo = QComboBox()
        self.ui_preset_combo.addItem("(none)")
        self._refresh_presets()
        self.ui_preset_combo.currentTextChanged.connect(
            self._on_preset_selected)
        sl.addWidget(self.ui_preset_combo)
        sl.addSpacing(8)

        # -- Settings form using QGridLayout for alignment --
        form_w = QWidget()
        settings_grid = QGridLayout(form_w)
        settings_grid.setContentsMargins(0, 0, 0, 0)
        settings_grid.setHorizontalSpacing(6)
        settings_grid.setVerticalSpacing(4)
        self.form_layout = settings_grid  # keep attribute for _mk_le

        srow = 0

        # Title (full row)
        settings_grid.addWidget(QLabel("Title"), srow, 0)
        self.ui_title = self._mk_le_no_row("")
        settings_grid.addWidget(self.ui_title, srow, 1, 1, 3)
        srow += 1

        # X Lim with Auto button
        settings_grid.addWidget(QLabel("X Lim"), srow, 0)
        self.ui_xlim = self._mk_le_no_row("")
        self.ui_xlim.setPlaceholderText("min, max")
        # User edit on X Lim resets the toolbar-zoom override
        self.ui_xlim.textEdited.connect(lambda _t: self._reset_user_zoom('x'))
        settings_grid.addWidget(self.ui_xlim, srow, 1, 1, 2)
        self.btn_auto_xrange = QPushButton("Auto")
        self.btn_auto_xrange.setStyleSheet(GlobalStyles.NORMAL_BTN)
        self.btn_auto_xrange.setToolTip("Auto-scale X axis to current data")
        self.btn_auto_xrange.clicked.connect(lambda: self._apply_auto_range('x'))
        settings_grid.addWidget(self.btn_auto_xrange, srow, 3)
        srow += 1

        # Y Lim with Auto button
        settings_grid.addWidget(QLabel("Y Lim"), srow, 0)
        self.ui_ylim = self._mk_le_no_row("")
        self.ui_ylim.setPlaceholderText("min, max")
        self.ui_ylim.textEdited.connect(lambda _t: self._reset_user_zoom('y'))
        settings_grid.addWidget(self.ui_ylim, srow, 1, 1, 2)
        self.btn_auto_yrange = QPushButton("Auto")
        self.btn_auto_yrange.setStyleSheet(GlobalStyles.NORMAL_BTN)
        self.btn_auto_yrange.setToolTip("Auto-scale Y axis to current data")
        self.btn_auto_yrange.clicked.connect(lambda: self._apply_auto_range('y'))
        settings_grid.addWidget(self.btn_auto_yrange, srow, 3)
        srow += 1

        # X Label / Y Label (paired)
        settings_grid.addWidget(QLabel("X Label"), srow, 0)
        self.ui_xlabel = self._mk_le_no_row("X Axis")
        settings_grid.addWidget(self.ui_xlabel, srow, 1)
        settings_grid.addWidget(QLabel("Y Label"), srow, 2)
        self.ui_ylabel = self._mk_le_no_row("Y Axis")
        settings_grid.addWidget(self.ui_ylabel, srow, 3)
        srow += 1

        # X Log / Y Log (paired checkable buttons)
        self.ui_xlog = self._mk_ckbtn("X Log")
        settings_grid.addWidget(self.ui_xlog, srow, 0, 1, 2)
        self.ui_ylog = self._mk_ckbtn("Y Log")
        settings_grid.addWidget(self.ui_ylog, srow, 2, 1, 2)
        srow += 1

        # Grid / Legend (paired checkable buttons)
        self.ui_grid = self._mk_ckbtn("Grid")
        settings_grid.addWidget(self.ui_grid, srow, 0, 1, 2)
        self.ui_legend = self._mk_ckbtn("Legend", True)
        settings_grid.addWidget(self.ui_legend, srow, 2, 1, 2)
        srow += 1

        # Separator before Keep Front
        sep_kf = QFrame(); sep_kf.setFrameShape(QFrame.Shape.HLine); sep_kf.setFrameShadow(QFrame.Shadow.Sunken)
        settings_grid.addWidget(sep_kf, srow, 0, 1, 4); srow += 1

        # Bring to Front (one-shot raise) / Keep Front
        self.ui_bring_front = QPushButton("Bring to Front")
        self.ui_bring_front.setStyleSheet(GlobalStyles.NORMAL_BTN)
        self.ui_bring_front.clicked.connect(self._bring_plot_to_front)
        settings_grid.addWidget(self.ui_bring_front, srow, 0, 1, 2)
        self.ui_keep_front = self._mk_ckbtn("Keep Front", False)
        settings_grid.addWidget(self.ui_keep_front, srow, 2, 1, 2)
        srow += 1

        # W (px) / H (px) (paired)
        settings_grid.addWidget(QLabel("W (px)"), srow, 0)
        self.ui_fig_w = self._mk_le_no_row("400")
        settings_grid.addWidget(self.ui_fig_w, srow, 1)
        settings_grid.addWidget(QLabel("H (px)"), srow, 2)
        self.ui_fig_h = self._mk_le_no_row("300")
        settings_grid.addWidget(self.ui_fig_h, srow, 3)
        srow += 1

        # Font / Legend font (paired)
        settings_grid.addWidget(QLabel("Font"), srow, 0)
        self.ui_font_size = self._mk_le_no_row("10")
        settings_grid.addWidget(self.ui_font_size, srow, 1)
        settings_grid.addWidget(QLabel("Legend"), srow, 2)
        self.ui_legend_font_size = self._mk_le_no_row("")
        self.ui_legend_font_size.setPlaceholderText("= Font")
        settings_grid.addWidget(self.ui_legend_font_size, srow, 3)
        srow += 1

        # Background color (saved PNG): empty = transparent
        settings_grid.addWidget(QLabel("BG Color"), srow, 0)
        self.ui_bg_color = self._mk_le_no_row("")
        self.ui_bg_color.setPlaceholderText("transparent")
        settings_grid.addWidget(self.ui_bg_color, srow, 1, 1, 3)
        srow += 1

        # Column stretch: label columns fixed, input columns stretch
        settings_grid.setColumnStretch(0, 0)
        settings_grid.setColumnStretch(1, 1)
        settings_grid.setColumnStretch(2, 0)
        settings_grid.setColumnStretch(3, 1)

        sl.addWidget(form_w)

        # -- [Add Horizontal Line] + [X -> -X if all negative] on same row --
        hline_negx_row = QHBoxLayout()
        self.btn_add_hline = QPushButton("Add H-Line")
        self.btn_add_hline.clicked.connect(self._add_horizontal_line)
        hline_negx_row.addWidget(self.btn_add_hline)
        self.ui_auto_neg_x = self._mk_ckbtn("X -> -X if neg.")
        self.ui_auto_neg_x.toggled.connect(self._apply_auto_neg_x)
        hline_negx_row.addWidget(self.ui_auto_neg_x)
        sl.addLayout(hline_negx_row)

        # -- Auto Assign Colors --
        sep_ac = QFrame(); sep_ac.setFrameShape(QFrame.Shape.HLine); sep_ac.setFrameShadow(QFrame.Shadow.Sunken)
        sl.addWidget(sep_ac)
        ac_row = QHBoxLayout()
        self.btn_auto_color = QPushButton("Auto Assign Colors")
        self.btn_auto_color.setStyleSheet(GlobalStyles.NORMAL_BTN)
        self.btn_auto_color.clicked.connect(self.auto_assign_colors)
        ac_row.addWidget(self.btn_auto_color, stretch=1)
        self.ui_auto_color = QPushButton("Auto")
        self.ui_auto_color.setCheckable(True)
        self.ui_auto_color.setChecked(True)
        self.ui_auto_color.setStyleSheet(GlobalStyles.CHECKABLE_BTN)
        self.ui_auto_color.setFixedWidth(50)
        self.ui_auto_color.setToolTip(
            "Reassign colors automatically whenever a curve is added")
        self.ui_auto_color.toggled.connect(self._auto_save_settings)
        ac_row.addWidget(self.ui_auto_color)
        sl.addLayout(ac_row)

        cs_row = QHBoxLayout()
        self.ui_color_scheme = QComboBox()
        # Populate with categorised color preview icons
        for group_label, cmap_names in GRADIENT_COLORMAP_GROUPS:
            # Add a disabled separator item as category header
            self.ui_color_scheme.addItem(f'[{group_label}]')
            sep_idx = self.ui_color_scheme.count() - 1
            model = self.ui_color_scheme.model()
            item = model.item(sep_idx)
            item.setEnabled(False)
            for cmap_name in cmap_names:
                icon = QtGui.QIcon(_generate_colormap_pixmap(cmap_name))
                self.ui_color_scheme.addItem(icon, cmap_name)
        # Select first real item (skip first header)
        self.ui_color_scheme.setCurrentIndex(1)
        self.ui_color_scheme.setIconSize(QtCore.QSize(120, 16))
        self.ui_color_scheme.currentTextChanged.connect(self._on_color_scheme_changed)
        self.ui_color_scheme.currentTextChanged.connect(self._auto_save_settings)
        cs_row.addWidget(self.ui_color_scheme, stretch=1)
        self.ui_inverse_colors = QPushButton("Inv")
        self.ui_inverse_colors.setCheckable(True)
        self.ui_inverse_colors.setStyleSheet(GlobalStyles.CHECKABLE_BTN)
        self.ui_inverse_colors.setFixedWidth(40)
        self.ui_inverse_colors.toggled.connect(self._on_color_scheme_changed)
        self.ui_inverse_colors.toggled.connect(self._auto_save_settings)
        cs_row.addWidget(self.ui_inverse_colors)
        sl.addLayout(cs_row)

        # -- Update --
        sep_upd = QFrame(); sep_upd.setFrameShape(QFrame.Shape.HLine); sep_upd.setFrameShadow(QFrame.Shadow.Sunken)
        sl.addWidget(sep_upd)
        self.btn_update = QPushButton("Update Plot")
        self.btn_update.setStyleSheet(GlobalStyles.NORMAL_BTN)
        self.btn_update.clicked.connect(self.do_update_plot)
        self.ui_auto_update = self._mk_ckbtn("Auto Update", True)
        update_row = QHBoxLayout()
        update_row.addWidget(self.ui_auto_update)
        update_row.addWidget(self.btn_update)
        sl.addLayout(update_row)

        # -- Save / Delete Preset --
        sep_preset = QFrame(); sep_preset.setFrameShape(QFrame.Shape.HLine); sep_preset.setFrameShadow(QFrame.Shadow.Sunken)
        sl.addWidget(sep_preset)
        p_row = QHBoxLayout()
        btn_sp = QPushButton("Save Preset")
        btn_sp.setStyleSheet(GlobalStyles.NORMAL_BTN)
        btn_sp.clicked.connect(self.save_preset)
        btn_dp = QPushButton("Delete Preset")
        btn_dp.setStyleSheet(GlobalStyles.NORMAL_BTN)
        btn_dp.clicked.connect(self.delete_preset)
        p_row.addWidget(btn_sp)
        p_row.addWidget(btn_dp)
        sl.addLayout(p_row)

        sl.addStretch()
        left_scroll.setWidget(left_w)

        # ═══════ MIDDLE PANEL (detail) ═══════
        middle_scroll = QScrollArea()
        middle_scroll.setWidgetResizable(True)
        middle_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        middle_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.stacked_widget = QStackedWidget()
        self.placeholder = QLabel(
            "Add a Curve or Grid to begin.\n\n"
            "Drag & drop data files here.")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color:#888; font-size:14pt;")
        self.stacked_widget.addWidget(self.placeholder)
        middle_scroll.setWidget(self.stacked_widget)

        # ═══════ RIGHT PANEL (entry list + figure preview) ═══════
        right_w = QWidget()
        right_w.setMinimumWidth(320)
        right_w.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        rl = QVBoxLayout(right_w)
        rl.setContentsMargins(5, 5, 5, 5)

        # Load file buttons
        load_row = QHBoxLayout()
        btn_load_curve = QPushButton("Load Curve File")
        btn_load_curve.setStyleSheet(GlobalStyles.NORMAL_BTN)
        btn_load_curve.clicked.connect(lambda: self._load_curve_file())
        btn_load_grid = QPushButton("Load Grid File")
        btn_load_grid.setStyleSheet(GlobalStyles.NORMAL_BTN)
        btn_load_grid.clicked.connect(lambda: self._load_grid_file())
        btn_load_clip = QPushButton("Load From Clipboard")
        btn_load_clip.setStyleSheet(GlobalStyles.NORMAL_BTN)
        btn_load_clip.clicked.connect(lambda: self._load_from_clipboard())
        for _btn in (btn_load_curve, btn_load_grid, btn_load_clip):
            _btn.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            load_row.addWidget(_btn, 1)
        rl.addLayout(load_row)

        # Add new entry buttons
        ar = QHBoxLayout()
        btn_ac = QPushButton("New Curve")
        btn_ac.setStyleSheet(GlobalStyles.NORMAL_BTN)
        btn_ac.clicked.connect(lambda: self.add_entry('curve'))
        btn_ag = QPushButton("New Grid")
        btn_ag.setStyleSheet(GlobalStyles.NORMAL_BTN)
        btn_ag.clicked.connect(lambda: self.add_entry('grid'))
        ar.addWidget(btn_ac)
        ar.addWidget(btn_ag)
        rl.addLayout(ar)



        # Property management dropdown
        self.ui_property_dropdown = QComboBox()
        self.ui_property_dropdown.addItems(INLINE_PROPERTY_OPTIONS)
        self.ui_property_dropdown.currentTextChanged.connect(
            self._on_property_dropdown_changed)
        self.ui_property_dropdown.setFixedHeight(30)

        # Move up/down buttons and Apply to All
        move_row = QHBoxLayout()
        btn_move_up = QPushButton("↑")
        btn_move_up.clicked.connect(self._move_selected_entry_up)
        btn_move_up.setFixedSize(30, 30)
        self.btn_move_up = btn_move_up
        btn_move_down = QPushButton("↓")
        btn_move_down.clicked.connect(self._move_selected_entry_down)
        btn_move_down.setFixedSize(30, 30)
        self.btn_move_down = btn_move_down
        self.btn_apply_all = QPushButton("Apply to All")
        self.btn_apply_all.setFixedHeight(30)
        self.btn_apply_all.setToolTip("Apply the selected entry's inline property value to all enabled entries")
        self.btn_apply_all.clicked.connect(self._apply_inline_property_to_all)
        move_row.addWidget(self.ui_property_dropdown)
        move_row.addWidget(self.btn_apply_all)
        move_row.addWidget(btn_move_up)
        move_row.addWidget(btn_move_down)
        rl.addLayout(move_row)
        rl.addSpacing(4)

        # Bulk label editor: edit every curve label in one multi-line window
        self.btn_edit_labels = QPushButton("Edit Labels")
        self.btn_edit_labels.setStyleSheet(GlobalStyles.NORMAL_BTN)
        self.btn_edit_labels.setToolTip(
            "Edit all curve labels at once, one per line "
            "(Alt+drag for column editing)")
        self.btn_edit_labels.clicked.connect(self._open_bulk_label_editor)
        rl.addWidget(self.btn_edit_labels)
        rl.addSpacing(4)

        self.entry_scroll = QScrollArea()
        self.entry_scroll.setWidgetResizable(True)
        # Show a horizontal scrollbar when a curve label is wider than the pane,
        # so long labels can be read in full by scrolling left/right. Each entry
        # button keeps a minimum width matching its text (see EntryButton), so
        # the inner widget grows past the viewport and this scrollbar appears.
        self.entry_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.entry_list_widget = QWidget()
        self.entry_list_layout = QVBoxLayout(self.entry_list_widget)
        self.entry_list_layout.setContentsMargins(0, 0, 12, 0)
        self.entry_list_layout.setSpacing(3)
        self.entry_list_layout.addStretch()
        self.entry_scroll.setWidget(self.entry_list_widget)

        # Delete Curves / Clear All buttons below the curve list
        del_clear_row = QHBoxLayout()
        btn_del_curves = QPushButton("Delete Curves")
        btn_del_curves.setStyleSheet(GlobalStyles.NORMAL_BTN)
        btn_del_curves.clicked.connect(self._confirm_delete_selected_entries)
        btn_cl = QPushButton("Clear All")
        btn_cl.setStyleSheet(GlobalStyles.NORMAL_BTN)
        btn_cl.clicked.connect(self._confirm_clear_entries)
        del_clear_row.addWidget(btn_del_curves)
        del_clear_row.addWidget(btn_cl)

        # Shared figure preview: an embedded copy of the floating figure
        # window, permanently docked at the bottom of the right panel.
        # The Plot widget itself is created lazily on the first plot update.
        self.preview_container = QWidget()
        self.preview_layout = QVBoxLayout(self.preview_container)
        self.preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_layout.setSpacing(0)
        self.preview_placeholder = QLabel("Figure preview will appear here.")
        self.preview_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_placeholder.setStyleSheet("color:#888; font-size:12pt;")
        self.preview_layout.addWidget(self.preview_placeholder)

        # Curve list (+ Delete/Clear buttons) on top, preview below, with a
        # user-adjustable splitter whose ratio is persisted.
        list_part = QWidget()
        list_part_l = QVBoxLayout(list_part)
        list_part_l.setContentsMargins(0, 0, 0, 0)
        list_part_l.addWidget(self.entry_scroll)
        list_part_l.addLayout(del_clear_row)

        self.right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.right_splitter.addWidget(list_part)
        self.right_splitter.addWidget(self.preview_container)
        self.right_splitter.setStretchFactor(0, 1)
        self.right_splitter.setStretchFactor(1, 1)
        self.right_splitter.splitterMoved.connect(self._on_right_splitter_moved)
        rl.addWidget(self.right_splitter, stretch=1)
        self._apply_right_split_ratio()

        # ── Shared Result area (docked below the active entry's data editor) ──
        self.result_group = QGroupBox("Result")
        self.result_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        result_layout = QVBoxLayout(self.result_group)
        result_layout.setContentsMargins(4, 4, 4, 4)

        # Program Output (Stdout) which also contains integration results
        self.ui_program_output = QtWidgets.QTextEdit()
        self.ui_program_output.setReadOnly(True)
        self.ui_program_output.setFont(QFont("Consolas", 9))
        self.ui_program_output.setMinimumHeight(60)
        self.ui_program_output.setPlaceholderText("Program output will appear here...")
        result_layout.addWidget(self.ui_program_output)

        # Not added to any layout here: hosted inside the active entry's
        # bottom slot, moving between entries as the selection changes.
        self.result_group.hide()

        # Stdout Redirect
        self._stdout_redirector = StreamRedirector()
        self._stdout_redirector.text_written.connect(self.ui_program_output.insertPlainText)
        self._stdout_redirector.text_written.connect(lambda: self.ui_program_output.ensureCursorVisible())
        sys.stdout = self._stdout_redirector

        # ═══════ Assemble ═══════
        left_scroll.setFixedWidth(340)
        right_w.setFixedWidth(340) # Ensure exact same width as left panel

        main_h.addWidget(left_scroll)
        main_h.addWidget(middle_scroll, stretch=1)
        main_h.addWidget(right_w)

    # ─── form-builder helpers ───
    def _mk_le(self, label, default):
        le = QLineEdit(default)
        le.editingFinished.connect(self.request_update)
        le.editingFinished.connect(self._auto_save_settings)
        le.textChanged.connect(self._auto_save_settings)
        self.form_layout.addRow(label, le)
        return le

    def _mk_le_no_row(self, default):
        """Create a QLineEdit with update/save signals but without adding to form layout."""
        le = QLineEdit(default)
        le.editingFinished.connect(self.request_update)
        le.editingFinished.connect(self._auto_save_settings)
        le.textChanged.connect(self._auto_save_settings)
        return le

    def _mk_ckbtn(self, text, checked=False):
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setStyleSheet(GlobalStyles.CHECKABLE_BTN)
        btn.toggled.connect(self.request_update)
        btn.toggled.connect(self._auto_save_settings)
        return btn

    def _bring_plot_to_front(self):
        """One-shot raise of the plot window: briefly enable keep_front then restore it."""
        if self.plot_window is None or not self.plot_window.isVisible():
            return
        self.plot_window.keep_front = True
        self.plot_window.keep_front = self.ui_keep_front.isChecked()
        self.plot_window.raise_()
        self.plot_window.activateWindow()

    def _get_selected_entries(self):
        return [entry for entry in self.entries if entry in self._selected_entries]

    @staticmethod
    def _modifier_value(modifiers):
        if modifiers is None:
            return 0
        return getattr(modifiers, 'value', modifiers)

    def _apply_selection_state(self, selected_entries, active_entry=None, anchor_entry=None):
        selected: set[DataEntryWidget] = {
            entry for entry in selected_entries
            if entry is not None and entry in self.entries
        }

        if not selected and active_entry is not None and active_entry in self.entries:
            selected = {active_entry}
        if not selected and self._active_entry is not None and self._active_entry in self.entries:
            selected = {self._active_entry}
        if not selected and self.entries:
            selected = {self.entries[0]}

        if active_entry not in selected:
            if self._active_entry in selected:
                active_entry = self._active_entry
            else:
                active_entry = next(iter(selected), None)

        if anchor_entry not in self.entries:
            anchor_entry = active_entry

        self._selected_entries = selected
        self._active_entry = active_entry
        self._selection_anchor = anchor_entry

        for entry, btn in zip(self.entries, self.entry_buttons):
            btn.blockSignals(True)
            btn.setChecked(entry in self._selected_entries)
            btn.blockSignals(False)

        if self._active_entry in self.entries:
            self.stacked_widget.setCurrentWidget(self._active_entry)
        elif not self.entries:
            self.stacked_widget.setCurrentWidget(self.placeholder)

        self._host_result_in_active_entry()
        self._refresh_bulk_action_buttons()

    def _set_active_entry(self, widget):
        if widget in self._selected_entries:
            self._apply_selection_state(
                self._selected_entries,
                active_entry=widget,
                anchor_entry=self._selection_anchor,
            )
        elif widget in self.entries:
            self._apply_selection_state({widget}, active_entry=widget, anchor_entry=widget)

    def _refresh_bulk_action_buttons(self):
        selected_entries = self._get_selected_entries()
        selected_curves = [entry for entry in selected_entries if isinstance(entry, CurveEntry)]

        if hasattr(self, 'btn_apply_all'):
            multi_selected = len(selected_entries) > 1
            self.btn_apply_all.setText("Apply to Selected" if multi_selected else "Apply to All")
            self.btn_apply_all.setToolTip(
                "Apply the selected entry's inline property value to the selected entries"
                if multi_selected else
                "Apply the selected entry's inline property value to all enabled entries"
            )

        if hasattr(self, 'btn_auto_color'):
            self.btn_auto_color.setText(
                "Auto Assign Colors to Selected" if len(selected_curves) > 1 else "Auto Assign Colors"
            )

    # ════════════════ CURVE COPY / PASTE ════════════════════
    def keyPressEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if event.key() == Qt.Key.Key_C:
                if self._copy_curve_properties():
                    event.accept()
                    return
            elif event.key() == Qt.Key.Key_V:
                if self._paste_curve_properties():
                    event.accept()
                    return
        super().keyPressEvent(event)

    def _copy_curve_properties(self):
        """Copy the active CurveEntry's properties. Returns True on success."""
        entry = self._active_entry
        if not isinstance(entry, CurveEntry):
            return False
        props = {}
        for _display, attr_name, widget_type, _section in CURVE_COPY_PROPS:
            widget = getattr(entry, attr_name, None)
            if widget is None:
                continue
            if widget_type in ('editable_combo', 'combo'):
                props[attr_name] = widget.currentText()
            elif widget_type == 'line_edit':
                props[attr_name] = widget.text()
            elif widget_type == 'checkable':
                props[attr_name] = widget.isChecked()
        self._copied_curve_props = props
        return True

    def _paste_curve_properties(self):
        """Paste copied properties to selected CurveEntries via dialog. Returns True on success."""
        if not self._copied_curve_props:
            return False
        targets = [e for e in self._get_selected_entries() if isinstance(e, CurveEntry)]
        if not targets:
            return False

        dlg = CurvePasteDialog(self._copied_curve_props, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return True  # consumed the key even if cancelled

        selected_attrs = dlg.get_selected_attrs()
        if not selected_attrs:
            return True

        self._suppress_update = True
        for entry in targets:
            for _display, attr_name, widget_type, _section in CURVE_COPY_PROPS:
                if attr_name not in selected_attrs:
                    continue
                if attr_name not in self._copied_curve_props:
                    continue
                widget = getattr(entry, attr_name, None)
                if widget is None:
                    continue
                value = self._copied_curve_props[attr_name]
                if widget_type in ('editable_combo', 'combo'):
                    widget.setCurrentText(value)
                elif widget_type == 'line_edit':
                    widget.setText(value)
                elif widget_type == 'checkable':
                    widget.setChecked(value)
        self._suppress_update = False
        self.request_update(from_entry_edit=True)
        return True

    # ════════════════ ENTRY MANAGEMENT ════════════════════
    def _load_curve_file(self):
        """Open file dialog and load a curve data file (CSV, XLSX, JSON)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Curve Data", "",
            "Data Files (*.csv *.xlsx *.json);;All Files (*.*)")
        if path:
            self.add_entry('curve', data_file=path)

    def _load_grid_file(self):
        """Open file dialog and load a grid data file (CSV, XLSX, JSON)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Grid Data", "",
            "Data Files (*.csv *.xlsx *.json);;All Files (*.*)")
        if path:
            self.add_entry('grid', data_file=path)

    def _load_from_clipboard(self):
        """Import curve data pasted from the clipboard, mirroring the drop
        behaviour: two numeric columns import directly, more open the
        column-picker dialog."""
        clip = QApplication.clipboard()
        text = clip.text() if clip is not None else ""
        if not text.strip():
            QMessageBox.warning(
                self, "Clipboard Empty",
                "The clipboard contains no text to load.")
            return
        delim_keys = self.get_drop_delim_keys()
        try:
            headers, columns = parse_data_text_columns(
                text, delim_keys=delim_keys)
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))
            return
        if len(columns) < 2:
            QMessageBox.warning(
                self, "Load Error",
                "The clipboard has fewer than 2 numeric columns to plot.")
            return
        if len(columns) == 2:
            df = columns_to_numeric_df([columns[0], columns[1]])
            if df.empty:
                QMessageBox.warning(
                    self, "Load Error",
                    "The clipboard has no rows where both columns are numeric.")
                return
            self._add_curve_from_dataframe(df, "Clipboard", "Clipboard")
            return
        dlg = ColumnPickerDialog(
            "Clipboard", self, parent=self, raw_text=text)
        dlg.curve_requested.connect(self._add_curve_from_dataframe)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.show()

    def _add_horizontal_line(self):
        """Add a horizontal line at user-specified Y value spanning all curves' X range."""
        # Get all curve entries (including disabled ones for X range calculation)
        curve_entries = [e for e in self.entries if isinstance(e, CurveEntry)]
        if not curve_entries:
            QMessageBox.warning(
                self, "No Curves",
                "Cannot add horizontal line: Horizontal lines should be added after all other lines has been added.")
            return

        # Prompt for Y value
        y_value, ok = QInputDialog.getDouble(
            self, "Add Horizontal Line",
            "Enter Y value for horizontal line:",
            decimals=6)
        if not ok:
            return

        # Calculate X range from all curves (including disabled ones)
        all_x_values = []
        for entry in curve_entries:
            X, Y, _ = entry.get_data_arrays()
            if X is not None and len(X) > 0:
                all_x_values.extend(X)

        if not all_x_values:
            QMessageBox.warning(
                self, "No Data",
                "Cannot determine X range: No valid curve data.")
            return

        x_min = float(np.min(all_x_values))
        x_max = float(np.max(all_x_values))

        # Create horizontal line data
        x_data = [x_min, x_max]
        y_data = [y_value, y_value]

        # Create new curve entry
        new_entry = self.add_entry('curve')
        assert isinstance(new_entry, CurveEntry), "Expected CurveEntry"
        
        # Set the data
        data_df = pd.DataFrame({'X': x_data, 'Y': y_data})
        new_entry.loaded_data = data_df
        new_entry._display_data_as_text()

        # Configure curve properties
        # Leave label empty to hide from legend, but set custom button text
        new_entry._custom_button_text = f"{HORIZONTAL_LINE_LABEL_PREFIX}{y_value}"  # type: ignore[attr-defined]
        new_entry.inp_label.setText("")  # Empty label = no legend entry
        new_entry.inp_color.setCurrentText("")
        new_entry.inp_color.lineEdit().setText(HORIZONTAL_LINE_COLOR)
        new_entry.inp_width.setText(str(HORIZONTAL_LINE_WIDTH))
        new_entry.inp_fmt.setCurrentText([k for k, v in LINE_FORMAT_DISPLAY.items() if v == HORIZONTAL_LINE_FORMAT][0] if HORIZONTAL_LINE_FORMAT in LINE_FORMAT_DISPLAY.values() else "")
        if not new_entry.inp_fmt.currentText():
            new_entry.inp_fmt.lineEdit().setText(HORIZONTAL_LINE_FORMAT)
        new_entry.inp_marker.setCurrentText("")
        new_entry.chk_interp.setChecked(False)

        # Update button text to reflect it's a baseline
        self._update_entry_button_text(new_entry)
        
        # Trigger update
        self.request_update()

    def add_entry(self, type_, data_file=None, select_new=True):
        widget = CurveEntry() if type_ == 'curve' else GridEntry()
        widget.data_changed_signal.connect(lambda: self.request_update(from_entry_edit=True))
        # Also update horizontal lines when curve data changes
        if isinstance(widget, CurveEntry):
            widget.data_changed_signal.connect(self._on_curve_data_changed)
        widget.delete_requested_signal.connect(
            lambda w=widget: self.remove_entry(w))
        if isinstance(widget, CurveEntry):
            widget._compute_hook = self._compute_curve_data
            widget._align_hook = self._compute_align
            widget.label_changed_signal.connect(
                lambda _t, w=widget: self._update_entry_button_text(w))
            widget.btn_integrate.clicked.connect(
                lambda _c, w=widget: self._compute_integration(w))

        # Adding a new entry while in multi-frame mode breaks the
        # entry ↔ frame-curve mapping, so discard stored frames.
        # (_suppress_update is True during initial loading, so this only
        # triggers for user-initiated additions.)
        if not self._suppress_update and (self._stored_curve_frames or self._stored_grid_frames):
            self._clear_stored_frames()

        self.entries.append(widget)
        self.stacked_widget.addWidget(widget)
        # Persist the editor/log split ratio when the user drags it
        widget.left_v_splitter.splitterMoved.connect(
            lambda _pos, _idx, w=widget: self._on_left_splitter_moved(w))

        # row in right list: [checkbox] [button]
        row_w = QWidget()
        row_l = QHBoxLayout(row_w)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.setSpacing(2)

        chk = QtWidgets.QCheckBox()
        chk.setChecked(True)
        chk.setFixedWidth(30)
        chk.setFixedHeight(25)
        chk.toggled.connect(
            lambda checked, w=widget: self._on_entry_enabled(w, checked))

        idx = len(self.entries)
        type_label = "Curve" if type_ == 'curve' else "Grid"
        btn = EntryButton(f"{type_label} {idx}")
        btn.setCheckable(True)
        btn.setStyleSheet(GlobalStyles.ENTRY_BTN)
        btn.pressed.connect(
            lambda b=btn: b.setProperty('_selection_modifiers', self._modifier_value(QApplication.keyboardModifiers())))
        btn.clicked.connect(
            lambda _c, w=widget, b=btn: self.select_entry(w, b.property('_selection_modifiers')))
        # Drag the button up/down to restack plot layers.
        btn.drag_move.connect(self._on_entry_drag_move)
        btn.drag_drop.connect(self._on_entry_drag_drop)
        # Keep a newly-added entry's label locked if the bulk dialog is open.
        if self._reorder_locked and isinstance(widget, CurveEntry):
            widget.inp_label.setEnabled(False)

        row_l.addWidget(chk)

        # Curve code box: a short alias (e.g. A1) used to reference this curve
        # from another curve's Label formula, e.g. ({{A1}} - 2*{{B2}})/2.
        # Placed to the LEFT of the label: a long label now overflows to the
        # right (with a horizontal scrollbar) instead of pushing this box off
        # the right edge, so the code stays visible at the default position.
        if isinstance(widget, CurveEntry):
            code_edit = QLineEdit()
            code_edit.setFixedWidth(40)
            code_edit.setFixedHeight(25)
            code_edit.setPlaceholderText("ID")
            code_edit.setToolTip(
                "Curve code (e.g. A1) — reference this curve from another\n"
                "curve's Label formula, e.g. ({{A1}} - 2*{{B2}})/2.\n"
                "Letters/digits, any length; leave blank if unused.")
            code_edit.setValidator(QtGui.QRegularExpressionValidator(
                QtCore.QRegularExpression(r'[A-Za-z0-9]*')))
            code_edit.setText(getattr(widget, 'curve_code', '') or '')
            code_edit.editingFinished.connect(
                lambda w=widget: self._on_code_changed(w))
            widget._code_edit = code_edit
            row_l.addWidget(code_edit)

        row_l.addWidget(btn, stretch=1)

        # Inline property widget container
        prop_container = QWidget()
        prop_layout = QHBoxLayout(prop_container)
        prop_layout.setContentsMargins(0, 0, 0, 0)
        prop_layout.setSpacing(0)
        prop_container.hide()
        self.entry_prop_containers.append(prop_container)
        row_l.addWidget(prop_container)

        self.entry_buttons.append(btn)
        self.entry_checkboxes.append(chk)
        cnt = self.entry_list_layout.count()
        self.entry_list_layout.insertWidget(cnt - 1, row_w)  # before stretch

        if data_file:
            widget.load_file(data_file)
        if select_new:
            self.select_entry(widget)
        else:
            self._refresh_bulk_action_buttons()
        # Rebuild inline property widget for the new entry
        self._rebuild_inline_property_widget(len(self.entries) - 1)
        # Update horizontal line button state
        self._update_hline_button_state()
        # Update horizontal lines X range if this is a regular curve
        if isinstance(widget, CurveEntry) and not self._is_horizontal_line(widget):
            self._update_horizontal_lines()
        # Auto-apply X -> -X if enabled and curve's X is all negative
        if self._maybe_apply_neg_x_to_entry(widget):
            self.request_update(from_entry_edit=True)
        # Auto-reassign colors after adding a new curve via file-load path.
        # The DataFrame import path also calls this directly after populating
        # data; calling it here is a no-op when data isn't loaded yet.
        if data_file and isinstance(widget, CurveEntry) and not self._is_horizontal_line(widget):
            self._auto_color_if_enabled()
        return widget

    def _is_horizontal_line(self, entry):
        """Check if an entry is a horizontal line (baseline)."""
        if not isinstance(entry, CurveEntry):
            return False
        return hasattr(entry, '_custom_button_text') and \
               isinstance(getattr(entry, '_custom_button_text', None), str) and \
               getattr(entry, '_custom_button_text', '').startswith(HORIZONTAL_LINE_LABEL_PREFIX)

    def _update_horizontal_lines(self):
        """Update X range of all horizontal lines to cover all regular curves."""
        # Find all horizontal lines and regular curves
        horizontal_lines = []
        regular_curves = []
        for entry in self.entries:
            if isinstance(entry, CurveEntry):
                if self._is_horizontal_line(entry):
                    horizontal_lines.append(entry)
                else:
                    regular_curves.append(entry)
        
        if not horizontal_lines or not regular_curves:
            return
        
        # Calculate X range from all regular curves
        all_x_values = []
        for entry in regular_curves:
            X, Y, _ = entry.get_data_arrays()
            if X is not None and len(X) > 0:
                all_x_values.extend(X)
        
        if not all_x_values:
            return
        
        x_min = float(np.min(all_x_values))
        x_max = float(np.max(all_x_values))
        
        # Update each horizontal line
        for hline in horizontal_lines:
            if hline.loaded_data is not None and len(hline.loaded_data) >= 2:
                # Get the Y value from the horizontal line
                y_value = float(hline.loaded_data['Y'].iloc[0])
                # Update data with new X range
                hline.loaded_data = pd.DataFrame({'X': [x_min, x_max], 'Y': [y_value, y_value]})
                hline._display_data_as_text()

    def remove_entry(self, widget):
        if widget not in self.entries:
            return
        idx = self.entries.index(widget)

        # Update stored multi-frame data: remove the corresponding curve/grid
        if isinstance(widget, CurveEntry) and self._stored_curve_frames:
            curve_idx = sum(1 for e in self.entries[:idx] if isinstance(e, CurveEntry))
            for frame in self._stored_curve_frames:
                if curve_idx < len(frame):
                    frame.pop(curve_idx)
        elif isinstance(widget, GridEntry) and self._stored_grid_frames:
            grid_idx = sum(1 for e in self.entries[:idx] if isinstance(e, GridEntry))
            for frame in self._stored_grid_frames:
                if grid_idx < len(frame):
                    frame.pop(grid_idx)

        self.entries.pop(idx)
        self._selected_entries.discard(widget)
        if self._active_entry is widget:
            self._active_entry = None
        if self._selection_anchor is widget:
            self._selection_anchor = None
        btn = self.entry_buttons.pop(idx)
        self.entry_checkboxes.pop(idx)
        self.entry_prop_containers.pop(idx)
        row_w = btn.parentWidget()
        self.entry_list_layout.removeWidget(row_w)
        row_w.deleteLater()
        self.stacked_widget.removeWidget(widget)
        # Rescue the shared Result log before the entry (its potential host) dies
        self._detach_result_from(widget)
        widget.deleteLater()
        self._update_all_button_texts()

        if self._selected_entries:
            active_entry = self._active_entry if self._active_entry in self._selected_entries else None
            if active_entry is None:
                active_entry = next(iter(self._selected_entries))
            anchor_entry = self._selection_anchor if self._selection_anchor in self.entries else active_entry
            self._apply_selection_state(self._selected_entries, active_entry=active_entry, anchor_entry=anchor_entry)
        elif self.entries:
            next_entry = self.entries[min(idx, len(self.entries) - 1)]
            self._apply_selection_state({next_entry}, active_entry=next_entry, anchor_entry=next_entry)
        else:
            self._selected_entries = set()
            self._active_entry = None
            self._selection_anchor = None
            self.stacked_widget.setCurrentWidget(self.placeholder)
            self._refresh_bulk_action_buttons()
        # Update horizontal line button state
        self._update_hline_button_state()
        # Update horizontal lines X range after removing an entry
        if isinstance(widget, CurveEntry) and not self._is_horizontal_line(widget):
            self._update_horizontal_lines()
        self.request_update(from_entry_edit=True)

    def select_entry(self, widget, modifiers=None):
        if widget not in self.entries:
            return

        modifiers = self._modifier_value(modifiers)
        ctrl_pressed = bool(modifiers & self._modifier_value(Qt.KeyboardModifier.ControlModifier))
        shift_pressed = bool(modifiers & self._modifier_value(Qt.KeyboardModifier.ShiftModifier))

        if shift_pressed and self._selection_anchor in self.entries:
            anchor_idx = self.entries.index(self._selection_anchor)
            idx = self.entries.index(widget)
            range_entries = {
                self.entries[i] for i in range(min(anchor_idx, idx), max(anchor_idx, idx) + 1)
            }
            if ctrl_pressed:
                selected_entries = set(self._selected_entries).union(range_entries)
            else:
                selected_entries = range_entries
            anchor_entry = self._selection_anchor
        elif ctrl_pressed:
            selected_entries = set(self._selected_entries)
            if widget in selected_entries and len(selected_entries) > 1:
                selected_entries.remove(widget)
            else:
                selected_entries.add(widget)
            anchor_entry = widget
        else:
            selected_entries = {widget}
            anchor_entry = widget

        self._apply_selection_state(selected_entries, active_entry=widget, anchor_entry=anchor_entry)
        self._update_hline_button_state()

    def _select_entry_for_inline_edit(self, entry):
        if entry in self.entries:
            self._set_active_entry(entry)

    def _get_entry_button_text(self, entry, idx):
        if hasattr(entry, '_custom_button_text') and entry._custom_button_text:
            return entry._custom_button_text
        if isinstance(entry, CurveEntry):
            label_text = entry.inp_label.text().strip()
            if label_text:
                return label_text
            return f"Curve {idx + 1}"
        return f"Grid {idx + 1}"

    def _update_hline_button_state(self):
        """Enable/disable horizontal line button based on whether any Grid entries exist."""
        has_grid = any(isinstance(entry, GridEntry) for entry in self.entries)
        self.btn_add_hline.setEnabled(not has_grid)

    def _on_curve_data_changed(self):
        """Called when any curve's data changes - update horizontal lines and apply auto-neg-x."""
        sender = self.sender()
        for entry in self.entries:
            if entry.data_changed_signal == sender:
                if isinstance(entry, CurveEntry) and not self._is_horizontal_line(entry):
                    self._update_horizontal_lines()
                    if self._maybe_apply_neg_x_to_entry(entry):
                        # neg_x just got applied — re-render so the flip is visible
                        self.request_update(from_entry_edit=True)
                break

    def _confirm_delete_selected_entries(self):
        """Delete selected entries after confirmation."""
        selected = list(self._selected_entries)
        if not selected:
            return
        reply = QMessageBox.question(
            self, "Delete Curves",
            f"Remove {len(selected)} selected {'entry' if len(selected) == 1 else 'entries'}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self._suppress_update = True
            for w in selected:
                self.remove_entry(w)
            self._suppress_update = False
            self.request_update(from_entry_edit=True)

    def _confirm_clear_entries(self):
        """Show confirmation dialog before clearing all entries."""
        if not self.entries:
            return
        reply = QMessageBox.question(
            self, "Clear All",
            f"Remove all {len(self.entries)} entries?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.clear_entries()

    def clear_entries(self):
        self._suppress_update = True
        while self.entries:
            self.remove_entry(self.entries[0])
        self._suppress_update = False
        self._clear_stored_frames()
        self._selected_entries = set()
        self._active_entry = None
        self._selection_anchor = None
        self.stacked_widget.setCurrentWidget(self.placeholder)
        # Update horizontal line button state
        self._update_hline_button_state()
        self._refresh_bulk_action_buttons()
        self.request_update(from_entry_edit=True)

    def _on_entry_enabled(self, widget, checked):
        widget._is_enabled = checked
        # Grid exclusivity: when enabling a Grid, disable all other Grids
        if checked and isinstance(widget, GridEntry):
            for i, entry in enumerate(self.entries):
                if isinstance(entry, GridEntry) and entry is not widget:
                    if entry._is_enabled:
                        entry._is_enabled = False
                        self.entry_checkboxes[i].blockSignals(True)
                        self.entry_checkboxes[i].setChecked(False)
                        self.entry_checkboxes[i].blockSignals(False)
        # Update horizontal line button state
        self._update_hline_button_state()
        self.request_update(from_entry_edit=True)

    def _update_entry_button_text(self, widget):
        if widget not in self.entries:
            return
        idx = self.entries.index(widget)
        self.entry_buttons[idx].setText(self._get_entry_button_text(widget, idx))

    def _update_all_button_texts(self):
        for i, (entry, btn) in enumerate(
                zip(self.entries, self.entry_buttons)):
            btn.setText(self._get_entry_button_text(entry, i))
        self._refresh_bulk_action_buttons()

    def _open_bulk_label_editor(self):
        """Edit every curve label at once in a non-modal, always-on-top window.

        Each line maps to one curve (horizontal lines/grids have no editable
        label and are excluded). Curves are bound by identity, so the line ↔
        curve mapping survives anything that happens in the main window; while
        the window is open, label editing and reordering there are locked."""
        if self._label_dialog is not None:
            self._label_dialog.raise_()
            self._label_dialog.activateWindow()
            return
        curves = [e for e in self.entries
                  if isinstance(e, CurveEntry) and not self._is_horizontal_line(e)]
        if not curves:
            QMessageBox.information(
                self, "Edit Labels", "There are no curves to label.")
            return
        labels = [c.inp_label.text() for c in curves]
        dlg = BulkLabelEditorDialog(labels, len(curves), self)
        self._label_dialog = dlg
        self._label_dialog_curves = list(curves)   # bind labels by identity
        self._set_label_order_locked(True)
        dlg.finished.connect(self._on_label_dialog_finished)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _on_label_dialog_finished(self, result):
        dlg = self._label_dialog
        curves = self._label_dialog_curves or []
        self._label_dialog = None
        self._label_dialog_curves = None
        self._set_label_order_locked(False)
        if dlg is None:
            return
        if result == int(QDialog.DialogCode.Accepted):
            new_lines = dlg.get_labels()
            changed = False
            for i, curve in enumerate(curves):
                if curve not in self.entries:
                    continue   # curve was deleted while the window was open
                new_label = new_lines[i] if i < len(new_lines) else ""
                if curve.inp_label.text() != new_label:
                    curve.inp_label.setText(new_label)
                    changed = True
            if changed:
                self._update_all_button_texts()
                self.request_update(from_entry_edit=True)
                self._auto_save_settings()
        dlg.deleteLater()

    def _set_label_order_locked(self, locked):
        """While the bulk-label window is open, lock label editing and curve
        reordering in the main window (other edits stay available)."""
        self._reorder_locked = locked
        enabled = not locked
        for entry in self.entries:
            if isinstance(entry, CurveEntry):
                entry.inp_label.setEnabled(enabled)
        # Inline right-panel editors (incl. inline Label) — the middle-pane
        # editor stays enabled for colours/widths.
        for container in self.entry_prop_containers:
            container.setEnabled(enabled)
        for btn_attr in ('btn_move_up', 'btn_move_down', 'btn_edit_labels'):
            btn = getattr(self, btn_attr, None)
            if btn is not None:
                btn.setEnabled(enabled)

    # ════════════════ MOVE ENTRIES UP/DOWN ═════════════════
    def _get_selected_index(self):
        """Return the index of the currently selected entry, or -1."""
        if self._active_entry in self.entries:
            return self.entries.index(self._active_entry)
        selected_entries = self._get_selected_entries()
        if selected_entries:
            return self.entries.index(selected_entries[0])
        return -1

    def _swap_entries(self, idx_a, idx_b):
        """Swap two entries in the list (data structures and UI row widgets)."""
        if idx_a == idx_b:
            return
        # Ensure idx_a < idx_b for consistency
        if idx_a > idx_b:
            idx_a, idx_b = idx_b, idx_a

        # Update stored frames if both entries are the same type
        entry_a, entry_b = self.entries[idx_a], self.entries[idx_b]
        if isinstance(entry_a, CurveEntry) and isinstance(entry_b, CurveEntry) and self._stored_curve_frames:
            curve_index_a = sum(1 for e in self.entries[:idx_a] if isinstance(e, CurveEntry))
            curve_index_b = sum(1 for e in self.entries[:idx_b] if isinstance(e, CurveEntry))
            for frame in self._stored_curve_frames:
                if curve_index_a < len(frame) and curve_index_b < len(frame):
                    frame[curve_index_a], frame[curve_index_b] = frame[curve_index_b], frame[curve_index_a]
        elif isinstance(entry_a, GridEntry) and isinstance(entry_b, GridEntry) and self._stored_grid_frames:
            grid_index_a = sum(1 for e in self.entries[:idx_a] if isinstance(e, GridEntry))
            grid_index_b = sum(1 for e in self.entries[:idx_b] if isinstance(e, GridEntry))
            for frame in self._stored_grid_frames:
                if grid_index_a < len(frame) and grid_index_b < len(frame):
                    frame[grid_index_a], frame[grid_index_b] = frame[grid_index_b], frame[grid_index_a]

        # Swap in data lists
        self.entries[idx_a], self.entries[idx_b] = self.entries[idx_b], self.entries[idx_a]
        self.entry_buttons[idx_a], self.entry_buttons[idx_b] = self.entry_buttons[idx_b], self.entry_buttons[idx_a]
        self.entry_checkboxes[idx_a], self.entry_checkboxes[idx_b] = self.entry_checkboxes[idx_b], self.entry_checkboxes[idx_a]
        self.entry_prop_containers[idx_a], self.entry_prop_containers[idx_b] = self.entry_prop_containers[idx_b], self.entry_prop_containers[idx_a]

        # Swap visual row widgets in layout
        # The layout has entry rows at positions 0..n-1, then a stretch at n
        row_a = self.entry_buttons[idx_a].parentWidget()
        row_b = self.entry_buttons[idx_b].parentWidget()
        self.entry_list_layout.removeWidget(row_a)
        self.entry_list_layout.removeWidget(row_b)
        # Re-insert: idx_a first (smaller index), then idx_b
        self.entry_list_layout.insertWidget(idx_a, row_a)
        self.entry_list_layout.insertWidget(idx_b, row_b)

        self._update_all_button_texts()

    def _move_selected_entry_up(self):
        idx = self._get_selected_index()
        if idx <= 0:
            return
        self._swap_entries(idx, idx - 1)
        self._rebuild_all_inline_property_widgets()
        self.request_update(from_entry_edit=True)

    def _move_selected_entry_down(self):
        idx = self._get_selected_index()
        if idx < 0 or idx >= len(self.entries) - 1:
            return
        self._swap_entries(idx, idx + 1)
        self._rebuild_all_inline_property_widgets()
        self.request_update(from_entry_edit=True)

    def _move_entry(self, from_idx, to_idx):
        """Move an entry to a new position, restacking plot layers to match.

        Implemented as a walk of adjacent swaps so all the bookkeeping in
        ``_swap_entries`` (lists, row widgets, stored multi-frame data) stays
        correct."""
        if from_idx == to_idx:
            return
        step = 1 if to_idx > from_idx else -1
        i = from_idx
        while i != to_idx:
            self._swap_entries(i, i + step)
            i += step
        self._rebuild_all_inline_property_widgets()
        self.request_update(from_entry_edit=True)

    # ════════════════ DRAG-TO-REORDER (curve buttons) ═══════
    def _drag_insertion_index(self, global_pos):
        """Slot in [0..n] where a drop at *global_pos* would insert a row."""
        container = self.entry_list_widget
        local_y = container.mapFromGlobal(global_pos).y()
        idx = 0
        for i, b in enumerate(self.entry_buttons):
            row = b.parentWidget()
            top = row.mapTo(container, QtCore.QPoint(0, 0)).y()
            if local_y > top + row.height() / 2:
                idx = i + 1
            else:
                break
        return idx

    def _show_drop_indicator(self, insert_idx):
        container = self.entry_list_widget
        if self._drop_indicator is None:
            self._drop_indicator = QFrame(container)
            self._drop_indicator.setStyleSheet(
                "background-color:#1976D2; border:none;")
            self._drop_indicator.setFixedHeight(3)
        n = len(self.entry_buttons)
        if n == 0:
            return
        if insert_idx >= n:
            row = self.entry_buttons[n - 1].parentWidget()
            y = row.mapTo(container, QtCore.QPoint(0, row.height())).y()
        else:
            row = self.entry_buttons[insert_idx].parentWidget()
            y = row.mapTo(container, QtCore.QPoint(0, 0)).y()
        self._drop_indicator.setGeometry(0, max(0, y - 1), container.width(), 3)
        self._drop_indicator.show()
        self._drop_indicator.raise_()

    def _hide_drop_indicator(self):
        if self._drop_indicator is not None:
            self._drop_indicator.hide()

    def _on_entry_drag_move(self, btn, global_pos):
        if self._reorder_locked or btn not in self.entry_buttons:
            return
        self._show_drop_indicator(self._drag_insertion_index(global_pos))

    def _on_entry_drag_drop(self, btn, global_pos):
        self._hide_drop_indicator()
        if self._reorder_locked or btn not in self.entry_buttons:
            return
        from_idx = self.entry_buttons.index(btn)
        insert_idx = self._drag_insertion_index(global_pos)
        # Convert insertion slot → target index after the row is removed.
        to_idx = insert_idx - 1 if insert_idx > from_idx else insert_idx
        to_idx = max(0, min(to_idx, len(self.entries) - 1))
        if to_idx != from_idx:
            self._move_entry(from_idx, to_idx)

    # ════════════════ AUTO-COLOR ══════════════════════════
    def _on_color_scheme_changed(self, *_args):
        pass

    def auto_assign_colors(self):
        selected_curves = [
            entry for entry in self._get_selected_entries()
            if isinstance(entry, CurveEntry)
        ]
        curves = selected_curves if len(selected_curves) > 1 else [
            entry for entry in self.entries if isinstance(entry, CurveEntry)
        ]
        n = len(curves)
        if n == 0:
            return
        inverse = self.ui_inverse_colors.isChecked()
        colors = get_gradient_colors(n, self.ui_color_scheme.currentText(), inverse=inverse)

        def _set_combo_text(combo, text):
            combo.blockSignals(True)
            combo.setCurrentText(text)
            if combo.isEditable() and combo.lineEdit() is not None:
                combo.lineEdit().setText(text)
            combo.blockSignals(False)

        for entry, c in zip(curves, colors):
            display_color = COLOR_DISPLAY_REV.get(c, c)
            _set_combo_text(entry.inp_color, display_color)
            _set_combo_text(entry.inp_dot_color, display_color)
        self.request_update(from_entry_edit=True)

    def _apply_auto_neg_x(self, checked):
        """When checked, enable X -> -X on curves whose X values are all negative."""
        if not checked:
            return
        for entry in self.entries:
            self._maybe_apply_neg_x_to_entry(entry)
        self.request_update(from_entry_edit=True)

    def _maybe_apply_neg_x_to_entry(self, entry):
        """If 'X -> -X if neg' is on and the entry's X values are all negative, enable X->-X.

        Returns True if the entry's chk_neg_x was changed.
        """
        if not getattr(self, 'ui_auto_neg_x', None) or not self.ui_auto_neg_x.isChecked():
            return False
        if not isinstance(entry, CurveEntry):
            return False
        X, _Y, _ = entry.get_data_arrays()
        if X is None or len(X) == 0:
            return False
        if np.all(X < 0) and not entry.chk_neg_x.isChecked():
            entry.chk_neg_x.blockSignals(True)
            entry.chk_neg_x.setChecked(True)
            entry.chk_neg_x.blockSignals(False)
            return True
        return False

    def _reset_user_zoom(self, axis):
        """Forget the toolbar-zoom override for an axis when the user edits its limit."""
        if axis == 'x':
            self._x_zoomed_by_user = False
            self._user_xlim = None
        else:
            self._y_zoomed_by_user = False
            self._user_ylim = None

    def _apply_auto_range(self, axis):
        """Clear the X or Y limit and trigger an auto-scale update.

        Auto-scaling Y while an X limit is in effect computes the range from
        only the data that falls inside that X window, rather than letting
        matplotlib consider every data point regardless of the visible X span.
        """
        if axis == 'x':
            self.ui_xlim.blockSignals(True)
            self.ui_xlim.setText("")
            self.ui_xlim.blockSignals(False)
            self._x_zoomed_by_user = False
            self._user_xlim = None
        else:
            yrange = None
            xr = self._effective_xrange()
            if xr is not None:
                yrange = self._compute_y_range_for_xrange(xr[0], xr[1])
            self.ui_ylim.blockSignals(True)
            if yrange is not None:
                self.ui_ylim.setText(f"{yrange[0]:g}, {yrange[1]:g}")
            else:
                self.ui_ylim.setText("")
            self.ui_ylim.blockSignals(False)
            self._y_zoomed_by_user = False
            self._user_ylim = None
        # Force the render-level zoom preservation to stand down for this one
        # redraw so the axis actually snaps back to the auto-scaled range.
        if self.plot_window is not None:
            self.plot_window._skip_preserve_zoom_once = True
        if self.preview_plot is not None:
            self.preview_plot._skip_preserve_zoom_once = True
        self._auto_save_settings()
        self.do_update_plot()

    def _effective_xrange(self):
        """Return the X window currently constraining the view, or None.

        Prefers an explicit X Lim field value, then a toolbar/zoom range the
        user dragged on the live axes. Returns a (xlo, xhi) tuple or None when
        the full X extent is shown (no constraint).
        """
        xr = self._tuple_or_none(self.ui_xlim.text())
        if xr is None and self._x_zoomed_by_user and self._user_xlim is not None:
            xr = tuple(self._user_xlim)
        if xr is None:
            return None
        xlo, xhi = xr
        if xlo > xhi:
            xlo, xhi = xhi, xlo
        return (xlo, xhi)

    def _compute_y_range_for_xrange(self, xlo, xhi):
        """Compute an auto Y range over only the data within [xlo, xhi].

        Considers each visible Curve's vertices inside the X window plus the
        interpolated Y at the window edges (so partially-visible segments are
        covered), error-bar extents, and the Y=0 baseline for filled curves.
        Returns a (ymin, ymax) tuple expanded by 5% on each side, or None when
        no data falls inside the window.
        """
        if self.plot_window is None:
            return None
        import numpy as np
        try:
            objs = self.plot_window.plot_objects
        except Exception:
            return None

        all_min, all_max = [], []
        for curve in objs:
            if not isinstance(curve, Curve):
                continue
            try:
                Xs = np.asarray(curve.Xs, dtype=float)
                Ys = np.asarray(curve.Ys, dtype=float)
            except Exception:
                continue
            if Xs.size == 0 or Ys.size != Xs.size:
                continue

            order = np.argsort(Xs)
            Xs, Ys = Xs[order], Ys[order]

            mask = (Xs >= xlo) & (Xs <= xhi)
            ys_in = list(Ys[mask])

            # Interpolated Y at the window edges, so a segment that merely
            # crosses the window (with no vertex inside) still contributes.
            for edge in (xlo, xhi):
                if Xs[0] <= edge <= Xs[-1]:
                    ys_in.append(float(np.interp(edge, Xs, Ys)))

            if not ys_in:
                continue

            lows = list(ys_in)
            highs = list(ys_in)

            # Include error-bar extents for in-window points.
            err = getattr(curve, 'Y_errorbar', None)
            if err is not None:
                try:
                    err_arr = np.asarray(err, dtype=float)
                    if err_arr.ndim == 2 and err_arr.shape[0] == 2:
                        err_low = err_arr[0][order][mask]
                        err_high = err_arr[1][order][mask]
                    else:
                        err_flat = err_arr.ravel()[order][mask]
                        err_low = err_high = err_flat
                    ys_pts = Ys[mask]
                    lows.extend(ys_pts - err_low)
                    highs.extend(ys_pts + err_high)
                except Exception:
                    pass

            # Filled curves draw down to (or up to) the Y=0 baseline.
            if getattr(curve, 'fill_color', None):
                lows.append(0.0)
                highs.append(0.0)

            all_min.append(min(lows))
            all_max.append(max(highs))

        if not all_min:
            return None

        final_min = min(all_min)
        final_max = max(all_max)
        span = final_max - final_min
        if span == 0:
            span = abs(final_max) * 0.1 if final_max != 0 else 1.0
        return (final_min - span * 0.05, final_max + span * 0.05)

    def _ensure_lim_callbacks(self):
        """Connect xlim/ylim_changed callbacks on the live axes once."""
        if self._lim_callbacks_connected:
            return
        if self.plot_window is None or not hasattr(self.plot_window, '_ax'):
            return
        ax = self.plot_window._ax
        ax.callbacks.connect('xlim_changed', self._on_axes_xlim_changed)
        ax.callbacks.connect('ylim_changed', self._on_axes_ylim_changed)
        self._lim_callbacks_connected = True

    def _on_axes_xlim_changed(self, ax):
        if self._suppress_lim_callback:
            return
        try:
            self._user_xlim = ax.get_xlim()
        except Exception:
            return
        self._x_zoomed_by_user = True

    def _on_axes_ylim_changed(self, ax):
        if self._suppress_lim_callback:
            return
        try:
            self._user_ylim = ax.get_ylim()
        except Exception:
            return
        self._y_zoomed_by_user = True

    # ════════════════ APPLY INLINE PROPERTY TO ALL ═══════
    def _apply_inline_property_to_all(self):
        """Copy the selected entry's inline property value to all entries of the same type."""
        prop_name = self.ui_property_dropdown.currentText()
        if prop_name == 'None':
            return

        # Find the currently selected entry
        sel_idx = self._get_selected_index()
        if sel_idx < 0:
            return
        source_entry = self.entries[sel_idx]

        # Determine which prop map to use
        if isinstance(source_entry, CurveEntry):
            prop_map = CURVE_INLINE_PROPS
        elif isinstance(source_entry, GridEntry):
            prop_map = GRID_INLINE_PROPS
        else:
            return

        if prop_name not in prop_map:
            return

        attr_name, widget_type, _options = prop_map[prop_name]
        source_widget = getattr(source_entry, attr_name)
        selected_same_type_entries = [
            entry for entry in self._get_selected_entries()
            if isinstance(entry, type(source_entry))
        ]
        apply_to_selected = len(selected_same_type_entries) > 1

        for idx, entry in enumerate(self.entries):
            if entry is source_entry:
                continue
            if not isinstance(entry, type(source_entry)):
                continue
            if apply_to_selected:
                if entry not in selected_same_type_entries:
                    continue
            elif idx >= len(self.entry_checkboxes) or not self.entry_checkboxes[idx].isChecked():
                continue
            if not hasattr(entry, attr_name):
                continue
            target_widget = getattr(entry, attr_name)
            if widget_type == 'line_edit':
                target_widget.setText(source_widget.text())
            elif widget_type in ('editable_combo', 'combo'):
                target_widget.setCurrentText(source_widget.currentText())
            elif widget_type == 'checkable':
                target_widget.setChecked(source_widget.isChecked())

        self.request_update()

    # ════════════════ INLINE PROPERTY MANAGEMENT ═════════
    def _on_property_dropdown_changed(self, _prop_name=None):
        self._rebuild_all_inline_property_widgets()

    def _rebuild_all_inline_property_widgets(self):
        for i in range(len(self.entries)):
            self._rebuild_inline_property_widget(i)

    def _rebuild_inline_property_widget(self, idx):
        entry = self.entries[idx]
        container = self.entry_prop_containers[idx]
        layout = container.layout()
        # Clear old contents
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        prop_name = self.ui_property_dropdown.currentText()
        if prop_name == 'None':
            container.hide()
            return

        container.show()
        inline_w = self._create_inline_widget(entry, prop_name)
        if inline_w is not None:
            layout.addWidget(inline_w)
        else:
            # Property doesn't apply to this entry type
            placeholder = QLabel("—")
            placeholder.setStyleSheet("color: #999;")
            layout.addWidget(placeholder)

    def _create_inline_widget(self, entry, prop_name):
        """Create an inline editing widget for the given property on the entry."""
        if isinstance(entry, CurveEntry):
            prop_map = CURVE_INLINE_PROPS
        elif isinstance(entry, GridEntry):
            prop_map = GRID_INLINE_PROPS
        else:
            return None

        if prop_name not in prop_map:
            return None

        attr_name, widget_type, options = prop_map[prop_name]
        source = getattr(entry, attr_name)

        if widget_type == 'line_edit':
            inline = QLineEdit()
            inline.setMinimumWidth(70)
            inline.setMaximumWidth(120)
            inline.setFixedHeight(25)
            inline.setText(source.text())

            def sync_from(text, i=inline):
                if self._suppress_prop_sync:
                    return
                try:
                    self._suppress_prop_sync = True
                    i.setText(text)
                    self._suppress_prop_sync = False
                except RuntimeError:
                    pass

            inline.textEdited.connect(
                lambda _text, e=entry: self._select_entry_for_inline_edit(e))
            inline.editingFinished.connect(
                lambda s=source, i=inline, e=entry: (
                    self._select_entry_for_inline_edit(e),
                    s.setText(i.text())
                ))
            source.textChanged.connect(sync_from)
            return inline

        elif widget_type in ('editable_combo', 'combo'):
            inline = QComboBox()
            if options:
                inline.addItems(options)
            if widget_type == 'editable_combo':
                inline.setEditable(True)
                inline.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            inline.setMinimumWidth(80)
            inline.setMaximumWidth(130)
            inline.setFixedHeight(25)
            inline.setMaxVisibleItems(min(len(options) if options else 10, 20))
            inline.setCurrentText(source.currentText())

            def sync_to_source():
                if self._suppress_prop_sync:
                    return
                self._suppress_prop_sync = True
                self._select_entry_for_inline_edit(entry)
                source.setCurrentText(inline.currentText())
                self._suppress_prop_sync = False

            def sync_from(text, i=inline):
                if self._suppress_prop_sync:
                    return
                try:
                    self._suppress_prop_sync = True
                    i.setCurrentText(text)
                    self._suppress_prop_sync = False
                except RuntimeError:
                    pass

            # For editable combos, sync when editing finishes or when item is selected
            if widget_type == 'editable_combo':
                inline.lineEdit().textEdited.connect(
                    lambda _text, e=entry: self._select_entry_for_inline_edit(e))
                inline.lineEdit().editingFinished.connect(sync_to_source)
            inline.activated.connect(sync_to_source)
            source.currentTextChanged.connect(sync_from)
            return inline

        elif widget_type == 'checkable':
            inline = QPushButton()
            inline.setCheckable(True)
            inline.setChecked(source.isChecked())
            inline.setFixedWidth(30)
            inline.setFixedHeight(25)
            inline.setStyleSheet(GlobalStyles.CHECKABLE_BTN)

            def sync_to(checked, s=source):
                if self._suppress_prop_sync:
                    return
                self._suppress_prop_sync = True
                self._select_entry_for_inline_edit(entry)
                s.setChecked(checked)
                self._suppress_prop_sync = False
                # Process events to flush UI updates before next fast click
                QApplication.processEvents()

            def sync_from(checked, i=inline):
                if self._suppress_prop_sync:
                    return
                try:
                    self._suppress_prop_sync = True
                    i.setChecked(checked)
                    self._suppress_prop_sync = False
                except RuntimeError:
                    pass

            inline.toggled.connect(sync_to)
            source.toggled.connect(sync_from)
            return inline

        return None

    # ════════════════ UPDATE PLOT ═════════════════════════
    def request_update(self, from_entry_edit=False):
        """Request a plot update.

        Parameters
        ----------
        from_entry_edit : bool
            Legacy parameter, kept for API compatibility. No longer clears
            stored multi-frame data — entry edits are now applied as
            overrides on top of stored frames.
        """
        if self._suppress_update:
            return
        if self.ui_auto_update.isChecked():
            self.do_update_plot()

    @staticmethod
    def _tuple_or_none(txt):
        try:
            parts = [p.strip() for p in txt.split(',') if p.strip()]
            if len(parts) >= 2:
                return (float(parts[0]), float(parts[1]))
        except (ValueError, TypeError):
            pass
        return None

    def _clear_stored_frames(self):
        """Clear all stored multi-frame data."""
        self._stored_curve_frames = None
        self._stored_grid_frames = None
        self._stored_frame_labels = None
        self._stored_current_frame_index = 0

    def _get_current_frame_index(self):
        """Return the current frame index from the plot window, or stored fallback."""
        if self.plot_window is not None and hasattr(self.plot_window, '_current_frame_index'):
            return self.plot_window._current_frame_index
        return self._stored_current_frame_index

    # -- Visual property names to copy from entry-built Curve onto each stored-frame Curve --
    _CURVE_VISUAL_ATTRS = (
        'Y_label',
        'curve_color', 'curve_width', 'plot_curve', 'curve_format',
        'plot_dot', 'dot_format', 'dot_color', 'dot_alpha',
        'dot_edge_color', 'dot_edge_width', 'dot_width',
        'do_interpolation', 'interpolation_kind',
        'interpolation_smoothing', 'interpolation_number',
        'curve_legend_color', 'curve_legend_format',
        'fill_color', 'fill_alpha',
    )

    # -- Visual property names for Grid objects --
    _GRID_VISUAL_ATTRS = (
        'interpolation_type', 'show_contour',
        'interpolation_density', 'show_colorbar',
    )

    def _build_effective_frames(self):
        """Build effective frames by applying editor entry states to stored frames.

        For each stored frame, curves/grids whose corresponding editor entry
        is disabled are filtered out, and visual properties are overridden from
        the editor entry.  Returns (effective_curve_frames, effective_grid_frames)
        or (None, None) if no stored frames exist.
        """
        if not self._stored_curve_frames and not self._stored_grid_frames:
            return None, None

        # Separate curve and grid entries (preserving order)
        curve_entries = [e for e in self.entries if isinstance(e, CurveEntry)]
        grid_entries = [e for e in self.entries if isinstance(e, GridEntry)]

        # Build reference Curve objects from entries (None if disabled)
        curve_ref_objs = [e.get_object() for e in curve_entries]
        grid_ref_objs = [e.get_object() for e in grid_entries]

        effective_curve_frames = None
        if self._stored_curve_frames:
            effective_curve_frames = []
            for frame_curves in self._stored_curve_frames:
                eff = []
                for j, frame_curve in enumerate(frame_curves):
                    if j >= len(curve_ref_objs):
                        # Extra curve in frame with no editor entry — keep as-is
                        eff.append(frame_curve)
                        continue
                    ref = curve_ref_objs[j]
                    if ref is None:
                        # Entry is disabled — skip this curve
                        continue
                    # Shallow-copy the frame curve and override visual props
                    modified = copy.copy(frame_curve)
                    for attr in self._CURVE_VISUAL_ATTRS:
                        if hasattr(ref, attr):
                            setattr(modified, attr, getattr(ref, attr))
                    eff.append(modified)
                effective_curve_frames.append(eff)

        effective_grid_frames = None
        if self._stored_grid_frames:
            effective_grid_frames = []
            for frame_grids in self._stored_grid_frames:
                eff = []
                for j, frame_grid in enumerate(frame_grids):
                    if j >= len(grid_ref_objs):
                        eff.append(frame_grid)
                        continue
                    ref = grid_ref_objs[j]
                    if ref is None:
                        continue
                    modified = copy.copy(frame_grid)
                    for attr in self._GRID_VISUAL_ATTRS:
                        if hasattr(ref, attr):
                            setattr(modified, attr, getattr(ref, attr))
                    eff.append(modified)
                effective_grid_frames.append(eff)

        return effective_curve_frames, effective_grid_frames

    # ───────── Curve-formula evaluation (inter-curve calculation) ─────────
    def _entry_code(self, entry):
        """Current curve code (alias) for an entry, or '' if none."""
        edit = getattr(entry, '_code_edit', None)
        if edit is not None:
            return edit.text().strip()
        return (getattr(entry, 'curve_code', '') or '').strip()

    def _build_code_index(self):
        """Map UPPER-cased curve code → CurveEntry (first occurrence wins)."""
        index = {}
        for entry in self.entries:
            if not isinstance(entry, CurveEntry):
                continue
            code = self._entry_code(entry)
            if code:
                index.setdefault(code.upper(), entry)
        return index

    @staticmethod
    def _entry_raw_xy(entry):
        """Raw (X, Y) float arrays for a data curve, or None when it has none."""
        X, Y, _Z = entry.get_data_arrays()
        if X is None or Y is None or len(X) == 0 or len(Y) == 0:
            return None
        return np.asarray(X, dtype=float), np.asarray(Y, dtype=float)

    def _resolve_entry_xy(self, entry, stack):
        """(X, Y) for a referenced curve: its computed data if it is itself a
        formula curve, otherwise its raw pasted data."""
        if isinstance(entry, CurveEntry) and entry.get_formula() is not None:
            return self._compute_curve_data(entry, stack)
        return self._entry_raw_xy(entry)

    def _compute_curve_data(self, entry, stack=None):
        """Evaluate a computed curve's Label formula → (X, Y) arrays.

        Returns None — and prints a message to the Result area — on any error
        (unknown/empty code, no common X range, eval failure, reference cycle)
        so a bad formula never crashes the render.  Results are cached on the
        entry and reused while the formula and every referenced curve's raw
        data are unchanged."""
        formula = entry.get_formula()
        if formula is None:
            return None
        stack = stack or []
        if entry in stack:
            print(f"[Curve formula] 循环引用，无法计算：{formula}")
            return None

        index = self._build_code_index()
        refs = [m.upper() for m in CURVE_CODE_REF_RE.findall(formula)]
        if not refs:
            return None

        resolved = {}
        missing = []
        for key in refs:
            if key in resolved:
                continue
            ref_entry = index.get(key)
            xy = (self._resolve_entry_xy(ref_entry, stack + [entry])
                  if ref_entry is not None else None)
            if xy is None:
                missing.append(key)
            else:
                resolved[key] = xy
        if missing:
            print(f"[Curve formula] 公式引用了不存在或无数据的曲线代号："
                  f"{', '.join(sorted(set(missing)))}  ←  {formula}")
            return None

        x_log = bool(self.ui_xlog.isChecked())
        cache_key = self._formula_cache_key(formula, resolved, x_log)
        cached = getattr(entry, '_formula_cache', None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        try:
            xy = self._evaluate_formula(formula, resolved, x_log)
        except Exception as exc:
            print(f"[Curve formula] 计算出错：{exc}  ←  {formula}")
            return None
        entry._formula_cache = (cache_key, xy)
        return xy

    def _compute_align(self, entry, X, Y):
        """Solve the optimal scale and/or vertical offset so *entry*'s data best
        overlaps the curve referenced in its "Align With" box, over the given X
        range.  Returns (scale, offset) or None when alignment is inactive or
        cannot be solved (prints a hint, never raises, so a bad config never
        crashes the render)."""
        want_scale = entry.chk_align_scale.isChecked()
        want_offset = entry.chk_align_offset.isChecked()
        if not (want_scale or want_offset):
            return None

        ref_text = entry.inp_align_ref.text()
        codes = CURVE_CODE_REF_RE.findall(ref_text)
        if not codes:
            if ref_text.strip():
                print(f"[Align] 未找到曲线代号（请用 {{{{a}}}} 形式引用）：{ref_text}")
            return None
        code = codes[0].upper()

        ref_entry = self._build_code_index().get(code)
        if ref_entry is entry:
            print(f"[Align] 引用了自身：代号 {code} 就是本曲线，无法对齐到自己。"
                  f"请把 Align With 填到要移动的那条曲线上，并指向参考曲线的代号")
            return None
        if ref_entry is None:
            print(f"[Align] 引用了不存在的曲线代号：{code}")
            return None
        ref_xy = self._resolve_entry_xy(ref_entry, [entry])
        if ref_xy is None:
            print(f"[Align] 参考曲线 {code} 没有数据")
            return None

        try:
            Xs = np.asarray(X, dtype=float)
            Ys = np.asarray(Y, dtype=float)
            Xr = np.asarray(ref_xy[0], dtype=float)
            Yr = np.asarray(ref_xy[1], dtype=float)
        except (TypeError, ValueError):
            return None
        if Xs.size == 0 or Xr.size == 0:
            return None

        # Parse the range; fall back to the curves' overlapping X span.
        nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',
                          entry.inp_align_range.text())
        if len(nums) >= 2:
            lo, hi = float(nums[0]), float(nums[1])
            if lo > hi:
                lo, hi = hi, lo
        else:
            lo, hi = -np.inf, np.inf
        lo = max(lo, float(np.nanmin(Xr)))
        hi = min(hi, float(np.nanmax(Xr)))

        mask = np.isfinite(Xs) & np.isfinite(Ys) & (Xs >= lo) & (Xs <= hi)
        need = 2 if (want_scale and want_offset) else 1
        if np.count_nonzero(mask) < need:
            print(f"[Align] 在范围 [{lo}, {hi}] 内可匹配的点不足")
            return None

        # Reference Y resampled onto this curve's X positions within the range.
        order = np.argsort(Xr, kind='stable')
        yr_on_x = np.interp(Xs[mask], Xr[order], Yr[order])
        ys = Ys[mask]

        # Ordinary linear least squares — minimise the vertical residual
        # Σ(a·y + b − r)² over the chosen range.
        if want_scale and want_offset:
            A = np.column_stack([ys, np.ones_like(ys)])
            sol, *_ = np.linalg.lstsq(A, yr_on_x, rcond=None)
            a, b = float(sol[0]), float(sol[1])
        elif want_scale:
            denom = float(np.dot(ys, ys))
            if denom == 0:
                return None
            a, b = float(np.dot(ys, yr_on_x) / denom), 0.0
        else:  # offset only
            a, b = 1.0, float(np.mean(yr_on_x - ys))
        return a, b

    @staticmethod
    def _formula_cache_key(formula, resolved, x_log):
        parts = [formula, '1' if x_log else '0']
        for key in sorted(resolved):
            X, Y = resolved[key]
            parts.append(key)
            parts.append(hashlib.md5(
                np.ascontiguousarray(X, dtype=float).tobytes()).hexdigest())
            parts.append(hashlib.md5(
                np.ascontiguousarray(Y, dtype=float).tobytes()).hexdigest())
        return '|'.join(parts)

    def _evaluate_formula(self, formula, resolved, x_log):
        """Turn the resolved {code: (X, Y)} map into the computed (X, Y)."""
        codes = list(resolved.keys())
        # Sort each curve by ascending X so comparison/interpolation is defined.
        data = {}
        for key in codes:
            X, Y = resolved[key]
            order = np.argsort(X, kind='stable')
            data[key] = (X[order], Y[order])

        ref_X = data[codes[0]][0]
        identical = all(
            data[k][0].shape == ref_X.shape and np.allclose(data[k][0], ref_X)
            for k in codes
        )
        if identical:
            target_X = ref_X
            values = {k: data[k][1] for k in codes}
        else:
            target_X, values = self._interpolate_to_common_grid(data, x_log)

        expr, var_map = formula_to_expr(formula)
        Y = eval_formula_pointwise(expr, var_map, values, len(target_X))
        return np.asarray(target_X, dtype=float), np.asarray(Y, dtype=float)

    @staticmethod
    def _interpolate_to_common_grid(data, x_log):
        """Resample every referenced curve onto a shared, fine, uniform grid
        over the curves' common X range.  Uniform means evenly spaced in x, or
        in log10(x) when the X axis is logarithmic.  The point count is
        max(min(common_range / (min_spacing/10), 100000), own_point_count)."""
        codes = list(data.keys())
        x_lo = max(float(data[k][0][0]) for k in codes)     # largest minimum
        x_hi = min(float(data[k][0][-1]) for k in codes)    # smallest maximum
        if not (x_hi > x_lo):
            raise ValueError("曲线之间没有公共的 X 区间")

        use_log = bool(x_log) and x_lo > 0

        def to_u(x):
            x = np.asarray(x, dtype=float)
            return np.log10(x) if use_log else x

        def from_u(u):
            return np.power(10.0, u) if use_log else u

        u_lo, u_hi = float(to_u(x_lo)), float(to_u(x_hi))

        min_spacing = np.inf
        max_own_points = 0
        for k in codes:
            X = data[k][0]
            in_range = int(np.count_nonzero((X >= x_lo) & (X <= x_hi)))
            max_own_points = max(max_own_points, in_range)
            u = to_u(X)
            u = u[np.isfinite(u)]
            du = np.diff(u)
            du = du[du > 0]
            if du.size:
                min_spacing = min(min_spacing, float(np.min(du)))
        if not np.isfinite(min_spacing) or min_spacing <= 0:
            min_spacing = (u_hi - u_lo) / 1000.0

        target_spacing = min_spacing / 10.0
        if target_spacing <= 0:
            n_by_spacing = max_own_points
        else:
            n_by_spacing = int(np.ceil((u_hi - u_lo) / target_spacing)) + 1
        n = max(min(n_by_spacing, FORMULA_MAX_POINTS), max_own_points, 2)

        u_grid = np.linspace(u_lo, u_hi, n)
        target_X = from_u(u_grid)
        values = {}
        for k in codes:
            X, Y = data[k]
            values[k] = np.interp(u_grid, to_u(X), Y)
        return target_X, values

    def _on_code_changed(self, entry):
        """A curve code was edited: sync the stored alias and re-render so any
        dependent formula curves update."""
        entry.curve_code = self._entry_code(entry)
        self.request_update(from_entry_edit=True)

    # ════════════════ DOCKED RESULT LOG + EMBEDDED FIGURE PREVIEW ═════════
    def _host_result_in_active_entry(self):
        """Dock the shared Result log into the active entry's bottom slot
        (below its data editor column), moving it away from whichever entry
        hosted it before."""
        if not hasattr(self, 'result_group'):
            return
        target = self._active_entry if self._active_entry in self.entries else None
        if target is None:
            if self.result_group.parentWidget() is not None:
                self.result_group.setParent(None)
            self.result_group.hide()
            return
        if self.result_group.parentWidget() is not target.bottom_slot:
            target.bottom_slot_layout.addWidget(self.result_group)
        self.result_group.show()
        self._apply_editor_log_split_ratio(target)

    def _detach_result_from(self, widget):
        """If the Result log lives inside `widget`, pull it out so the widget
        can be deleted without destroying the log."""
        if hasattr(self, 'result_group') and widget.isAncestorOf(self.result_group):
            self.result_group.setParent(None)
            self.result_group.hide()

    @staticmethod
    def _set_splitter_ratio(splitter, ratio, fallback_total):
        sizes = splitter.sizes()
        total = sum(sizes)
        if total <= 0:
            total = max(splitter.height(), fallback_total)
        ratio = min(max(ratio, 0.05), 0.95)
        top = round(total * ratio)
        splitter.setSizes([top, max(total - top, 0)])

    @staticmethod
    def _get_splitter_ratio(splitter):
        sizes = splitter.sizes()
        total = sum(sizes)
        return (sizes[0] / total) if total > 0 else None

    def _apply_editor_log_split_ratio(self, entry):
        """Set the entry's editor/log splitter sizes from the stored ratio."""
        self._applying_splitter_ratio = True
        try:
            self._set_splitter_ratio(entry.left_v_splitter,
                                     self._editor_log_split_ratio, 400)
        finally:
            self._applying_splitter_ratio = False

    def _on_left_splitter_moved(self, entry):
        """User dragged the editor/log divider: remember and persist the ratio."""
        if self._applying_splitter_ratio:
            return
        ratio = self._get_splitter_ratio(entry.left_v_splitter)
        if ratio is None:
            return
        self._editor_log_split_ratio = ratio
        self._auto_save_settings()

    def _apply_right_split_ratio(self):
        """Set the right panel's list/preview splitter sizes from the stored ratio."""
        self._applying_splitter_ratio = True
        try:
            self._set_splitter_ratio(self.right_splitter,
                                     self._right_split_ratio, 600)
        finally:
            self._applying_splitter_ratio = False

    def _on_right_splitter_moved(self, _pos=None, _idx=None):
        """User dragged the list/preview divider: remember and persist the ratio."""
        if self._applying_splitter_ratio:
            return
        ratio = self._get_splitter_ratio(self.right_splitter)
        if ratio is None:
            return
        self._right_split_ratio = ratio
        self._auto_save_settings()

    def _destroy_preview_plot(self):
        """Remove the embedded preview Plot widget (if any) and show the placeholder."""
        if self.preview_plot is not None:
            try:
                self.preview_layout.removeWidget(self.preview_plot)
                self.preview_plot.setParent(None)
                self.preview_plot.deleteLater()
            except Exception:
                pass
            self.preview_plot = None
        if hasattr(self, 'preview_placeholder'):
            self.preview_placeholder.show()

    def _update_preview_plot(self, objs, opts, eff_curve_frames, eff_grid_frames,
                             has_frames, bg_color):
        """Mirror the floating figure window into the embedded preview pane.

        The preview is a second Plot instance kept permanently geometry-locked:
        it fills whatever space the splitter pane provides (so the aspect ratio
        may differ from the floating window) while everything else — toolbar,
        zoom, curve settings — behaves identically. Its secondary toolbar row
        (coordinate label + save/raise/pause buttons) is hidden to save space.
        """
        try:
            pv = self.preview_plot
            if pv is None:
                # Construct empty (no objects) so the Plot doesn't show itself
                # as a top-level window, then reparent it into the preview pane.
                ctor_opts = {k: v for k, v in opts.items() if k != 'keep_front'}
                pv = Plot(window_title="Preview", **ctor_opts)
                pv._geometry_locked = True
                # Compact preview: hide the coord-label + save/raise/pause row
                pv.set_secondary_toolbar_row_visible(False)
                pv.on_plot_json_saved = self._on_plot_json_saved
                pv.on_curve_clicked = self._on_curve_clicked
                pv.on_canvas_clicked = self._on_canvas_clicked
                # on_canvas_resized deliberately NOT connected: pane resizing
                # must not overwrite the W/H fields of the floating window.
                self.preview_placeholder.hide()
                self.preview_layout.addWidget(pv)
                self.preview_plot = pv
                # Track toolbar zoom done inside the preview the same way as
                # zoom done in the floating window.
                pv._ax.callbacks.connect('xlim_changed', self._on_axes_xlim_changed)
                pv._ax.callbacks.connect('ylim_changed', self._on_axes_ylim_changed)

            # Apply the same properties as the floating window, except
            # keep_front (window flags are meaningless on an embedded child).
            for k in ('x_axis_label', 'y_axis_label', 'x_lim', 'y_lim',
                      'x_log', 'y_log', 'show_grid', 'plot_legend',
                      'figure_title'):
                setattr(pv, k, opts[k])
            pv.font_size = opts['font_size']
            pv.legend_font_size = opts['legend_font_size']

            if has_frames:
                current_idx = self._get_current_frame_index()
                pv.set_plot_frames(
                    Curve_objects_frames=eff_curve_frames,
                    Grid_objects_frames=eff_grid_frames,
                    current_frame_index=current_idx,
                    frame_labels=self._stored_frame_labels,
                )
            else:
                pv._plot_objects = list(objs)
                pv._update_plot()

            pv.set_figure_title(opts['figure_title'])
            pv.save_bg_color = bg_color or None
        except Exception as e:
            print(f"Error updating figure preview: {e}")
            import traceback
            traceback.print_exc()
            self._destroy_preview_plot()

    def do_update_plot(self):
        if self._suppress_update:
            return

        try:
            objs = [e.get_object() for e in self.entries]
        except Exception as e:
            print(f"Error building plot objects: {e}")
            import traceback
            traceback.print_exc()
            # Destroy the broken plot but keep the editor alive
            if self.plot_window is not None:
                self.plot_window.close()
                self.plot_window = None
            self._destroy_preview_plot()
            return

        objs = [o for o in objs if o is not None]

        # If no objects remain, close the plot window
        if not objs:
            if self.plot_window is not None:
                self.plot_window.close()
                self.plot_window = None
            self._destroy_preview_plot()
            return

        ui_xlim_val = self._tuple_or_none(self.ui_xlim.text())
        ui_ylim_val = self._tuple_or_none(self.ui_ylim.text())

        # If the user has zoomed via the matplotlib toolbar and hasn't typed an
        # explicit limit, preserve the toolbar-zoomed view across redraws — even
        # if the data has just been transformed (scale factor, integration
        # bounds, …) so that the zoom range no longer overlaps the new data.
        # The user clears the zoom explicitly via the Home button or by editing
        # the limit fields.
        eff_xlim = ui_xlim_val
        if eff_xlim is None and self._x_zoomed_by_user and self._user_xlim is not None:
            eff_xlim = tuple(self._user_xlim)

        eff_ylim = ui_ylim_val
        if eff_ylim is None and self._y_zoomed_by_user and self._user_ylim is not None:
            eff_ylim = tuple(self._user_ylim)

        opts = {
            'figure_title': self.ui_title.text(),
            'x_axis_label': self.ui_xlabel.text(),
            'y_axis_label': self.ui_ylabel.text(),
            'x_lim': eff_xlim,
            'y_lim': eff_ylim,
            'x_log': self.ui_xlog.isChecked(),
            'y_log': self.ui_ylog.isChecked(),
            'show_grid': self.ui_grid.isChecked(),
            'plot_legend': self.ui_legend.isChecked(),
            'keep_front': self.ui_keep_front.isChecked(),
        }
        # Convert pixel dimensions to inches for fig_size_inch
        width_px = safe_eval_number(self.ui_fig_w.text(), 400)
        height_px = safe_eval_number(self.ui_fig_h.text(), 300)
        opts['fig_size_inch'] = (width_px / DEFAULT_FIG_DPI, height_px / DEFAULT_FIG_DPI)
        opts['fig_size_pixel'] = (width_px, height_px)
        opts['font_size'] = int(safe_eval_number(self.ui_font_size.text(), 10))
        legend_fs = safe_eval_number(self.ui_legend_font_size.text())
        opts['legend_font_size'] = int(legend_fs) if legend_fs is not None else None

        # Build effective frames that incorporate editor entry states
        eff_curve_frames, eff_grid_frames = self._build_effective_frames()
        has_frames = eff_curve_frames is not None or eff_grid_frames is not None

        bg_color = (self.ui_bg_color.text() or '').strip() if hasattr(self, 'ui_bg_color') else ''

        self._suppress_lim_callback = True
        try:
            if self.plot_window is None:
                if has_frames:
                    self.plot_window = Plot(
                        Curve_objects_frames=eff_curve_frames,
                        Grid_objects_frames=eff_grid_frames,
                        current_frame_index=self._stored_current_frame_index,
                        frame_labels=self._stored_frame_labels,
                        **opts,
                    )
                else:
                    self.plot_window = Plot(objs, **opts)
                self.plot_window.on_plot_json_saved = self._on_plot_json_saved
                self.plot_window.on_curve_clicked = self._on_curve_clicked
                self.plot_window.on_canvas_clicked = self._on_canvas_clicked
                self.plot_window.on_canvas_resized = self._on_canvas_resized
                # Set the figure title explicitly after creation
                self.plot_window.set_figure_title(opts['figure_title'])
                self._last_applied_fig_size = opts['fig_size_pixel']
                self._last_applied_font_size = opts['font_size']
                self._last_applied_legend_font_size = opts['legend_font_size']
            else:
                p = self.plot_window
                p.on_plot_json_saved = self._on_plot_json_saved
                p.on_curve_clicked = self._on_curve_clicked
                p.on_canvas_clicked = self._on_canvas_clicked
                p.on_canvas_resized = self._on_canvas_resized
                # Set all properties BEFORE triggering _update_plot
                for k in ('x_axis_label', 'y_axis_label', 'x_lim', 'y_lim',
                          'x_log', 'y_log', 'show_grid', 'plot_legend',
                          'figure_title', 'keep_front'):
                    setattr(p, k, opts[k])
                p.font_size = opts['font_size']
                p.legend_font_size = opts['legend_font_size']

                # Only update figure size when size or font fields actually changed,
                # so that manual window resizing by the user is preserved.
                current_fig_size = opts['fig_size_pixel']
                layout_changed = (
                    current_fig_size != self._last_applied_fig_size
                    or opts['font_size'] != self._last_applied_font_size
                    or opts['legend_font_size'] != self._last_applied_legend_font_size
                )
                self._last_applied_fig_size = current_fig_size
                self._last_applied_font_size = opts['font_size']
                self._last_applied_legend_font_size = opts['legend_font_size']
                if layout_changed:
                    if current_fig_size is not None:
                        p._fig_size_pixel = tuple(current_fig_size)
                    else:
                        p._fig_size_pixel = (opts['fig_size_inch'][0] * DEFAULT_FIG_DPI, opts['fig_size_inch'][1] * DEFAULT_FIG_DPI)
                    p._setup_figure_size()

                # When layout didn't change, temporarily lock geometry so
                # _update_plot won't resize the window (preserving user's manual resize).
                prev_locked = p._geometry_locked
                if not layout_changed:
                    p._geometry_locked = True

                # Update objects and redraw, preserving current frame position
                try:
                    if has_frames:
                        current_idx = self._get_current_frame_index()
                        p.set_plot_frames(
                            Curve_objects_frames=eff_curve_frames,
                            Grid_objects_frames=eff_grid_frames,
                            current_frame_index=current_idx,
                            frame_labels=self._stored_frame_labels,
                        )
                    else:
                        # _update_plot calls resize_window_to_fig at the end
                        p._plot_objects = list(objs)
                        p._update_plot()
                finally:
                    p._geometry_locked = prev_locked

                p.set_figure_title(opts['figure_title'])
            
            # Tell the plot which background color to use when saving PNG
            if self.plot_window is not None:
                self.plot_window.save_bg_color = bg_color or None

            # Mirror the same state into the embedded preview pane
            self._update_preview_plot(objs, opts, eff_curve_frames,
                                      eff_grid_frames, has_frames, bg_color)

            # Ensure the plot window is visible and updated.
            # Only raise/activate on first show or when Keep Front is checked,
            # so routine setting changes don't steal focus.
            first_show = not self.plot_window.isVisible()
            if first_show:
                self.plot_window.show()
            if first_show or self.ui_keep_front.isChecked():
                self.plot_window.raise_()
                self.plot_window.activateWindow()
        except Exception as e:
            print(f"Error updating plot: {e}")
            import traceback
            traceback.print_exc()
            # Close broken plot window but keep the editor alive
            if self.plot_window is not None:
                try:
                    self.plot_window.close()
                except Exception:
                    pass
                self.plot_window = None
                self._lim_callbacks_connected = False
        finally:
            self._suppress_lim_callback = False
            # Connect axes-limit callbacks the first time after a window exists
            self._ensure_lim_callbacks()

    # ════════════════ COMPUTE INTEGRATION ═════════════════
    def _compute_integration(self, entry):
        """Compute integration for a CurveEntry and print result to stdout."""
        try:
            X, Y, _Z = entry.get_data_arrays()
            if Y is None:
                print("Error: No data loaded for this curve.")
                return

            start_text = entry.inp_integ_start.text().strip()
            stop_text = entry.inp_integ_stop.text().strip()
            if not start_text or not stop_text:
                print("Error: Start and Stop values are required.")
                return

            start = float(start_text)
            stop = float(stop_text)

            # Weights
            weights_path = entry.inp_integ_weights.currentText().strip().strip('"').strip("'")
            weights = None
            if weights_path:
                if os.path.isfile(weights_path):
                    weights = weights_path
                    add_to_weights_history(weights_path)
                    entry._refresh_weights_combo(weights_path)
                else:
                    print(f"Error: Weights file not found: {weights_path}")
                    return

            # Handle Negative X
            if entry.chk_neg_x.isChecked():
                X = -X

            # Create temporary Curve and integrate
            curve = Curve(X=X, Y=Y)
            result = curve.integrate(start, stop, weights)

            # Normalize integration result
            normalize_text = entry.inp_integ_normalize.text().strip()
            normalize_val = safe_eval_number(normalize_text) if normalize_text else None

            label = entry.inp_label.text() or "Untitled"
            weights_info = f", weights={os.path.basename(weights_path)}" if weights else ""
            if normalize_val is not None and result != 0:
                factor = normalize_val / result
                print(f"∫ {label} [{start}, {stop}]{weights_info} = {result}  (normalized to {normalize_val}, factor = {factor})")
            else:
                print(f"∫ {label} [{start}, {stop}]{weights_info} = {result}")
        except Exception as e:
            print(f"Integration error: {e}")

    # ════════════════ SETTINGS PERSISTENCE ════════════════
    def _get_settings_dict(self):
        return {
            'title': self.ui_title.text(),
            'xlabel': self.ui_xlabel.text(),
            'ylabel': self.ui_ylabel.text(),
            'xlim': self.ui_xlim.text(),
            'ylim': self.ui_ylim.text(),
            'xlog': self.ui_xlog.isChecked(),
            'ylog': self.ui_ylog.isChecked(),
            'grid': self.ui_grid.isChecked(),
            'legend': self.ui_legend.isChecked(),
            'keep_front': self.ui_keep_front.isChecked(),
            'fig_w': self.ui_fig_w.text(),
            'fig_h': self.ui_fig_h.text(),
            'font_size': self.ui_font_size.text(),
            'auto_update': self.ui_auto_update.isChecked(),
            'color_scheme': self.ui_color_scheme.currentText(),
            'legend_font_size': self.ui_legend_font_size.text(),
            'auto_neg_x': self.ui_auto_neg_x.isChecked(),
            'inverse_colors': self.ui_inverse_colors.isChecked(),
            'auto_color_on_add': self.ui_auto_color.isChecked(),
            'drop_delims': list(self.get_drop_delim_keys()),
            'bg_color': self.ui_bg_color.text(),
            'editor_log_split_ratio': self._editor_log_split_ratio,
            'right_split_ratio': self._right_split_ratio,
        }

    def _apply_settings_dict(self, s):
        self._suppress_update = True
        self.ui_title.setText(s.get('title', ''))
        self.ui_xlabel.setText(s.get('xlabel', 'X Axis'))
        self.ui_ylabel.setText(s.get('ylabel', 'Y Axis'))
        self.ui_xlim.setText(s.get('xlim', ''))
        self.ui_ylim.setText(s.get('ylim', ''))
        self.ui_xlog.setChecked(s.get('xlog', False))
        self.ui_ylog.setChecked(s.get('ylog', False))
        self.ui_grid.setChecked(s.get('grid', False))
        self.ui_legend.setChecked(s.get('legend', True))
        self.ui_keep_front.setChecked(s.get('keep_front', False))
        # Migrate old inch-based values to pixel-based values
        fig_w_str = s.get('fig_w', '400')
        fig_h_str = s.get('fig_h', '300')
        try:
            if float(fig_w_str) < 50:  # Likely old inch-based value
                fig_w_str = str(int(float(fig_w_str) * 100))
            if float(fig_h_str) < 50:  # Likely old inch-based value
                fig_h_str = str(int(float(fig_h_str) * 100))
        except (ValueError, TypeError):
            fig_w_str, fig_h_str = '400', '300'
        self.ui_fig_w.setText(fig_w_str)
        self.ui_fig_h.setText(fig_h_str)
        self.ui_font_size.setText(s.get('font_size', '10'))
        self.ui_auto_update.setChecked(s.get('auto_update', True))
        idx = self.ui_color_scheme.findText(
            s.get('color_scheme', 'Tab colors'))
        if idx >= 0:
            self.ui_color_scheme.setCurrentIndex(idx)
        self.ui_legend_font_size.setText(s.get('legend_font_size', ''))
        self.ui_auto_neg_x.setChecked(s.get('auto_neg_x', False))
        self.ui_inverse_colors.setChecked(s.get('inverse_colors', False))
        self.ui_auto_color.setChecked(s.get('auto_color_on_add', True))
        saved_delims = s.get('drop_delims', list(DEFAULT_DROP_DELIMS))
        self._drop_delim_keys = tuple(
            k for k, _, _ in DROP_DELIM_OPTIONS if k in saved_delims)
        self.ui_bg_color.setText(s.get('bg_color', ''))
        try:
            self._editor_log_split_ratio = float(
                s.get('editor_log_split_ratio', self._editor_log_split_ratio))
        except (TypeError, ValueError):
            pass
        try:
            self._right_split_ratio = float(
                s.get('right_split_ratio', self._right_split_ratio))
        except (TypeError, ValueError):
            pass
        if self._active_entry in self.entries:
            self._apply_editor_log_split_ratio(self._active_entry)
        if hasattr(self, 'right_splitter'):
            self._apply_right_split_ratio()
        self._suppress_update = False

    def _auto_save_settings(self):
        if not getattr(self, '_init_done', False):
            return
        try:
            ensure_profile_dir()
            # Atomic write: dump to temp file in the same dir, then replace.
            # Prevents tail-byte corruption if two saves race or a write is interrupted.
            tmp = LAST_SETTINGS_FILE + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self._get_settings_dict(), f,
                          indent=2, ensure_ascii=False)
            os.replace(tmp, LAST_SETTINGS_FILE)
        except Exception as e:
            import traceback
            print(f"Error auto-saving settings: {e}")
            traceback.print_exc()

    def _load_last_settings(self):
        if not os.path.exists(LAST_SETTINGS_FILE):
            return
        try:
            with open(LAST_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            # File was corrupted (e.g. concurrent-write tail bytes).
            # Fall back to defaults silently; next save will overwrite atomically.
            return
        except Exception as e:
            import traceback
            print(f"Error loading last settings: {e}")
            traceback.print_exc()
            return
        try:
            self._apply_settings_dict(data)
        except Exception as e:
            import traceback
            print(f"Error applying last settings: {e}")
            traceback.print_exc()

    # ════════════════ PRESETS ═════════════════════════════
    def _refresh_presets(self):
        self.ui_preset_combo.blockSignals(True)
        cur = self.ui_preset_combo.currentText()
        self.ui_preset_combo.clear()
        self.ui_preset_combo.addItem("(none)")
        for name in get_preset_names():
            self.ui_preset_combo.addItem(name)
        idx = self.ui_preset_combo.findText(cur)
        if idx >= 0:
            self.ui_preset_combo.setCurrentIndex(idx)
        self.ui_preset_combo.blockSignals(False)

    def _on_preset_selected(self, name):
        if name == "(none)":
            return
        fp = os.path.join(PROFILE_DIR, f"preset_{name}.json")
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                self._apply_settings_dict(json.load(f))
            self.request_update()
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Could not load preset: {e}")

    def save_preset(self):
        existing = get_preset_names()
        name, ok = QInputDialog.getItem(
            self, "Save Preset",
            "Choose an existing preset to overwrite, or type a new name:",
            existing, 0, True)
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in existing:
            if QMessageBox.question(
                    self, "Overwrite Preset",
                    f"Preset '{name}' already exists. Overwrite it?"
                    ) != QMessageBox.StandardButton.Yes:
                return
        fp = os.path.join(PROFILE_DIR, f"preset_{name}.json")
        try:
            ensure_profile_dir()
            with open(fp, 'w', encoding='utf-8') as f:
                json.dump(self._get_settings_dict(), f,
                          indent=2, ensure_ascii=False)
            self._refresh_presets()
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Could not save preset: {e}")

    def delete_preset(self):
        cur = self.ui_preset_combo.currentText()
        if cur == "(none)":
            return
        fp = os.path.join(PROFILE_DIR, f"preset_{cur}.json")
        if os.path.exists(fp):
            os.remove(fp)
        self._refresh_presets()

    # ════════════════ SESSION SAVE / LOAD ═════════════════
    @staticmethod
    def _is_session_payload(data):
        return isinstance(data, dict) and 'settings' in data and 'entries' in data

    @staticmethod
    def _is_plot_payload(data):
        if not isinstance(data, dict):
            return False
        if 'Curve_objects' in data or 'Grid_objects' in data:
            return True
        if 'Curve_objects_frames' in data or 'Grid_objects_frames' in data:
            return True
        plot_keys = {
            'x_axis_label', 'y_axis_label', 'x_lim', 'y_lim',
            'x_log', 'y_log', 'show_grid', 'plot_legend',
            'figure_title', 'window_title', 'fig_size_pixel',
            'fig_size_inch', 'font_size', 'legend_font_size',
        }
        return len(plot_keys.intersection(set(data.keys()))) >= 3

    @staticmethod
    def _format_limit_text(limit_val):
        if not isinstance(limit_val, (list, tuple)) or len(limit_val) < 2:
            return ''
        left = '' if limit_val[0] is None else format_float_repr(limit_val[0])
        right = '' if limit_val[1] is None else format_float_repr(limit_val[1])
        return f"{left}, {right}"

    @staticmethod
    def _df_to_jsonable(df):
        if df is None:
            return None
        if not isinstance(df, pd.DataFrame):
            return df
        safe_df = df.replace({np.nan: None})
        return {
            '__type__': 'dataframe',
            'values': safe_df.values.tolist(),
        }

    @staticmethod
    def _jsonable_to_df(data):
        if data is None:
            return None
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, dict) and data.get('__type__') == 'dataframe':
            return pd.DataFrame(data.get('values', []))
        if isinstance(data, list):
            return pd.DataFrame(data)
        return None

    def _load_from_session_payload(self, data):
        s = data.get('settings', {})
        self._apply_settings_dict(s)
        self.clear_entries()

        for ed in data.get('entries', []):
            w = self.add_entry(ed['type'])
            # backward compatible: 'enabled' or old 'checked'
            enabled = ed.get('enabled', ed.get('checked', True))
            w._is_enabled = enabled
            idx = self.entries.index(w)
            if idx < len(self.entry_checkboxes):
                self.entry_checkboxes[idx].setChecked(enabled)

            w._file_path = ed.get('file_path')
            ds_text = ed.get('data_source')
            if ds_text is None and w._file_path:
                ds_text = os.path.basename(w._file_path)
            if ds_text:
                w.inp_data_source.blockSignals(True)
                w.inp_data_source.setText(ds_text)
                w.inp_data_source.blockSignals(False)
            if ed.get('text_paste'):
                w._is_normalizing = True
                w.text_paste.blockSignals(True)
                w.text_paste.setPlainText(ed['text_paste'])
                w.text_paste.blockSignals(False)
                w._is_normalizing = False
            if ed.get('loaded_data') is not None:
                loaded_df = self._jsonable_to_df(ed.get('loaded_data'))
                if loaded_df is not None:
                    w.loaded_data = loaded_df
                    shape_str = (f"{w.loaded_data.shape[0]} × "
                                 f"{w.loaded_data.shape[1]}")
                    if w._file_path:
                        w.data_label.setText(
                            f"<b>Data Source</b> ({shape_str})  "
                            f"{os.path.basename(w._file_path)}")
                    else:
                        w.data_label.setText(
                            f"<b>Data Source</b> ({shape_str})")

            p = ed.get('params', {})
            if isinstance(w, CurveEntry):
                w.inp_label.setText(p.get('label', ''))
                # Backward compat: formulas used to live in the Label.  If this
                # is an old computed curve (no data, no formula in Data Source,
                # but a {{code}} formula in the Label), copy the formula into the
                # Data Source field so it still computes.  The Label is left
                # unchanged so the legend looks exactly as it did before.
                _label = w.inp_label.text()
                if (w.loaded_data is None
                        and not w.text_paste.toPlainText().strip()
                        and not CURVE_CODE_REF_RE.search(w.inp_data_source.text())
                        and _label and CURVE_CODE_REF_RE.search(_label)):
                    w.inp_data_source.blockSignals(True)
                    w.inp_data_source.setText(_label)
                    w.inp_data_source.blockSignals(False)
                code_val = p.get('code', '') or ''
                w.curve_code = code_val
                if getattr(w, '_code_edit', None) is not None:
                    w._code_edit.blockSignals(True)
                    w._code_edit.setText(code_val)
                    w._code_edit.blockSignals(False)
                # Backward compat: convert old program values to display
                color_t = p.get('color', 'Blue')
                w.inp_color.setCurrentText(
                    COLOR_DISPLAY_REV.get(color_t, color_t))
                w.inp_width.setText(p.get('width', '0.8'))
                fmt_t = p.get('fmt', '-')
                w.inp_fmt.setCurrentText(
                    LINE_FORMAT_DISPLAY_REV.get(fmt_t, fmt_t))
                marker_t = p.get('marker', '')
                w.inp_marker.setCurrentText(
                    MARKER_DISPLAY_REV.get(marker_t, marker_t))
                dot_color_t = p.get('dot_color', '')
                w.inp_dot_color.setCurrentText(
                    COLOR_DISPLAY_REV.get(dot_color_t, dot_color_t))
                w.inp_dot_width.setText(p.get('dot_width', '5'))
                w.chk_neg_x.setChecked(p.get('neg_x', False))
                w.chk_interp.setChecked(p.get('interp', True))
                w.inp_interp_kind.setCurrentText(
                    p.get('interp_kind', 'linear'))
                w.inp_interp_smoothing.setText(p.get('interp_smoothing', '0'))
                w.inp_interp_number.setText(p.get('interp_number', '5000'))
                w.inp_scale.setText(p.get('scale', ''))
                w.inp_offset.setText(p.get('offset', ''))
                w.inp_normalize.setText(p.get('normalize', ''))
                w.inp_align_ref.setText(p.get('align_ref', ''))
                w.inp_align_range.setText(p.get('align_range', ''))
                w.chk_align_scale.setChecked(p.get('align_scale', False))
                w.chk_align_offset.setChecked(p.get('align_offset', False))
                legend_color_t = p.get('legend_color', '')
                w.inp_legend_color.setCurrentText(
                    COLOR_DISPLAY_REV.get(legend_color_t, legend_color_t))
                legend_fmt_t = p.get('legend_format', '')
                w.inp_legend_format.setCurrentText(
                    LINE_FORMAT_DISPLAY_REV.get(legend_fmt_t, legend_fmt_t))
                w.chk_errorbar.setChecked(p.get('errorbar', True))
                w.inp_errorbar_capsize.setText(p.get('errorbar_capsize', '2'))
                w.inp_integ_start.setText(p.get('integ_start', ''))
                w.inp_integ_stop.setText(p.get('integ_stop', ''))
                integ_w = p.get('integ_weights', '')
                if integ_w:
                    w.inp_integ_weights.setCurrentText(integ_w)
            elif isinstance(w, GridEntry):
                w.inp_interp_type.setCurrentText(
                    p.get('interp_type', 'linear'))
                w.chk_contour.setChecked(p.get('contour', False))
                w.inp_density.setText(p.get('density', '100'))
                w.chk_colorbar.setChecked(p.get('colorbar', True))

        self._update_all_button_texts()
        self.request_update()

    @staticmethod
    def _extract_curve_objects_from_plot_payload(data):
        curve_objects = data.get('Curve_objects')
        if isinstance(curve_objects, dict):
            curve_objects = [curve_objects]
        if isinstance(curve_objects, list) and curve_objects:
            return [curve_data for curve_data in curve_objects if isinstance(curve_data, dict)]

        curve_frames = data.get('Curve_objects_frames')
        if isinstance(curve_frames, list) and curve_frames:
            try:
                frame_index = int(data.get('current_frame_index', 0))
            except (TypeError, ValueError):
                frame_index = 0
            frame_index = max(0, min(frame_index, len(curve_frames) - 1))
            selected_frame = curve_frames[frame_index]
            if isinstance(selected_frame, list):
                return [curve_data for curve_data in selected_frame if isinstance(curve_data, dict)]

        return []

    def _apply_curve_plot_payload_to_entry(self, entry, curve_data):
        xs = curve_data.get('X')
        ys = curve_data.get('Y')
        y_errorbar = curve_data.get('Y_errorbar', None)
        if isinstance(xs, (list, tuple)) and isinstance(ys, (list, tuple)) and len(xs) == len(ys) and len(xs) > 0:
            df_dict = {'X': xs, 'Y': ys}
            if y_errorbar is not None:
                if isinstance(y_errorbar, (list, tuple)) and len(y_errorbar) > 0:
                    if isinstance(y_errorbar[0], (list, tuple)):
                        # Asymmetric: [[lower...], [upper...]]
                        if len(y_errorbar) >= 2 and len(y_errorbar[0]) == len(xs):
                            df_dict['E_lower'] = y_errorbar[0]
                            df_dict['E_upper'] = y_errorbar[1]
                    elif len(y_errorbar) == len(xs):
                        # Symmetric: [err...]
                        df_dict['E'] = y_errorbar
            entry.loaded_data = pd.DataFrame(df_dict)
            entry._display_data_as_text()
            entry.data_label.setText(
                f"<b>Data Source</b> ({entry.loaded_data.shape[0]} × {entry.loaded_data.shape[1]})")

        entry.inp_label.setText(str(curve_data.get('Y_label', '')))
        curve_color = curve_data.get('curve_color', '') or ''
        entry.inp_color.setCurrentText(COLOR_DISPLAY_REV.get(curve_color, str(curve_color)))
        entry.inp_width.setText(str(curve_data.get('curve_width', 0.8)))
        curve_fmt = curve_data.get('curve_format', '') or ''
        entry.inp_fmt.setCurrentText(LINE_FORMAT_DISPLAY_REV.get(curve_fmt, str(curve_fmt)))
        marker_fmt = curve_data.get('dot_format', '') or ''
        entry.inp_marker.setCurrentText(MARKER_DISPLAY_REV.get(marker_fmt, str(marker_fmt)))
        dot_color = curve_data.get('dot_color', '') or ''
        entry.inp_dot_color.setCurrentText(COLOR_DISPLAY_REV.get(dot_color, str(dot_color)))
        entry.inp_dot_width.setText(str(curve_data.get('dot_width', 5)))
        dot_alpha = curve_data.get('dot_alpha', None)
        entry.inp_dot_alpha.setText('' if dot_alpha is None else str(dot_alpha))
        dot_edge_color = curve_data.get('dot_edge_color', '') or ''
        entry.inp_dot_edge_color.setCurrentText(COLOR_DISPLAY_REV.get(dot_edge_color, str(dot_edge_color)))
        dot_edge_width = curve_data.get('dot_edge_width', None)
        entry.inp_dot_edge_width.setText('' if dot_edge_width is None else str(dot_edge_width))
        entry.chk_interp.setChecked(bool(curve_data.get('do_interpolation', True)))
        entry.inp_interp_kind.setCurrentText(str(curve_data.get('interpolation_kind', 'linear')))
        entry.inp_interp_smoothing.setText(str(curve_data.get('interpolation_smoothing', 0)))
        entry.inp_interp_number.setText(str(curve_data.get('interpolation_number', 5000)))
        scale_factor = curve_data.get('scale_factor', None)
        entry.inp_scale.setText('' if scale_factor is None else str(scale_factor))
        normalize_to = curve_data.get('normalize_to', None)
        if isinstance(normalize_to, (list, tuple)) and len(normalize_to) >= 2:
            entry.inp_normalize.setText(f"{normalize_to[0]}, {normalize_to[1]}")
        elif normalize_to is None:
            entry.inp_normalize.setText('')
        else:
            entry.inp_normalize.setText(str(normalize_to))

        legend_color = curve_data.get('curve_legend_color', '') or ''
        entry.inp_legend_color.setCurrentText(COLOR_DISPLAY_REV.get(legend_color, str(legend_color)))
        legend_format = curve_data.get('curve_legend_format', '') or ''
        entry.inp_legend_format.setCurrentText(LINE_FORMAT_DISPLAY_REV.get(legend_format, str(legend_format)))
        fill_color = curve_data.get('fill_color', None)
        entry.inp_fill_color.setCurrentText('' if fill_color is None else str(fill_color))
        entry.chk_errorbar.setChecked(y_errorbar is not None and bool(y_errorbar))
        entry._update_errorbar_status()

    def _append_curves_from_plot_payload(self, data):
        curve_objects = self._extract_curve_objects_from_plot_payload(data)
        if not curve_objects:
            QMessageBox.information(self, "No Curves", "The selected Plot file does not contain curve data to append.")
            return None

        self._suppress_update = True
        appended_entries = []
        for curve_data in curve_objects:
            entry = self.add_entry('curve', select_new=False)
            if not isinstance(entry, CurveEntry):
                continue
            self._apply_curve_plot_payload_to_entry(entry, curve_data)
            appended_entries.append(entry)
        self._suppress_update = False

        if appended_entries:
            self._clear_associated_plot_json_path()
            self._apply_selection_state(
                set(appended_entries),
                active_entry=appended_entries[-1],
                anchor_entry=appended_entries[0],
            )
            self.request_update(from_entry_edit=True)
            return True
        return None

    def _prompt_plot_load_mode(self, path):
        if not self.entries:
            return PLOT_LOAD_REPLACE

        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Question)
        msg_box.setWindowTitle("Load Plot File")
        msg_box.setText(f"How should {os.path.basename(path)} be loaded?")
        msg_box.setInformativeText(
            "Replace will clear the current editor state. Append Curves will only add the new file's curves and curve settings."
        )
        replace_btn = msg_box.addButton("Replace", QMessageBox.ButtonRole.AcceptRole)
        append_btn = msg_box.addButton("Append Curves", QMessageBox.ButtonRole.ActionRole)
        cancel_btn = msg_box.addButton(QMessageBox.StandardButton.Cancel)
        msg_box.setDefaultButton(replace_btn)
        msg_box.exec()

        clicked = msg_box.clickedButton()
        if clicked == replace_btn:
            return PLOT_LOAD_REPLACE
        if clicked == append_btn:
            return PLOT_LOAD_APPEND
        if clicked == cancel_btn:
            return PLOT_LOAD_CANCEL
        return PLOT_LOAD_CANCEL

    def _load_from_plot_payload(self, data):
        curve_objects = self._extract_curve_objects_from_plot_payload(data)
        fig_px = data.get('fig_size_pixel')
        fig_in = data.get('fig_size_inch')
        fig_w = '400'
        fig_h = '300'
        if isinstance(fig_px, (list, tuple)) and len(fig_px) >= 2:
            fig_w = str(int(safe_eval_number(str(fig_px[0]), 400)))
            fig_h = str(int(safe_eval_number(str(fig_px[1]), 300)))
        elif isinstance(fig_in, (list, tuple)) and len(fig_in) >= 2:
            fig_w = str(int(float(fig_in[0]) * DEFAULT_FIG_DPI))
            fig_h = str(int(float(fig_in[1]) * DEFAULT_FIG_DPI))

        settings = {
            'title': data.get('figure_title', ''),
            'xlabel': data.get('x_axis_label', 'X Axis'),
            'ylabel': data.get('y_axis_label', 'Y Axis'),
            'xlim': self._format_limit_text(data.get('x_lim')),
            'ylim': self._format_limit_text(data.get('y_lim')),
            'xlog': bool(data.get('x_log', False)),
            'ylog': bool(data.get('y_log', False)),
            'grid': bool(data.get('show_grid', False)),
            'legend': bool(data.get('plot_legend', True)),
            'keep_front': bool(data.get('keep_front', False)),
            'fig_w': fig_w,
            'fig_h': fig_h,
            'font_size': str(data.get('font_size', 10)),
            'legend_font_size': '' if data.get('legend_font_size') is None else str(data.get('legend_font_size')),
        }
        self._apply_settings_dict(settings)
        self.clear_entries()

        for curve_data in curve_objects:
            w = self.add_entry('curve')
            if not isinstance(w, CurveEntry):
                continue
            self._apply_curve_plot_payload_to_entry(w, curve_data)

        grid_objects = data.get('Grid_objects')
        if isinstance(grid_objects, dict):
            grid_objects = [grid_objects]
        if not isinstance(grid_objects, list):
            grid_objects = []

        for grid_data in grid_objects:
            if not isinstance(grid_data, dict):
                continue
            w = self.add_entry('grid')
            if not isinstance(w, GridEntry):
                continue

            xyz = grid_data.get('XYZ_triples')
            if isinstance(xyz, (list, tuple)) and len(xyz) > 0:
                w.loaded_data = pd.DataFrame(xyz)
                w._display_data_as_text()
                w.data_label.setText(
                    f"<b>Data Source</b> ({w.loaded_data.shape[0]} × {w.loaded_data.shape[1]})")

            w.inp_interp_type.setCurrentText(str(grid_data.get('interpolation_type', 'linear')))
            w.chk_contour.setChecked(bool(grid_data.get('show_contour', False)))
            w.inp_density.setText(str(grid_data.get('interpolation_density', 100)))
            w.chk_colorbar.setChecked(bool(grid_data.get('show_colorbar', True)))

        # ── Multi-frame support ──
        # Deserialize Curve_objects_frames and Grid_objects_frames into actual
        # Curve / Grid instances so that do_update_plot can pass them straight
        # to the Plot window.
        curve_frames_raw = data.get('Curve_objects_frames')
        grid_frames_raw = data.get('Grid_objects_frames')
        has_frames = bool(curve_frames_raw) or bool(grid_frames_raw)

        if has_frames:
            if curve_frames_raw:
                self._stored_curve_frames = [
                    [Curve_from_JSON(c) for c in frame]
                    for frame in curve_frames_raw
                ]
            else:
                self._stored_curve_frames = None
            if grid_frames_raw:
                self._stored_grid_frames = [
                    [Grid_from_JSON(g) for g in frame]
                    for frame in grid_frames_raw
                ]
            else:
                self._stored_grid_frames = None
            self._stored_frame_labels = data.get('frame_labels')
            self._stored_current_frame_index = data.get('current_frame_index', 0)
        else:
            self._stored_curve_frames = None
            self._stored_grid_frames = None
            self._stored_frame_labels = None
            self._stored_current_frame_index = 0

        self._update_all_button_texts()
        self.request_update()

    def _try_load_session_or_plot_file(self, path, show_error=True):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if self._is_session_payload(data):
                self._load_from_session_payload(data)
                self._clear_associated_plot_json_path()
                # Bind to this .Plot_Editor file so Quick Save overwrites it.
                self._session_save_path = os.path.abspath(path)
                self._refresh_editor_title()
                self._mark_session_saved()
                return True
            if self._is_plot_payload(data):
                load_mode = self._prompt_plot_load_mode(path)
                if load_mode == PLOT_LOAD_CANCEL:
                    return None
                if load_mode == PLOT_LOAD_APPEND:
                    return self._append_curves_from_plot_payload(data)
                self._load_from_plot_payload(data)
                # A Plot JSON is not a session file; Quick Save should write a
                # new dated session rather than overwriting the source.
                self._session_save_path = None
                self._set_associated_plot_json_path(path)
                self._mark_session_saved()
                return True

            if show_error:
                raise ValueError(
                    "File is neither a Plot Editor session nor a Plot JSON dump.")
            return False
        except Exception as e:
            if show_error:
                QMessageBox.critical(self, "Error", f"Could not load: {e}")
                import traceback
                traceback.print_exc()
            return False

    def _capture_live_zoom_into_fields(self):
        """Sync the live plot window's current zoom into the xlim/ylim fields."""
        if self.plot_window is not None and hasattr(self.plot_window, '_ax'):
            try:
                if self.plot_window.isVisible():
                    xl = self.plot_window._ax.get_xlim()
                    yl = self.plot_window._ax.get_ylim()
                    self.ui_xlim.setText(f"{xl[0]}, {xl[1]}")
                    self.ui_ylim.setText(f"{yl[0]}, {yl[1]}")
            except Exception:
                pass

    def _build_session_dict(self):
        """Build the serializable .Plot_Editor session payload from the
        current editor state. Pure read — no UI side effects."""
        data = {'settings': self._get_settings_dict(), 'entries': []}
        for entry in self.entries:
            es = {
                'type': 'curve' if isinstance(entry, CurveEntry) else 'grid',
                'enabled': entry._is_enabled,
                # A computed curve's data box shows formula output, not user
                # data — persist it empty so the formula still rules on load.
                'text_paste': ('' if entry._formula_display_active
                               else entry.text_paste.toPlainText()),
                'loaded_data': self._df_to_jsonable(entry.loaded_data),
                'file_path': entry._file_path,
                'data_source': entry.inp_data_source.text(),
            }
            if isinstance(entry, CurveEntry):
                es['params'] = {
                    'label': entry.inp_label.text(),
                    'code': self._entry_code(entry),
                    'color': entry.inp_color.currentText(),
                    'width': entry.inp_width.text(),
                    'fmt': entry.inp_fmt.currentText(),
                    'marker': entry.inp_marker.currentText(),
                    'dot_color': entry.inp_dot_color.currentText(),
                    'dot_width': entry.inp_dot_width.text(),
                    'neg_x': entry.chk_neg_x.isChecked(),
                    'interp': entry.chk_interp.isChecked(),
                    'interp_kind': entry.inp_interp_kind.currentText(),
                    'interp_smoothing': entry.inp_interp_smoothing.text(),
                    'interp_number': entry.inp_interp_number.text(),
                    'scale': entry.user_scale_text(),
                    'offset': entry.user_offset_text(),
                    'normalize': entry.inp_normalize.text(),
                    'align_ref': entry.inp_align_ref.text(),
                    'align_range': entry.inp_align_range.text(),
                    'align_scale': entry.chk_align_scale.isChecked(),
                    'align_offset': entry.chk_align_offset.isChecked(),
                    'legend_color': entry.inp_legend_color.currentText(),
                    'legend_format': entry.inp_legend_format.currentText(),
                    'errorbar': entry.chk_errorbar.isChecked(),
                    'errorbar_capsize': entry.inp_errorbar_capsize.text(),
                    'integ_start': entry.inp_integ_start.text(),
                    'integ_stop': entry.inp_integ_stop.text(),
                    'integ_weights': entry.inp_integ_weights.currentText(),
                }
            elif isinstance(entry, GridEntry):
                es['params'] = {
                    'interp_type': entry.inp_interp_type.currentText(),
                    'contour': entry.chk_contour.isChecked(),
                    'density': entry.inp_density.text(),
                    'colorbar': entry.chk_colorbar.isChecked(),
                }
            data['entries'].append(es)
        return data

    def _write_session_file(self, path, data):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ────── unsaved-changes tracking ──────
    def _session_snapshot(self):
        """Canonical JSON string of the current session, for change detection."""
        return json.dumps(self._build_session_dict(),
                          ensure_ascii=False, sort_keys=True)

    def _mark_session_saved(self):
        """Record the current state as the last saved/loaded baseline."""
        try:
            self._saved_snapshot = self._session_snapshot()
        except Exception:
            self._saved_snapshot = None

    def _has_unsaved_changes(self):
        if getattr(self, '_saved_snapshot', None) is None:
            return False
        try:
            return self._session_snapshot() != self._saved_snapshot
        except Exception:
            return False

    # ────── new window / quick save ──────
    def new_window(self):
        """Open a fresh, independent editor window."""
        PlotController()  # self-registers in _open_windows and shows itself

    def _common_ancestor_dir(self):
        """Deepest directory containing every loaded data file, or None."""
        paths = []
        for entry in self.entries:
            fp = getattr(entry, '_file_path', None)
            if fp and os.path.isfile(fp):
                paths.append(os.path.abspath(fp))
        if not paths:
            return None
        try:
            common = os.path.commonpath(paths)
        except ValueError:
            # Paths span different drives (Windows) — no common ancestor.
            return None
        if os.path.isfile(common):
            common = os.path.dirname(common)
        return common or None

    def _auto_quick_save_path(self):
        """Dated filename inside the common-ancestor folder of the data files,
        falling back to the user's home directory."""
        folder = self._common_ancestor_dir() or os.path.expanduser("~")
        date_str = datetime.date.today().isoformat()
        candidate = os.path.join(folder, f"{date_str}.Plot_Editor")
        i = 2
        while os.path.exists(candidate):
            candidate = os.path.join(folder, f"{date_str}_{i}.Plot_Editor")
            i += 1
        return candidate

    def _suggest_save_dir(self):
        """Initial path/dir for the Save As dialog."""
        if self._session_save_path:
            return self._session_save_path
        return self._common_ancestor_dir() or ""

    def quick_save(self):
        """Save the session without a dialog. Overwrites the bound
        .Plot_Editor file if there is one; otherwise writes a dated file into
        the closest common-ancestor folder of the loaded data files.

        Returns True on success, False if nothing was written."""
        self._capture_live_zoom_into_fields()
        target = self._session_save_path or self._auto_quick_save_path()
        try:
            parent = os.path.dirname(target)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._write_session_file(target, self._build_session_dict())
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save: {e}")
            return False
        self._session_save_path = os.path.abspath(target)
        self._mark_session_saved()
        self._refresh_editor_title()
        self.ui_program_output.append(f"Saved: {self._session_save_path}")
        return True

    def save_session(self):
        # Capture current zoom state from the live plot
        self._capture_live_zoom_into_fields()

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", self._suggest_save_dir(),
            "Plot Editor Session (*.Plot_Editor)")
        if not path:
            return False
        if not path.lower().endswith('.plot_editor'):
            path += '.Plot_Editor'

        try:
            self._write_session_file(path, self._build_session_dict())
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save: {e}")
            return False
        self._session_save_path = os.path.abspath(path)
        self._mark_session_saved()
        self._refresh_editor_title()
        return True

    def load_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "",
            "Session / Plot Files (*.Plot_Editor *.json *.json.Plot *.Plot);;"
            "Session Files (*.Plot_Editor);;"
            "Plot JSON Files (*.json *.json.Plot *.Plot);;"
            "All Files (*.*)")
        if not path:
            return
        self._try_load_session_or_plot_file(path, show_error=True)


def register_windows_context_menu():
    if sys.platform != 'win32':
        return
    import winreg
    import os
    
    # Use python.exe to keep the console window (for debugging/errors)
    python_exe = sys.executable
        
    script_path = os.path.abspath(__file__)
    
    extensions = [".json", ".Plot", ".Plot_Editor"]
    menu_name = "LoadWithPlotEditor"
    menu_text = "Load with Plot Editor"
    command_text = f'"{python_exe}" "{script_path}" "%1"'
    
    for ext in extensions:
        try:
            key_path = rf"Software\Classes\SystemFileAssociations\{ext}\shell\{menu_name}"
            # Check if it already exists and has the correct command
            try:
                cmd_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path + r"\command")
                val, _ = winreg.QueryValueEx(cmd_key, "")
                winreg.CloseKey(cmd_key)
                if val == command_text:
                    continue  # Already registered correctly
            except FileNotFoundError:
                pass
            
            # Create or update key
            key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path)
            winreg.SetValue(key, "", winreg.REG_SZ, menu_text)
            
            cmd_key = winreg.CreateKey(key, "command")
            winreg.SetValue(cmd_key, "", winreg.REG_SZ, command_text)
            
            winreg.CloseKey(cmd_key)
            winreg.CloseKey(key)
        except Exception as e:
            print(f"Failed to register context menu for {ext}: {e}")


# ═════════════════════════ MAIN ══════════════════════════
if __name__ == '__main__':
    register_windows_context_menu()
    
    app = QApplication(sys.argv)
    font = QtGui.QFont("Arial", 10)
    app.setFont(font)
    window = PlotController()
    
    # If a file is passed via command line (e.g., from context menu)
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.isfile(file_path):
            window._try_load_session_or_plot_file(file_path, show_error=True)
            
    sys.exit(app.exec())
