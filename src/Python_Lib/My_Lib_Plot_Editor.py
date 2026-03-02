import sys
import os
import pathlib
import json
import pickle
import pandas as pd
import numpy as np
import io
import re
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QComboBox,
                             QFileDialog, QScrollArea, QFrame, QGroupBox,
                             QFormLayout, QMessageBox, QSizePolicy,
                             QStackedWidget, QInputDialog, QButtonGroup)
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont
from PyQt6.QtCore import Qt, pyqtSignal

parent_path = str(pathlib.Path(__file__).parent.resolve())
sys.path.insert(0, parent_path)

from My_Lib_Plot import (Plot, Curve, Grid, DEFAULT_FIG_DPI,
                         MARKER_DISPLAY, MARKER_DISPLAY_REV,
                         LINE_FORMAT_DISPLAY, LINE_FORMAT_DISPLAY_REV,
                         COLOR_DISPLAY, COLOR_DISPLAY_REV,
                         GRADIENT_COLORMAP_GROUPS, GRADIENT_COLORMAPS,
                         FILL_SCHEMES)

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

# ─── Inline property editing in curve list ───
INLINE_PROPERTY_OPTIONS = [
    'None',
    'Label', 'Color', 'Width', 'Curve Format', 'Marker Format',
    'Interpolation', 'Interp Kind', 'Scale Factor', 'Normalize To', 'X -> -X',
    'Method', 'Contours', 'Density', 'Colorbar',
]

# Maps property name → (entry_attr, widget_type, combo_options_or_None)
# widget_type: 'line_edit', 'editable_combo', 'combo', 'checkable'
CURVE_INLINE_PROPS = {
    'Label':              ('inp_label',             'line_edit',      None),
    'Color':              ('inp_color',             'editable_combo', list(COLOR_DISPLAY.keys())),
    'Width':              ('inp_width',             'line_edit',      None),
    'Curve Format':       ('inp_fmt',               'editable_combo', list(LINE_FORMAT_DISPLAY.keys())),
    'Marker Format':      ('inp_marker',            'editable_combo', list(MARKER_DISPLAY.keys())),
    'Dot Color':          ('inp_dot_color',         'editable_combo', list(COLOR_DISPLAY.keys())),
    'Dot Width':          ('inp_dot_width',         'line_edit',      None),
    'Interpolation':      ('chk_interp',            'checkable',      None),
    'Interp Kind':        ('inp_interp_kind',       'combo',          ["linear", "cubic", "quadratic", "nearest", "zero", "slinear", "spline", "nearest-up", "previous", "next"]),
    'Interp Smoothing':   ('inp_interp_smoothing',  'line_edit',      None),
    'Interp Number':      ('inp_interp_number',     'line_edit',      None),
    'Scale Factor':       ('inp_scale',             'line_edit',      None),
    'Normalize To':       ('inp_normalize',         'line_edit',      None),
    'Legend Color':       ('inp_legend_color',      'editable_combo', list(COLOR_DISPLAY.keys())),
    'Legend Format':      ('inp_legend_format',     'editable_combo', list(LINE_FORMAT_DISPLAY.keys())),
    'X -> -X':             ('chk_neg_x',             'checkable',      None),
}

GRID_INLINE_PROPS = {
    'Method':   ('inp_interp_type', 'combo',     ["linear", "cubic", "nearest", "multiquadric", "gaussian"]),
    'Contours': ('chk_contour',     'checkable', None),
    'Density':  ('inp_density',     'line_edit', None),
    'Colorbar': ('chk_colorbar',    'checkable', None),
}


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


# Profile directory: 3 levels up from this script → project root
PROFILE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "My_Lib_Plot_Editor_Profiles",
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


def format_data_to_aligned_text(data_rows):
    """Format row-wise data into aligned text with minimal spacing."""
    if not data_rows:
        return ""

    # Convert all to string first using format_float_repr
    str_rows = [[format_float_repr(v) for v in row] for row in data_rows]

    # Calculate max width per column
    if not str_rows:
        return ""
    num_cols = max(len(r) for r in str_rows)
    col_widths = [0] * num_cols

    for row in str_rows:
        for i, s in enumerate(row):
            if i < num_cols and len(s) > col_widths[i]:
                col_widths[i] = len(s)

    # Format rows
    lines = []
    for row in str_rows:
        parts = []
        for i, s in enumerate(row):
            if i < num_cols:
                parts.append(s.rjust(col_widths[i]))
        # Use 4 spaces separation for clear column distinction
        lines.append("    ".join(parts))

    return "\n".join(lines)


def normalize_data_text(text):
    """Parse data in X,Y / X Y / X\\tY format → Aligned text.
    Returns normalised text or None on parse failure."""
    lines = text.strip().split('\n')
    parsed_rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = re.split(r'[,\t]+|\s+', line)
        parts = [p.strip() for p in parts if p.strip()]
        try:
            nums = [float(p) for p in parts]
            parsed_rows.append(nums)
        except ValueError:
            return None

    if not parsed_rows:
        return None

    return format_data_to_aligned_text(parsed_rows)


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

    CHECKABLE_BTN = """
        QPushButton {
            background-color: #DDDDDD;
            border: 1px solid #AAAAAA;
            min-height: 30px; max-height: 30px;
        }
        QPushButton:checked {
            background-color: #90CAF9;
            font-weight: bold;
        }
    """

    ENTRY_BTN = """
        QPushButton {
            text-align: left;
            border: 1px solid #CCCCCC;
            min-height: 30px;
            max-height: 30px;
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

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # ── Top row: Data Source info and Delete button ──
        top_row = QHBoxLayout()
        
        # Data Source section
        data_source_layout = QVBoxLayout()
        data_source_layout.setSpacing(2)
        self.data_label = QLabel("<b>Data Source</b>")
        self.data_label.setWordWrap(True)
        data_source_layout.addWidget(self.data_label)
        
        top_row.addLayout(data_source_layout, stretch=1)
        
        # Delete button on the right
        self.btn_delete = QPushButton(f"Delete {entry_type_name}")
        self.btn_delete.setStyleSheet(GlobalStyles.DELETE_BTN)
        self.btn_delete.clicked.connect(self.delete_requested_signal.emit)
        top_row.addWidget(self.btn_delete, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        
        root.addLayout(top_row)
        root.addSpacing(8)

        # ── Horizontal: (left) data + paste | (right) properties ──
        h_split = QHBoxLayout()
        h_split.setSpacing(20)  # Add more horizontal spacing between columns

        # Left column: text paste area
        left = QVBoxLayout()

        self.text_paste = DroppableTextEdit()
        paste_font = QFont("Consolas", 9)
        self.text_paste.setFont(paste_font)
        fm = QtGui.QFontMetrics(paste_font)
        self.text_paste.setTabStopDistance(fm.averageCharWidth() * 16)
        self.text_paste.setPlaceholderText(
            "Paste data here (X,Y  or  X Y  or  X\\tY)\n"
            "or drop CSV / XLSX files …")
        # Don't connect textChanged for immediate updates
        self.text_paste.installEventFilter(self)
        self.text_paste.file_dropped.connect(self.load_file)
        left.addWidget(self.text_paste, 1)

        # Right column: property form (populated by subclass)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_w = QWidget()
        self.form_layout = QFormLayout(right_w)
        self.form_layout.setContentsMargins(0, 0, 0, 0)
        right_scroll.setWidget(right_w)

        h_split.addLayout(left, 1)
        h_split.addWidget(right_scroll, 1)
        root.addLayout(h_split, 1)

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

    def parse_text_data(self):
        text = self.text_paste.toPlainText()
        if not text.strip():
            self.loaded_data = None
            self.data_label.setText("<b>Data Source</b>")
            self.emit_change()
            return
        try:
            df = pd.read_csv(io.StringIO(text), sep=None,
                             engine='python', header=None)
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
                f"{self.loaded_data.shape[1]})<br>"
                f"<span style='line-height: 1.5; margin-top: 3px; display: inline-block;'>{os.path.basename(filepath)}</span>")
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
        self._setup_ui()

    def _setup_ui(self):
        self.inp_label = self.add_line_edit("Label", "")
        self.inp_label.editingFinished.connect(
            lambda: self.label_changed_signal.emit(self.inp_label.text()))
        self.inp_color = self.add_editable_combo(
            "Color", list(COLOR_DISPLAY.keys()), "Blue")
        self.inp_width = self.add_line_edit("Width", "0.8")
        self.inp_fmt = self.add_editable_combo(
            "Curve Format", list(LINE_FORMAT_DISPLAY.keys()), "Solid (-)")
        self.inp_marker = self.add_editable_combo(
            "Marker Format", list(MARKER_DISPLAY.keys()), "")
        self.inp_dot_color = self.add_editable_combo(
            "Dot Color", list(COLOR_DISPLAY.keys()), "")
        self.inp_dot_width = self.add_line_edit("Dot Width", "5")
        self.chk_interp = self.add_checkable_button("Interpolation", True)
        self.inp_interp_kind = self.add_combo(
            "Interp Kind",
            ["linear", "cubic", "quadratic", "nearest", "zero", "slinear", "spline", "nearest-up", "previous", "next"], "linear")
        self.inp_interp_smoothing = self.add_line_edit("Interp Smoothing", "0")
        self.inp_interp_number = self.add_line_edit("Interp Number", "5000")
        self.inp_scale = self.add_line_edit("Scale Factor", "")
        self.inp_normalize = self.add_line_edit(
            "Normalize To", "")
        self.inp_legend_color = self.add_editable_combo(
            "Legend Color", list(COLOR_DISPLAY.keys()), "")
        self.inp_legend_format = self.add_editable_combo(
            "Legend Format", list(LINE_FORMAT_DISPLAY.keys()), "")
        # Fill color: empty = no fill, color name = solid fill, scheme name = sectioned fill
        fill_options = [''] + list(COLOR_DISPLAY.keys())[1:] + list(FILL_SCHEMES.keys())
        self.inp_fill_color = self.add_editable_combo("Fill Color", fill_options, "")
        self.chk_neg_x = self.add_checkable_button("X -> -X", False)

        # ── Integration ──
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        self.form_layout.addRow(sep)
        self.form_layout.addRow(QLabel("<b>Integration</b>"))
        self.inp_integ_start = self.add_line_edit("Start", "")
        self.inp_integ_stop = self.add_line_edit("Stop", "")

        # Weights file with browse
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
        self.form_layout.addRow("Weights", weights_row)

        self.inp_integ_normalize = self.add_line_edit("Normalize Integration to", "")

        self.btn_integrate = QPushButton("Compute Integration")
        self.btn_integrate.setStyleSheet("font-weight:bold; background:#E3F2FD;")
        self.form_layout.addRow(self.btn_integrate)

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

    def get_object(self):
        if not self._is_enabled:
            return None
        X, Y, _Z = self.get_data_arrays()
        if Y is None:
            return None
        width = safe_eval_number(self.inp_width.text(), 0.8)
        dot_width = safe_eval_number(self.inp_dot_width.text(), 5)
        scale = safe_eval_number(self.inp_scale.text(), None) if self.inp_scale.text().strip() else None
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
        
        return Curve(
            X=X, Y=Y,
            X_label="",
            Y_label=self.inp_label.text(),
            curve_color=color if color else None,
            curve_width=width,
            plot_curve=bool(fmt),
            curve_format=fmt,
            plot_dot=bool(marker),
            dot_format=marker if marker else '.',
            dot_color=dot_color if dot_color else None,
            dot_width=dot_width,
            do_interpolation=self.chk_interp.isChecked(),
            interpolation_kind=self.inp_interp_kind.currentText(),
            interpolation_smoothing=interp_smoothing,
            interpolation_number=interp_number,
            curve_legend_color=legend_color if legend_color else "",
            curve_legend_format=legend_format if legend_format else "",
            fill_color=fill_color,
            scale_factor=scale,
            normalize_to=normalize_to,
        )


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
        if Z is None:
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


class StreamRedirector(QtCore.QObject):
    text_written = pyqtSignal(str)
    
    def write(self, text):
        self.text_written.emit(str(text))
        
    def flush(self):
        pass


# ────────────────────── PlotController ───────────────────
class PlotController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plot Controller GUI")
        self.resize(1500, 900)
        self.setStyleSheet(GlobalStyles.STYLE_SHEET)
        self.setAcceptDrops(True)

        self.plot_window = None
        self.entries: list[DataEntryWidget] = []
        self.entry_buttons: list[QPushButton] = []
        self.entry_checkboxes: list[QtWidgets.QCheckBox] = []
        self.entry_prop_containers: list[QWidget] = []
        self._suppress_update = False
        self._suppress_prop_sync = False
        self._init_done = False

        self._center_window()
        self._init_ui()
        self._load_last_settings()
        self._update_hline_button_state()  # Initialize button state
        self._init_done = True
        self.show()

    # ────── window helpers ──────
    def _center_window(self):
        qr = self.frameGeometry()
        cp = QtGui.QGuiApplication.primaryScreen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):
        self._auto_save_settings()
        super().closeEvent(event)

    # ────── drag & drop on main window ──────
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            f = url.toLocalFile()
            if os.path.isfile(f):
                self.add_entry('curve', f)

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

        # -- Load Preset --
        sl.addWidget(QLabel("<b>Load Preset</b>"))
        self.ui_preset_combo = QComboBox()
        self.ui_preset_combo.addItem("(none)")
        self._refresh_presets()
        self.ui_preset_combo.currentTextChanged.connect(
            self._on_preset_selected)
        sl.addWidget(self.ui_preset_combo)
        sl.addSpacing(8)

        # -- Settings form --
        form_w = QWidget()
        self.form_layout = QFormLayout(form_w)
        self.form_layout.setContentsMargins(0, 0, 0, 0)

        self.ui_title     = self._mk_le("Title", "My Plot")
        self.ui_xlim      = self._mk_le("X Lim (min, max)", "")
        self.ui_ylim      = self._mk_le("Y Lim (min, max)", "")

        # Paired row: X Label / Y Label
        xy_label_row = QHBoxLayout()
        xy_label_row.addWidget(QLabel("X Label"))
        self.ui_xlabel = self._mk_le_no_row("X Axis")
        xy_label_row.addWidget(self.ui_xlabel)
        xy_label_row.addWidget(QLabel("Y Label"))
        self.ui_ylabel = self._mk_le_no_row("Y Axis")
        xy_label_row.addWidget(self.ui_ylabel)
        self.form_layout.addRow(xy_label_row)

        row1 = QHBoxLayout()
        self.ui_xlog = self._mk_ckbtn("X Log")
        self.ui_ylog = self._mk_ckbtn("Y Log")
        row1.addWidget(self.ui_xlog)
        row1.addWidget(self.ui_ylog)
        self.form_layout.addRow(row1)

        row2 = QHBoxLayout()
        self.ui_grid   = self._mk_ckbtn("Grid")
        self.ui_legend = self._mk_ckbtn("Legend", True)
        row2.addWidget(self.ui_grid)
        row2.addWidget(self.ui_legend)
        self.form_layout.addRow(row2)

        # Keep Front option
        self.ui_keep_front = self._mk_ckbtn("Keep Front", False)
        self.form_layout.addRow(self.ui_keep_front)

        # Paired row: Width / Height
        wh_row = QHBoxLayout()
        wh_row.addWidget(QLabel("W (px)"))
        self.ui_fig_w = self._mk_le_no_row("400")
        wh_row.addWidget(self.ui_fig_w)
        wh_row.addWidget(QLabel("H (px)"))
        self.ui_fig_h = self._mk_le_no_row("300")
        wh_row.addWidget(self.ui_fig_h)
        self.form_layout.addRow(wh_row)

        # Paired row: Font Size / Legend Font Size
        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("Font"))
        self.ui_font_size = self._mk_le_no_row("10")
        font_row.addWidget(self.ui_font_size)
        font_row.addWidget(QLabel("Legend"))
        self.ui_legend_font_size = self._mk_le_no_row("")
        self.ui_legend_font_size.setPlaceholderText("= Font")
        font_row.addWidget(self.ui_legend_font_size)
        self.form_layout.addRow(font_row)

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

        # -- Auto Assign Colors (checkable) --
        sl.addSpacing(5)
        self.btn_auto_color = QPushButton("Auto Assign Colors")
        self.btn_auto_color.setCheckable(True)
        self.btn_auto_color.setStyleSheet(GlobalStyles.CHECKABLE_BTN)
        self.btn_auto_color.toggled.connect(self._on_auto_color_toggled)
        sl.addWidget(self.btn_auto_color)

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
        cs_row.addWidget(self.ui_color_scheme, stretch=1)
        self.ui_inverse_colors = QPushButton("Inv")
        self.ui_inverse_colors.setCheckable(True)
        self.ui_inverse_colors.setStyleSheet(GlobalStyles.CHECKABLE_BTN)
        self.ui_inverse_colors.setFixedWidth(40)
        self.ui_inverse_colors.toggled.connect(self._on_color_scheme_changed)
        cs_row.addWidget(self.ui_inverse_colors)
        sl.addLayout(cs_row)

        # -- Update --
        sl.addSpacing(8)
        self.btn_update = QPushButton("Update Plot")
        self.btn_update.setStyleSheet(
            "font-weight:bold; font-size:10pt; background:#DDDDDD; "
            "min-height:30px; max-height:30px;")
        self.btn_update.clicked.connect(self.do_update_plot)
        self.ui_auto_update = self._mk_ckbtn("Auto Update", True)
        update_row = QHBoxLayout()
        update_row.addWidget(self.ui_auto_update)
        update_row.addWidget(self.btn_update)
        sl.addLayout(update_row)

        # -- Save / Delete Preset --
        sl.addSpacing(8)
        p_row = QHBoxLayout()
        btn_sp = QPushButton("Save Preset")
        btn_sp.clicked.connect(self.save_preset)
        btn_dp = QPushButton("Delete Preset")
        btn_dp.clicked.connect(self.delete_preset)
        p_row.addWidget(btn_sp)
        p_row.addWidget(btn_dp)
        sl.addLayout(p_row)

        # -- Save / Load Session --
        s_row = QHBoxLayout()
        btn_ss = QPushButton("Save Session")
        btn_ss.clicked.connect(self.save_session)
        btn_ls = QPushButton("Load Session")
        btn_ls.clicked.connect(self.load_session)
        s_row.addWidget(btn_ss)
        s_row.addWidget(btn_ls)
        sl.addLayout(s_row)

        # -- Copy Python Code --
        sl.addSpacing(5)
        btn_copy_code = QPushButton("Copy Python Code")
        btn_copy_code.setStyleSheet("font-weight:bold; background:#E8F5E9;")
        btn_copy_code.clicked.connect(self._copy_python_code)
        sl.addWidget(btn_copy_code)

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

        # ═══════ RIGHT PANEL (entry list + result) ═══════
        right_w = QWidget()
        right_w.setMinimumWidth(320)
        right_w.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        rl = QVBoxLayout(right_w)
        rl.setContentsMargins(5, 5, 5, 5)

        # Load file buttons
        load_row = QHBoxLayout()
        btn_load_curve = QPushButton("Load Curve File")
        btn_load_curve.clicked.connect(lambda: self._load_curve_file())
        btn_load_grid = QPushButton("Load Grid File")
        btn_load_grid.clicked.connect(lambda: self._load_grid_file())
        load_row.addWidget(btn_load_curve)
        load_row.addWidget(btn_load_grid)
        rl.addLayout(load_row)

        # Add new entry buttons
        ar = QHBoxLayout()
        btn_ac = QPushButton("New Curve")
        btn_ac.clicked.connect(lambda: self.add_entry('curve'))
        btn_ag = QPushButton("New Grid")
        btn_ag.clicked.connect(lambda: self.add_entry('grid'))
        ar.addWidget(btn_ac)
        ar.addWidget(btn_ag)
        rl.addLayout(ar)

        btn_cl = QPushButton("Clear All")
        btn_cl.clicked.connect(self._confirm_clear_entries)
        rl.addWidget(btn_cl)

        # Property management dropdown
        self.ui_property_dropdown = QComboBox()
        self.ui_property_dropdown.addItems(INLINE_PROPERTY_OPTIONS)
        self.ui_property_dropdown.currentTextChanged.connect(
            self._on_property_dropdown_changed)
        self.ui_property_dropdown.setFixedHeight(30)

        # Move up/down buttons
        move_row = QHBoxLayout()
        btn_move_up = QPushButton("↑")
        btn_move_up.clicked.connect(self._move_selected_entry_up)
        btn_move_up.setFixedSize(30, 30)
        btn_move_down = QPushButton("↓")
        btn_move_down.clicked.connect(self._move_selected_entry_down)
        btn_move_down.setFixedSize(30, 30)
        move_row.addWidget(self.ui_property_dropdown)
        move_row.addWidget(btn_move_up)
        move_row.addWidget(btn_move_down)
        rl.addLayout(move_row)
        rl.addSpacing(4)

        self.entry_scroll = QScrollArea()
        self.entry_scroll.setWidgetResizable(True)
        self.entry_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.entry_list_widget = QWidget()
        self.entry_list_layout = QVBoxLayout(self.entry_list_widget)
        self.entry_list_layout.setContentsMargins(0, 0, 0, 0)
        self.entry_list_layout.setSpacing(3)
        self.entry_list_layout.addStretch()
        self.entry_scroll.setWidget(self.entry_list_widget)
        rl.addWidget(self.entry_scroll)

        self.entry_button_group = QButtonGroup(self)
        self.entry_button_group.setExclusive(True)

        # ── Shared Result area (right panel bottom) ──
        result_group = QGroupBox("Result")
        result_group.setMinimumWidth(320)
        result_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        result_layout = QVBoxLayout(result_group)
        result_layout.setContentsMargins(4, 4, 4, 4)

        # Program Output (Stdout) which also contains integration results
        self.ui_program_output = QtWidgets.QTextEdit()
        self.ui_program_output.setReadOnly(True)
        self.ui_program_output.setFont(QFont("Consolas", 9))
        self.ui_program_output.setFixedHeight(80)
        self.ui_program_output.setPlaceholderText("Program output will appear here...")
        result_layout.addWidget(self.ui_program_output)

        rl.addWidget(result_group)

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

    def add_entry(self, type_, data_file=None):
        widget = CurveEntry() if type_ == 'curve' else GridEntry()
        widget.data_changed_signal.connect(self.request_update)
        # Also update horizontal lines when curve data changes
        if isinstance(widget, CurveEntry):
            widget.data_changed_signal.connect(self._on_curve_data_changed)
        widget.delete_requested_signal.connect(
            lambda w=widget: self.remove_entry(w))
        if isinstance(widget, CurveEntry):
            widget.label_changed_signal.connect(
                lambda _t, w=widget: self._update_entry_button_text(w))
            widget.btn_integrate.clicked.connect(
                lambda _c, w=widget: self._compute_integration(w))

        self.entries.append(widget)
        self.stacked_widget.addWidget(widget)

        # row in right list: [checkbox] [button]
        row_w = QWidget()
        row_l = QHBoxLayout(row_w)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.setSpacing(2)

        chk = QtWidgets.QCheckBox()
        chk.setChecked(True)
        chk.setFixedWidth(30)
        chk.toggled.connect(
            lambda checked, w=widget: self._on_entry_enabled(w, checked))

        idx = len(self.entries)
        type_label = "Curve" if type_ == 'curve' else "Grid"
        btn = QPushButton(f"{type_label} {idx}")
        btn.setCheckable(True)
        btn.setStyleSheet(GlobalStyles.ENTRY_BTN)
        btn.clicked.connect(
            lambda _c, w=widget: self.select_entry(w))
        self.entry_button_group.addButton(btn)

        row_l.addWidget(chk)
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
        self.select_entry(widget)
        # Rebuild inline property widget for the new entry
        self._rebuild_inline_property_widget(len(self.entries) - 1)
        # Auto-assign colours when the checkable button is on
        if self.btn_auto_color.isChecked():
            self.auto_assign_colors()
        # Update horizontal line button state
        self._update_hline_button_state()
        # Update horizontal lines X range if this is a regular curve
        if isinstance(widget, CurveEntry) and not self._is_horizontal_line(widget):
            self._update_horizontal_lines()
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
        self.entries.pop(idx)
        btn = self.entry_buttons.pop(idx)
        self.entry_checkboxes.pop(idx)
        self.entry_prop_containers.pop(idx)
        self.entry_button_group.removeButton(btn)
        row_w = btn.parentWidget()
        self.entry_list_layout.removeWidget(row_w)
        row_w.deleteLater()
        self.stacked_widget.removeWidget(widget)
        widget.deleteLater()
        self._update_all_button_texts()
        if self.entries:
            self.select_entry(
                self.entries[min(idx, len(self.entries) - 1)])
        else:
            self.stacked_widget.setCurrentWidget(self.placeholder)
        # Update horizontal line button state
        self._update_hline_button_state()
        # Update horizontal lines X range after removing an entry
        if isinstance(widget, CurveEntry) and not self._is_horizontal_line(widget):
            self._update_horizontal_lines()
        self.request_update()

    def select_entry(self, widget):
        if widget in self.entries:
            self.stacked_widget.setCurrentWidget(widget)
            idx = self.entries.index(widget)
            self.entry_buttons[idx].setChecked(True)
        # Update horizontal line button state
        self._update_hline_button_state()

    def _update_hline_button_state(self):
        """Enable/disable horizontal line button based on whether any Grid entries exist."""
        has_grid = any(isinstance(entry, GridEntry) for entry in self.entries)
        self.btn_add_hline.setEnabled(not has_grid)

    def _on_curve_data_changed(self):
        """Called when any curve's data changes - update horizontal lines."""
        # Get the sender (the curve that changed)
        sender = self.sender()
        # Only update if the changed curve is not a horizontal line itself
        for entry in self.entries:
            if entry.data_changed_signal == sender:
                if isinstance(entry, CurveEntry) and not self._is_horizontal_line(entry):
                    self._update_horizontal_lines()
                break

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
        self.stacked_widget.setCurrentWidget(self.placeholder)
        # Update horizontal line button state
        self._update_hline_button_state()
        self.request_update()

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
        self.request_update()

    def _update_entry_button_text(self, widget):
        if widget not in self.entries:
            return
        idx = self.entries.index(widget)
        tp = "Curve" if isinstance(widget, CurveEntry) else "Grid"
        # Check for custom button text (e.g., for horizontal lines)
        if hasattr(widget, '_custom_button_text') and widget._custom_button_text:
            self.entry_buttons[idx].setText(widget._custom_button_text)
        elif isinstance(widget, CurveEntry) and widget.inp_label.text().strip():
            self.entry_buttons[idx].setText(widget.inp_label.text().strip())
        else:
            self.entry_buttons[idx].setText(f"{tp} {idx + 1}")

    def _update_all_button_texts(self):
        for i, (entry, btn) in enumerate(
                zip(self.entries, self.entry_buttons)):
            tp = "Curve" if isinstance(entry, CurveEntry) else "Grid"
            if isinstance(entry, CurveEntry) and entry.inp_label.text().strip():
                btn.setText(entry.inp_label.text().strip())
            else:
                btn.setText(f"{tp} {i + 1}")

    # ════════════════ MOVE ENTRIES UP/DOWN ═════════════════
    def _get_selected_index(self):
        """Return the index of the currently selected entry, or -1."""
        for i, btn in enumerate(self.entry_buttons):
            if btn.isChecked():
                return i
        return -1

    def _swap_entries(self, idx_a, idx_b):
        """Swap two entries in the list (data structures and UI row widgets)."""
        if idx_a == idx_b:
            return
        # Ensure idx_a < idx_b for consistency
        if idx_a > idx_b:
            idx_a, idx_b = idx_b, idx_a

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
        self.request_update()

    def _move_selected_entry_down(self):
        idx = self._get_selected_index()
        if idx < 0 or idx >= len(self.entries) - 1:
            return
        self._swap_entries(idx, idx + 1)
        self._rebuild_all_inline_property_widgets()
        self.request_update()

    # ════════════════ AUTO-COLOR ══════════════════════════
    def _on_auto_color_toggled(self, checked):
        if checked:
            self.auto_assign_colors()

    def _on_color_scheme_changed(self, *_args):
        if self.btn_auto_color.isChecked():
            self.auto_assign_colors()

    def auto_assign_colors(self):
        curves = [e for e in self.entries if isinstance(e, CurveEntry)]
        n = len(curves)
        if n == 0:
            return
        inverse = self.ui_inverse_colors.isChecked()
        colors = get_gradient_colors(n, self.ui_color_scheme.currentText(), inverse=inverse)
        for entry, c in zip(curves, colors):
            entry.inp_color.setCurrentText(COLOR_DISPLAY_REV.get(c, c))
        self.request_update()

    def _apply_auto_neg_x(self, checked):
        """When checked, enable X -> -X on curves whose X values are all negative."""
        if not checked:
            return
        for entry in self.entries:
            if not isinstance(entry, CurveEntry):
                continue
            X, Y, _ = entry.get_data_arrays()
            if X is not None and len(X) > 0 and np.all(X < 0):
                entry.chk_neg_x.setChecked(True)
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

            inline.editingFinished.connect(lambda s=source, i=inline: s.setText(i.text()))
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
            inline.setMaxVisibleItems(min(len(options) if options else 10, 20))
            inline.setCurrentText(source.currentText())

            def sync_to_source():
                if self._suppress_prop_sync:
                    return
                self._suppress_prop_sync = True
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
                inline.lineEdit().editingFinished.connect(sync_to_source)
            inline.activated.connect(sync_to_source)
            source.currentTextChanged.connect(sync_from)
            return inline

        elif widget_type == 'checkable':
            inline = QPushButton()
            inline.setCheckable(True)
            inline.setChecked(source.isChecked())
            inline.setFixedWidth(30)
            inline.setStyleSheet(GlobalStyles.CHECKABLE_BTN)

            def sync_to(checked, s=source):
                if self._suppress_prop_sync:
                    return
                self._suppress_prop_sync = True
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
    def request_update(self):
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
            return

        objs = [o for o in objs if o is not None]

        # If no objects remain, close the plot window
        if not objs:
            if self.plot_window is not None:
                self.plot_window.close()
                self.plot_window = None
            return

        opts = {
            'figure_title': self.ui_title.text(),
            'x_axis_label': self.ui_xlabel.text(),
            'y_axis_label': self.ui_ylabel.text(),
            'x_lim': self._tuple_or_none(self.ui_xlim.text()),
            'y_lim': self._tuple_or_none(self.ui_ylim.text()),
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

        try:
            if self.plot_window is None:
                self.plot_window = Plot(objs, **opts)
                # Set the figure title explicitly after creation
                self.plot_window.set_figure_title(opts['figure_title'])
            else:
                p = self.plot_window
                # Set all properties BEFORE triggering _update_plot
                for k in ('x_axis_label', 'y_axis_label', 'x_lim', 'y_lim',
                          'x_log', 'y_log', 'show_grid', 'plot_legend',
                          'figure_title', 'keep_front'):
                    setattr(p, k, opts[k])
                p.font_size = opts['font_size']
                p.legend_font_size = opts['legend_font_size']

                # Update pixel target and recalculate figure size via centralized method
                if opts['fig_size_pixel'] is not None:
                    p._fig_size_pixel = tuple(opts['fig_size_pixel'])
                else:
                    p._fig_size_pixel = (opts['fig_size_inch'][0] * DEFAULT_FIG_DPI, opts['fig_size_inch'][1] * DEFAULT_FIG_DPI)
                p._setup_figure_size()

                # Update objects and redraw
                # _update_plot calls resize_window_to_fig at the end
                p._plot_objects = list(objs)
                p._update_plot()


                p.set_figure_title(opts['figure_title'])
            
            # Ensure the plot window is visible and updated
            if not self.plot_window.isVisible():
                self.plot_window.show()
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

    # ════════════════ COPY PYTHON CODE ════════════════════
    def _copy_python_code(self):
        """Generate Python code that recreates the current plot and copy to clipboard."""
        lines = []
        lines.append("from Python_Lib.My_Lib_Plot import *\n")

        curve_vars = []
        grid_vars = []
        var_idx = 0

        for i, entry in enumerate(self.entries):
            if not entry._is_enabled:
                continue

            if isinstance(entry, CurveEntry):
                var_idx += 1
                X, Y, _Z = entry.get_data_arrays()
                if Y is None:
                    continue
                var_name = f"curve_{var_idx}"
                curve_vars.append(var_name)

                # Data arrays
                x_list = X.tolist() if hasattr(X, 'tolist') else list(X)
                y_list = Y.tolist() if hasattr(Y, 'tolist') else list(Y)

                if entry.chk_neg_x.isChecked():
                    lines.append(f"X_{var_idx} = [-x for x in {repr(x_list)}]")
                else:
                    lines.append(f"X_{var_idx} = {repr(x_list)}")
                lines.append(f"Y_{var_idx} = {repr(y_list)}")

                # Curve parameters
                params = {}
                params['X'] = f"X_{var_idx}"
                params['Y'] = f"Y_{var_idx}"

                label = entry.inp_label.text()
                if label:
                    params['Y_label'] = repr(label)

                color = resolve_display_value(entry.inp_color.currentText().strip(), COLOR_DISPLAY)
                if color:
                    params['curve_color'] = repr(color)

                try:
                    width = float(entry.inp_width.text())
                    if width != 0.8:
                        params['curve_width'] = repr(width)
                except (ValueError, TypeError):
                    pass

                fmt = resolve_display_value(entry.inp_fmt.currentText().strip(), LINE_FORMAT_DISPLAY)
                if fmt:
                    params['plot_curve'] = 'True'
                    params['curve_format'] = repr(fmt)
                else:
                    params['plot_curve'] = 'False'

                marker = resolve_display_value(entry.inp_marker.currentText().strip(), MARKER_DISPLAY)
                if marker:
                    params['plot_dot'] = 'True'
                    params['dot_format'] = repr(marker)
                else:
                    params['plot_dot'] = 'False'

                dot_color = resolve_display_value(entry.inp_dot_color.currentText().strip(), COLOR_DISPLAY)
                if dot_color:
                    params['dot_color'] = repr(dot_color)

                try:
                    dot_width = float(entry.inp_dot_width.text())
                    if dot_width != 5:
                        params['dot_width'] = repr(dot_width)
                except (ValueError, TypeError):
                    pass

                interp = entry.chk_interp.isChecked()
                params['do_interpolation'] = repr(interp)

                interp_kind = entry.inp_interp_kind.currentText()
                if interp_kind != 'linear':
                    params['interpolation_kind'] = repr(interp_kind)

                try:
                    interp_smoothing = float(entry.inp_interp_smoothing.text())
                    if interp_smoothing != 0:
                        params['interpolation_smoothing'] = repr(interp_smoothing)
                except (ValueError, TypeError):
                    pass

                try:
                    interp_number = int(float(entry.inp_interp_number.text()))
                    if interp_number != 5000:
                        params['interpolation_number'] = repr(interp_number)
                except (ValueError, TypeError):
                    pass

                scale_text = entry.inp_scale.text().strip()
                if scale_text:
                    try:
                        params['scale_factor'] = repr(float(scale_text))
                    except (ValueError, TypeError):
                        pass

                normalize_text = entry.inp_normalize.text().strip()
                if normalize_text:
                    if ',' in normalize_text:
                        try:
                            nt = tuple(float(x.strip()) for x in normalize_text.split(',') if x.strip())
                            params['normalize_to'] = repr(nt)
                        except (ValueError, TypeError):
                            pass
                    else:
                        try:
                            params['normalize_to'] = repr(float(normalize_text))
                        except (ValueError, TypeError):
                            pass

                legend_color = resolve_display_value(entry.inp_legend_color.currentText().strip(), COLOR_DISPLAY)
                if legend_color:
                    params['curve_legend_color'] = repr(legend_color)

                legend_format = resolve_display_value(entry.inp_legend_format.currentText().strip(), LINE_FORMAT_DISPLAY)
                if legend_format:
                    params['curve_legend_format'] = repr(legend_format)

                # Build Curve(...) call
                param_strs = []
                for k, v in params.items():
                    param_strs.append(f"    {k}={v},")
                lines.append(f"{var_name} = Curve(")
                lines.extend(param_strs)
                lines.append(")")
                lines.append("")

            elif isinstance(entry, GridEntry):
                var_idx += 1
                X, Y, Z = entry.get_data_arrays()
                if Z is None:
                    continue
                var_name = f"grid_{var_idx}"
                grid_vars.append(var_name)

                xyz = list(zip(
                    X.tolist() if hasattr(X, 'tolist') else list(X),
                    Y.tolist() if hasattr(Y, 'tolist') else list(Y),
                    Z.tolist() if hasattr(Z, 'tolist') else list(Z),
                ))
                lines.append(f"XYZ_{var_idx} = {repr(xyz)}")

                params = {}
                params['XYZ_triples'] = f"XYZ_{var_idx}"
                params['interpolation_type'] = repr(entry.inp_interp_type.currentText())
                params['show_contour'] = repr(entry.chk_contour.isChecked())
                try:
                    dens = int(entry.inp_density.text())
                    params['interpolation_density'] = repr(dens)
                except (ValueError, TypeError):
                    pass
                params['show_colorbar'] = repr(entry.chk_colorbar.isChecked())

                param_strs = []
                for k, v in params.items():
                    param_strs.append(f"    {k}={v},")
                lines.append(f"{var_name} = Grid(")
                lines.extend(param_strs)
                lines.append(")")
                lines.append("")

        if not curve_vars and not grid_vars:
            QMessageBox.information(self, "Copy Python Code", "No enabled entries with data to generate code for.")
            return

        # Plot call
        all_objs = curve_vars + grid_vars
        objs_str = ", ".join(all_objs)
        if len(all_objs) == 1:
            objs_str = objs_str  # single object

        plot_params = {}
        plot_params['figure_title'] = repr(self.ui_title.text())
        plot_params['x_axis_label'] = repr(self.ui_xlabel.text())
        plot_params['y_axis_label'] = repr(self.ui_ylabel.text())

        xlim = self._tuple_or_none(self.ui_xlim.text())
        if xlim:
            plot_params['x_lim'] = repr(xlim)
        ylim = self._tuple_or_none(self.ui_ylim.text())
        if ylim:
            plot_params['y_lim'] = repr(ylim)

        if self.ui_xlog.isChecked():
            plot_params['x_log'] = 'True'
        if self.ui_ylog.isChecked():
            plot_params['y_log'] = 'True'
        if self.ui_grid.isChecked():
            plot_params['show_grid'] = 'True'
        if not self.ui_legend.isChecked():
            plot_params['plot_legend'] = 'False'

        try:
            width_px = float(self.ui_fig_w.text())
            height_px = float(self.ui_fig_h.text())
            plot_params['fig_size_pixel'] = f"({width_px}, {height_px})"
        except (ValueError, TypeError):
            pass

        try:
            font_size = int(float(self.ui_font_size.text()))
            if font_size != 8:
                plot_params['font_size'] = repr(font_size)
        except (ValueError, TypeError):
            pass

        if self.ui_keep_front.isChecked():
            plot_params['keep_front'] = 'True'

        plot_param_strs = []
        for k, v in plot_params.items():
            plot_param_strs.append(f"    {k}={v},")

        lines.append(f"plot = Plot(")
        lines.append(f"    [{objs_str}],")
        lines.extend(plot_param_strs)
        lines.append(")")
        lines.append("plot.pause()")

        code = "\n".join(lines)

        clipboard = QApplication.clipboard()
        clipboard.setText(code)
        QMessageBox.information(self, "Copy Python Code", "Python code copied to clipboard.")

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
            'auto_color': self.btn_auto_color.isChecked(),
        }

    def _apply_settings_dict(self, s):
        self._suppress_update = True
        self.ui_title.setText(s.get('title', 'My Plot'))
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
        self.btn_auto_color.setChecked(s.get('auto_color', False))
        self._suppress_update = False

    def _auto_save_settings(self):
        if not getattr(self, '_init_done', False):
            return
        try:
            ensure_profile_dir()
            with open(LAST_SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._get_settings_dict(), f,
                          indent=2, ensure_ascii=False)
        except Exception as e:
            import traceback
            print(f"Error auto-saving settings: {e}")
            traceback.print_exc()

    def _load_last_settings(self):
        try:
            if os.path.exists(LAST_SETTINGS_FILE):
                with open(LAST_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    self._apply_settings_dict(json.load(f))
        except Exception as e:
            import traceback
            print(f"Error loading last settings: {e}")
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
        name, ok = QInputDialog.getText(
            self, "Save Preset", "Enter preset name:")
        if not ok or not name.strip():
            return
        fp = os.path.join(PROFILE_DIR, f"preset_{name.strip()}.json")
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
    def save_session(self):
        # Capture current zoom state from the live plot
        if self.plot_window is not None and hasattr(self.plot_window, '_ax'):
            try:
                if self.plot_window.isVisible():
                    xl = self.plot_window._ax.get_xlim()
                    yl = self.plot_window._ax.get_ylim()
                    self.ui_xlim.setText(f"{xl[0]}, {xl[1]}")
                    self.ui_ylim.setText(f"{yl[0]}, {yl[1]}")
            except Exception:
                pass

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "", "Plot Session (*.pkl)")
        if not path:
            return

        data = {'settings': self._get_settings_dict(), 'entries': []}
        for entry in self.entries:
            es = {
                'type': 'curve' if isinstance(entry, CurveEntry) else 'grid',
                'enabled': entry._is_enabled,
                'text_paste': entry.text_paste.toPlainText(),
                'loaded_data': entry.loaded_data,
                'file_path': entry._file_path,
            }
            if isinstance(entry, CurveEntry):
                es['params'] = {
                    'label': entry.inp_label.text(),
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
                    'scale': entry.inp_scale.text(),
                    'normalize': entry.inp_normalize.text(),
                    'legend_color': entry.inp_legend_color.currentText(),
                    'legend_format': entry.inp_legend_format.currentText(),
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

        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save: {e}")

    def load_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "", "Plot Session (*.pkl)")
        if not path:
            return
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

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
                if ed.get('text_paste'):
                    w._is_normalizing = True
                    w.text_paste.blockSignals(True)
                    w.text_paste.setPlainText(ed['text_paste'])
                    w.text_paste.blockSignals(False)
                    w._is_normalizing = False
                if ed.get('loaded_data') is not None:
                    w.loaded_data = ed['loaded_data']
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
                    w.inp_normalize.setText(p.get('normalize', ''))
                    legend_color_t = p.get('legend_color', '')
                    w.inp_legend_color.setCurrentText(
                        COLOR_DISPLAY_REV.get(legend_color_t, legend_color_t))
                    legend_fmt_t = p.get('legend_format', '')
                    w.inp_legend_format.setCurrentText(
                        LINE_FORMAT_DISPLAY_REV.get(legend_fmt_t, legend_fmt_t))
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

            self.request_update()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load: {e}")
            import traceback
            traceback.print_exc()


# ═════════════════════════ MAIN ══════════════════════════
if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QtGui.QFont("Arial", 10)
    app.setFont(font)
    window = PlotController()
    sys.exit(app.exec())
