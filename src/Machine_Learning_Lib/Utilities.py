# -*- coding: utf-8 -*-

import sys
import pathlib
Python_Lib_path = str(pathlib.Path(__file__).parent.resolve())
if Python_Lib_path not in sys.path:
    sys.path.insert(0, Python_Lib_path)

import math
from PyQt6 import QtWidgets, QtCore, QtGui

from My_Lib import *
from My_Lib_PyQt6 import *
from My_Lib_Plot import *


# Module-level HEADLESS flag — set to False by show_Train_NN_Network_progress
# before any polling begins.  Default: False (assume GUI available).
HEADLESS = False

MAX_SCREEN_WIDTH_RATIO = 0.8
MIN_LINE_EDIT_WIDTH = 60
# FIXED_LABEL_WIDTH = 50
VALUE_COLUMN_WIDTH = MIN_LINE_EDIT_WIDTH * 2 + 10 # Allow space for 2 line edits + spacing
GROUP_SPACING = 10

ESTIMATED_ITEM_WIDTH = 30 + VALUE_COLUMN_WIDTH + GROUP_SPACING 
DEFAULT_PANEL_HEIGHT = 150
DEFAULT_UNIT_HEIGHT = 400

class _Epoch_LineEdit(QtWidgets.QLineEdit):
    """QLineEdit subclass that emits wheel_delta_signal on mouse wheel events."""
    wheel_delta_signal = QtCore.pyqtSignal(int)   # +1 = scroll up, -1 = scroll down

    def wheelEvent(self, event: QtGui.QWheelEvent):
        delta = event.angleDelta().y()
        if delta > 0:
            self.wheel_delta_signal.emit(1)
        elif delta < 0:
            self.wheel_delta_signal.emit(-1)
        event.accept()


class Training_Control_Panel(QtWidgets.QWidget, Qt_Widget_Common_Functions):
    stop_signal = QtCore.pyqtSignal()
    pause_monitor_signal = QtCore.pyqtSignal(bool)    # True = paused
    epoch_changed_signal = QtCore.pyqtSignal(int)     # target epoch requested

    def __init__(self, parent=None):
        super().__init__(parent)
        self._app = Global_QApplication.get_app()
        self.setWindowTitle("Training Control Panel")
        self.setStyleSheet("font-family: Arial; font-size: 10pt;")
        
        self.comrades = []
        self._param_widgets = {} # Key: base_name, Value: (ValueContainer, {suffix: QLineEdit}, QLabel)
        self._epoch_line_edit = None  # Will be set when Epoch widget is created
        
        # Sorted list of available epoch numbers (set by Monitor)
        self._available_epochs = []   # sorted list of ints
        self._max_epoch = 0
        self._current_browse_epoch = None  # None means "live / latest"
        self._suppress_epoch_signals = False  # guard against signal loops
        
        # Main Wrapper Layout (Vertical)
        self.outer_layout = QtWidgets.QVBoxLayout(self)
        self.outer_layout.setContentsMargins(5, 5, 5, 5)
        self.outer_layout.setSpacing(10)

        # Top: Save Path Stem
        self.save_path_stem = ""
        self.txt_save_path = QtWidgets.QLineEdit(self.save_path_stem)
        self.txt_save_path.setFont(QtGui.QFont("Arial", 10))
        self.txt_save_path.setReadOnly(True)
        self.outer_layout.addWidget(self.txt_save_path)

        # --- Epoch Slider Row ---
        self.slider_widget = QtWidgets.QWidget()
        slider_layout = QtWidgets.QHBoxLayout(self.slider_widget)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(5)

        slider_label = QtWidgets.QLabel("Browse Epoch:")
        slider_label.setFixedHeight(24)
        font_bold = slider_label.font()
        font_bold.setBold(True)
        slider_label.setFont(font_bold)
        slider_layout.addWidget(slider_label)

        self.epoch_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.epoch_slider.setMinimum(0)
        self.epoch_slider.setMaximum(0)
        self.epoch_slider.setFixedHeight(24)
        self.epoch_slider.valueChanged.connect(self._on_slider_value_changed)
        slider_layout.addWidget(self.epoch_slider, stretch=1)

        self.outer_layout.addWidget(self.slider_widget)
        self.slider_widget.setVisible(False)  # hidden until epochs are known

        # Grid Container
        self.grid_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QGridLayout(self.grid_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(10)
        self.outer_layout.addWidget(self.grid_widget)
        
        # Helper storage for buttons so we can re-add them to grid
        self.btn_raise = QtWidgets.QPushButton("Raise All")
        self.btn_raise.setFixedHeight(30)
        self.btn_raise.clicked.connect(self.raise_all_windows)
        
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_pause.setFixedHeight(30)
        self.btn_pause.setCheckable(True)
        self.btn_pause.clicked.connect(self.on_pause_clicked)

        self.btn_pause_monitor = QtWidgets.QPushButton("Pause Monitor")
        self.btn_pause_monitor.setFixedHeight(30)
        self.btn_pause_monitor.setCheckable(True)
        self.btn_pause_monitor.clicked.connect(self._on_pause_monitor_clicked)
        
        self.btn_stop = QtWidgets.QPushButton("Stop Training")
        self.btn_stop.setFixedHeight(30)
        self.btn_stop.clicked.connect(self.on_stop_clicked)
        
        self.has_adjusted_layout = False
        
    def on_stop_clicked(self):
        self.stop_signal.emit()

    def _on_pause_monitor_clicked(self):
        checked = self.btn_pause_monitor.isChecked()
        self.pause_monitor_signal.emit(checked)

    def on_pause_clicked(self):
        checked = self.btn_pause.isChecked()
        for window in self.comrades:
            if hasattr(window, 'pause_button'):
                window.pause_button.setChecked(checked)

    # ---------- epoch browsing helpers ----------

    def set_available_epochs(self, epochs: list, max_epoch: int = 0):
        """Called by Monitor to update the slider range and available epochs list."""
        self._available_epochs = sorted(set(epochs))
        self._max_epoch = max_epoch if max_epoch else (self._available_epochs[-1] if self._available_epochs else 0)

        self._suppress_epoch_signals = True
        self.epoch_slider.setMinimum(0)
        self.epoch_slider.setMaximum(self._max_epoch)
        if self._current_browse_epoch is None:
            self.epoch_slider.setValue(self._max_epoch)
        self._suppress_epoch_signals = False

        self.slider_widget.setVisible(self._max_epoch > 0)

    def _on_slider_value_changed(self, value: int):
        if self._suppress_epoch_signals:
            return
        # Snap to nearest available epoch
        target = self._snap_to_nearest_epoch(value)
        self._set_browse_epoch(target, from_slider=True)

    def _on_epoch_wheel(self, direction: int):
        """Called when user scrolls wheel on the Epoch line-edit. direction: +1 up, -1 down."""
        if not self._available_epochs:
            return
        current = self._current_browse_epoch
        if current is None:
            current = self._max_epoch

        if direction > 0:
            # scroll up → next higher epoch
            candidates = [e for e in self._available_epochs if e > current]
            target = candidates[0] if candidates else current
        else:
            # scroll down → next lower epoch
            candidates = [e for e in self._available_epochs if e < current]
            target = candidates[-1] if candidates else current

        self._set_browse_epoch(target)

    def _on_epoch_edited(self):
        """Called when user finishes editing the Epoch line-edit."""
        edit_widget = self._epoch_line_edit
        if edit_widget is None:
            return
        text = edit_widget.text().strip()
        try:
            value = int(text)
        except ValueError:
            return
        target = self._snap_to_nearest_epoch(value)
        self._set_browse_epoch(target)

    def _snap_to_nearest_epoch(self, value: int) -> int:
        """Snap *value* to the closest epoch in _available_epochs."""
        if not self._available_epochs:
            return value
        import bisect
        idx = bisect.bisect_left(self._available_epochs, value)
        candidates = []
        if idx < len(self._available_epochs):
            candidates.append(self._available_epochs[idx])
        if idx > 0:
            candidates.append(self._available_epochs[idx - 1])
        return min(candidates, key=lambda e: abs(e - value))

    def _set_browse_epoch(self, epoch: int, from_slider: bool = False):
        """Unified setter: auto-pauses monitor and emits epoch_changed_signal."""
        self._current_browse_epoch = epoch
        # Auto-pause monitor
        if not self.btn_pause_monitor.isChecked():
            self.btn_pause_monitor.setChecked(True)
            self.pause_monitor_signal.emit(True)
        # Update slider position (suppress re-entrant signal)
        if not from_slider:
            self._suppress_epoch_signals = True
            self.epoch_slider.setValue(epoch)
            self._suppress_epoch_signals = False
        # Emit epoch change
        self.epoch_changed_signal.emit(epoch)

    def resume_live(self):
        """Resume live tracking (un-pause and go back to latest epoch)."""
        self._current_browse_epoch = None
        self.btn_pause_monitor.setChecked(False)
        self.pause_monitor_signal.emit(False)
        # Slider to max
        self._suppress_epoch_signals = True
        self.epoch_slider.setValue(self._max_epoch)
        self._suppress_epoch_signals = False

    def raise_all_windows(self):
        self.activateWindow()
        self.raise_()
        for window in self.comrades:
            window.activateWindow()
            window.raise_()

    def update_info(self, info_dict: dict):
        if not info_dict:
            return

        # 1. Identify Items (Group Train/Test)
        current_keys = list(info_dict.keys())
        processed_keys = set()
        display_items = [] # List of (base_name, [list of actual keys])"

        # 1.1 Handle Special Pairs explicitly
        special_pairs = [("Epoch", "Max_Epoch"), ("Time", "Speed")]
        for p1, p2 in special_pairs:
            if p1 in current_keys and p2 in current_keys:
                display_items.append((p1, [p1, p2]))
                processed_keys.add(p1)
                processed_keys.add(p2)

        for key in current_keys:
            if key in processed_keys:
                continue
            
            if key.endswith("_Train"):
                base_name = key[:-6]
                test_key = base_name + "_Test"
                if test_key in current_keys:
                    display_items.append((base_name, [key, test_key]))
                    processed_keys.add(key)
                    processed_keys.add(test_key)
                else:
                    display_items.append((key, [key]))
                    processed_keys.add(key)
            elif key.endswith("_Test"):
                pass 
            else:
                display_items.append((key, [key]))
                processed_keys.add(key)

        # 2. Update/Create Widgets
        for base_name, keys in display_items:
            if base_name not in self._param_widgets:
                # Label
                label_text = base_name.replace("_", " ") if len(keys) > 1 else base_name
                
                # Special case label naming for pairs
                if base_name == "Epoch": label_text = "Epoch / Max"
                if base_name == "Time": label_text = "Time / Speed"

                lbl = QtWidgets.QLabel(label_text, self)
                lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                lbl.setFixedHeight(30)
                # make it bold
                font = lbl.font()
                font.setBold(True)
                lbl.setFont(font)
                # lbl.setFixedWidth(FIXED_LABEL_WIDTH) # Allow variable width
                
                # Value Container (for one or two LineEdits)
                val_container = QtWidgets.QWidget(self)
                h_layout = QtWidgets.QHBoxLayout(val_container)
                h_layout.setContentsMargins(0, 0, 0, 0)
                h_layout.setSpacing(5)
                
                line_edits = {}
                for k in keys:
                    # Use _Epoch_LineEdit for the "Epoch" field so it
                    # supports wheel-scrolling and user editing.
                    if k == "Epoch":
                        val = _Epoch_LineEdit()
                        val.setReadOnly(False)
                        val.wheel_delta_signal.connect(self._on_epoch_wheel)
                        val.editingFinished.connect(self._on_epoch_edited)
                        self._epoch_line_edit = val
                    else:
                        val = QtWidgets.QLineEdit()
                        val.setReadOnly(True)
                    val.setFixedHeight(30)
                    val.setMinimumWidth(MIN_LINE_EDIT_WIDTH) # Enforce minimum width
                    # set align middle
                    val.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    line_edits[k] = val
                    h_layout.addWidget(val)
                
                # If pair, ensure equal width
                if len(keys) > 1:
                     for k in keys:
                         h_layout.setStretchFactor(line_edits[k], 1)
                
                val_container.setLayout(h_layout)
                val_container.setMinimumWidth(VALUE_COLUMN_WIDTH) # Enforce group width
                
                self._param_widgets[base_name] = (val_container, line_edits, lbl)
            
            # Update values
            val_container, line_edits, lbl = self._param_widgets[base_name]
            for k in keys:
                if k in line_edits:
                    line_edits[k].setText(str(info_dict[k]))
            
            val_container.show()
            lbl.show()

        # Hide missing
        active_base_names = [item[0] for item in display_items]
        for base in self._param_widgets:
            if base not in active_base_names:
                self._param_widgets[base][0].hide()
                self._param_widgets[base][2].hide()

        # 3. Calculate Layout
        # Estimate items
        
        screen = self._app.primaryScreen()
        if self.window().windowHandle():
             screen = self.window().windowHandle().screen()
        
        available_width = screen.availableGeometry().width() * MAX_SCREEN_WIDTH_RATIO
        buttons_width = 120 # Approx
        
        available_for_params = available_width - buttons_width
        num_items = len(display_items)
        
        # Calculate how many "data columns" we can have
        # Each data column is (Label + ValueParam + GroupSpacing)
        
        max_screen_cols = math.floor(available_for_params / ESTIMATED_ITEM_WIDTH)
        if max_screen_cols < 1: max_screen_cols = 1
        
        # Prefer 3 rows layout distribution (e.g. 10 items -> 4 cols: 4, 4, 2)
        preferred_cols = math.ceil(num_items / 3)
        if preferred_cols < 1: preferred_cols = 1
        
        # Use preferred cols if it fits within screen width
        num_data_cols = min(preferred_cols, max_screen_cols)
        
        # 4. Rearrange Grid
        # Clear layout items (remove from layout but keep widgets)
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            if item.widget() in [self.btn_raise, self.btn_pause, self.btn_pause_monitor, self.btn_stop]:
                 continue
            # Don't delete widgets
            
        # Add Widgets
        for i, (base_name, _) in enumerate(display_items):
            val_container, _, lbl = self._param_widgets[base_name]
            
            # Row-major index (Horizontal first, then Vertical)
            col_group = i % num_data_cols
            row_in_group = i // num_data_cols
            
            grid_col_label = col_group * 3
            
            self.main_layout.addWidget(lbl, row_in_group, grid_col_label)
            self.main_layout.addWidget(val_container, row_in_group, grid_col_label + 1)
            self.main_layout.setColumnMinimumWidth(grid_col_label + 2, GROUP_SPACING)

        # Add buttons at the end
        btn_col = num_data_cols * 3
        
        # Add buttons to grid directly, vertically aligned with rows
        self.main_layout.addWidget(self.btn_raise, 0, btn_col)
        self.main_layout.addWidget(self.btn_pause, 1, btn_col)
        self.main_layout.addWidget(self.btn_pause_monitor, 2, btn_col)
        self.main_layout.addWidget(self.btn_stop, 3, btn_col)
        
        self.adjustSize()
        # self.adjust_window_layout() # Reposition after size adjustment (Moved to update_plots in Train_NN)

    def adjust_window_layout(self):
        if self.has_adjusted_layout:
             return
             
        # Ensure size is calculated
        self.adjustSize() 
        
        # Get own dimensions (including frame)
        frame_geo = self.frameGeometry()
        panel_height = frame_geo.height()
        
        # Shift comrades down by half of this panel's height
        offset_y = panel_height // 2
        for w in self.comrades:
            if hasattr(w, 'additional_offset_px'):
                w.additional_offset_px = (0, offset_y)

        # Find the highest (smallest Y) top edge among visible comrades
        min_comrade_y = float('inf')
        has_visible_comrade = False
        for w in self.comrades:
            if w.isVisible():
                has_visible_comrade = True
                y = w.frameGeometry().y()
                if y < min_comrade_y:
                    min_comrade_y = y
        
        # Calculate Position
        screen = self._app.primaryScreen()
        if self.window().windowHandle():
             screen = self.window().windowHandle().screen()
        screen_geo = screen.availableGeometry()
        
        # Center Horizontally
        target_x = screen_geo.center().x() - frame_geo.width() // 2
        
        if has_visible_comrade and min_comrade_y != float('inf'):
            # Place panel's bottom edge at the highest comrade's top edge
            target_y = min_comrade_y - panel_height
        else:
            # Fallback: Center Vertically
            target_y = screen_geo.center().y() - panel_height // 2
        
        # Apply manual offset
        target_y -= 30  # Manual adjustment (panel tends to position slightly lower than calculated)
        
        # Boundary check: Ensure panel stays within screen bounds
        target_x = max(screen_geo.left(), min(target_x, screen_geo.right() - frame_geo.width()))
        target_y = max(screen_geo.top(), min(target_y, screen_geo.bottom() - panel_height))
        
        print(f"[Control Panel] Positioning: x={target_x}, y={target_y}, screen={screen_geo}, panel_size=({frame_geo.width()}, {panel_height})")
            
        self.move(int(target_x), int(target_y))
        self.has_adjusted_layout = True

            
    def closeEvent(self, event):
        # Handle close event if necessary, or let parent handle it
        super().closeEvent(event)


"""
Remote monitoring for Train_NN_Network progress.

This module watches a JSON_Progress folder for JSON files produced by
Train_NN_Network and displays them in live Plot windows and a
Training_Control_Panel, all in a standalone process.

File naming convention produced by Train_NN_Network::

    <tag>_[E<epoch>].json      e.g. 0_Losses_[E00042].json

Usage
-----
From the command line::

    python My_Lib_MachineLearning_Monitor.py  <path_to_JSON_Progress_folder>

Or from Python::

    from My_Lib_MachineLearning_Monitor import show_Train_NN_Network_progress
    show_Train_NN_Network_progress(r"C:\\...\\Checkpoints\\run_20250101\\JSON_Progress")
"""

# Pattern:  0_Losses_[E00042].json  →  tag="0_Losses", epoch=42
_JSON_FILENAME_RE = re.compile(r"^(.+)_\[E(\d{5})\]\.json$")

def _parse_json_filename(basename: str):
    """Return (epoch: int, tag: str) or None if the name doesn't match."""
    m = _JSON_FILENAME_RE.match(basename)
    if m:
        return int(m.group(2)), m.group(1)
    return None


def _latest_jsons_in_folder(folder: str):
    """
    Scan *folder* and return a dict  {tag: filepath}  keeping only the
    file with the **highest epoch number** for each tag.
    Also includes non-epoch-tagged JSON files (e.g. ``0_Losses.json``).
    Also return the maximum epoch seen overall.
    """
    best = {}       # tag → (epoch, filepath)
    max_epoch = 0
    for name in os.listdir(folder):
        if not name.endswith('.json'):
            continue
        parsed = _parse_json_filename(name)
        if parsed is not None:
            epoch, tag = parsed
            if epoch > max_epoch:
                max_epoch = epoch
            if tag not in best or epoch > best[tag][0]:
                best[tag] = (epoch, os.path.join(folder, name))
        else:
            # Non-epoch-tagged JSON file (e.g. 0_Losses.json)
            tag = name[:-5]  # strip .json
            if tag not in best:
                best[tag] = (0, os.path.join(folder, name))
    return {tag: path for tag, (_, path) in best.items()}, max_epoch


def _all_epoch_numbers_in_folder(folder: str):
    """
    Scan *folder* and return a **sorted list** of all unique epoch numbers
    found across all epoch-tagged JSON files.
    """
    epochs = set()
    for name in os.listdir(folder):
        if not name.endswith('.json'):
            continue
        parsed = _parse_json_filename(name)
        if parsed is not None:
            epochs.add(parsed[0])
    return sorted(epochs)


def _jsons_for_target_epoch(folder: str, target_epoch: int):
    """
    For each *tag* in *folder*, find the JSON file whose epoch is the
    **closest ≤ target_epoch**.  Returns ``{tag: filepath}``.

    Tags that only have epochs greater than *target_epoch* are omitted.
    Non-epoch-tagged files (epoch=0) are always included.
    """
    # Collect ALL (tag, epoch, filepath) triples
    all_files = {}  # tag → list of (epoch, filepath)
    for name in os.listdir(folder):
        if not name.endswith('.json'):
            continue
        parsed = _parse_json_filename(name)
        if parsed is not None:
            epoch, tag = parsed
            all_files.setdefault(tag, []).append((epoch, os.path.join(folder, name)))
        else:
            tag = name[:-5]
            all_files.setdefault(tag, []).append((0, os.path.join(folder, name)))

    result = {}
    for tag, entries in all_files.items():
        # Filter to epochs ≤ target_epoch
        candidates = [(e, p) for e, p in entries if e <= target_epoch]
        if candidates:
            # Pick the one with the highest epoch ≤ target_epoch
            best = max(candidates, key=lambda x: x[0])
            result[tag] = best[1]
        # else: no data at or before target_epoch for this tag — skip
    return result


def _is_training_finished(folder: str) -> bool:
    """
    Check if the training has finished by looking for the termination file
    in the Checkpoints folder.
    """
    return os.path.isfile(os.path.join(folder, "0_Optimization_Finished_Successfully.txt"))


# ---------------------------------------------------------------------------
#   Training_Progress_Monitor
# ---------------------------------------------------------------------------

class Training_Progress_Monitor:
    """
    Polls Plots and Control_Panel_History folders and keeps live Plot
    windows + a control panel.

    Existing Plot windows are **updated in-place** rather than destroyed and
    recreated, so the user's window positions are preserved and flicker is
    avoided.

    Parameters
    ----------
    folder : str
        Absolute path to the Checkpoints folder (containing ``Plots`` and
        ``Control_Panel_History`` subfolders).
    poll_interval_ms : int
        How often (milliseconds) to scan the folders for new files.
    auto_close_on_finish : bool
        If True, stop polling (but keep windows open) once the training
        ``0_Optimization_Finished_Successfully.txt`` sentinel is detected.
    """

    def __init__(self, folder: str, poll_interval_ms: int = 2000, auto_close_on_finish: bool = True):
        self.folder = os.path.abspath(folder)
        self.plots_folder = os.path.join(self.folder, "Plots")
        self.control_panel_folder = os.path.join(self.folder, "Control_Panel_History")
        self.base_poll_interval_ms = poll_interval_ms
        self.auto_close_on_finish = auto_close_on_finish

        self._app = Global_QApplication.get_app()

        self._last_epoch = 0          # track highest epoch we've rendered
        self._windows = {}            # tag → Plot  (kept across polls)
        self._control_panel = None    # Training_Control_Panel (created lazily)
        self._layout_adjusted = False
        self._poll_count = 0
        self._is_paused = False       # True when monitor is paused by user
        
        # Load metadata to get expected window count
        self._expected_window_count = None
        self._expected_window_tags = None
        metadata_path = os.path.join(self.folder, "_metadata.json")
        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self._expected_window_count = metadata.get('window_count')
                self._expected_window_tags = metadata.get('window_tags')
                print(f"[Monitor] Loaded metadata: {self._expected_window_count} windows expected")
                if self._expected_window_tags:
                    print(f"[Monitor]   Tags: {self._expected_window_tags}")
            except Exception as e:
                print(f"[Monitor] Warning: Failed to load metadata: {e}")
        
        # Pre-calculate fixed positions for expected windows
        self._fixed_positions = {}
        if self._expected_window_tags and self._expected_window_count:
            n = self._expected_window_count
            if n <= max(WINDOW_POSITIONS.keys()):
                for i, tag in enumerate(self._expected_window_tags):
                    self._fixed_positions[tag] = WINDOW_POSITIONS[n][i]
                print(f"[Monitor] Pre-calculated positions for {n} windows")

        print(f"[Monitor] Training_Progress_Monitor initialized.")
        print(f"[Monitor]   plots_folder:         {self.plots_folder}")
        print(f"[Monitor]   control_panel_folder:  {self.control_panel_folder}")
        print(f"[Monitor]   base_poll_interval:   {self.base_poll_interval_ms}ms")
        print(f"[Monitor]   HEADLESS={HEADLESS}")

    # ----- polling logic --------------------------------------------------

    def _poll(self):
        self._poll_count += 1

        # Skip polling when paused (user is browsing epochs)
        if self._is_paused:
            return

        if not os.path.isdir(self.plots_folder):
            if self._poll_count <= 3:
                print(f"[Monitor] Poll #{self._poll_count}: Plots folder not found yet: {self.plots_folder}")
            return

        latest, max_epoch = _latest_jsons_in_folder(self.plots_folder)

        if self._poll_count <= 5 or self._poll_count % 20 == 0:
            print(f"[Monitor] Poll #{self._poll_count}: found {len(latest)} JSON tags, "
                  f"max_epoch={max_epoch}, last_epoch={self._last_epoch}, "
                  f"windows={list(self._windows.keys())}, "
                  f"control_panel={'yes' if self._control_panel else 'no'}")

        # Update available epochs for the control panel slider
        all_epochs = _all_epoch_numbers_in_folder(self.plots_folder)
        if self._control_panel is not None:
            self._control_panel.set_available_epochs(all_epochs, max_epoch)

        if max_epoch <= self._last_epoch and self._last_epoch > 0:
            # No new epoch yet — check training done
            if self.auto_close_on_finish and _is_training_finished(self.folder):
                self._timer.stop()
                print("[Monitor] Training finished. Polling stopped.")
            return

        self._last_epoch = max_epoch

        # --- Update control panel from Control_Panel_History ---
        if os.path.isdir(self.control_panel_folder):
            cp_files, cp_max_epoch = _latest_jsons_in_folder(self.control_panel_folder)
            for tag, filepath in cp_files.items():
                if tag == "Control_Panel":
                    self._update_control_panel(filepath)

        # --- Update / create windows ---
        # Use fixed positions from metadata, or fallback to dynamic calculation
        
        newly_created_tags = []
        
        # Process windows in expected order if available, else sorted order
        tags_to_process = self._expected_window_tags if self._expected_window_tags else sorted(latest.keys())
        
        for tag in tags_to_process:
            filepath = latest.get(tag)
            if not filepath or HEADLESS:
                continue
            
            existing_window = self._windows.get(tag)
            is_new = (existing_window is None)
            
            # Load JSON data
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    plot_data = json.load(f)
            except Exception as e:
                print(f"[Monitor] Error loading {filepath}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Set position ONLY for NEW windows using fixed positions
            if is_new:
                position = self._fixed_positions.get(tag)
                if position:
                    plot_data['shift_window'] = position
                    print(f"[Monitor] Setting fixed position {position} for new window '{tag}'")
                else:
                    # Fallback: calculate based on current window count
                    n = len(self._windows) + 1
                    if n <= max(WINDOW_POSITIONS.keys()):
                        idx = list(tags_to_process).index(tag)
                        if idx < len(WINDOW_POSITIONS[n]):
                            plot_data['shift_window'] = WINDOW_POSITIONS[n][idx]
                            print(f"[Monitor] Fallback position for '{tag}': {plot_data['shift_window']}")
            
            # Create or update Plot window
            try:
                plot_window = Plot_from_JSON(plot_data, target_window=existing_window)
            except Exception as e:
                print(f"[Monitor] Error updating Plot from {filepath}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Show and store new windows
            if is_new:
                newly_created_tags.append(tag)
                plot_window.show()
                print(f"[Monitor] Created new Plot window for tag='{tag}'")
            
            self._windows[tag] = plot_window
            
            # Update window title to show current epoch
            plot_window.setWindowTitle(f"{tag} [Epoch {max_epoch}]")
            
            # Save SVG
            try:
                svg_path = os.path.splitext(filepath)[0] + ".svg"
                plot_window.save_svg(svg_path, save_plot_history=False)
            except Exception as e:
                print(f"[Monitor] Warning: Failed to save SVG for tag='{tag}': {e}")
        
        # --- Comrade linking & control panel positioning ---
        # Always update comrades list when there are windows
        active_windows = [self._windows[t] for t in sorted(self._windows.keys())]
        if active_windows:
            all_widgets = list(active_windows)
            if self._control_panel is not None:
                all_widgets.insert(0, self._control_panel)
            
            for w in all_widgets:
                w.comrades = all_widgets
            
            # Position control panel only when layout changes
            if self._control_panel is not None and (newly_created_tags or not self._layout_adjusted):
                self._app.processEvents()  # ensure geometry is up-to-date
                self._control_panel.has_adjusted_layout = False
                self._control_panel.adjust_window_layout()
                self._layout_adjusted = True
                print(f"[Monitor] Layout adjusted: {len(all_widgets)} widgets "
                      f"(control_panel + {len(active_windows)} plots)")

        # Check stop
        if self.auto_close_on_finish and _is_training_finished(self.folder):
            self._timer.stop()
            print("[Monitor] Training finished. Polling stopped.")

    # ----- control panel --------------------------------------------------

    def _update_control_panel(self, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                info_dict = json.load(f)
        except Exception as e:
            print(f"[Monitor] Error loading control panel JSON: {e}")
            return

        if self._control_panel is None:
            print(f"[Monitor] Creating Training_Control_Panel from {os.path.basename(filepath)}")
            self._control_panel = Training_Control_Panel()
            self._control_panel.setWindowTitle("Training Control Panel (Monitor)")
            # Disable stop/pause — they have no effect when merely monitoring
            self._control_panel.btn_stop.setEnabled(False)
            self._control_panel.btn_pause.setEnabled(False)
            # Connect monitor-specific signals
            self._control_panel.pause_monitor_signal.connect(self._on_pause_monitor)
            self._control_panel.epoch_changed_signal.connect(self._on_epoch_changed)
            self._control_panel.show()
            print(f"[Monitor] Training_Control_Panel shown.")

        self._control_panel.update_info(info_dict)

    # ----- public API -----------------------------------------------------

    def start(self):
        """Start polling. Called automatically by show_Train_NN_Network_progress."""
        # Adaptive polling: fast initially (500ms), then slow down after 20 epochs
        initial_interval = 500  # 500ms for first ~20 epochs
        print(f"[Monitor] Starting adaptive polling (initial={initial_interval}ms, later={self.base_poll_interval_ms}ms)...")
        self._timer = QTimer()
        self._timer.timeout.connect(self._poll_with_adaptive_interval)
        self._timer.start(initial_interval)
        # Immediate first poll
        self._poll()
    
    def _poll_with_adaptive_interval(self):
        """Poll and adjust interval based on epoch progress."""
        self._poll()
        # After 20 epochs, slow down polling
        if self._last_epoch >= 20 and self._timer.interval() < self.base_poll_interval_ms:
            self._timer.setInterval(self.base_poll_interval_ms)
            print(f"[Monitor] Slowing down to {self.base_poll_interval_ms}ms after epoch {self._last_epoch}")

    def stop(self):
        """Stop polling (windows remain open)."""
        self._timer.stop()

    def exec(self):
        """Enter the Qt event loop (blocking). Returns when all windows are closed."""
        self._app.exec()


# ---------------------------------------------------------------------------
#   Public entry-point
# ---------------------------------------------------------------------------

def show_Train_NN_Network_progress(folder: str, poll_interval_ms: int = 1000):
    """
    Launch a live monitoring GUI that tracks training progress from JSON files.

    This function is designed to run in a **completely separate process**
    from the training script.  It creates its own QApplication, reads the
    ``Plots`` and ``Control_Panel_History`` subfolders written by
    ``Train_NN_Network``, and displays Plot windows + a Training Control
    Panel that update as new epochs are written.

    Parameters
    ----------
    folder : str
        Path to the Checkpoints run folder (containing Plots and
        Control_Panel_History subfolders).
    poll_interval_ms : int
        Polling interval in milliseconds (default 2 000 = 2 s).

    Example
    -------
    >>> show_Train_NN_Network_progress(r"D:\\Runs\\Checkpoints\\run_20250101_120000")
    
    Note
    ----
    When called from a separate process (e.g., via multiprocessing.Process),
    ensure your main script has proper `if __name__ == '__main__':` guards
    to prevent recursive process spawning on Windows.
    """
    print(f"[Monitor] show_Train_NN_Network_progress called for folder: {folder}")
    print(f"[Monitor] Process PID: {os.getpid()}")

    # Force HEADLESS to False for monitoring (we want to show windows)
    # This is necessary on Windows where Qt might not be initialized during module import
    import matplotlib
    global HEADLESS
    HEADLESS = False
    
    # Ensure matplotlib uses Qt backend for display
    try:
        import matplotlib.pyplot as plt
        if matplotlib.get_backend() != 'QtAgg':
            matplotlib.use('QtAgg', force=True)
            plt.switch_backend('QtAgg')
    except Exception as e:
        print(f"[Monitor] Warning: Failed to set QtAgg backend: {e}")
    
    # Initialize QApplication early to ensure proper Qt setup
    try:
        _ = Global_QApplication.get_app()
        print("[Monitor] QApplication initialized successfully")
    except Exception as e:
        print(f"[Monitor] Warning: Failed to initialize QApplication: {e}")
    
    folder = os.path.abspath(folder)

    # Verify expected folder structure
    plots_folder = os.path.join(folder, "Plots")
    control_panel_folder = os.path.join(folder, "Control_Panel_History")

    if not os.path.isdir(plots_folder):
        print(f"[Monitor] Plots folder not found: {plots_folder}")
        print("[Monitor] Waiting for training to create the Plots folder…")
        while not os.path.isdir(plots_folder):
            time.sleep(1)

    print(f"[Monitor] Watching: {folder}")
    monitor = Training_Progress_Monitor(folder, poll_interval_ms=poll_interval_ms)
    monitor.start()
    monitor.exec()



if __name__ == "__main__":

    if len(sys.argv) < 2:
        target_folder = input("Folder to monitor:").strip('"')
    else:
        target_folder = sys.argv[1]
    
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    show_Train_NN_Network_progress(target_folder, poll_interval_ms=interval)

    # def test_control_panel():
    #     import sys
        
    #     info_dict = {
    #         'Epoch': '1',
    #         'Max_Epoch': '200',
    #         'Loss_Train': '0.02669',
    #         'Loss_Test': '0.04541',
    #         'RMSE_Train': '0.2725',
    #         'RMSE_Test': '0.3111',
    #         'R2_Train': '-0.5869',
    #         'R2_Test': '-4.687',
    #         'T1_Train': '0.07428',
    #         'T1_Test': '0.09676',
    #         'W1_Train': '0.1797',
    #         'W1_Test': '0.1787',
    #         'T2_Train': '0.01627',
    #         'T2_Test': '0.03424',
    #         'W2_Train': '0.8203',
    #         'W2_Test': '0.8213',
    #         'LR': '0.000109',
    #         'Time': '0:01',
    #         'Speed': '0:01/1 Epoch'
    #     }

    #     # Ensure app exists
    #     app = Global_QApplication.get_app()
        
    #     panel = Training_Control_Panel()
    #     panel.update_info(info_dict)
    #     # panel.adjust_window_layout() # Usually needs comrades, run safely without
    #     panel.show()
        
    #     print("Panel shown with test data.")
    #     sys.exit(app.exec())

    # test_control_panel()
