# -*- coding: utf-8 -*-
import math
from PyQt6 import QtWidgets, QtCore, QtGui
from My_Lib import *
from My_Lib_PyQt6 import *
from My_Lib_Plot import Global_QApplication

MAX_SCREEN_WIDTH_RATIO = 0.8
MIN_LINE_EDIT_WIDTH = 60
# FIXED_LABEL_WIDTH = 50
VALUE_COLUMN_WIDTH = MIN_LINE_EDIT_WIDTH * 2 + 10 # Allow space for 2 line edits + spacing
GROUP_SPACING = 10

ESTIMATED_ITEM_WIDTH = 30 + VALUE_COLUMN_WIDTH + GROUP_SPACING 
DEFAULT_PANEL_HEIGHT = 150
DEFAULT_UNIT_HEIGHT = 400

class Training_Control_Panel(QtWidgets.QWidget, Qt_Widget_Common_Functions):
    stop_signal = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._app = Global_QApplication.get_app()
        self.setWindowTitle("Training Control Panel")
        self.setStyleSheet("font-family: Arial; font-size: 10pt;")
        
        self.comrades = []
        self._param_widgets = {} # Key: base_name, Value: (ValueContainer, {suffix: QLineEdit}, QLabel)
        
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
        
        self.btn_stop = QtWidgets.QPushButton("Stop Training")
        self.btn_stop.setFixedHeight(30)
        self.btn_stop.clicked.connect(self.on_stop_clicked)
        
        self.has_adjusted_layout = False
        
    def on_stop_clicked(self):
        self.stop_signal.emit()

    def on_pause_clicked(self):
        checked = self.btn_pause.isChecked()
        for window in self.comrades:
            if hasattr(window, 'pause_button'):
                window.pause_button.setChecked(checked)
        
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
                    val = QtWidgets.QLineEdit()
                    val.setFixedHeight(30)
                    val.setReadOnly(True)
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
            if item.widget() in [self.btn_raise, self.btn_pause, self.btn_stop]:
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
        self.main_layout.addWidget(self.btn_stop, 2, btn_col)
        
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
            
        self.move(int(target_x), int(target_y)-30) # 不知道为什么算出来总是低一点，不管了直接手动改30px
        self.has_adjusted_layout = True

            
    def closeEvent(self, event):
        # Handle close event if necessary, or let parent handle it
        super().closeEvent(event)

if __name__ == "__main__":
    def test_control_panel():
        import sys
        
        info_dict = {
            'Epoch': '1',
            'Max_Epoch': '200',
            'Loss_Train': '0.02669',
            'Loss_Test': '0.04541',
            'RMSE_Train': '0.2725',
            'RMSE_Test': '0.3111',
            'R2_Train': '-0.5869',
            'R2_Test': '-4.687',
            'T1_Train': '0.07428',
            'T1_Test': '0.09676',
            'W1_Train': '0.1797',
            'W1_Test': '0.1787',
            'T2_Train': '0.01627',
            'T2_Test': '0.03424',
            'W2_Train': '0.8203',
            'W2_Test': '0.8213',
            'LR': '0.000109',
            'Time': '0:01',
            'Speed': '0:01/1 Epoch'
        }

        # Ensure app exists
        app = Global_QApplication.get_app()
        
        panel = Training_Control_Panel()
        panel.update_info(info_dict)
        # panel.adjust_window_layout() # Usually needs comrades, run safely without
        panel.show()
        
        print("Panel shown with test data.")
        sys.exit(app.exec())

    test_control_panel()
