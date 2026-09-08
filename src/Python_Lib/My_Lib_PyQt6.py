# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import sys
import pathlib
import platform
import faulthandler
# faulthandler dumps low-level crash tracebacks to sys.stderr and needs a real
# fileno() for it. If stderr has been replaced by something without one (a tee /
# logger / pythonw's None), enabling it must not take down GUI startup — it's only
# a debugging aid. Fall back to a no-op when it can't attach.
try:
    faulthandler.enable()
except (AttributeError, ValueError, OSError):
    pass

try:
    from PyQt6 import QtGui, QtCore, QtWidgets, uic
    from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QMessageBox, \
        QFileDialog, QGraphicsPixmapItem, QGraphicsScene, QInputDialog, QDialog, \
        QListView, QAbstractItemView, QTreeView, QWidget, QLayout, QVBoxLayout, QHBoxLayout, QGridLayout, \
        QTextEdit, QSpinBox, QAbstractSpinBox, \
        QPushButton, QToolButton, QRadioButton, QCheckBox, QLineEdit, QDoubleSpinBox, \
        QTableWidgetItem, QFrame, QSpacerItem, QSizePolicy, QTableWidget
    from PyQt6.QtGui import QPixmap, QColor, QPainter, QPen, QFont, QDropEvent, QIcon, QTextCursor, QScreen, QKeyEvent, QTextCharFormat, QSyntaxHighlighter
    from PyQt6.QtCore import QPoint, QTimer, QMimeData, QSize, pyqtSignal, QProcess, QObject
    from PyQt6.QtCore import Qt as QtCore_Qt
    from PyQt6.QtCore import QEvent
    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False

    class _QtStubMeta(type):
        """Metaclass that returns _QtStubClass for any undefined class attribute."""
        def __getattr__(cls, name):
            return _QtStubClass

    class _QtStubClass(metaclass=_QtStubMeta):
        """Placeholder for all Qt classes — allows subclassing and attribute
        chaining at class-definition time.  Instantiation is a silent no-op
        so that patterns like QCoreApplication.instance() don't crash."""
        def __init__(self, *args, **kwargs):
            pass
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)

    class _QtStubModule:
        """Module-like stub — attribute access returns _QtStubClass."""
        def __getattr__(self, name):
            return _QtStubClass

    class _pyqtSignalStub:
        """Descriptor stub so that `closing = pyqtSignal()` in a class body
        succeeds at definition time."""
        def __init__(self, *args, **kwargs):
            pass
        def emit(self, *args, **kwargs):
            pass
        def connect(self, *args, **kwargs):
            pass
        def disconnect(self, *args, **kwargs):
            pass

    QtGui = _QtStubModule()
    QtCore = _QtStubModule()
    QtWidgets = _QtStubModule()
    uic = _QtStubModule()

    QApplication = QMainWindow = QLabel = QMessageBox = _QtStubClass
    QFileDialog = QGraphicsPixmapItem = QGraphicsScene = QInputDialog = QDialog = _QtStubClass
    QListView = QAbstractItemView = QTreeView = QWidget = QLayout = _QtStubClass
    QVBoxLayout = QHBoxLayout = QGridLayout = QTextEdit = QSpinBox = QAbstractSpinBox = _QtStubClass
    QPushButton = QToolButton = QRadioButton = QCheckBox = QLineEdit = QDoubleSpinBox = _QtStubClass
    QTableWidgetItem = QFrame = QSpacerItem = QSizePolicy = QTableWidget = _QtStubClass
    QPixmap = QColor = QPainter = QPen = QFont = QDropEvent = QIcon = _QtStubClass
    QTextCursor = QScreen = QKeyEvent = QTextCharFormat = QSyntaxHighlighter = _QtStubClass
    QPoint = QTimer = QMimeData = QSize = QProcess = QObject = _QtStubClass
    QEvent = _QtStubClass
    QtCore_Qt = _QtStubClass
    pyqtSignal = _pyqtSignalStub
    QFontDatabase = QCoreApplication = _QtStubClass

from Python_Lib.My_Lib_Stock import *
from Python_Lib.My_Lib_System import is_headless

if platform.system() == 'Windows':
    os.environ['QT_QPA_FONTDIR'] = 'C:/Windows/Fonts'
    os.environ.setdefault('QT_STYLE_OVERRIDE', 'windowsvista')

if PYQT6_AVAILABLE:
    Qt_Keys = QtCore_Qt.Key
    Qt_Colors = QtCore_Qt.GlobalColor

    QAspectRatioMode = QtCore_Qt.AspectRatioMode
    QKeepAspectRatio = QAspectRatioMode.KeepAspectRatio

    QAlignmentFlag = QtCore_Qt.AlignmentFlag
    QAlignCenter = QAlignmentFlag.AlignCenter

    QTransformationMode = QtCore_Qt.TransformationMode
    QSmoothTransformation = QTransformationMode.SmoothTransformation

    QMessageBox_Abort = QMessageBox.StandardButton.Abort
    QMessageBox_Cancel = QMessageBox.StandardButton.Cancel
    QMessageBox_Close = QMessageBox.StandardButton.Close
    QMessageBox_Discard = QMessageBox.StandardButton.Discard
    QMessageBox_Ignore = QMessageBox.StandardButton.Ignore
    QMessageBox_No = QMessageBox.StandardButton.No
    QMessageBox_NoToAll = QMessageBox.StandardButton.NoToAll
    QMessageBox_Ok = QMessageBox.StandardButton.Ok
    QMessageBox_Save = QMessageBox.StandardButton.Save
    QMessageBox_SaveAll = QMessageBox.StandardButton.SaveAll
    QMessageBox_Yes = QMessageBox.StandardButton.Yes
    QMessageBox_YesToAll = QMessageBox.StandardButton.YesToAll

    QTextCursor_End = QTextCursor.MoveOperation.End

    QCrossCursor = QtCore_Qt.CursorShape.CrossCursor
else:
    Qt_Keys = Qt_Colors = _QtStubClass
    QAspectRatioMode = QKeepAspectRatio = _QtStubClass
    QAlignmentFlag = QAlignCenter = _QtStubClass
    QTransformationMode = QSmoothTransformation = _QtStubClass
    QMessageBox_Abort = QMessageBox_Cancel = QMessageBox_Close = None
    QMessageBox_Discard = QMessageBox_Ignore = QMessageBox_No = None
    QMessageBox_NoToAll = QMessageBox_Ok = QMessageBox_Save = None
    QMessageBox_SaveAll = QMessageBox_Yes = QMessageBox_YesToAll = None
    QTextCursor_End = None
    QCrossCursor = None

import platform

import sys
import pathlib

import platform
import os
import shutil
import glob
import matplotlib
import matplotlib.font_manager as font_manager
if PYQT6_AVAILABLE:
    from PyQt6.QtGui import QFontDatabase
    from PyQt6.QtCore import QCoreApplication

def check_and_install_fonts():

    # Locate local fonts dir relative to this file
    # This file: .../src/Python_Lib/My_Lib_PyQt6.py
    # Fonts dir: .../src_Attachments/fonts/
    current_dir = pathlib.Path(__file__).parent.resolve()
    project_root = current_dir.parent.parent
    local_fonts_dir = project_root / "src_Attachments" / "fonts"

    if not local_fonts_dir.exists():
        # print(f"Fonts folder not found at {local_fonts_dir}")
        return

    # Find all font files (ttf, otf)
    font_files = list(local_fonts_dir.glob("*.[tT][tT][fF]")) + \
                 list(local_fonts_dir.glob("*.[oO][tT][fF]"))

    if not font_files:
        return

    system = platform.system()
    
    # ---------------------------------------------------------
    # 1. System/User Installation (Persistent)
    # ---------------------------------------------------------
    if system == 'Linux':
        user_font_dir = os.path.expanduser("~/.local/share/fonts")
        try:
            if not os.path.exists(user_font_dir):
                os.makedirs(user_font_dir)

            need_rebuild = False
            for font_path in font_files:
                font_name = font_path.name
                dest_path = os.path.join(user_font_dir, font_name)
                
                if not os.path.exists(dest_path):
                    # print(f"Installing font {font_name} on Linux...")
                    try:
                        shutil.copy2(font_path, dest_path)
                        need_rebuild = True
                    except Exception as e:
                        # print(f"Failed to copy {font_name}: {e}")
                        pass
            
            if need_rebuild:
                # print("Rebuilding matplotlib font cache...")
                try:
                    # font_manager._load_fontmanager(try_read_cache=False)
                    import subprocess
                    subprocess.run(['fc-cache', '-f', '-v'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except:
                    pass
        except Exception:
            pass

    elif system == 'Windows':
        # On Windows, we just try to copy to User Fonts if not present in System Fonts
        # Fully installing usually requires Registry modification which is risky in a generic script.
        # But we can check standard paths.
        
        # Standard System Fonts
        win_fonts = pathlib.Path(os.environ.get('WINDIR', 'C:\\Windows')) / 'Fonts'
        
        # User Fonts (Windows 10+)
        local_appdata = os.environ.get('LOCALAPPDATA')
        user_fonts = None
        if local_appdata:
            user_fonts = pathlib.Path(local_appdata) / 'Microsoft' / 'Windows' / 'Fonts'

        for font_path in font_files:
            font_name = font_path.name
            
            try:
                # Check if exists in C:\Windows\Fonts
                if (win_fonts / font_name).exists():
                    continue
                
                # Check if exists in User Fonts
                if user_fonts and (user_fonts / font_name).exists():
                    continue
                    
                # If missing, try to install to User Fonts (easiest without Admin)
                if user_fonts:
                    if not user_fonts.exists():
                        try:
                            os.makedirs(user_fonts, exist_ok=True)
                        except:
                            pass
                    
                    dest = user_fonts / font_name
                    try:
                        # print(f"Installing font {font_name} on Windows (User Scope)...")
                        shutil.copy2(font_path, dest)
                        # Note: Without registry key, this survives but isn't registered system-wide on reboot.
                        # But QFontDatabase below handles the runtime session.
                    except Exception as e:
                        # print(f"Failed to install font {font_name} on Windows: {e}")
                        pass
            except Exception:
                pass

    # ---------------------------------------------------------
    # 2. Runtime Loading (Immediate usage for this process)
    # ---------------------------------------------------------
    # This ensures PyQt and Matplotlib see the fonts regardless of installation success
    
    # Initialize ctypes for Windows fallback
    add_font_resource_ex = None
    if system == 'Windows':
        try:
            import ctypes
            # FR_PRIVATE = 0x10 protects the font from other processes and doesn't require install
            # But here we want it available to the process.
            add_font_resource_ex = ctypes.windll.gdi32.AddFontResourceExW
        except Exception:
            pass

    has_qapp = (QCoreApplication.instance() is not None)

    for font_path in font_files:
        try:
            str_path = str(font_path)
            
            # Add to Matplotlib (needed even in headless for plot generation)
            try:
                font_manager.fontManager.addfont(str_path)
            except Exception:
                pass

            # Add to PyQt6 
            if not is_headless():
                # Use QFontDatabase if App exists (Safest/Correct way)
                if has_qapp:
                    QFontDatabase.addApplicationFont(str_path)
                
                # Windows Fallback: If no App, use GDI to load font for process
                elif add_font_resource_ex:
                    try:
                        # FR_PRIVATE = 0x10
                        add_font_resource_ex(str_path, 0x10, 0)
                    except Exception:
                        pass
                
        except Exception:
            pass

if __name__ != '__main__':
    # print("Checking fonts...")
    check_and_install_fonts()
    # print("Font checking complete.")

#
# def set_Windows_scaling_factor_env_var():
#
#     # Sometimes, the scaling factor of PyQt is different from the Windows system scaling factor, reason unknown
#     # For example, on a 4K screen sets to 250% scaling on Windows, PyQt reads a default 300% scaling,
#     # causing everything to be too large, this function is to determine the ratio of the real DPI and the PyQt DPI
#
#     import platform
#     if platform.system() == 'Windows':
#         import ctypes
#         try:
#             import win32api
#             MDT_EFFECTIVE_DPI = 0
#             monitor = win32api.EnumDisplayMonitors()[0]
#             dpiX,dpiY = ctypes.c_uint(),ctypes.c_uint()
#             ctypes.windll.shcore.GetDpiForMonitor(monitor[0].handle,MDT_EFFECTIVE_DPI,ctypes.byref(dpiX),ctypes.byref(dpiY))
#             DPI_ratio_for_monitor = (dpiX.value+dpiY.value)/2/96
#         except Exception as e:
#             traceback.print_exc()
#             print(e)
#             DPI_ratio_for_monitor = 0
#
#         DPI_ratio_for_device = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100
#         PyQt_scaling_ratio = QApplication.primaryScreen().devicePixelRatio()
#         print(f"Windows 10 High-DPI debug:",end=' ')
#         Windows_DPI_ratio = DPI_ratio_for_monitor if DPI_ratio_for_monitor else DPI_ratio_for_device
#         if DPI_ratio_for_monitor:
#             print("Using monitor DPI.")
#             ratio_of_ratio = DPI_ratio_for_monitor / PyQt_scaling_ratio
#         else:
#             print("Using device DPI.")
#             ratio_of_ratio = DPI_ratio_for_device / PyQt_scaling_ratio
#
#         if ratio_of_ratio>1.05 or ratio_of_ratio<0.95:
#             use_ratio = "{:.2f}".format(ratio_of_ratio)
#             print(f"{DPI_ratio_for_monitor=}, {DPI_ratio_for_device=}, {PyQt_scaling_ratio=}")
#             print(f"Using GUI high-DPI ratio: {use_ratio}")
#             print("----------------------------------------------------------------------------")
#             os.environ["QT_SCALE_FACTOR"] = use_ratio
#         else:
#             print("Ratio of ratio near 1. Not scaling.")
#
#         return Windows_DPI_ratio,PyQt_scaling_ratio
#
#

def get_matplotlib_DPI_setting(Windows_DPI_ratio):
    matplotlib_DPI_setting = 60
    if platform.system() == 'Windows':
        matplotlib_DPI_setting = 60 / Windows_DPI_ratio
    if os.path.isfile("__matplotlib_DPI_Manual_Setting.txt"):
        matplotlib_DPI_manual_setting = open("__matplotlib_DPI_Manual_Setting.txt").read()
        if is_int(matplotlib_DPI_manual_setting):
            matplotlib_DPI_setting = matplotlib_DPI_manual_setting
    else:
        with open("__matplotlib_DPI_Manual_Setting.txt", 'w') as matplotlib_DPI_Manual_Setting_file:
            matplotlib_DPI_Manual_Setting_file.write("")
    matplotlib_DPI_setting = int(matplotlib_DPI_setting)
    print(
        f"\nMatplotlib DPI: {matplotlib_DPI_setting}. \n"
        f"Set an appropriate integer in __matplotlib_DPI_Manual_Setting.txt if the preview size doesn't match the output.\n")

    return matplotlib_DPI_setting


def get_open_directories():
    if not QApplication.instance():
        QApplication(sys.argv)

    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.FileMode.Directory)
    file_dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    file_view = file_dialog.findChild(QListView, 'listView')

    # to make it possible to select multiple directories:
    if file_view:
        file_view.setSelectionMode(QAbstractItemView.MultiSelection)
    f_tree_view = file_dialog.findChild(QTreeView)
    if f_tree_view:
        f_tree_view.setSelectionMode(QAbstractItemView.MultiSelection)

    if file_dialog.exec():
        return file_dialog.selectedFiles()

    return []


def toggle_layout(layout, hide=-1, show=-1):
    """
    Hide (or show) all elements in layout
    :param layout:
    :param hide: to hide layout
    :param show: to show layout
    :return:
    """

    for i in reversed(range(layout.count())):
        assert hide != -1 or show != -1
        assert isinstance(hide, bool) or isinstance(show, bool)

        if isinstance(show, bool):
            hide = not show

        if hide:
            if layout.itemAt(i).widget():
                layout.itemAt(i).widget().hide()
        else:
            if layout.itemAt(i).widget():
                layout.itemAt(i).widget().show()


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget:
            widget.deleteLater()


def set_slider_to_line(textedit, line_number):
    scroll_bar = textedit.verticalScrollBar()
    line_height = textedit.fontMetrics().lineSpacing()
    position = line_number * line_height
    position = max(scroll_bar.minimum(), position)
    position = min(scroll_bar.maximum(), position)
    scroll_bar.setValue(position)


def vertical_scroll_to_end(textEdit):
    scroll_bar = textEdit.verticalScrollBar()
    scroll_bar.setSliderPosition(scroll_bar.maximum())


class Qt_Widget_Common_Functions:
    closing = pyqtSignal()

    def center_the_widget(self, activate_window=True):
        frame_geometry = self.frameGeometry()
        screen_center = QtGui.QGuiApplication.primaryScreen().availableGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())
        if activate_window:
            self.window().activateWindow()

    def closeEvent(self, a0: QtGui.QCloseEvent):
        # print("Window {} closed".format(self))
        self.closing.emit()
        if hasattr(super(), "closeEvent"):
            return super().closeEvent(a0)

    def open_config_file(self):
        self.config = open_config_file()

    def get_config(self, key, absence_return=""):
        return get_config(self.config, key, absence_return)

    # backward compatible
    def load_config(self, key, absence_return=""):
        return self.get_config(key, absence_return)

    def save_config(self):
        save_config(self.config)


class Drag_Drop_TextEdit(QtWidgets.QTextEdit):
    drop_accepted_signal = pyqtSignal(list)

    def __init__(self):
        super(self.__class__, self).__init__()
        self.setText(" Drop Area")
        self.setAcceptDrops(True)

        font = QFont()
        font.setFamily("arial")
        font.setPointSize(13)
        self.setFont(font)

        self.setAlignment(QAlignCenter)

    def dropEvent(self, event):
        if event.mimeData().urls():
            event.accept()
            self.drop_accepted_signal.emit([x.toLocalFile() for x in event.mimeData().urls()])
            self.reset_dropEvent(event)

    def reset_dropEvent(self, event):
        mimeData = QMimeData()
        mimeData.setText("")
        dummyEvent = QDropEvent(event.posF(), event.possibleActions(),
                                mimeData, event.mouseButtons(), event.keyboardModifiers())

        super(self.__class__, self).dropEvent(dummyEvent)


def default_signal_for_connection(signal):
    if isinstance(signal, QPushButton) or isinstance(signal, QToolButton) or isinstance(signal, QRadioButton) or \
            isinstance(signal, QCheckBox):
        signal = signal.clicked
    elif isinstance(signal, QLineEdit):
        signal = signal.textChanged
    elif isinstance(signal, QDoubleSpinBox) or isinstance(signal, QSpinBox):
        signal = signal.valueChanged
    return signal


def disconnect_all(signal, slot):
    signal = default_signal_for_connection(signal)
    marker = False
    while not marker:
        try:
            signal.disconnect(slot)
        except Exception as e:  # TODO: determine what's the specific exception?
            # traceback.print_exc()
            # print(e)
            marker = True


def connect_once(signal, slot):
    signal = default_signal_for_connection(signal)
    disconnect_all(signal, slot)
    signal.connect(slot)


def build_fileDialog_filter(allowed_appendix: list, tags=()):
    """

    :param allowed_appendix: a list of list, each group shows together [[xlsx,log,out],[txt,com,gjf]]
    :param tags: list, tag for each group, default ""
    :return: a compiled filter ready for QFileDialog.getOpenFileNames or other similar functions
            e.g. "Input File (*.gjf *.inp *.com *.sdf *.xyz)\n Output File (*.out *.log *.xlsx *.txt)"
    """

    if not tags:
        tags = [""] * len(allowed_appendix)
    else:
        assert len(tags) == len(allowed_appendix)

    ret = ""
    for count, appendix_group in enumerate(allowed_appendix):
        ret += tags[count].strip()
        ret += "(*."
        ret += ' *.'.join(appendix_group)
        ret += ')'
        if count + 1 != len(allowed_appendix):
            ret += '\n'

    return ret


def alert_UI(message="", title="", parent=None):
    # 旧版本的alert UI定义是alert_UI(parent=None，message="")
    if not isinstance(message, str) and isinstance(title, str) and parent is None:
        parent, message, title = message, title, ""
    elif not isinstance(message, str) and isinstance(title, str) and isinstance(parent, str):
        parent, message, title = message, title, parent
    print(message)
    if not QApplication.instance():
        QApplication(sys.argv)
    if not title:
        title = message
    QMessageBox.critical(parent, title, message)


def warning_UI(message="", parent=None):
    # 旧版本的alert UI定义是alert_UI(parent=None，message="")
    if not isinstance(message, str):
        message, parent = parent, message
    print(message)
    if not QApplication.instance():
        QApplication(sys.argv)
    QMessageBox.warning(parent, message, message)


def information_UI(message="", parent=None):
    # 旧版本的alert UI定义是alert_UI(parent=None，message="")

    if not isinstance(message, str):
        message, parent = parent, message
    print(message)
    if not QApplication.instance():
        QApplication(sys.argv)
    QMessageBox.information(parent, message, message)


def wait_confirmation_UI(parent=None, message=""):
    if not QApplication.instance():
        QApplication(sys.argv)
    button = QMessageBox.warning(parent, message, message, QMessageBox_Ok | QMessageBox_Cancel)
    if button == QMessageBox_Ok:
        return True
    else:
        return False


def get_open_file_UI(parent, start_path: str, allowed_appendix, title="No Title", tags=(), single=False, save=False):
    """

    :param parent
    :param start_path:
    :param allowed_appendix: same as function (build_fileDialog_filter)
            but allow single str "txt" or single list ['txt','gjf'] as input, list of list is not necessary
    :param title:
    :param tags:
    :param single:
    :param save: use the save file UI
    :return: a list of files if not single, a single filepath if single
    """

    if not QApplication.instance():
        QApplication(sys.argv)

    if isinstance(allowed_appendix, str):  # single str
        allowed_appendix = [[allowed_appendix]]
    if [x for x in allowed_appendix if isinstance(x, str)]:  # single list not list of list
        allowed_appendix = [allowed_appendix]

    filename_filter_string = build_fileDialog_filter(allowed_appendix, tags)

    if save:
        ret = QFileDialog.getSaveFileName(parent, title, start_path, filename_filter_string)
    else:
        if single:
            ret = QFileDialog.getOpenFileName(parent, title, start_path, filename_filter_string)
        else:
            ret = QFileDialog.getOpenFileNames(parent, title, start_path, filename_filter_string)

    if ret:  # 上面返回 (['E:/My_Program/Python_Lib/elements_dict.txt'], '(*.txt)')
        return ret[0]


def get_save_file_UI(parent, start_path: str, allowed_appendix, title="No Title", tags=()):
    return get_open_file_UI(parent, start_path, allowed_appendix, title=title, tags=tags)


def show_pixmap(image_filename, graphicsView_object):
    # must call widget.show() holding the graphicsView, otherwise the View.size() will get a wrong (100,30) value
    if os.path.isfile(image_filename):
        pixmap = QPixmap()
        pixmap.load(image_filename)

        print(graphicsView_object.size())

        if pixmap.width() > graphicsView_object.width() or pixmap.height() > graphicsView_object.height():
            pixmap = pixmap.scaled(graphicsView_object.size(), QKeepAspectRatio, QSmoothTransformation)
    else:
        pixmap = QPixmap()

    graphicsPixmapItem = QGraphicsPixmapItem(pixmap)
    graphicsScene = QGraphicsScene()
    graphicsScene.addItem(graphicsPixmapItem)
    graphicsView_object.setScene(graphicsScene)


def update_UI():
    QtCore.QCoreApplication.processEvents()


def exit_UI():
    QtCore.QCoreApplication.instance().quit()


def clear_layout(layout):
    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()


def add_list_to_layout(layout, list_of_item):
    for item in list_of_item:
        if isinstance(item, QWidget):
            layout.addWidget(item)
        if isinstance(item, QLayout):
            layout.addLayout(item)


def pyqt_ui_compile(filename):
    """

    :param filename:
    :return:
    """

    # 允许将.ui文件放在命名为UI的文件夹下，或程序目录下，但只输入文件名，而不必输入“UI/”

    if filename[:3] in ['UI\\', 'UI/']:
        filename = filename[3:]

    ui_filename = filename_class(filename).replace_append_to('ui')
    # print(os.path.abspath(ui_filename))
    if not os.path.isfile(ui_filename):
        ui_filename = 'UI/' + ui_filename
    modify_log_filename = filename_class(ui_filename).replace_append_to('txt')
    py_file = filename_class(ui_filename).replace_append_to('py')

    modify_time = ""
    if os.path.isfile(modify_log_filename):
        with open(modify_log_filename) as modify_log_file:
            modify_time = modify_log_file.read()

    if modify_time != str(int(os.path.getmtime(ui_filename))):
        print("GUI MODIFIED:", ui_filename)
        with open(modify_log_filename, 'w') as modify_log_file:
            modify_log_file.write(str(int(os.path.getmtime(ui_filename))))

        ui_File_Compile = open(py_file, 'w')
        uic.compileUi(ui_filename, ui_File_Compile)
        ui_File_Compile.close()
        with open(py_file, encoding='gbk') as ui_File_Compile_object:
            ui_File_Compile_content = ui_File_Compile_object.read()
        with open(py_file, 'w', encoding='utf-8') as ui_File_Compile_object:
            ui_File_Compile_object.write(ui_File_Compile_content)


# ──────────── Column-aware multi-line text editor ────────────
class _LineNumberArea(QtWidgets.QWidget):
    """Gutter widget that paints line numbers for ColumnEditTextEdit."""

    def __init__(self, editor):
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self):
        return QtCore.QSize(self._editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self._editor.line_number_area_paint_event(event)


class ColumnEditTextEdit(QtWidgets.QPlainTextEdit):
    """Plain-text editor with a line-number gutter and Alt+drag column editing.

    Features
    --------
    * Consolas font, no line wrapping.
    * A line-number gutter; numbers past ``curve_count`` are painted grey (those
      lines map to no curve) but remain fully editable.
    * Alt+drag marks a rectangular block; typing / Backspace / Delete then edits
      every line in the block at the same columns. Short lines are space-padded
      on insert. Escape (or a plain click / arrow key) leaves column mode.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        font = QFont("Consolas", 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)

        self._curve_count = 0

        self._line_number_area = _LineNumberArea(self)
        self.blockCountChanged.connect(self._update_line_number_area_width)
        self.updateRequest.connect(self._update_line_number_area)
        self._update_line_number_area_width(0)

        # Column-selection state. Anchor/caret are (block_number, column) pairs.
        self._col_active = False     # a column block (or multi-line caret) exists
        self._col_dragging = False   # mouse is currently Alt-dragging
        self._col_anchor = (0, 0)
        self._col_caret = (0, 0)

    # ── grey-line-number threshold ──
    def set_curve_count(self, n):
        self._curve_count = max(0, int(n))
        self._line_number_area.update()

    # ── line-number gutter (standard QPlainTextEdit pattern) ──
    def line_number_area_width(self):
        digits = max(1, len(str(max(1, self.blockCount()))))
        return 12 + self.fontMetrics().horizontalAdvance('9') * digits

    def _update_line_number_area_width(self, _count=0):
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def _update_line_number_area(self, rect, dy):
        if dy:
            self._line_number_area.scroll(0, dy)
        else:
            self._line_number_area.update(
                0, rect.y(), self._line_number_area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_line_number_area_width(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._line_number_area.setGeometry(
            QtCore.QRect(cr.left(), cr.top(),
                         self.line_number_area_width(), cr.height()))

    def line_number_area_paint_event(self, event):
        painter = QtGui.QPainter(self._line_number_area)
        painter.fillRect(event.rect(), QtGui.QColor("#f0f0f0"))

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(self.blockBoundingGeometry(block)
                  .translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())
        normal_color = QtGui.QColor("#606060")
        grey_color = QtGui.QColor("#b8b8b8")
        flags = int(QtCore_Qt.AlignmentFlag.AlignRight | QtCore_Qt.AlignmentFlag.AlignVCenter)
        width = self._line_number_area.width() - 4
        height = self.fontMetrics().height()

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                # Lines past the curve count are "extra" → grey number.
                painter.setPen(grey_color if block_number >= self._curve_count
                               else normal_color)
                painter.drawText(0, top, width, height, flags,
                                 str(block_number + 1))
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_number += 1
        painter.end()

    # ── column-mode geometry helpers ──
    def _pos_to_line_col(self, pos):
        cursor = self.cursorForPosition(pos)
        return cursor.blockNumber(), cursor.positionInBlock()

    def _make_cursor(self, line, col):
        block = self.document().findBlockByNumber(line)
        cursor = QtGui.QTextCursor(block)
        col = min(max(0, col), block.length() - 1)  # length counts the separator
        cursor.setPosition(block.position() + col)
        return cursor

    def _column_bounds(self):
        a_line, a_col = self._col_anchor
        c_line, c_col = self._col_caret
        return (min(a_line, c_line), max(a_line, c_line),
                min(a_col, c_col), max(a_col, c_col))

    def _clear_column_mode(self):
        if self._col_active or self._col_dragging:
            self._col_active = False
            self._col_dragging = False
            self.viewport().update()

    # ── mouse: Alt+drag starts / extends a column block ──
    def mousePressEvent(self, event):
        if (event.button() == QtCore_Qt.MouseButton.LeftButton
                and event.modifiers() & QtCore_Qt.KeyboardModifier.AltModifier):
            line, col = self._pos_to_line_col(event.pos())
            self._col_anchor = (line, col)
            self._col_caret = (line, col)
            self._col_dragging = True
            self._col_active = True
            self.setTextCursor(self._make_cursor(line, col))
            self.viewport().update()
            event.accept()
            return
        self._clear_column_mode()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._col_dragging:
            line, col = self._pos_to_line_col(event.pos())
            self._col_caret = (line, col)
            self.setTextCursor(self._make_cursor(line, col))
            self.viewport().update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._col_dragging:
            self._col_dragging = False
            line, col = self._pos_to_line_col(event.pos())
            self._col_caret = (line, col)
            top, bottom, left, right = self._column_bounds()
            # A zero-area Alt-click is just a normal caret.
            if top == bottom and left == right:
                self._col_active = False
            self.viewport().update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ── paint the column selection / multi-line caret over the text ──
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._col_active:
            return
        top_line, bottom_line, left_col, right_col = self._column_bounds()
        painter = QtGui.QPainter(self.viewport())
        sel_color = QtGui.QColor(51, 153, 255, 70)
        caret_color = QtGui.QColor(51, 102, 204)
        for line in range(top_line, bottom_line + 1):
            block = self.document().findBlockByNumber(line)
            if not block.isValid() or not block.isVisible():
                continue
            text_len = block.length() - 1
            left_rect = self.cursorRect(self._make_cursor(line, min(left_col, text_len)))
            if left_col == right_col:
                painter.fillRect(left_rect.left(), left_rect.top(),
                                 2, left_rect.height(), caret_color)
            else:
                right_rect = self.cursorRect(
                    self._make_cursor(line, min(right_col, text_len)))
                w = max(right_rect.left() - left_rect.left(), 2)
                painter.fillRect(left_rect.left(), left_rect.top(),
                                 w, left_rect.height(), sel_color)
        painter.end()

    # ── keyboard: route edits to every line of the column block ──
    def keyPressEvent(self, event):
        if not self._col_active:
            super().keyPressEvent(event)
            return
        key = event.key()
        ctrl = bool(event.modifiers() & QtCore_Qt.KeyboardModifier.ControlModifier)
        if key == QtCore_Qt.Key.Key_Escape:
            self._clear_column_mode()
            event.accept(); return
        if ctrl and key == QtCore_Qt.Key.Key_C:
            self._column_copy(); event.accept(); return
        if ctrl and key == QtCore_Qt.Key.Key_X:
            self._column_copy(); self._column_replace(""); event.accept(); return
        if ctrl and key == QtCore_Qt.Key.Key_V:
            self._column_paste(); event.accept(); return
        if ctrl:
            self._clear_column_mode()
            super().keyPressEvent(event); return
        # Bare modifier presses (e.g. Shift before an uppercase letter) must
        # not drop column mode, or Shift+letter typing would break.
        if key in (QtCore_Qt.Key.Key_Shift, QtCore_Qt.Key.Key_Control, QtCore_Qt.Key.Key_Alt,
                   QtCore_Qt.Key.Key_Meta, QtCore_Qt.Key.Key_AltGr, QtCore_Qt.Key.Key_CapsLock,
                   QtCore_Qt.Key.Key_NumLock):
            super().keyPressEvent(event); return
        if key == QtCore_Qt.Key.Key_Backspace:
            self._column_backspace(); event.accept(); return
        if key == QtCore_Qt.Key.Key_Delete:
            self._column_delete(); event.accept(); return
        if key in (QtCore_Qt.Key.Key_Left, QtCore_Qt.Key.Key_Right, QtCore_Qt.Key.Key_Up,
                   QtCore_Qt.Key.Key_Down, QtCore_Qt.Key.Key_Home, QtCore_Qt.Key.Key_End,
                   QtCore_Qt.Key.Key_Return, QtCore_Qt.Key.Key_Enter, QtCore_Qt.Key.Key_Tab,
                   QtCore_Qt.Key.Key_PageUp, QtCore_Qt.Key.Key_PageDown):
            self._clear_column_mode()
            super().keyPressEvent(event); return
        text = event.text()
        if text and text.isprintable():
            self._column_replace(text); event.accept(); return
        self._clear_column_mode()
        super().keyPressEvent(event)

    # ── per-line column edits ──
    def _replace_line_columns(self, line, left, right, text):
        """On one line, replace the columns [left, right) with *text*.

        Lines shorter than *left* are space-padded before a non-empty insert;
        an empty *text* on such a line is a no-op (so Backspace/Delete never
        add padding)."""
        block = self.document().findBlockByNumber(line)
        if not block.isValid():
            return
        text_len = block.length() - 1
        cursor = QtGui.QTextCursor(block)
        if left >= text_len:
            if not text:
                return
            cursor.setPosition(block.position() + text_len)
            cursor.insertText(' ' * (left - text_len) + text)
        else:
            cursor.setPosition(block.position() + left)
            cursor.setPosition(block.position() + min(right, text_len),
                               QtGui.QTextCursor.MoveMode.KeepAnchor)
            cursor.insertText(text)

    def _column_replace(self, text):
        """Replace the current block-selection columns on every line with *text*
        (a single line), then collapse to a multi-line caret after it."""
        top, bottom, left, right = self._column_bounds()
        edit = QtGui.QTextCursor(self.document())
        edit.beginEditBlock()
        for line in range(top, bottom + 1):
            self._replace_line_columns(line, left, right, text)
        edit.endEditBlock()
        new_col = left + len(text)
        self._col_anchor = (top, new_col)
        self._col_caret = (bottom, new_col)
        self._col_active = True
        self.setTextCursor(self._make_cursor(bottom, new_col))
        self.viewport().update()

    def _column_backspace(self):
        top, bottom, left, right = self._column_bounds()
        if left != right:
            self._column_replace("")
            return
        if left == 0:
            return
        edit = QtGui.QTextCursor(self.document())
        edit.beginEditBlock()
        for line in range(top, bottom + 1):
            self._replace_line_columns(line, left - 1, left, "")
        edit.endEditBlock()
        new_col = left - 1
        self._col_anchor = (top, new_col)
        self._col_caret = (bottom, new_col)
        self.setTextCursor(self._make_cursor(bottom, new_col))
        self.viewport().update()

    def _column_delete(self):
        top, bottom, left, right = self._column_bounds()
        if left != right:
            self._column_replace("")
            return
        edit = QtGui.QTextCursor(self.document())
        edit.beginEditBlock()
        for line in range(top, bottom + 1):
            self._replace_line_columns(line, left, left + 1, "")
        edit.endEditBlock()
        self.setTextCursor(self._make_cursor(bottom, left))
        self.viewport().update()

    def _column_copy(self):
        top, bottom, left, right = self._column_bounds()
        rows = []
        for line in range(top, bottom + 1):
            block = self.document().findBlockByNumber(line)
            s = block.text() if block.isValid() else ""
            rows.append(s[min(left, len(s)):min(right, len(s))])
        QApplication.clipboard().setText("\n".join(rows))

    def _column_paste(self):
        clip = QApplication.clipboard().text()
        top, bottom, left, right = self._column_bounds()
        nrows = bottom - top + 1
        parts = clip.split('\n')
        if len(parts) == nrows and nrows > 1:
            # One clipboard line per selected row → distribute.
            edit = QtGui.QTextCursor(self.document())
            edit.beginEditBlock()
            for i, line in enumerate(range(top, bottom + 1)):
                self._replace_line_columns(line, left, right, parts[i])
            edit.endEditBlock()
            self._clear_column_mode()
            self.setTextCursor(self._make_cursor(bottom, left + len(parts[-1])))
            return
        # Otherwise insert the first clipboard line on every row.
        self._column_replace(parts[0] if parts else "")


class ResizableLabel(QtWidgets.QLabel):
    def __init__(self, text="", parent=None, max_font_size = 10):
        super().__init__(text, parent)

        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(30)
        self.setMaximumHeight(60)
        if not text:
            self.setFixedHeight(0)

        max_font_size = int(round(max_font_size))

        # Base font setup
        self.base_font = QtGui.QFont("Arial", max_font_size)
        self.setFont(self.base_font)
        self.max_font_size = max_font_size
        self.min_font_size = 6  # prevent disappearing text
        self.current_font_size = self.max_font_size

        # Enable word wrap so text can adjust height
        self.setWordWrap(True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjustFontSize()

    def setText(self, text: str):
        super().setText(text)
        self.adjustFontSize()

        # if self.text().strip() == "":
        #     self.setMaximumHeight(0)
        #     self.setMinimumHeight(0)
        #     self.hide()
        # else:
        #     self.show()
        #     self.adjustFontSize()

    def adjustFontSize(self):
        """Adjust font size to fit width while keeping height flexible."""
        if not self.text().strip():
            return

        available_width = self.width()
        font = QtGui.QFont(self.base_font)

        for font_size in range(self.max_font_size,self.min_font_size-1,-1):
            font.setPointSize(font_size)
            fm = QtGui.QFontMetrics(font)
            text_width = fm.horizontalAdvance(self.text())

            if text_width <= available_width or font_size == self.min_font_size:
                self.current_font_size = font_size
                break
        # print("Font size:", font_size)
        self.setFont(font)

        # Adjust height based on text contents
        fm = QtGui.QFontMetrics(font)
        text_rect = fm.boundingRect(
            0, 0, available_width, 0,
            QtCore.Qt.TextFlag.TextWordWrap,
            self.text()
        )
        new_height = max(text_rect.height() + 6, 30)  # +6 for padding
        self.setMinimumHeight(new_height)
        self.setMaximumHeight(new_height)


class Ui_Wait_Message_Form(object):
    def setupUi(self, Wait_Message_Form):
        Wait_Message_Form.setObjectName("Wait_Message_Form")
        Wait_Message_Form.resize(446, 82)
        self.horizontalLayout = QHBoxLayout(Wait_Message_Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(Wait_Message_Form)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setAlignment(QAlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)

        self.retranslateUi(Wait_Message_Form)
        QtCore.QMetaObject.connectSlotsByName(Wait_Message_Form)

    def retranslateUi(self, Wait_Message_Form):
        _translate = QtCore.QCoreApplication.translate
        Wait_Message_Form.setWindowTitle(_translate("Wait_Message_Form", "Message"))
        self.label.setText(_translate("Wait_Message_Form", "Doing someting...Please wait..."))


class Wait_MessageBox(Ui_Wait_Message_Form, QWidget, Qt_Widget_Common_Functions):
    def __init__(self, message):
        super(self.__class__, self).__init__()
        print(message)
        self.setupUi(self)
        self.label.setText(message)
        timer = QTimer()
        timer.start(10)

    def setText(self, text):
        self.label.setText(text)

    def pop_out(self):
        self.show()
        self.center_the_widget()


def wait_messageBox(message, title="Please Wait..."):
    if not QApplication.instance():
        QApplication(sys.argv)

    message_box = QMessageBox()
    message_box.setWindowTitle(title)
    message_box.setText(message)

    return message_box
