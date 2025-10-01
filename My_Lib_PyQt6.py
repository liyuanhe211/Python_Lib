# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import os
from PyQt6 import QtGui, QtCore, QtWidgets, uic
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QMessageBox, \
    QFileDialog, QGraphicsPixmapItem, QGraphicsScene, QInputDialog, QDialog, \
    QListView, QAbstractItemView, QTreeView, QWidget, QLayout, QVBoxLayout, QHBoxLayout, QGridLayout, \
    QTextEdit, QSpinBox, QAbstractSpinBox, \
    QPushButton, QToolButton, QRadioButton, QCheckBox, QLineEdit, QDoubleSpinBox, \
    QTableWidgetItem, QFrame, QSpacerItem, QSizePolicy, QTableWidget
from PyQt6.QtGui import QPixmap, QColor, QPainter, QPen, QFont, QDropEvent, QIcon, QTextCursor, QScreen, QKeyEvent, QTextCharFormat, QSyntaxHighlighter
from PyQt6.QtCore import QPoint, QTimer, QMimeData, QSize, pyqtSignal, QProcess
from PyQt6.QtCore import Qt as QtCore_Qt
from PyQt6.QtCore import QEvent

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

import platform

# import matplotlib
# matplotlib.use("QtAgg")
# from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as MpFigureCanvas
# from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as MpNavToolBar
# from matplotlib import pyplot
# import matplotlib.patches as patches
# from matplotlib.figure import Figure as MpFigure
# from matplotlib import pylab

import sys
import pathlib

Python_Lib_path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(Python_Lib_path)
from My_Lib_Stock import *


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

    def closeEvent(self, event: QtGui.QCloseEvent):
        # print("Window {} closed".format(self))
        self.closing.emit()
        if hasattr(super(), "closeEvent"):
            return super().closeEvent(event)

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


class ResizableLabel(QtWidgets.QLabel):
    def __init__(self, text="", parent=None, max_font_size = 10):
        super().__init__(text, parent)

        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(30)

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
