# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import sys
import os

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QTextEdit, QSplitter,
                             QSizePolicy)
from PyQt6.QtGui import QFont, QPainter, QColor, QFontMetrics, QTextCursor
from PyQt6.QtCore import Qt, pyqtSignal

from Python_Lib.My_Lib_Stock import natural_language_sort
from Python_Lib.My_Lib_PyQt6 import Drag_Drop_TextEdit, get_open_file_UI


FONT_EDIT = QFont("Consolas", 9)
FONT_BUTTON = QFont("Arial", 10)
BUTTON_HEIGHT = 25


class LineNumberArea(QWidget):
    """Renders line numbers aligned to blocks in a paired QTextEdit."""

    def __init__(self, text_edit):
        super().__init__()
        self._te = text_edit
        self.setFont(text_edit.font())
        self._update_width()
        text_edit.document().blockCountChanged.connect(self._update_width)
        text_edit.verticalScrollBar().valueChanged.connect(self.update)
        text_edit.document().contentsChanged.connect(self.update)

    def _update_width(self):
        digits = len(str(max(1, self._te.document().blockCount())))
        fm = QFontMetrics(self.font())
        w = fm.horizontalAdvance('9') * digits + 12
        self.setFixedWidth(w)
        self.update()

    def paintEvent(self, a0):
        if a0 is None:
            return
        painter = QPainter(self)
        painter.fillRect(a0.rect(), QColor(240, 240, 240))
        # separator line on the right edge
        painter.setPen(QColor(190, 190, 190))
        painter.drawLine(self.width() - 1, 0, self.width() - 1, self.height())

        painter.setFont(self.font())
        painter.setPen(QColor(120, 120, 120))

        doc = self._te.document()
        block = doc.begin()
        line_num = 1
        while block.isValid():
            cursor = QTextCursor(block)
            rect = self._te.cursorRect(cursor)
            y = rect.top()
            h = rect.height()
            if y > self.height():
                break
            if y + h > 0:
                painter.drawText(
                    0, y, self.width() - 6, h,
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    str(line_num)
                )
            block = block.next()
            line_num += 1


class InputSection(QWidget):
    """One input section: toolbar buttons + text edit."""
    delete_requested = pyqtSignal(object)
    content_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)

        self.btn_load = QPushButton("Load File")
        self.btn_string_sort = QPushButton("String Sorting")
        self.btn_natural_sort = QPushButton("Natural Language Sorting")
        self.btn_delete = QPushButton("Delete Section")

        for btn in (self.btn_load, self.btn_string_sort, self.btn_natural_sort, self.btn_delete):
            btn.setFont(FONT_BUTTON)
            btn.setFixedHeight(BUTTON_HEIGHT)
            btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            toolbar.addWidget(btn)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Text edit
        self.text_edit = Drag_Drop_TextEdit()
        self.text_edit.clear()
        self.text_edit.setFont(FONT_EDIT)
        self.text_edit.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.text_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.text_edit.drop_accepted_signal.connect(self._append_dropped_files)
        self.text_edit.textChanged.connect(self.content_changed)

        # Wrap text edit with line number area
        te_wrapper = QWidget()
        te_wrapper_layout = QHBoxLayout(te_wrapper)
        te_wrapper_layout.setContentsMargins(0, 0, 0, 0)
        te_wrapper_layout.setSpacing(0)
        self._line_number_area = LineNumberArea(self.text_edit)
        te_wrapper_layout.addWidget(self._line_number_area)
        te_wrapper_layout.addWidget(self.text_edit)
        layout.addWidget(te_wrapper)

        # Connections
        self.btn_load.clicked.connect(self._load_file)
        self.btn_string_sort.clicked.connect(self._string_sort)
        self.btn_natural_sort.clicked.connect(self._natural_sort)
        self.btn_delete.clicked.connect(lambda: self.delete_requested.emit(self))

    def _append_dropped_files(self, paths):
        paths = [path for path in paths if path]
        if not paths:
            return
        existing = self.text_edit.toPlainText()
        if existing and not existing.endswith("\n"):
            existing += "\n"
        self.text_edit.setPlainText(existing + "\n".join(paths))

    def _load_file(self):
        start_path = os.getcwd()
        selected = get_open_file_UI(
            self,
            start_path,
            [["txt", "csv", "json", "md", "py", "bat", "log", "yaml", "yml", "ini", "cfg", "gjf", "com", "out"]],
            title="Load File",
            single=True,
        )
        if isinstance(selected, list):
            path = selected[0] if selected else ""
        else:
            path = selected
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(path, "r", encoding="latin-1") as f:
                    content = f.read()
            self.text_edit.setPlainText(content)

    def _string_sort(self):
        lines = self.get_lines()
        lines.sort()
        self.text_edit.setPlainText("\n".join(lines))

    def _natural_sort(self):
        lines = self.get_lines()
        lines = natural_language_sort(lines)
        self.text_edit.setPlainText("\n".join(lines))

    def get_lines(self):
        """Return lines, stripping trailing empty lines (only truly empty \"\", not whitespace-only)."""
        text = self.text_edit.toPlainText()
        if not text:
            return []
        lines = text.split("\n")
        while lines and lines[-1] == "":
            lines.pop()
        return lines


class BatchCommandGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Batch Command Generator")
        self.resize(1500, 850)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # --- Left panel ---
        left_widget = QWidget()
        self.left_layout = QVBoxLayout(left_widget)
        self.left_layout.setContentsMargins(4, 4, 4, 4)
        self.left_layout.setSpacing(6)

        self.sections: list[InputSection] = []

        # Add-section button
        self.btn_add_section = QPushButton("+ Add Section")
        self.btn_add_section.setFont(FONT_BUTTON)
        self.btn_add_section.setFixedHeight(BUTTON_HEIGHT)
        self.btn_add_section.clicked.connect(self._add_section)

        self.left_layout.addWidget(self.btn_add_section)

        splitter.addWidget(left_widget)

        # --- Right panel ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)

        # Output toolbar (matching style with left side)
        output_toolbar = QHBoxLayout()
        output_label = QLabel("Output")
        output_label.setFont(FONT_BUTTON)
        output_toolbar.addWidget(output_label)
        output_toolbar.addStretch()
        right_layout.addLayout(output_toolbar)

        self.output_edit = QTextEdit()
        self.output_edit.setFont(FONT_EDIT)
        self.output_edit.setReadOnly(True)
        self.output_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        right_layout.addWidget(self.output_edit)

        splitter.addWidget(right_widget)

        # Equal widths
        splitter.setSizes([600, 600])

        # Start with three sections
        self._add_section()
        self._add_section()
        self._add_section()

    def _add_section(self):
        section = InputSection()
        section.delete_requested.connect(self._remove_section)
        section.content_changed.connect(self._update_output)
        self.sections.append(section)

        # Insert before the add-button
        idx = self.left_layout.count() - 1  # before the btn_add_section
        self.left_layout.insertWidget(idx, section)

        self._distribute_heights()
        self._update_output()

    def _remove_section(self, section: InputSection):
        if section in self.sections:
            self.sections.remove(section)
            section.setParent(None)
            section.deleteLater()
            self._distribute_heights()
            self._update_output()

    def _distribute_heights(self):
        for i in range(self.left_layout.count()):
            item = self.left_layout.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), InputSection):
                self.left_layout.setStretch(i, 1)
            else:
                self.left_layout.setStretch(i, 0)

    def _update_output(self):
        section_lines = [s.get_lines() for s in self.sections]

        max_len = max((len(lines) for lines in section_lines), default=0)
        if max_len == 0:
            self.output_edit.setPlainText("")
            return

        output_lines = []
        for i in range(max_len):
            parts = []
            for lines in section_lines:
                if not lines:
                    continue
                else:
                    parts.append(lines[i % len(lines)])
            # Join parts; respect existing trailing/leading whitespace
            combined = " ".join(parts)
            output_lines.append(combined)

        display = "\n".join(output_lines)
        self.output_edit.setPlainText(display)


def main():
    app = QApplication(sys.argv)
    window = BatchCommandGenerator()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
