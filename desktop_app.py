import sys
import os
import time
from pathlib import Path

# IMPORT TORCH FIRST ON WINDOWS TO AVOID DLL CONFLICTS WITH PYQT
import torch

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QListWidget, QListWidgetItem, QPushButton, QMessageBox, QDialog, QLabel,
                             QCheckBox, QTextEdit, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QSize
from PyQt6.QtGui import QColor, QBrush, QPixmap

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.cli.predict import predict_image
from sharp.utils.gaussians import save_ply

# Import viewer components
from viewer import RenderWorker, LoadWorker, ParallaxViewer

class CalcWorker(QThread):
    log_msg = pyqtSignal(str, str, bool) # text, color, bold
    finished_calc = pyqtSignal(float)
    error_occurred = pyqtSignal(str)

    def __init__(self, image_paths, checkpoint_path, output_dir):
        super().__init__()
        self.image_paths = image_paths
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir

    def run(self):
        try:
            total_start_time = time.time()
            total_images = len(self.image_paths)
            
            # Setup device
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
                
            self.log_msg.emit(f"✓ {total_images} 个待处理的图片。", "#00FFFF", True)
            self.log_msg.emit(f"✓ 使用计算设备: {device}", "#FFFF00", True)
            
            self.log_msg.emit(f"○ 正在从 {self.checkpoint_path} 加载模型权重...", "#FF00FF", False)
            
            start_load = time.time()
            # Load model
            state_dict = torch.load(self.checkpoint_path, weights_only=True)
            gaussian_predictor = create_predictor(PredictorParams())
            gaussian_predictor.load_state_dict(state_dict)
            gaussian_predictor.eval()
            gaussian_predictor.to(device)
            load_time = time.time() - start_load
            
            self.log_msg.emit(f"✓ 模型加载完成！耗时 {load_time:.2f} 秒\n", "#00FF00", False)
            
            self.output_dir.mkdir(exist_ok=True, parents=True)
            
            for i, image_path in enumerate(self.image_paths, 1):
                img_start_time = time.time()
                self.log_msg.emit(f"[{i}/{total_images}] 正在处理图像: {image_path.name}...", "#4da6ff", True)
                
                # Load image and focal length
                image, _, f_px = io.load_rgb(image_path)
                height, width = image.shape[:2]
                
                # Predict 3D Gaussians
                gaussians = predict_image(gaussian_predictor, image, f_px, torch.device(device))
                
                # Save PLY file
                output_file = self.output_dir / f"{image_path.stem}.ply"
                save_ply(gaussians, f_px, (height, width), output_file)
                
                img_time = time.time() - img_start_time
                self.log_msg.emit(f"  ✓ 单图处理完成！模型已成功导出至 {output_file.name} (单图耗时 {img_time:.2f}s)\n", "#00FF00", False)
                
            total_time = time.time() - total_start_time
            avg_time = total_time / total_images if total_images > 0 else 0
            self.log_msg.emit(f"✓ 全部处理完成！总耗时: {total_time:.2f}s (平均耗时: {avg_time:.2f}s/个)", "#00FF00", True)
            self.finished_calc.emit(total_time)
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class DetailedProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("计算进度")
        self.resize(700, 500)
        self.setModal(True)
        self.setStyleSheet("QDialog { background-color: #2b2b2b; }")
        
        layout = QVBoxLayout(self)
        
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 14px;
                border: 1px solid #444;
                padding: 10px;
            }
        """)
        layout.addWidget(self.text_edit)
        
        self.close_btn = QPushButton("完成")
        self.close_btn.setStyleSheet("""
            QPushButton { 
                background-color: #3a3a3a; color: white; border: 1px solid #555; 
                padding: 10px 40px; font-size: 14px; border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #4CAF50; border: 1px solid #4CAF50;}
            QPushButton:pressed { background-color: #388E3C; }
            QPushButton:disabled { background-color: #2b2b2b; color: #777; border: 1px solid #444; }
        """)
        self.close_btn.clicked.connect(self.close)
        self.close_btn.setEnabled(False)
        layout.addWidget(self.close_btn, alignment=Qt.AlignmentFlag.AlignRight)
        
    def append_log(self, text, color="#d4d4d4", bold=False):
        weight = "bold" if bold else "normal"
        # Convert newline character to HTML <br> if needed, though we already emit them separately
        text = text.replace('\n', '<br>')
        html = f'<span style="color: {color}; font-weight: {weight};">{text}</span><br>'
        
        cursor = self.text_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.text_edit.setTextCursor(cursor)
        self.text_edit.insertHtml(html)
        
        # Scroll to bottom
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def calculation_finished(self):
        self.close_btn.setEnabled(True)


class PreviewDialog(QDialog):
    def __init__(self, filepath, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3DGS 实时视差预览")
        self.resize(1000, 700)
        self.setStyleSheet("QDialog { background-color: #2b2b2b; }")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.render_worker = RenderWorker()
        self.viewer = ParallaxViewer(self.render_worker)
        self.viewer.setText("")
        layout.addWidget(self.viewer)
        
        self.loader = LoadWorker(self.render_worker, filepath)
        self.loader.finished_loading.connect(self.on_load_finished)
        self.loader.error.connect(self.on_load_error)
        
        # UI prompt while loading
        self.loading_label = QLabel("正在将 3D 模型加载至 GPU，请稍候...", self.viewer)
        self.loading_label.setStyleSheet("color: white; font-size: 18px;")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout_overlay = QVBoxLayout(self.viewer)
        layout_overlay.addWidget(self.loading_label)
        
        self.loader.start()
        
    def on_load_finished(self, filepath):
        self.loading_label.hide()
        self.viewer.is_loaded = True
        
    def on_load_error(self, err_msg):
        self.loading_label.setText(f"加载失败: {err_msg}")
        QMessageBox.critical(self, "错误", f"加载模型失败: {err_msg}")


class CardWidget(QFrame):
    selection_changed = pyqtSignal()
    
    def __init__(self, img_path, has_ply):
        super().__init__()
        self.img_path = img_path
        self.has_ply = has_ply
        
        self.setFixedSize(220, 280)
        self.setStyleSheet("""
            CardWidget {
                background-color: #333333;
                border-radius: 8px;
                border: 1px solid #444;
            }
            CardWidget:hover { border: 1px solid #666; background-color: #3a3a3a; }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Checkbox and label
        self.checkbox = QCheckBox(img_path.name)
        font_metrics = self.checkbox.fontMetrics()
        elided_text = font_metrics.elidedText(img_path.name, Qt.TextElideMode.ElideRight, 160)
        self.checkbox.setText(elided_text)
        self.checkbox.setToolTip(img_path.name)
        self.checkbox.setStyleSheet("""
            QCheckBox { color: white; font-weight: bold; font-size: 14px; spacing: 8px; outline: none; }
            QCheckBox::indicator { width: 18px; height: 18px; }
        """)
        self.checkbox.toggled.connect(self.selection_changed.emit)
        layout.addWidget(self.checkbox)
        
        # Image
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setStyleSheet("background-color: #222; border-radius: 6px;")
        self.img_label.setFixedSize(196, 170)
        
        pixmap = QPixmap(str(img_path))
        if not pixmap.isNull():
            scaled = pixmap.scaled(196, 170, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.img_label.setPixmap(scaled)
        else:
            self.img_label.setText("无图片预览")
            self.img_label.setStyleSheet("color: #888; background-color: #222; border-radius: 6px;")
            
        layout.addWidget(self.img_label)
        
        # Status
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if has_ply:
            self.status_label.setText("✓ 已生成模型 (.ply)")
            self.status_label.setStyleSheet("color: #4CAF50; font-size: 13px; font-weight: bold; padding-top: 5px;")
        else:
            self.status_label.setText("○ 未生成模型")
            self.status_label.setStyleSheet("color: #9e9e9e; font-size: 13px; padding-top: 5px;")
            
        layout.addWidget(self.status_label)
        
    def mousePressEvent(self, event):
        # Allow clicking entire card to toggle checkbox preview state
        if event.button() == Qt.MouseButton.LeftButton:
            self.checkbox.setChecked(not self.checkbox.isChecked())
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Apple ml-sharp Desktop")
        self.resize(1000, 700)
        
        self.input_dir = Path("data/input")
        self.output_dir = Path("data/output")
        self.checkpoint_path = Path("model/sharp_2572gikvuh.pt")
        
        self.setup_ui()
        self.load_file_list()
        
    def setup_ui(self):
        self.setStyleSheet('''
            QWidget { font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif; }
            QMainWindow { background-color: #2b2b2b; }
            QListWidget { background-color: #1e1e1e; border: 1px solid #333; outline: none; padding: 10px; border-radius: 8px; }
            QListWidget::item { background-color: transparent; border: none; }
            QListWidget::item:selected { background-color: transparent; border: none; }
            
            QPushButton { 
                background-color: #3a3a3a; color: white; border: 1px solid #555; 
                padding: 12px 30px; font-size: 15px; border-radius: 6px; font-weight: bold;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton:pressed { background-color: #2a2a2a; }
            QPushButton:disabled { background-color: #2b2b2b; color: #777; border: 1px solid #444; }
            
            #CalcBtn:enabled { background-color: #007acc; border: 1px solid #005c99; }
            #CalcBtn:enabled:hover { background-color: #0088e6; }
            #CalcBtn:enabled:pressed { background-color: #005c99; }
        ''')
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Top banner / Title
        title_label = QLabel("项目列表")
        title_label.setStyleSheet("color: white; font-size: 20px; font-weight: bold; font-family: 'Microsoft YaHei', sans-serif;")
        layout.addWidget(title_label)
        
        # File List (Card Layout)
        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_widget.setGridSize(QSize(240, 300))
        self.list_widget.setUniformItemSizes(True)
        self.list_widget.setSpacing(0)
        self.list_widget.setMovement(QListWidget.Movement.Static)
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.NoSelection) # Rely purely on Card checkboxes
        layout.addWidget(self.list_widget)
        
        # Buttons Layout
        btn_layout = QHBoxLayout()
        
        self.calc_btn = QPushButton("计 算")
        self.calc_btn.setObjectName("CalcBtn")
        self.calc_btn.clicked.connect(self.on_calc_clicked)
        self.calc_btn.setEnabled(False)
        
        self.preview_btn = QPushButton("预 览")
        self.preview_btn.clicked.connect(self.on_preview_clicked)
        self.preview_btn.setEnabled(False)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.calc_btn)
        btn_layout.addWidget(self.preview_btn)
        
        layout.addLayout(btn_layout)
        
    def load_file_list(self):
        self.list_widget.clear()
        self.cards = []
        
        if not self.input_dir.exists():
            return
            
        extensions = io.get_supported_image_extensions()
        image_paths_set = set()
        for ext in extensions:
            for p in self.input_dir.glob(f"**/*{ext}"):
                image_paths_set.add(p.resolve())
                
        image_paths = sorted(list(image_paths_set))
        
        for img_path in image_paths:
            output_file = self.output_dir / f"{img_path.stem}.ply"
            has_ply = output_file.exists()
            
            card = CardWidget(img_path, has_ply)
            card.selection_changed.connect(self.on_selection_changed)
            self.cards.append(card)
            
            item = QListWidgetItem()
            # Must explicitly set the size hint to match our fixed widget size
            item.setSizeHint(card.sizeHint())
            item.setFlags(Qt.ItemFlag.ItemIsEnabled) # User cannot blue-highlight it
            
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, card)
            
        self.on_selection_changed()

    def on_selection_changed(self):
        selected_cards = [c for c in self.cards if c.checkbox.isChecked()]
        num_selected = len(selected_cards)
        
        # Calculation button logic
        self.calc_btn.setEnabled(num_selected > 0)
        
        # Preview button logic
        if num_selected == 1:
            self.preview_btn.setEnabled(selected_cards[0].has_ply)
        else:
            self.preview_btn.setEnabled(False)
            
    def on_calc_clicked(self):
        selected_cards = [c for c in self.cards if c.checkbox.isChecked()]
        if not selected_cards:
            return
            
        if not self.checkpoint_path.exists():
            QMessageBox.warning(self, "错误", f"找不到模型权重文件: {self.checkpoint_path}")
            return
            
        image_paths = [c.img_path for c in selected_cards]
        
        self.progress_dialog = DetailedProgressDialog(self)
        
        self.calc_worker = CalcWorker(image_paths, self.checkpoint_path, self.output_dir)
        self.calc_worker.log_msg.connect(self.progress_dialog.append_log)
        self.calc_worker.finished_calc.connect(self.on_calc_finished)
        self.calc_worker.error_occurred.connect(self.on_calc_error)
        
        self.calc_worker.start()
        self.progress_dialog.exec() # Blocks interaction with main window
        
    def on_calc_finished(self, total_time):
        self.progress_dialog.calculation_finished()
        self.load_file_list() # Refresh list to update colors
        
    def on_calc_error(self, err_msg):
        self.progress_dialog.append_log(f"发生错误:\n{err_msg}", "#FF0000", True)
        self.progress_dialog.calculation_finished()
        self.load_file_list()

    def on_preview_clicked(self):
        selected_cards = [c for c in self.cards if c.checkbox.isChecked()]
        if len(selected_cards) != 1:
            return
            
        img_path = selected_cards[0].img_path
        output_file = self.output_dir / f"{img_path.stem}.ply"
        
        if not output_file.exists():
            QMessageBox.warning(self, "错误", "该项目尚未生成 3D 模型 (.ply文件)。")
            return
            
        self.preview_dialog = PreviewDialog(str(output_file), self)
        self.preview_dialog.exec()

if __name__ == "__main__":
    # Optimize for PyTorch + PyQt integration avoiding context conflicts
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    app = QApplication(sys.argv)
    
    from PyQt6.QtGui import QFont
    font = QFont("Microsoft YaHei", 10)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
