import sys
import os
import glob
import time

# IMPORT TORCH FIRST ON WINDOWS TO AVOID DLL CONFLICTS WITH PYQT
import torch

# ml-sharp dependencies
from sharp.utils import camera, gsplat
from sharp.utils.gaussians import load_ply

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QListWidget, QLabel, QSplitter, QMessageBox, QProgressDialog)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage

class RenderWorker(QObject):
    render_done = pyqtSignal(QImage)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.gaussians = None
        self.camera_model = None
        self.renderer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_offset = None
        self._is_rendering = False
        self._pending_request = None  # (dx_ratio, dy_ratio)

    def load_model(self, filepath):
        try:
            gaussians, metadata = load_ply(filepath)
            
            # Reduce resolution to ensure realtime performance on 4060
            width, height = metadata.resolution_px
            # Scale down to approx 720p height if original is very large
            if height > 720:
                scale = 720 / height
                width = int(width * scale)
                height = 720
            # Ensure even dimensions
            width = width + 1 if width % 2 != 0 else width
            height = height + 1 if height % 2 != 0 else height
            resolution_px = (width, height)
            
            f_px = metadata.focal_length_px * (height / metadata.resolution_px[1])
            
            intrinsics = torch.tensor([
                [f_px, 0, (width - 1) / 2.0, 0],
                [0, f_px, (height - 1) / 2.0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], device=self.device, dtype=torch.float32)

            params = camera.TrajectoryParams()
            self.max_offset = camera.compute_max_offset(gaussians, params, resolution_px, f_px)
            
            self.gaussians = gaussians.to(self.device)
            self.camera_model = camera.create_camera_model(
                self.gaussians, intrinsics, resolution_px=resolution_px
            )
            # Override to our resized resolution
            self.camera_model.screen_resolution_px = resolution_px
            
            self.renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)
            
            # Initial render
            self.request_render(0.0, 0.0)
            
        except Exception as e:
            self.error.emit(str(e))

    def request_render(self, dx_ratio, dy_ratio):
        self._pending_request = (dx_ratio, dy_ratio)
        if not self._is_rendering:
            self._process_render()

    def _process_render(self):
        if not self._pending_request or self.gaussians is None:
            return
            
        self._is_rendering = True
        dx_ratio, dy_ratio = self._pending_request
        self._pending_request = None
        
        try:
            # dx_ratio and dy_ratio are in [-1, 1]
            max_x, max_y, _ = self.max_offset
            dx = dx_ratio * max_x
            dy = dy_ratio * max_y
            
            # The coordinate system in camera.py: up is -Y in image space usually
            eye_pos = torch.tensor([-dx, -dy, 0.0], dtype=torch.float32)
            
            with torch.no_grad():
                camera_info = self.camera_model.compute(eye_pos)
                rendering_output = self.renderer(
                    self.gaussians,
                    extrinsics=camera_info.extrinsics[None].to(self.device),
                    intrinsics=camera_info.intrinsics[None].to(self.device),
                    image_width=camera_info.width,
                    image_height=camera_info.height,
                )
                
                color = rendering_output.color[0].permute(1, 2, 0)
                color = (color * 255.0).clamp(0, 255).to(dtype=torch.uint8).cpu().numpy()
            
            h, w, c = color.shape
            bytes_per_line = w * c
            qimg = QImage(color.data.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.render_done.emit(qimg)
            
        except Exception as e:
            print("Render error:", e)
        finally:
            self._is_rendering = False
            # Check if new requests came in while rendering
            if self._pending_request:
                # Use QTimer to schedule the next render and yield control to event loop
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(0, self._process_render)

class LoadWorker(QThread):
    finished_loading = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, render_worker, filepath):
        super().__init__()
        self.render_worker = render_worker
        self.filepath = filepath

    def run(self):
        try:
            self.render_worker.load_model(self.filepath)
            self.finished_loading.emit(self.filepath)
        except Exception as e:
            self.error.emit(str(e))

class ParallaxViewer(QLabel):
    def __init__(self, render_worker):
        super().__init__()
        self.render_worker = render_worker
        self.render_worker.render_done.connect(self.update_image)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("双击左侧列表中的 .ply 文件开始加载 3D 模型\\n加载完成后在画面上滑动鼠标查看实时视差")
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff; font-size: 16px;")
        self.is_loaded = False

    @pyqtSlot(QImage)
    def update_image(self, qimg):
        # Scale to fit label size while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(scaled_pixmap)

    def mouseMoveEvent(self, event):
        if not self.is_loaded:
            return
            
        # Map mouse position to [-1, 1]
        width = self.width()
        height = self.height()
        
        x = event.position().x()
        y = event.position().y()
        
        # Calculate ratio: center is 0, edges are -1 / +1
        # Mouse moving UP (y decreases) -> cursor represents eye moving UP -> view shifts accordingly
        dx_ratio = ((x / width) * 2.0) - 1.0
        dy_ratio = ((y / height) * 2.0) - 1.0
        
        # Clamp between -1 and 1
        dx_ratio = max(-1.0, min(1.0, dx_ratio))
        dy_ratio = max(-1.0, min(1.0, dy_ratio))
        
        self.render_worker.request_render(dx_ratio, dy_ratio)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-render to update scaling
        if self.is_loaded:
            self.render_worker.request_render(0.0, 0.0)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3DGS Real-time Parallax Viewer (ml-sharp)")
        self.resize(1200, 800)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QListWidget { background-color: #1e1e1e; color: #ffffff; border: 1px solid #333; font-size: 14px; }
            QListWidget::item { padding: 8px; }
            QListWidget::item:selected { background-color: #007acc; }
        """)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        splitter.addWidget(self.list_widget)

        # Setup render worker in main thread for simplicity, 
        # CUDA calls are fast enough and we only render when free
        self.render_worker = RenderWorker()
        self.viewer = ParallaxViewer(self.render_worker)
        splitter.addWidget(self.viewer)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        self.load_file_list()

    def load_file_list(self):
        data_dir = os.path.join("data", "output")
        if not os.path.exists(data_dir):
            return
            
        # Only load the actual 3DGS models
        ply_files = glob.glob(os.path.join(data_dir, "*.ply"))
        for f in ply_files:
            basename = os.path.basename(f)
            self.list_widget.addItem(basename)

    def on_item_double_clicked(self, item):
        filename = item.text()
        filepath = os.path.join("data", "output", filename)
        
        self.progress = QProgressDialog("正在将 3D 模型加载至 GPU，请稍候...", "取消", 0, 0, self)
        self.progress.setWindowTitle("加载中")
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setCancelButton(None)
        self.progress.show()

        self.viewer.is_loaded = False
        self.loader = LoadWorker(self.render_worker, filepath)
        self.loader.finished_loading.connect(self.on_load_finished)
        self.loader.error.connect(self.on_load_error)
        self.loader.start()

    def on_load_finished(self, filepath):
        if hasattr(self, 'progress'):
            self.progress.close()
        self.viewer.is_loaded = True
        self.viewer.setText("")

    def on_load_error(self, err_msg):
        if hasattr(self, 'progress'):
            self.progress.close()
        QMessageBox.critical(self, "错误", f"加载模型失败: {err_msg}")


if __name__ == "__main__":
    # Optimize for PyTorch + PyQt integration avoiding context conflicts
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
