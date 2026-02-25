# ML-SHARP Image to 3DGS Viewer - 简易使用说明书

这是一个针对 `ml-sharp` 官方库进行简化的个人工作流说明书。快速从二维图像生成 3D 高斯泼溅模型（3DGS, `.ply`），并支持交互式实时渲染出类似官方主页演示里的 3D 深度视差效果。

---

## 1. 批量推理生成 3D 模型
**核心脚本：** `run.py`

### 用法 / Usage
```bash
uv run python run.py
```

### 原理 / Principle
- **自动扫描 (Scanner):** 脚本会自动扫描 `data/input/` 文件夹下所有支持的格式且未被处理过的图片文件。
- **单图建模 (Inference):** 底层调用官方的预训练模型 `sharp_2572gikvuh.pt`，一步前馈直接从单张 2D 照片中推断并估计出完整的 3D 高斯（3D几何+颜色）数据。
- **批处理 (Efficiency):** 脚本会将2.6GB的神经网络模型仅向显存中加载一次，然后以队列的形式连续推理完所有的图片。它会自动跳过那些在 `data/output/` 下已经存在同名`.ply`结果的图片。该脚本未使用了视频生成渲染（`--render`）流程，只保存3D模型数据。

---

## 2. 实时 3D 视差查看互动器
**核心脚本：** `viewer.py`

### 用法 / Usage
```bash
uv run python viewer.py
```

### 原理 / Principle
- **交互界面 (Interactive UI):** 基于PyQt6编写。只需要双击程序左侧检测到的`.ply`文件，即可直接将推理完成的整个3D模型拉取到显卡显存里。
- **实时渲染 (Real-time Rendering):** 该程序并非播放视频，而是利用虚拟相机（`PinholeCameraModel`）实现了动态观察。将鼠标光标移动到画面中滑动，光标的`X`、`Y`轴坐标将转化为这台虚拟摄像机的物理位移（`dx`, `dy`）。
- **视差联动 (True 3D Parallax):** 借助CUDA光栅化渲染包 `gsplat.GSplatRenderer`，这套界面能够在 PyTorch 内部运算并实时重建新视角的场景，复现出真正的2.5D/3D相机漫游与视差效果。
