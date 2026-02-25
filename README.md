# ML-SHARP 桌面端应用使用说明

这是一个针对 `ml-sharp` 官方库进行封装和优化的图形化桌面端程序。用户可以通过现代化的 UI 界面快速完成2D图像到3D高斯泼溅模型（3DGS, `.ply`）的转换，并在软件内直接进行交互式视差预览。

---

## 快速开始

**核心程序：** `desktop_app.py`

### 1. 环境配置与模型下载

本项目推荐使用 `uv` 包管理器配置Python虚拟环境。

**配置安装：**
```bash
# 1. 全局安装 uv (若系统中已安装请跳过此步)
pip install uv

# 2. 一键创建虚拟环境并在其中安装所有相关依赖库
uv venv
uv pip install -r requirements.txt
```

**下载模型：**
在正式运行前，请确保提前下载好官方提供的预训练模型权重文件 `sharp_2572gikvuh.pt`，并将其直接放置于项目根目录下的 `model/` 文件夹中。

### 2. 启动程序
请在配置好环境的终端中运行以下命令启动桌面端：
```bash
# 如果使用 venv
.venv\Scripts\python.exe desktop_app.py

# 如果使用 uv
uv run python desktop_app.py
```

### 3. 准备数据
软件启动后，会自动读取 `data/input` 文件夹。请确保你已提前将需要处理的图像存入该目录内。
程序主界面中列出的卡片便是你的全部项目：
- 卡片会显示图像的预览图。
- 底部状态提示“未生成模型”表示图像尚未被处理。
- 底部状态提示“已生成模型 (.ply)”表示该图像此前已经成功转换出 3D 文件（保存在 `data/output` 中）。

### 4. 生成3D模型 (计算)
1. 勾选想要处理的卡片左上角的复选框，你可以单选也可以一次性多选。
2. 选中完成后，点击右下角的“计算”按钮。
3. 进度窗口将弹出，它会在后台加载显卡计算所需的模型权重 `sharp_2572gikvuh.pt` 并批量推断所选照片。
4. 进度窗口会详细输出每张照片的计算耗时，全部完成后自动保存到 `data/output` 文件夹。

### 5. 实时3D视差互动 (预览)
1. 在主界面的卡片中仅勾选1个已经标识为“已生成模型”的卡片项目。
2. 点击右下角的“预览”按钮，会弹出一个独立的互动视窗。
3. 将鼠标光标放置于该画面上方并晃动。鼠标如同控制着一个相机探头，左右上下移动你的鼠标即可预览景深视差效果。

## 关于ml-sharp

---

# Sharp Monocular View Synthesis in Less Than a Second

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://apple.github.io/ml-sharp/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.10685-b31b1b.svg)](https://arxiv.org/abs/2512.10685)

This software project accompanies the research paper: _Sharp Monocular View Synthesis in Less Than a Second_
by _Lars Mescheder, Wei Dong, Shiwei Li, Xuyang Bai, Marcel Santos, Peiyun Hu, Bruno Lecouat, Mingmin Zhen, Amaël Delaunoy,
Tian Fang, Yanghai Tsin, Stephan Richter and Vladlen Koltun_.

![](data/teaser.jpg)

We present SHARP, an approach to photorealistic view synthesis from a single image. Given a single photograph, SHARP regresses the parameters of a 3D Gaussian representation of the depicted scene. This is done in less than a second on a standard GPU via a single feedforward pass through a neural network. The 3D Gaussian representation produced by SHARP can then be rendered in real time, yielding high-resolution photorealistic images for nearby views. The representation is metric, with absolute scale, supporting metric camera movements. Experimental results demonstrate that SHARP delivers robust zero-shot generalization across datasets. It sets a new state of the art on multiple datasets, reducing LPIPS by 25–34% and DISTS by 21–43% versus the best prior model, while lowering the synthesis time by three orders of magnitude.

## Getting started

We recommend to first create a python environment:

```
conda create -n sharp python=3.13
```

Afterwards, you can install the project using

```
pip install -r requirements.txt
```

To test the installation, run

```
sharp --help
```

## Using the CLI

To run prediction:

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians
```

The model checkpoint will be downloaded automatically on first run and cached locally at `~/.cache/torch/hub/checkpoints/`.

Alternatively, you can download the model directly:

```
wget https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt
```

To use a manually downloaded checkpoint, specify it with the `-c` flag:

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians -c sharp_2572gikvuh.pt
```

The results will be 3D gaussian splats (3DGS) in the output folder. The 3DGS `.ply` files are compatible to various public 3DGS renderers. We follow the OpenCV coordinate convention (x right, y down, z forward). The 3DGS scene center is roughly at (0, 0, +z). When dealing with 3rdparty renderers, please scale and rotate to re-center the scene accordingly.

### Rendering trajectories (CUDA GPU only)

Additionally you can render videos with a camera trajectory. While the gaussians prediction works for all CPU, CUDA, and MPS, rendering videos via the `--render` option currently requires a CUDA GPU. The gsplat renderer takes a while to initialize at the first launch.

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians --render

# Or from the intermediate gaussians:
sharp render -i /path/to/output/gaussians -o /path/to/output/renderings
```

## Evaluation

Please refer to the paper for both quantitative and qualitative evaluations.
Additionally, please check out this [qualitative examples page](https://apple.github.io/ml-sharp/) containing several video comparisons against related work.

## Citation

If you find our work useful, please cite the following paper:

```bibtex
@inproceedings{Sharp2025:arxiv,
  title      = {Sharp Monocular View Synthesis in Less Than a Second},
  author     = {Lars Mescheder and Wei Dong and Shiwei Li and Xuyang Bai and Marcel Santos and Peiyun Hu and Bruno Lecouat and Mingmin Zhen and Ama\"{e}l Delaunoy and Tian Fang and Yanghai Tsin and Stephan R. Richter and Vladlen Koltun},
  journal    = {arXiv preprint arXiv:2512.10685},
  year       = {2025},
  url        = {https://arxiv.org/abs/2512.10685},
}
```

## Acknowledgements

Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details.

## License

Please check out the repository [LICENSE](LICENSE) before using the provided code and
[LICENSE_MODEL](LICENSE_MODEL) for the released models.
