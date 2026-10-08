# Astra Scene

面向机器人场景变化的采集、局部3D Gaussian Splatting重建、区域融合和导航栅格候选生成。

本仓库整理了截至2026-10-08的项目源码、前端与测试。它是依赖外部GPU引擎、模型和已有地图的数据处理系统，不是下载后即可完成实机导航的独立程序。

## 目录

| 路径 | 用途 |
|---|---|
| `astra_scene/` | 场景处理、配准/几何、渲染及旧任务工作流核心 |
| `astra_mapping_lab/` | 采集界面、云任务服务、自动融合与栅格编排 |
| `astra_mapping_lab/static/` | 建图模式前端 |
| `tests/` | 核心回归测试 |
| `astra_mapping_lab/tests/` | 采集、云服务、融合与栅格测试 |
| `configs/` | 不含机器凭据的配置示例 |
| `docs/` | 安装说明、当前架构和验证记录 |

## 当前流程

数据包和ROI → 图像筛选/人物遮罩 → Spirula SfM与束调整 → RGB-D尺度检查 → 原地图配准 → Boxer/DeepSeek区域审核 → SH0重建 → 有界设备/地面融合 → 多视角审核 → 3DGS表面栅格 → 二值化 → 同设备三维支持补缝/参考净空检查 → 候选报告。

默认融合 `bounded_context_v2`，默认栅格后端 `gaussian_surface`，栅格版本 `object_clearance_binary_v1`。最终只有黑色不可通行和白色可通行；不确定证据在导出时归为不可通行。ROI限定允许更新范围，不代表整块替换。

**当前自动化止于候选生成。** 执行完成与质量/导航验收分开；尺度、机器人净空和独立回访尚未完成认证。此代码不会因报告完成就自动发布机器人导航地图。普通导航期间不应启用建图采集。

## 开始使用

见 [环境与配置](docs/SETUP.md)、[架构与版本](docs/ARCHITECTURE.md)、[验证记录](docs/VALIDATION.md)。

```bash
python -m pip install -r requirements.txt
python scripts/init_workspace.py
python -m unittest discover -s astra_mapping_lab/tests -p 'test_binary_grid.py'
python -m unittest discover -s astra_mapping_lab/tests -p 'test_object_clearance_grid.py'
```

GPU服务及完整测试建议在Linux运行；ROS采集需要既有ROS环境。仓库没有包含原始图像/视频、地图点云、训练产物、第三方模型/二进制、私有引擎及服务凭据。第三方软件应按各自许可单独安装。
