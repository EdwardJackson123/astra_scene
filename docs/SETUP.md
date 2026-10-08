# 环境与配置

## 基础测试环境

Python 3.10或更新版本；基础依赖见requirements.txt。依赖版本尚未锁定为跨机器复现环境，应根据本机平台验证。GPU/服务代码含fcntl等Linux接口，Windows仅适合部分纯栅格单元测试。

## 外部运行依赖

完整链路还需要NVIDIA GPU/CUDA、兼容的PyTorch/gsplat、Spirula可执行程序、VPS引擎、分割模型、已有地图和参考图库。ROS采集需要rclpy、sensor_msgs、message_filters及既有相机话题，不直接争抢相机设备。

外部模块`vps`由引擎提供，放在项目vendor/engine或配置PYTHONPATH；分割路径和Spirula路径见artifact_cache.py、reconstruct.py等运行入口。正式实验使用Spirula2026.9.20；10.6训练深度损失未设为默认。不同软件版本需要重新验证。

源码保留部分原部署路径，特别是astra_scene/pipeline.py中的CUDA路径/架构、PATH以及重建/分割运行时路径。新机器必须审核并配置这些路径，不能直接复制已有环境命令。仓库不提供第三方引擎/模型自动下载器。

## 地图注册

复制configs/map_update_registry.example.json到astra_mapping_lab/map_update_registry.json，填写项目根、Python解释器、地图PLY、指纹及参考数据路径。该机器配置已被.gitignore排除。示例中的占位符不能直接用于重建。

## 自有服务

令牌应由环境变量提供，长度至少32字符；不要写入Git。DeepSeek密钥按代码读取路径存入本地configs/secrets/deepseek.key，整个secrets目录不跟踪。

```bash
# 已配置ASTRA_CLOUD_TOKEN后，在可信本地网络运行
python astra_mapping_lab/cloud_server.py --root ./jobs --port 8030
# 独立采集页面，默认只绑定本机
python astra_mapping_lab/server.py --host 127.0.0.1 --port 8021 --data ./data
```

页面与云端连接仍需按cloud_client/server配置设置服务地址及令牌。不要公开暴露开发服务；未经验证不要修改已有机器人nav或停止其服务。此仓库没有自动启动/停止机器人导航的部署脚本。

## 初始化与完整测试

```bash
python scripts/init_workspace.py
python -m unittest discover -s tests
python -m unittest discover -s astra_mapping_lab/tests
```

采集端通过环境变量ASTRA_CLOUD_URL和ASTRA_CLOUD_TOKEN连接云端；未设置云端地址时会使用本地实验接口。初始化脚本只创建空运行目录，不写凭据、不下载模型。
