# JumpJump 项目

## 项目简介
这是一个自动玩跳跃游戏的项目，使用计算机视觉和机器学习来自动计算跳跃距离。

## 环境要求
- Window11
- Python 3.11

- 其他依赖项（见requirements.txt）

## 安装步骤
1. 克隆仓库：
```bash
git clone https://github.com/yourusername/jumpjump.git
```
2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法
1. 启动跳一跳电脑微信小程序游戏并运行主程序(训练了模型jump_jump_helper.py之后)：
```bash
python connectprogram.py
```
2. 程序会自动识别游戏界面并计算跳跃距离
  
4. 训练模型（可选）：
```bash
python jump_jump_helper.py --train
```

## 注意事项
- 确保游戏开始后窗口在屏幕可见区域（可以让vs小窗，调出微信小程序游戏跳一跳与他并列）
- 首次使用需要训练模型或下载预训练模型
- 在connectprogram.py 中 PRESS_COEFFICIENT根据个人电脑的分辨率来测试 (分辨率为2560x1440 缩放150% 大概为2.6)
  
## 贡献指南
欢迎提交Pull Request，请确保代码风格一致并通过测试。
