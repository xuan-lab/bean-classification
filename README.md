# 豆类图像分类模型

本项目包含一个用于分类豆类植物图像的深度学习模型。该模型能够将图像分类为三种类别：健康、锈病和角斑病。模型基于 ResNet18 架构，并使用 Hugging Face 的 `beans` 数据集进行训练。

## 项目结构

```
├── 27aa014ce09...          # 原始训练用数据集
├── bean_use.py              # 用于图像预测的主脚本
├── beans_classifier.pth     # 训练好的模型权重文件
├── resnet18-f37072fd.pth    # ResNet18 预训练模型权重
├── test.png                 # 测试图像
└── README.md                # 项目的说明文件
```

## 环境要求

- Python 3.6 或更高版本
- PyTorch
- torchvision
- Pillow

## 安装依赖

使用以下命令安装所需的 Python 包：

```bash
pip install torch torchvision pillow
```

## 使用方法

1. 确保你在项目目录中，并且所有文件都在相同的目录下。
2. 更新 `bean_use.py` 中的图像路径，指向你想要分类的图像。
3. 运行脚本进行预测：

```bash
python bean_use.py
```

4. 预测结果将显示在终端。

## 数据集

该项目使用的训练数据集为 Hugging Face 的 `beans` 数据集，包含健康、锈病和角斑病的豆类植物图像。请根据需要下载并使用该数据集进行模型训练。

## 许可证

该项目采用 MIT 许可证。有关更多详细信息，请参见 [LICENSE](LICENSE) 文件。

## 作者

- **姓名**: Xiexuan
- **邮箱**: xiexuan@njfu.edu.cn
