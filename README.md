# AI_Model_fine-tuning

基于 DistilBERT 的文本相似度匹配模型，用于在预定义的答案集中找到最匹配的结果。

## 项目结构

```
AI_Model_fine-tuning/
├── dataset/
│   ├── train_data/
│   │   ├── train.json      # 训练数据
│   │   └── model_info.json # 预定义答案集
│   ├── model/              # 预训练模型缓存目录
│   └── save_model/         # 训练后的模型保存目录
├── results/                # 训练过程中的检查点
├── logs/                   # 训练日志
├── train.py               # 模型训练脚本
└── predict.py             # 模型预测脚本
```

## 功能说明

- 基于 DistilBERT 预训练模型
- 使用文本相似度匹配方法
- 在预定义的答案集（model_info）中找到最匹配的结果
- 支持模型训练和预测功能

## 数据格式

### train.json
```json
[
    {
        "input": "输入文本",
        "output": "期望输出"
    }
]
```

### model_info.json
```json
[
    "可能的输出值1",
    "可能的输出值2",
    ...
]
```

## 使用方法

### 1. 环境配置
```bash
# 安装依赖
pip install transformers torch scikit-learn

# 如果需要上传到 GitHub，安装 Git LFS
# Windows
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
git lfs install

# Linux
sudo apt-get install git-lfs
git lfs install
```

### 2. Git LFS 配置
由于模型文件较大，需要使用 Git LFS 进行版本控制：

```bash
# 在项目根目录创建 .gitattributes 文件
git lfs track "dataset/model/**/*"
git lfs track "dataset/save_model/**/*"
git add .gitattributes
```

### 3. 训练模型
```bash
python train.py
```

### 4. 预测
```bash
python predict.py
```

## .gitignore 建议

建议创建 .gitignore 文件，内容如下：
```
# 缓存文件
__pycache__/
*.py[cod]

# 日志和结果
logs/
results/

# 如果不使用 Git LFS，也要忽略模型文件
# dataset/model/
# dataset/save_model/
```

## 训练参数

- 训练轮数：3 轮
- 批次大小：16
- 模型：distilbert-base-uncased
- 验证集比例：20%

## 模型说明

模型使用 DistilBERT 提取文本特征，通过计算相似度找到最匹配的预定义答案。主要特点：
- 轻量级 BERT 变体，运行更快
- 支持文本相似度匹配
- 可以处理不在训练集中的新输入

## 注意事项

1. 确保 dataset 目录结构正确
2. 训练数据需符合指定格式
3. 预测结果将从 model_info 中选择最相似的答案
4. 模型文件较大，请使用 Git LFS 进行版本控制
5. 首次运行会下载预训练模型，需要稳定的网络连接

## 性能优化建议

- 增加训练轮数以提高准确率
- 调整批次大小以适应内存
- 增加训练数据量以提升模型表现

## 维护者

[Miver]
