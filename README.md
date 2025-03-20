# 文本摘要生成项目

这是一个基于 BART 模型的文本摘要生成项目，使用 PyTorch 和 Transformers 库实现。该项目包含模型训练和测试两个主要部分。

## 项目结构

.
├── dataset/
│ ├── train_data/
│ │ └── train.json
│ └── save_model/
│ └── model/
├── train.py
├── main.py
└── README.md


## 环境要求

- Python 3.6+
- PyTorch
- Transformers
- NLTK
- JSON

安装依赖：
```bash
pip install torch transformers nltk
```

## 数据格式

训练和测试数据应为 JSON 格式，结构如下：
```json
[
    {
        "input": "原文内容",
        "output": "摘要内容"
    },
    ...
]
```

## 使用说明

### 1. 训练模型

运行 `train.py` 来训练模型：

```bash
python train.py
```

训练过程包括：
- 加载 BART 预训练模型和分词器
- 处理训练数据
- 训练模型（默认 5 个 epochs）
- 保存模型到指定路径

主要参数：
- `max_len`：输入文本最大长度（默认 512）
- `summary_len`：摘要最大长度（默认 128）
- `batch_size`：批次大小（默认 4）
- `learning_rate`：学习率（默认 2e-5）

### 2. 测试模型

运行 `main.py` 来测试模型：

```bash
python main.py
```

测试过程包括：
- 加载训练好的模型
- 处理测试数据
- 生成摘要
- 计算 BLEU 分数
- 输出评估结果

测试输出包含：
- 原文内容
- 生成的摘要
- 参考摘要
- BLEU 分数
- 总体评估结果

## 主要功能

### train.py
- `SummaryDataset` 类：处理训练数据
- `train_model` 函数：执行模型训练
- 支持日志记录
- 自动保存模型和分词器

### main.py
- `load_test_data` 函数：加载测试数据
- `evaluate_model` 函数：评估模型性能
- 支持 GPU 加速（如果可用）
- 详细的评估输出

## 注意事项

1. 确保数据集格式正确
2. 检查模型保存路径是否正确
3. 根据实际需求调整模型参数
4. GPU 内存不足时可调整 batch size
5. 可以通过修改 `test_limit` 参数控制测试样本数量

## 性能评估

使用 BLEU 分数评估模型性能，输出包括：
- 每个样本的单独评分
- 整体平均分数
- 测试样本总数

## 可能的改进方向

1. 添加更多评估指标（如 ROUGE）
2. 实现交叉验证
3. 支持更多模型选项
4. 添加早停机制
5. 支持配置文件

## 许可证

[添加许可证信息]

## 联系方式

[qq:201900465]