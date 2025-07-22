# 中文情感分类 (LSTM)

本项目基于 **PyTorch** 实现中文文本情感分类模型，使用 **LSTM + Global Max Pooling** 结构。
项目代码已按照最佳实践进行了模块化拆分，目录结构如下：

```
├── dataset/                # 数据集文件放置目录
├── outputs/                # 训练模型及日志输出目录
├── service/                # ONNX 模型转换与在线服务
│   ├── convert_to_onnx.py
│   └── onnx_server.py
├── src/                    # 业务核心代码
│   ├── __init__.py
│   ├── dataset.py          # 数据集定义
│   ├── model.py            # LSTM 模型
│   ├── train.py            # 训练脚本
│   └── utils.py            # 通用工具
├── requirements.txt        # 运行依赖
└── README.md
```

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 训练模型

```bash
python -m src.train --train ./data/usual_train.txt --dev ./data/usual_eval_labeled.txt --word2id ./data/word2id.json
```

3. 将最佳模型转换为 ONNX

```bash
python service/convert_to_onnx.py --checkpoint outputs/checkpoints/best.pt --vocab_size <VOCAB_SIZE>
```

4. 启动在线服务

```bash
python service/onnx_server.py
```

然后通过 `POST /predict` 发送形如 `[[1, 2, 3], [4, 5, 6]]` 的 `JSON` 数据即可获取预测结果。

## 依赖说明
详见 `requirements.txt`。