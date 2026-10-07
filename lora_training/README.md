# StarVector-1B 动物插画 LoRA 训练

这个目录把数据接口、PEFT LoRA 注入和训练入口集中起来，便于复盘，且不修改官方仓库内部源码。

## 文件

- `train_lora.py`：正式训练程序。读取 JSONL，完成图像预处理、PEFT LoRA 注入、官方内部 loss、梯度累积、验证和 checkpoint 保存。
- `run_smoke.sh`：只训练 2 个 optimizer step，每步验证并保存，用于修改代码后的快速检查。
- `run_train.sh`：正式训练入口，默认 3 epoch、batch size 1、梯度累积 8。

## 为什么调用内部 forward

仓库当前外层 `StarVectorForCausalLM.forward` 使用 Hugging Face 风格的生成参数，与官方 `train.py` 传入的 batch 字典不兼容。官方真正的训练前向在内部 `StarVectorBase.forward(batch)`，因此训练程序调用：

```python
loss = model.get_base_model().model(batch)
```

这个内部函数负责图像编码、投影、SVG token 化、attention mask、labels 和交叉熵 loss。LoRA 仍由 PEFT 注入同一批 StarCoder 线性层。

## LoRA 位置

24 层 StarCoder，每层挂载：

- `attn.c_attn`：QKV 联合投影
- `attn.c_proj`：注意力输出投影
- `mlp.c_fc`：FFN 扩张层
- `mlp.c_proj`：FFN 回投影层

共 96 个目标模块、192 个 LoRA A/B 张量。CLIP、图像投影层和基础 StarCoder 权重保持冻结。

## 数据

默认使用：

- `data/animal-lora/train.jsonl`
- `data/animal-lora/validation.jsonl`

每行至少包含 `image` 和 `svg`，训练监督目标是 target SVG 源码。

## 运行

先执行烟雾测试：

```bash
bash lora_training/run_smoke.sh
```

正式运行并写日志：

```bash
nohup bash lora_training/run_train.sh > outputs/animal-lora-train.log 2>&1 &
tail -f outputs/animal-lora-train.log
```

## 输出

- `outputs/animal-lora/checkpoint-XXXXXX/`：定期 Adapter 与 optimizer/scheduler/scaler 状态。
- `outputs/animal-lora/final-adapter/`：训练结束后的 PEFT Adapter。

基础模型不会被复制进每个 checkpoint，只保存 LoRA 参数及训练状态。
