# Qwen模型Embedding层Norm分析工具

这个工具专门用于分析Qwen2.5和Qwen2.5-VL系列模型中embedding层的norm值以及embedding相关的norm层参数。

## 功能特性

- **Embedding层分析**：计算每个embedding向量的L2 norm的统计信息（均值、标准差、最小值、最大值）
- **Embedding相关Norm层分析**：自动识别与embedding相关的norm层（如input_layernorm等）
- **详细统计信息**：提供embedding权重和norm层参数的完整统计分析
- **多设备支持**：支持CPU、CUDA、MPS设备
- **结构化输出**：生成详细的JSON报告

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python analyze_qwen_embeddings.py --model "Qwen/Qwen2.5-7B"
```

### 完整参数

```bash
python analyze_qwen_embeddings.py \
    --model "Qwen/Qwen2.5-7B" \
    --device auto \
    --output "embedding_analysis_results.json"
```

### 参数说明

- `--model`: 模型名称或本地路径（必需）
  - 支持的模型：Qwen/Qwen2.5-7B, Qwen/Qwen2.5-14B, Qwen/Qwen2.5-VL-7B等
- `--device`: 计算设备（可选，默认auto）
  - auto: 自动选择最佳设备
  - cpu: 使用CPU
  - cuda: 使用CUDA GPU
  - mps: 使用Apple Silicon GPU
- `--output`: 输出文件路径（可选，默认qwen_embedding_analysis.json）

## 批量分析

使用提供的脚本批量分析多个模型：

```bash
chmod +x run_analysis.sh
./run_analysis.sh
```

## 输出格式

工具会输出两类分析结果：

### 1. Embedding层分析
- **形状信息**：词汇表大小和嵌入维度
- **Norm统计**：每个embedding向量的L2 norm的均值、标准差、最小值、最大值
- **权重统计**：embedding权重参数的统计信息

### 2. Embedding相关Norm层分析
- **层名称和类型**：RMSNorm、LayerNorm等
- **参数统计**：norm层参数的均值、标准差、最小值、最大值
- **参数形状和数量**

## 示例输出

```
=== Embedding层分析 ===
Embedding层: embed_tokens
  形状: [152064, 4096]
  词汇表大小: 152064
  嵌入维度: 4096
  每个embedding向量的norm均值: 0.707124
  每个embedding向量的norm标准差: 0.089456
  每个embedding向量的norm最小值: 0.234567
  每个embedding向量的norm最大值: 1.123456
  权重参数均值: 0.000123
  权重参数标准差: 0.012345
------------------------------------------------------------

=== Embedding相关的Norm层分析 ===
找到 3 个embedding相关的norm层:
------------------------------------------------------------
Norm层名称: input_layernorm
  参数类型: RMSNorm
  参数形状: [4096]
  参数均值: 1.000023
  参数标准差: 0.000456
  参数最小值: 0.998234
  参数最大值: 1.001789
  参数数量: 4096
------------------------------------------------------------
```

## 支持的模型

- **Qwen2.5系列**：Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-7B, Qwen2.5-14B, Qwen2.5-32B, Qwen2.5-72B
- **Qwen2.5-VL系列**：Qwen2.5-VL-2B, Qwen2.5-VL-7B, Qwen2.5-VL-72B

## 分析内容说明

### Embedding向量Norm
- 计算每个token的embedding向量的L2范数
- 提供norm值的分布统计，帮助了解embedding的规模和分布特性

### Embedding相关Norm层
- 自动识别与embedding处理相关的normalization层
- 分析这些层的参数分布，了解模型的归一化特性

## 注意事项

1. **内存需求**：大型模型需要足够的GPU内存或系统内存
2. **网络连接**：首次运行需要下载模型文件
3. **设备选择**：对于大模型建议使用GPU以提高处理速度
4. **结果解读**：embedding norm的均值反映了词向量的平均大小，标准差反映了不同词向量大小的差异程度