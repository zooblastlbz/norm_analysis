# 多模态模型Embedding层Norm分析工具

这个工具支持分析多种模型（Qwen2.5、Qwen2.5-VL、ViT、SigLIP、Molmo VL）的embedding层norm值以及相关norm层参数。

## 功能特性

- **多模型支持**：支持Qwen2.5、Qwen2.5-VL、ViT、SigLIP、Molmo VL等多种模型
- **智能模型检测**：自动检测模型类型并采用相应的分析策略
- **Embedding层分析**：计算每个embedding向量的L2 norm统计信息（均值、标准差、最小值、最大值）
- **Norm层分析**：分析norm层参数的绝对值均值和其他统计信息
- **SigLIP特殊支持**：对于SigLIP模型，专门分析所有norm层
- **多设备支持**：支持CPU、CUDA、MPS设备
- **结构化输出**：生成详细的JSON报告

## 支持的模型类型

### 1. Qwen2.5系列
- **模型**：Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-7B, Qwen2.5-14B, Qwen2.5-32B, Qwen2.5-72B
- **分析内容**：embedding层 + embedding相关norm层

### 2. Qwen2.5-VL系列
- **模型**：Qwen2.5-VL-2B, Qwen2.5-VL-7B, Qwen2.5-VL-72B
- **分析内容**：embedding层 + embedding相关norm层

### 3. ViT（Vision Transformer）
- **模型**：google/vit-base-patch16-224, google/vit-large-patch16-224等
- **分析内容**：patch embedding、position embedding + 相关norm层

### 4. SigLIP
- **模型**：google/siglip-base-patch16-224, google/siglip-large-patch16-256等
- **分析内容**：**仅分析所有norm层**（按照用户要求）

### 5. Molmo VL
- **模型**：allenai/Molmo-7B-D-0924等
- **分析内容**：embedding层 + embedding相关norm层

### 6. MoonVit
- **模型**：MoonVit/moonvit-base-patch16-224等
- **分析内容**：patch embedding、position embedding + 相关norm层

### 7. KimiVL-A3B  
- **模型**：kimivl-A3B等
- **分析内容**：embedding层 + embedding相关norm层

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
# 分析Qwen模型
python analyze_qwen_embeddings.py --model "Qwen/Qwen2.5-7B"

# 分析ViT模型
python analyze_qwen_embeddings.py --model "google/vit-base-patch16-224"

# 分析SigLIP模型（仅norm层）
python analyze_qwen_embeddings.py --model "google/siglip-base-patch16-224"

# 分析Molmo模型
python analyze_qwen_embeddings.py --model "allenai/Molmo-7B-D-0924"
```

### 完整参数

```bash
python analyze_qwen_embeddings.py \
    --model "google/siglip-base-patch16-224" \
    --device auto \
    --output "siglip_analysis_results.json"
```

### 参数说明

- `--model`: 模型名称或本地路径（必需）
- `--device`: 计算设备（可选，默认auto）
  - auto: 自动选择最佳设备
  - cpu: 使用CPU
  - cuda: 使用CUDA GPU
  - mps: 使用Apple Silicon GPU
- `--output`: 输出文件路径（可选，默认embedding_analysis.json）

## 批量分析

使用提供的脚本批量分析多个模型：

```bash
chmod +x run_analysis.sh
./run_analysis.sh
```

## 输出格式

根据模型类型，工具会输出不同的分析结果：

### 对于SigLIP模型
仅输出norm层分析：
- **层名称和类型**：LayerNorm、GroupNorm等
- **参数统计**：均值、**绝对值均值**、标准差、最小值、最大值
- **参数形状和数量**

### 对于其他模型
输出embedding层和norm层分析：

#### 1. Embedding层分析
- **形状信息**：词汇表大小/特征数量和嵌入维度
- **Norm统计**：每个embedding向量的L2 norm的均值、标准差、最小值、最大值
- **权重统计**：embedding权重参数的统计信息（包括绝对值均值）

#### 2. Embedding相关Norm层分析
- **层名称和类型**：RMSNorm、LayerNorm等
- **参数统计**：norm层参数的均值、**绝对值均值**、标准差、最小值、最大值
- **参数形状和数量**

## 示例输出

### SigLIP模型输出示例
```
=== 模型类型: SIGLIP ===

=== SigLIP Norm层分析 ===
找到 25 个norm层:
------------------------------------------------------------
Norm层名称: vision_model.encoder.layers.0.layer_norm1
  参数类型: LayerNorm
  参数形状: [768]
  参数均值: 1.000023
  参数绝对值均值: 1.000023
  参数标准差: 0.000456
  参数最小值: 0.998234
  参数最大值: 1.001789
  参数数量: 768
------------------------------------------------------------
```

### 其他模型输出示例
```
=== 模型类型: VIT ===

=== Embedding层分析 ===
Embedding层: embeddings.patch_embeddings.projection
  形状: [768, 3, 16, 16]
  权重参数均值: 0.000123
  权重参数绝对值均值: 0.042156
  权重参数标准差: 0.067234
------------------------------------------------------------

=== Embedding相关的Norm层分析 ===
找到 3 个embedding相关的norm层:
------------------------------------------------------------
Norm层名称: layernorm
  参数类型: LayerNorm
  参数形状: [768]
  参数均值: 1.000023
  参数绝对值均值: 1.000023
  参数标准差: 0.000456
  参数数量: 768
------------------------------------------------------------
```

## 重要特性说明

### 绝对值均值分析
- 对于所有norm层的scale参数，工具计算**绝对值的均值**而不是直接均值
- 这有助于理解norm层参数的实际规模，避免正负值相互抵消的影响

### 模型特定处理
- **SigLIP**：按照要求仅分析norm层，不分析embedding层
- **ViT**：重点分析patch embedding和position embedding相关的norm
- **Qwen/Molmo**：分析文本embedding和相关的input norm层

## 注意事项

1. **内存需求**：大型模型需要足够的GPU内存或系统内存
2. **网络连接**：首次运行需要下载模型文件
3. **设备选择**：对于大模型建议使用GPU以提高处理速度
4. **模型检测**：工具会自动检测模型类型，无需手动指定