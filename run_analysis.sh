#!/bin/bash
# 示例脚本：分析不同的Qwen模型

echo "开始分析Qwen2.5和Qwen2.5-VL系列模型的embedding norm层..."

# 创建输出目录
mkdir -p results

# 分析Qwen2.5-7B模型
echo "正在分析 Qwen2.5-7B..."
python analyze_qwen_embeddings.py \
    --model "Qwen/Qwen2.5-7B" \
    --output "results/qwen2.5_7b_analysis.json" \
    --device auto

# 分析Qwen2.5-VL-7B模型
echo "正在分析 Qwen2.5-VL-7B..."
python analyze_qwen_embeddings.py \
    --model "Qwen/Qwen2.5-VL-7B" \
    --output "results/qwen2.5_vl_7b_analysis.json" \
    --device auto

# 分析Qwen2.5-14B模型
echo "正在分析 Qwen2.5-14B..."
python analyze_qwen_embeddings.py \
    --model "Qwen/Qwen2.5-14B" \
    --output "results/qwen2.5_14b_analysis.json" \
    --device auto

echo "分析完成！结果保存在 results/ 目录中"