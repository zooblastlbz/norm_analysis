#!/bin/bash
# 示例脚本：分析多种模态的模型

echo "开始分析多种模态模型的embedding norm层..."

# 创建输出目录
mkdir -p results

echo "正在分析 KimiVL-A3B..."
python analyze_qwen_embeddings.py \
    --model "/ytech_m2v5_hdd/workspace/kling_mm/Models/Kimi-VL-A3B-Thinking-2506" \
    --output "results/kimivl_a3b_analysis.json" \
    --device auto
# 分析Qwen2.5系列
echo "正在分析 Qwen2.5-7B..."
python analyze_qwen_embeddings.py \
    --model "/pfs/Models/Qwen2.5-7B" \
    --output "results/qwen2.5_7b_analysis.json" \
    --device auto

echo "正在分析 Qwen2.5-7B-Instruct..."
python analyze_qwen_embeddings.py \
    --model "/pfs/Models/Qwen2.5-7B-Instruct" \
    --output "results/qwen2.5_7b_instruct_analysis.json" \
    --device auto

# 分析Qwen2.5-VL系列
echo "正在分析 Qwen2.5-VL-7B..."
python analyze_qwen_embeddings.py \
    --model "/pfs/Models/Qwen2.5-VL-7B-Instruct" \
    --output "results/qwen2.5_vl_7b_analysis.json" \
    --device auto

# 分析ViT模型
#echo "正在分析 ViT-Base..."
#python analyze_qwen_embeddings.py \
#    --model "google/vit-base-patch16-224" \
#    --output "results/vit_base_analysis.json" \
#    --device auto

# 分析SigLIP模型（仅norm层）
#echo "正在分析 SigLIP-Base..."
#python analyze_qwen_embeddings.py \
#    --model "google/siglip-base-patch16-224" \
#    --output "results/siglip_base_analysis.json" \
#    --device auto

# 分析Molmo VL模型
#echo "正在分析 Molmo-7B-D..."
#python analyze_qwen_embeddings.py \
#    --model "allenai/Molmo-7B-D-0924" \
#    --output "results/molmo_7b_analysis.json" \
#    --device auto

# 分析MoonVit模型
echo "正在分析 MoonVit..."
python analyze_qwen_embeddings.py \
    --model "/ytech_m2v5_hdd/workspace/kling_mm/Models/MoonViT-SO-400M" \
    --output "results/moonvit_analysis.json" \
    --device auto




echo "分析完成！结果保存在 results/ 目录中"