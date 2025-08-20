#!/usr/bin/env python3
"""
分析Qwen2.5和Qwen2.5-VL系列模型的embedding norm层参数
"""

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import argparse
import os
from typing import Dict, List, Tuple
import warnings

# 忽略警告信息
warnings.filterwarnings("ignore")

class QwenEmbeddingAnalyzer:
    """Qwen模型embedding分析器"""
    
    def __init__(self, model_name: str, device: str = "auto"):
        """
        初始化分析器
        
        Args:
            model_name: 模型名称或路径
            device: 设备类型 (auto, cpu, cuda)
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        
    def _get_device(self, device: str) -> str:
        """获取可用设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """加载模型和分词器"""
        print(f"正在加载模型: {self.model_name}")
        print(f"使用设备: {self.device}")
        
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 加载模型
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device != "cpu" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            print(f"模型加载成功！")
            return True
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def find_embedding_norm_layers(self) -> Dict[str, torch.nn.Module]:
        """查找模型中embedding相关的norm层"""
        embedding_norm_layers = {}
        
        def _find_embedding_norms(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # 查找embedding相关的norm层
                is_norm_layer = any(norm_type in child.__class__.__name__.lower() for norm_type in 
                                  ['layernorm', 'rmsnorm', 'norm'])
                
                # 判断是否为embedding相关的norm层
                is_embedding_related = any(embed_keyword in full_name.lower() for embed_keyword in [
                    'embed', 'input_layernorm', 'input_norm', 'norm', 'post_attention_layernorm'
                ])
                
                if is_norm_layer and (is_embedding_related or 'embed' in full_name.lower()):
                    embedding_norm_layers[full_name] = child
                
                # 递归查找子模块
                _find_embedding_norms(child, full_name)
        
        _find_embedding_norms(self.model)
        
        # 如果没有找到明确的embedding norm，则查找所有norm层中可能相关的
        if not embedding_norm_layers:
            all_norm_layers = self.find_all_norm_layers()
            # 选择前几层的norm，通常embedding相关的norm在模型前部
            for layer_name, layer_module in list(all_norm_layers.items())[:10]:
                embedding_norm_layers[layer_name] = layer_module
        
        return embedding_norm_layers
    
    def find_all_norm_layers(self) -> Dict[str, torch.nn.Module]:
        """查找模型中的所有norm层（作为备用方法）"""
        norm_layers = {}
        
        def _find_norms(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # 查找LayerNorm、RMSNorm等norm层
                if any(norm_type in child.__class__.__name__.lower() for norm_type in 
                       ['layernorm', 'rmsnorm', 'norm']):
                    norm_layers[full_name] = child
                
                # 递归查找子模块
                _find_norms(child, full_name)
        
        _find_norms(self.model)
        return norm_layers

    def analyze_embedding_layer(self) -> Dict[str, any]:
        """分析embedding层本身的参数"""
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model()")
        
        embedding_stats = {}
        
        # 查找embedding层
        for name, module in self.model.named_modules():
            if 'embed' in name.lower() and hasattr(module, 'weight'):
                # 分析embedding权重
                weight = module.weight.data.cpu().numpy()
                
                # 计算每个embedding向量的norm
                embedding_norms = np.linalg.norm(weight, axis=1)
                
                embedding_stats[name] = {
                    'embedding_shape': list(weight.shape),
                    'vocab_size': weight.shape[0],
                    'embedding_dim': weight.shape[1],
                    'norm_mean': float(np.mean(embedding_norms)),
                    'norm_std': float(np.std(embedding_norms)),
                    'norm_min': float(np.min(embedding_norms)),
                    'norm_max': float(np.max(embedding_norms)),
                    'weight_mean': float(np.mean(weight)),
                    'weight_std': float(np.std(weight))
                }
        
        return embedding_stats

    def analyze_embedding_norms(self) -> Dict[str, Dict[str, float]]:
        """分析embedding相关的norm层"""
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model()")
        
        # 分析embedding层本身
        embedding_stats = self.analyze_embedding_layer()
        
        # 分析embedding相关的norm层
        norm_layers = self.find_embedding_norm_layers()
        norm_results = {}
        
        print(f"\n=== Embedding层分析 ===")
        for layer_name, stats in embedding_stats.items():
            print(f"Embedding层: {layer_name}")
            print(f"  形状: {stats['embedding_shape']}")
            print(f"  词汇表大小: {stats['vocab_size']}")
            print(f"  嵌入维度: {stats['embedding_dim']}")
            print(f"  每个embedding向量的norm均值: {stats['norm_mean']:.6f}")
            print(f"  每个embedding向量的norm标准差: {stats['norm_std']:.6f}")
            print(f"  每个embedding向量的norm最小值: {stats['norm_min']:.6f}")
            print(f"  每个embedding向量的norm最大值: {stats['norm_max']:.6f}")
            print(f"  权重参数均值: {stats['weight_mean']:.6f}")
            print(f"  权重参数标准差: {stats['weight_std']:.6f}")
            print("-" * 60)
        
        print(f"\n=== Embedding相关的Norm层分析 ===")
        print(f"找到 {len(norm_layers)} 个embedding相关的norm层:")
        print("-" * 60)
        
        for layer_name, layer_module in norm_layers.items():
            # 获取层的参数
            layer_stats = self._analyze_layer_parameters(layer_module)
            norm_results[layer_name] = layer_stats
            
            print(f"Norm层名称: {layer_name}")
            print(f"  参数类型: {layer_module.__class__.__name__}")
            print(f"  参数形状: {layer_stats['shape']}")
            print(f"  参数均值: {layer_stats['mean']:.6f}")
            print(f"  参数标准差: {layer_stats['std']:.6f}")
            print(f"  参数最小值: {layer_stats['min']:.6f}")
            print(f"  参数最大值: {layer_stats['max']:.6f}")
            print(f"  参数数量: {layer_stats['param_count']}")
            print("-" * 60)
        
        # 合并结果
        all_results = {
            'embedding_layers': embedding_stats,
            'norm_layers': norm_results
        }
        
        return all_results
    
    def _analyze_layer_parameters(self, layer: torch.nn.Module) -> Dict[str, float]:
        """分析单个层的参数统计信息"""
        stats = {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'shape': None,
            'param_count': 0
        }
        
        all_params = []
        param_count = 0
        
        for param_name, param in layer.named_parameters():
            if param.requires_grad:
                param_data = param.data.cpu().numpy().flatten()
                all_params.extend(param_data)
                param_count += param.numel()
                
                if stats['shape'] is None:
                    stats['shape'] = list(param.shape)
        
        if all_params:
            all_params = np.array(all_params)
            stats['mean'] = float(np.mean(all_params))
            stats['std'] = float(np.std(all_params))
            stats['min'] = float(np.min(all_params))
            stats['max'] = float(np.max(all_params))
            stats['param_count'] = param_count
        
        return stats
    
    def save_results(self, results: Dict[str, any], output_file: str):
        """保存分析结果到文件"""
        import json
        
        # 处理新的结果格式
        output_data = {
            'model_name': self.model_name,
            'device': self.device,
            'embedding_layers': results.get('embedding_layers', {}),
            'norm_layers': results.get('norm_layers', {}),
            'total_embedding_layers': len(results.get('embedding_layers', {})),
            'total_norm_layers': len(results.get('norm_layers', {}))
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {output_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="分析Qwen2.5系列模型的embedding norm层")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="模型名称或路径 (例如: Qwen/Qwen2.5-7B, Qwen/Qwen2.5-VL-7B)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="计算设备"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="qwen_embedding_analysis.json",
        help="输出文件路径"
    )
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = QwenEmbeddingAnalyzer(args.model, args.device)
    
    # 加载模型
    if not analyzer.load_model():
        print("模型加载失败，程序退出")
        return
    
    try:
        # 分析embedding norm层
        results = analyzer.analyze_embedding_norms()
        
        # 保存结果
        analyzer.save_results(results, args.output)
        
        # 打印总结
        print(f"\n分析完成！")
        print(f"模型: {args.model}")
        print(f"总共分析了 {len(results)} 个norm层")
        
    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()