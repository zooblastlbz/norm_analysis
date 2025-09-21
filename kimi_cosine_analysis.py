#!/usr/bin/env python3
"""
通用多模态语言模型层间余弦相似度和L2 Norm分析工具
支持Kimi-VL、Gemma-3等多种多模态模型的hidden states分析
计算每一层hidden states与前一层的余弦相似度，按层和token类型统计
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer,Gemma3Processor,Gemma3ForCausalLM
import argparse
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
from collections import defaultdict
from PIL import Image
import requests
import io

import os

# 忽略警告信息
warnings.filterwarnings("ignore")

class MultiModalMetricsAnalyzer:
    """通用多模态模型层间余弦相似度和L2 Norm分析器"""
    
    def __init__(self, model_name: str, device: str = "auto"):
        """
        初始化分析器
        
        Args:
            model_name: 模型名称或路径
            device: 设备类型 (auto, cpu, cuda)
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.model_type = self._detect_model_type(model_name)
        self.special_tokens = self._get_special_tokens()
        
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
    
    def _detect_model_type(self, model_name: str) -> str:
        """检测模型类型"""
        model_name_lower = model_name.lower()
        if "kimi" in model_name_lower:
            return "kimi"
        elif "gemma" in model_name_lower or "paligemma" in model_name_lower:
            return "gemma"
        elif "qwen" in model_name_lower:
            return "qwen"
        elif "llava" in model_name_lower:
            return "llava"
        else:
            return "generic"
    
    def _get_special_tokens(self) -> Dict[str, str]:
        """获取不同模型的特殊token"""
        token_map = {
            "kimi": {"image_token": "<|media_pad|>"},
            "gemma": {"image_token": "<image_soft_token>"},  # Gemma-3的图像token
            "qwen": {"image_token": "<|vision_start|>"},
            "llava": {"image_token": "<image>"},
            "generic": {"image_token": "<image>"}
        }
        return token_map.get(self.model_type, token_map["generic"])
    
    def load_model(self):
        """加载模型和处理器"""
        print(f"正在加载多模态模型: {self.model_name}")
        print(f"检测到模型类型: {self.model_type}")
        print(f"使用设备: {self.device}")
        
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 加载处理器
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True
                )
                print("处理器加载成功")
            except Exception as e:
                print(f"处理器加载失败，尝试仅使用tokenizer: {e}")
                self.processor = None
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto" if self.device != "cpu" else None,
                trust_remote_code=True,
                output_hidden_states=True  # 确保输出hidden states
            )
            
            # 设置为评估模式
            self.model.eval()
            
            print(f"模型加载成功！")
            print(f"图像token标记: {self.special_tokens['image_token']}")
            return True
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False

    def create_sample_dataset(self) -> List[Dict]:
        """创建示例数据集"""
        if self.model_type == "gemma":
            # Gemma-3的示例数据集
            dataset = [
                {
                    "text": "Describe what you see in these images.",
                    "image_paths": ["https://picsum.photos/224/224?random=1", "https://picsum.photos/224/224?random=2"]
                },
                {
                    "text": "What is the main subject of this image?",
                    "image_paths": ["https://picsum.photos/224/224?random=3"]
                },
                {
                    "text": "This is a text-only question: What is artificial intelligence?",
                    "image_paths": []
                }
            ]
        else:
            # 其他模型的示例数据集
            dataset = [
                {
                    "text": "Please describe the content in these images step by step.",
                    "image_paths": ["https://picsum.photos/224/224?random=1", "https://picsum.photos/224/224?random=2"]
                },
                {
                    "text": "分析这张图片中的内容。",
                    "image_paths": ["https://picsum.photos/224/224?random=3"]
                },
                {
                    "text": "这是一个纯文本的例子，没有图片。请回答这个问题：什么是人工智能？",
                    "image_paths": []
                }
            ]
        return dataset
    
    def load_images(self, image_paths: List[str]) -> List[Image.Image]:
        """加载多张图片"""
        images = []
        for path in image_paths:
            try:
                if path.startswith('http'):
                    response = requests.get(path, timeout=10)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content))
                else:
                    image = Image.open(path)
                images.append(image.convert('RGB'))
            except Exception as e:
                print(f"图片加载失败 {path}: {e}")
        return images
    
    def _prepare_inputs(self, text: str, image_data: Optional[str] = None) -> Optional[Dict]:
        """准备模型输入，支持不同模型架构"""
        if not self.model or not self.processor:
            print("模型或处理器未加载")
            return None
            
        try:
            # 处理图像
            image = None
            if image_data:
                try:
                    image = Image.open(image_data).convert('RGB')
                except Exception as e:
                    print(f"图像解码失败: {e}")
                    return None
            
            # 根据模型类型准备输入
            if self.model_type == "kimi":
                # Kimi-VL特殊处理
                if hasattr(self.processor, 'apply_chat_template'):
                    conversation = [{"role": "user", "content": text}]
                    text_input = self.processor.apply_chat_template(
                        conversation, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                else:
                    text_input = text
                
                if image:
                    inputs = self.processor(
                        text=text_input,
                        images=image,
                        return_tensors="pt"
                    ).to(self.device)
                else:
                    inputs = self.processor(
                        text=text_input,
                        return_tensors="pt"
                    ).to(self.device)
                    
            elif self.model_type == "gemma":
                # Gemma-3 (PaliGemma)处理
                if image:
                    inputs = self.processor(
                        text=text,
                        images=[image],
                        return_tensors="pt"
                    ).to(self.device)
                else:
                    # 纯文本输入
                    if self.tokenizer:
                        inputs = self.tokenizer(
                            text,
                            return_tensors="pt",
                            padding=True,
                            truncation=True
                        ).to(self.device)
                    else:
                        inputs = self.processor(
                            text=text,
                            return_tensors="pt"
                        ).to(self.device)
                        
            elif self.model_type == "qwen":
                # Qwen-VL处理
                if image:
                    inputs = self.processor(
                        text=text,
                        images=image,
                        return_tensors="pt"
                    ).to(self.device)
                else:
                    if self.tokenizer:
                        inputs = self.tokenizer(
                            text,
                            return_tensors="pt",
                            padding=True,
                            truncation=True
                        ).to(self.device)
                    else:
                        inputs = self.processor(
                            text=text,
                            return_tensors="pt"
                        ).to(self.device)
                        
            else:
                # 通用处理（LLaVA等）
                if image:
                    inputs = self.processor(
                        text=text,
                        images=image,
                        return_tensors="pt"
                    ).to(self.device)
                else:
                    if self.tokenizer:
                        inputs = self.tokenizer(
                            text,
                            return_tensors="pt",
                            padding=True,
                            truncation=True
                        ).to(self.device)
                    else:
                        inputs = self.processor(
                            text=text,
                            return_tensors="pt"
                        ).to(self.device)
            
            return inputs
            
        except Exception as e:
            print(f"输入准备失败: {e}")
            return None

    def _extract_hidden_states(self, inputs: Dict) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
        """提取隐藏状态"""
        if not self.model:
            print("模型未加载")
            return None, []
            
        try:
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # 检查输出格式
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    hidden_states = outputs.hidden_states
                    input_ids = inputs.get('input_ids')
                    return input_ids, list(hidden_states)
                else:
                    print("模型输出中没有找到hidden_states")
                    return None, []
                    
        except Exception as e:
            print(f"提取隐藏状态失败: {e}")
            return None, []

    def _identify_tokens(self, input_ids: torch.Tensor) -> Dict[str, List[int]]:
        """识别特殊token位置"""
        if not self.tokenizer:
            return {"text": list(range(input_ids.size(-1))), "image": []}
            
        try:
            # 获取token ID
            image_token_id = None
            if self.special_tokens['image_token']:
                try:
                    image_token_id = self.tokenizer.convert_tokens_to_ids(self.special_tokens['image_token'])
                except:
                    # 如果转换失败，尝试直接查找
                    vocab = getattr(self.tokenizer, 'vocab', {})
                    image_token_id = vocab.get(self.special_tokens['image_token'])
            
            # 将tensor转换为列表进行处理
            token_ids = input_ids.squeeze().cpu().tolist()
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            
            # 识别图像token位置
            image_positions = []
            if image_token_id is not None:
                image_positions = [i for i, token_id in enumerate(token_ids) if token_id == image_token_id]
            
            # 文本token位置（非图像token）
            text_positions = [i for i in range(len(token_ids)) if i not in image_positions]
            
            return {
                "text": text_positions,
                "image": image_positions
            }
            
        except Exception as e:
            print(f"Token识别失败: {e}")
            # 返回默认值
            total_tokens = input_ids.size(-1)
            return {"text": list(range(total_tokens)), "image": []}

    def _compute_cosine_similarity(self, hidden_states: List[torch.Tensor], 
                                 token_positions: Dict[str, List[int]]) -> Dict[str, List[float]]:
        """计算层间余弦相似度"""
        if len(hidden_states) < 2:
            return {"text": [], "image": []}
        
        similarities = {"text": [], "image": []}
        
        try:
            for i in range(len(hidden_states) - 1):
                current_layer = hidden_states[i]
                next_layer = hidden_states[i + 1]
                
                # 确保维度匹配
                if current_layer.shape != next_layer.shape:
                    continue
                
                # 计算文本token的相似度
                if token_positions["text"]:
                    text_indices = torch.tensor(token_positions["text"], device=current_layer.device)
                    if len(text_indices) > 0 and text_indices.max() < current_layer.size(1):
                        current_text = current_layer[0, text_indices, :].mean(dim=0)
                        next_text = next_layer[0, text_indices, :].mean(dim=0)
                        
                        text_sim = F.cosine_similarity(
                            current_text.unsqueeze(0), 
                            next_text.unsqueeze(0)
                        ).item()
                        similarities["text"].append(text_sim)
                    else:
                        similarities["text"].append(0.0)
                else:
                    similarities["text"].append(0.0)
                
                # 计算图像token的相似度
                if token_positions["image"]:
                    image_indices = torch.tensor(token_positions["image"], device=current_layer.device)
                    if len(image_indices) > 0 and image_indices.max() < current_layer.size(1):
                        current_image = current_layer[0, image_indices, :].mean(dim=0)
                        next_image = next_layer[0, image_indices, :].mean(dim=0)
                        
                        image_sim = F.cosine_similarity(
                            current_image.unsqueeze(0), 
                            next_image.unsqueeze(0)
                        ).item()
                        similarities["image"].append(image_sim)
                    else:
                        similarities["image"].append(0.0)
                else:
                    similarities["image"].append(0.0)
            
            return similarities
            
        except Exception as e:
            print(f"余弦相似度计算失败: {e}")
            return {"text": [], "image": []}

    def _compute_l2_norms(self, hidden_states: List[torch.Tensor], 
                          token_positions: Dict[str, List[int]]) -> Dict[str, List[float]]:
        """计算每层hidden states的L2 Norm"""
        if not hidden_states:
            return {"text": [], "image": []}
        
        norms = {"text": [], "image": []}
        
        try:
            for i, layer_hidden_state in enumerate(hidden_states):
                # 计算文本token的L2 Norm
                if token_positions["text"]:
                    text_indices = torch.tensor(token_positions["text"], device=layer_hidden_state.device)
                    if len(text_indices) > 0 and text_indices.max() < layer_hidden_state.size(1):
                        text_vectors = layer_hidden_state[0, text_indices, :]
                        text_norm = torch.norm(text_vectors, p=2, dim=-1).mean().item()
                        norms["text"].append(text_norm)
                    else:
                        norms["text"].append(0.0)
                else:
                    norms["text"].append(0.0)
                
                # 计算图像token的L2 Norm
                if token_positions["image"]:
                    image_indices = torch.tensor(token_positions["image"], device=layer_hidden_state.device)
                    if len(image_indices) > 0 and image_indices.max() < layer_hidden_state.size(1):
                        image_vectors = layer_hidden_state[0, image_indices, :]
                        image_norm = torch.norm(image_vectors, p=2, dim=-1).mean().item()
                        norms["image"].append(image_norm)
                    else:
                        norms["image"].append(0.0)
                else:
                    norms["image"].append(0.0)
            
            return norms
            
        except Exception as e:
            print(f"L2 Norm计算失败: {e}")
            return {"text": [], "image": []}

    def _compute_statistics(self, similarities: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """计算统计信息"""
        stats = {}
        
        for modality in ["text", "image"]:
            values = similarities[modality]
            if values:
                stats[modality] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values)
                }
            else:
                stats[modality] = {
                    "mean": 0.0, "std": 0.0, "min": 0.0, 
                    "max": 0.0, "median": 0.0
                }
        
        return stats

    def print_results(self, results: List[Dict]):
        """打印分析结果"""
        print("\n" + "="*80)
        print(f"多模态余弦相似度分析结果 - {self.model_name}")
        print("="*80)
        
        for i, result in enumerate(results):
            print(f"\n样本 {i+1}: {result['text'][:50]}...")
            print("-" * 60)
            
            # 文本模态结果
            text_stats = result["statistics"]["text"]
            print(f"文本模态统计:")
            print(f"  平均值: {text_stats['mean']:.4f}")
            print(f"  标准差: {text_stats['std']:.4f}")
            print(f"  最小值: {text_stats['min']:.4f}")
            print(f"  最大值: {text_stats['max']:.4f}")
            print(f"  中位数: {text_stats['median']:.4f}")
            
            # 图像模态结果
            if any(result["similarities"]["image"]):
                image_stats = result["statistics"]["image"]
                print(f"\n图像模态统计:")
                print(f"  平均值: {image_stats['mean']:.4f}")
                print(f"  标准差: {image_stats['std']:.4f}")
                print(f"  最小值: {image_stats['min']:.4f}")
                print(f"  最大值: {image_stats['max']:.4f}")
                print(f"  中位数: {image_stats['median']:.4f}")
            else:
                print(f"\n图像模态: 无图像数据")
            
            # 层间相似度
            if len(result["similarities"]["text"]) > 0:
                print(f"\n层间相似度变化:")
                for j, (text_sim, image_sim) in enumerate(zip(
                    result["similarities"]["text"], 
                    result["similarities"]["image"]
                )):
                    print(f"  层 {j}-{j+1}: 文本={text_sim:.4f}, 图像={image_sim:.4f}")

    def print_dataset_results(self, results: Dict):
        """打印数据集分析结果"""
        print("\n" + "="*100)
        print(f"多模态模型层间余弦相似度分析结果 - {self.model_name}")
        print(f"模型类型: {self.model_type}")
        print("="*100)
        
        # 核心二维矩阵结果
        print("\n【核心结果：各层各类型Token与前一层的余弦相似度及L2 Norm】")
        print("-" * 120)
        
        if 'layer_token_matrix' in results:
            print(f"{'层':<12} {'图像Token (余弦/L2 Norm)':<50} {'文本Token (余弦/L2 Norm)':<50}")
            print(f"{'':<12} {'均值±标准差(样本数)[范围]':<50} {'均值±标准差(样本数)[范围]':<50}")
            print("-" * 120)
            
            sorted_layers = sorted(results['layer_token_matrix'].keys())

            for layer_name in sorted_layers:
                layer_data = results['layer_token_matrix'][layer_name]
                
                # 图像token统计
                image_cos_stats = layer_data.get('image', {}).get('cosine', {})
                image_norm_stats = layer_data.get('image', {}).get('norm', {})
                
                image_str = ""
                if image_cos_stats.get('mean') is not None:
                    cos_str = f"Cos: {image_cos_stats['mean']:.3f}±{image_cos_stats['std']:.3f}({image_cos_stats['count']})"
                else:
                    cos_str = "Cos: 无数据"

                if image_norm_stats.get('mean') is not None:
                    norm_str = f"Norm: {image_norm_stats['mean']:.2f}±{image_norm_stats['std']:.2f}"
                else:
                    norm_str = "Norm: 无数据"
                image_str = f"{cos_str:<30} {norm_str:<20}"


                # 文本token统计
                text_cos_stats = layer_data.get('text', {}).get('cosine', {})
                text_norm_stats = layer_data.get('text', {}).get('norm', {})

                text_str = ""
                if text_cos_stats.get('mean') is not None:
                    cos_str = f"Cos: {text_cos_stats['mean']:.3f}±{text_cos_stats['std']:.3f}({text_cos_stats['count']})"
                else:
                    cos_str = "Cos: 无数据"

                if text_norm_stats.get('mean') is not None:
                    norm_str = f"Norm: {text_norm_stats['mean']:.2f}±{text_norm_stats['std']:.2f}"
                else:
                    norm_str = "Norm: 无数据"
                text_str = f"{cos_str:<30} {norm_str:<20}"
                
                print(f"{layer_name:<12} {image_str:<50} {text_str:<50}")
        
        # 按层汇总
        print(f"\n【按层汇总 (余弦相似度)】")
        print("-" * 50)
        for layer_name, stats in sorted(results['layer_averages'].items()):
            print(f"{layer_name}: {stats['mean']:.4f}±{stats['std']:.4f} (样本数:{stats['count']})")
        
        # 按token类型汇总
        print(f"\n【按Token类型汇总 (余弦相似度)】")
        print("-" * 50)
        for token_type, stats in results['token_type_averages'].items():
            type_name = f"图像Token({self.special_tokens['image_token']})" if token_type == "image" else "文本Token"
            print(f"{type_name}: {stats['mean']:.4f}±{stats['std']:.4f} (样本数:{stats['count']})")
        
        # 总体统计
        if 'summary' in results and results['summary']:
            print(f"\n【全局统计】")
            print("-" * 40)
            summary = results['summary']
            print(f"所有token所有层平均相似度: {summary['overall_mean']:.4f}")
            print(f"全局标准差: {summary['overall_std']:.4f}")
            print(f"总token比较次数: {summary['total_comparisons']}")
            print(f"分析的层数: {summary['num_layers']}")
        
        # 结果解读
        print(f"\n【结果解读】")
        print("-" * 40)
        print("• 余弦相似度：衡量token表示在层间变化的方向一致性。接近1.0表示方向稳定。")
        print("• L2 Norm：衡量token表示的向量长度或“强度”。")
        print("• 图像Token vs 文本Token：观察多模态信息在不同层的处理差异。")
        print(f"• {self.special_tokens['image_token']}：{self.model_type}模型的图像token标记。")

    def save_results(self, results: List[Dict], output_file: str):
        """保存结果到文件"""
        try:
            # 准备保存的数据
            save_data = {
                "model_name": self.model_name,
                "model_type": self.model_type,
                "device": str(self.device),
                "total_samples": len(results),
                "results": results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n结果已保存到: {output_file}")
            
        except Exception as e:
            print(f"保存结果失败: {e}")

    def analyze_dataset(self, dataset: List[Dict]) -> Dict:
        """分析整个数据集"""
        print(f"\n开始分析数据集，共{len(dataset)}个样本...")
        
        # 二维统计：layer x token_type
        layer_token_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        all_layer_similarities = defaultdict(list)
        all_token_type_similarities = defaultdict(list)
        
        for idx, sample in enumerate(dataset):
            print(f"\n{'='*50}")
            print(f"处理样本 {idx + 1}/{len(dataset)}")
            print(f"{'='*50}")
            
            # 处理输入数据
            text = sample.get('text', '')

            image_path = sample.get('image' )
            
            if image_path:
            # 构建messages格式
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path} 
                        ] + [{"type": "text", "text": text}],
                    },
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": text,
                    },
                ]
                
                
            if "gemma" in self.model_type:
                inputs =self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                    )
            else:
                inputs=self.processor.apply_chat_template(
                            messages, 
                            return_tensors="pt",
                        )
 
            # 如果有image_paths，加载第一张图片并转换为base64
            

            

            
            if inputs is None:
                print(f"样本{idx + 1}: 输入准备失败，跳过")
                continue
            
            # 提取隐藏状态
            input_ids, hidden_states = self._extract_hidden_states(inputs)
            #print(f"input_ids shape: {input_ids}")
            if len(hidden_states) == 0:
                print(f"样本{idx + 1}: 无法获取hidden states，跳过")
                continue
            
            # 识别token位置
            token_positions = self._identify_tokens(input_ids) if input_ids is not None else {"text": [], "image": []}
            print(token_positions)
            
            # 计算余弦相似度
            similarities = self._compute_cosine_similarity(hidden_states, token_positions)
            
            # 计算L2 Norm
            norms = self._compute_l2_norms(hidden_states, token_positions)

            # 统计结果 - 按层和token类型分别统计
            for layer_idx in range(len(similarities["text"])):
                # 文本token相似度
                if layer_idx < len(similarities["text"]):
                    text_sim = similarities["text"][layer_idx]
                    layer_token_metrics[layer_idx]["text"]["cosine"].append(text_sim)
                    all_token_type_similarities["text"].append(text_sim)
                    all_layer_similarities[layer_idx].append(text_sim)
                
                # 图像token相似度
                if layer_idx < len(similarities["image"]) and similarities["image"][layer_idx] != 0.0:
                    image_sim = similarities["image"][layer_idx]
                    layer_token_metrics[layer_idx]["image"]["cosine"].append(image_sim)
                    all_token_type_similarities["image"].append(image_sim)
            
            # 统计L2 Norm
            for layer_idx in range(len(norms["text"])):
                if layer_idx < len(norms["text"]):
                    layer_token_metrics[layer_idx]["text"]["norm"].append(norms["text"][layer_idx])
                if layer_idx < len(norms["image"]) and norms["image"][layer_idx] != 0.0:
                    layer_token_metrics[layer_idx]["image"]["norm"].append(norms["image"][layer_idx])

        # 计算统计结果
        results = self._compute_dataset_statistics(layer_token_metrics, 
                                                 all_layer_similarities, 
                                                 all_token_type_similarities)
        
        return results
    
    def _compute_dataset_statistics(self, layer_token_metrics, all_layer_similarities, all_token_type_similarities):
        """计算数据集统计结果"""
        results = {
            'layer_token_matrix': {},  # 二维矩阵：[layer][token_type]
            'layer_averages': {},      # 按层统计
            'token_type_averages': {}, # 按token类型统计
            'summary': {}
        }
        
        # 二维矩阵统计
        num_layers = len(layer_token_metrics)
        for layer_idx in range(num_layers):
            
            if layer_idx + 1 < num_layers:
                layer_name = f'L{layer_idx+1}_vs_L{layer_idx}'
                cos_layer_idx = layer_idx
            else: # 最后一层只有norm
                layer_name = f'L{layer_idx}'
                cos_layer_idx = -1


            results['layer_token_matrix'][layer_name] = {}
            
            for token_type in ['image', 'text']:
                results['layer_token_matrix'][layer_name][token_type] = {}
                
                # Cosine Similarity
                if cos_layer_idx != -1 and \
                   token_type in layer_token_metrics[cos_layer_idx] and \
                   "cosine" in layer_token_metrics[cos_layer_idx][token_type] and \
                   layer_token_metrics[cos_layer_idx][token_type]["cosine"]:
                    
                    similarities = layer_token_metrics[cos_layer_idx][token_type]["cosine"]
                    results['layer_token_matrix'][layer_name][token_type]['cosine'] = {
                        'mean': float(np.mean(similarities)),
                        'std': float(np.std(similarities)),
                        'count': len(similarities),
                        'min': float(np.min(similarities)),
                        'max': float(np.max(similarities)),
                    }

                # L2 Norm
                if token_type in layer_token_metrics[layer_idx] and \
                   "norm" in layer_token_metrics[layer_idx][token_type] and \
                   layer_token_metrics[layer_idx][token_type]["norm"]:

                    norms = layer_token_metrics[layer_idx][token_type]["norm"]
                    results['layer_token_matrix'][layer_name][token_type]['norm'] = {
                        'mean': float(np.mean(norms)),
                        'std': float(np.std(norms)),
                        'count': len(norms),
                        'min': float(np.min(norms)),
                        'max': float(np.max(norms)),
                    }

        # 按层平均 (cosine)
        for layer_idx, similarities in all_layer_similarities.items():
            layer_name = f'L{layer_idx+1}_vs_L{layer_idx}'
            results['layer_averages'][layer_name] = {
                'mean': float(np.mean(similarities)),
                'std': float(np.std(similarities)),
                'count': len(similarities)
            }
        
        # 按token类型平均 (cosine)
        for token_type, similarities in all_token_type_similarities.items():
            if similarities:
                results['token_type_averages'][token_type] = {
                    'mean': float(np.mean(similarities)),
                    'std': float(np.std(similarities)),
                    'count': len(similarities)
                }
        
        # 总体统计
        all_similarities = []
        for similarities in all_layer_similarities.values():
            all_similarities.extend(similarities)
        
        if all_similarities:
            results['summary'] = {
                'overall_mean': float(np.mean(all_similarities)),
                'overall_std': float(np.std(all_similarities)),
                'total_comparisons': len(all_similarities),
                'num_layers': len(all_layer_similarities)
            }
        
        return results

def main():
    parser = argparse.ArgumentParser(description='多模态模型层间余弦相似度与L2 Norm分析')
    parser.add_argument('--model', type=str, default="/ytech_m2v5_hdd/workspace/kling_mm/Models/gemma-3-4b-it", help='模型名称或路径')
    parser.add_argument('--data', type=str, default='sample_data.json', help='数据集文件路径')
    parser.add_argument('--output', type=str, help='输出文件路径（可选）')
    parser.add_argument('--device', type=str, default='cpu', help='设备选择')
    parser.add_argument('--max-samples', type=int, default=10, help='最大样本数量')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = MultiModalMetricsAnalyzer(args.model, args.device)
    
    # 加载模型
    print("正在加载模型...")
    if not analyzer.load_model():
        print("模型加载失败")
        return
    
    # 准备数据集
    try:
        # 尝试加载自定义数据集
        if  os.path.exists(args.data):
            print(f"正在加载数据集: {args.data}")
            with open(args.data, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        else:
            print("使用示例数据集")
            dataset = analyzer.create_sample_dataset()
        
        # 限制样本数量
        if len(dataset) > args.max_samples:
            dataset = dataset[:args.max_samples]
            print(f"限制数据集大小为 {args.max_samples} 个样本")
            
    except Exception as e:
        print(f"数据集加载失败: {e}")
        print("使用示例数据集")
        dataset = analyzer.create_sample_dataset()
    
    # 运行分析
    print("开始分析...")
    results = analyzer.analyze_dataset(dataset)
    
    if results:
        # 使用新的打印方法显示数据集分析结果
        analyzer.print_dataset_results(results)
        
        # 保存结果
        if args.output:
            output_file = args.output
        else:
            # 默认输出文件名
            model_safe_name = args.model.replace('/', '_').replace('\\', '_')
            output_file = f"multimodal_dataset_analysis_{model_safe_name}.json"
        
        try:
            # 保存数据集分析结果
            save_data = {
                "model_name": analyzer.model_name,
                "model_type": analyzer.model_type,
                "device": str(analyzer.device),
                "total_samples": len(dataset),
                "analysis_results": results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n数据集分析结果已保存到: {output_file}")
            
        except Exception as e:
            print(f"保存结果失败: {e}")
    else:
        print("分析失败或无结果")

if __name__ == "__main__":
    main()