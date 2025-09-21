import torch
import numpy as np
from transformers import AutoModel

def calculate_rmsnorm_scale_for_llama_alignment(model_name="/ytech_m2v5_hdd/workspace/kling_mm/Models/Llama-3.2-3B-Instruct/"):
    """
    计算RMSNorm的scale参数，使输出向量L2范数与Llama 3.2 embedding范数均值相等
    """
    
    # 1. 获取Llama 3.2模型的embedding L2范数均值
    print(f"正在加载模型: {model_name}")
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
    
    embedding_weights = model.embed_tokens.weight.data  # shape: [vocab_size, hidden_dim]
    hidden_dim = embedding_weights.shape[1]
    
    # 计算非零embedding的L2范数均值
    embedding_norms = torch.norm(embedding_weights, dim=1)
    non_zero_mask = embedding_norms > 1e-8
    non_zero_norms = embedding_norms[non_zero_mask]
    target_l2_norm = non_zero_norms.mean().item()
    
    print(f"Hidden维度: {hidden_dim}")
    print(f"目标L2范数(embedding均值): {target_l2_norm:.6f}")
    
    # 2. RMSNorm数学推导
    print(f"\n=== RMSNorm Scale参数计算 ===")
    
    # RMSNorm公式: y = x * scale / sqrt(mean(x^2))
    # 我们要求: ||y||_2 = target_l2_norm
    
    # 对于输入向量x ~ N(0,1)^d (标准正态分布):
    # - mean(x^2) = 1 (标准正态分布的方差)
    # - ||x||_2的期望值 ≈ sqrt(d)
    
    # 推导过程:
    # ||y||_2 = ||x * scale / sqrt(mean(x^2))||_2
    #         = scale * ||x||_2 / sqrt(mean(x^2))
    #         = scale * ||x||_2 / sqrt(1)
    #         = scale * ||x||_2
    
    # 要使 ||y||_2 = target_l2_norm:
    # scale * ||x||_2 = target_l2_norm
    # scale = target_l2_norm / ||x||_2
    
    # 对于标准正态分布向量，||x||_2的期望值约为sqrt(d)
    expected_input_norm = np.sqrt(hidden_dim)
    optimal_scale = target_l2_norm / expected_input_norm
    
    print(f"输入向量期望L2范数: sqrt({hidden_dim}) = {expected_input_norm:.6f}")
    print(f"计算得到的最优scale: {target_l2_norm:.6f} / {expected_input_norm:.6f} = {optimal_scale:.6f}")
    
    # 3. 验证计算结果
    print(f"\n=== 验证计算结果 ===")
    
    def rmsnorm(x, scale):
        """RMSNorm函数"""
        rms = torch.sqrt(torch.mean(x ** 2) + 1e-8)  # 加小值避免除零
        return x * scale / rms
    
    # 测试多个随机向量
    num_tests = 1000
    output_norms = []
    
    for i in range(num_tests):
        # 生成标准正态分布向量
        x = torch.randn(hidden_dim)
        y = rmsnorm(x, optimal_scale)
        output_norm = torch.norm(y).item()
        output_norms.append(output_norm)
    
    mean_output_norm = np.mean(output_norms)
    std_output_norm = np.std(output_norms)
    
    print(f"测试{num_tests}个随机向量:")
    print(f"输出L2范数均值: {mean_output_norm:.6f}")
    print(f"输出L2范数标准差: {std_output_norm:.6f}")
    print(f"目标L2范数: {target_l2_norm:.6f}")
    print(f"误差: {abs(mean_output_norm - target_l2_norm):.6f}")
    print(f"相对误差: {abs(mean_output_norm - target_l2_norm) / target_l2_norm * 100:.2f}%")
    
    # 4. 返回结果
    results = {
        'target_l2_norm': target_l2_norm,
        'hidden_dim': hidden_dim,
        'optimal_scale': optimal_scale,
        'verification_mean': mean_output_norm,
        'verification_std': std_output_norm
    }
    
    print(f"\n=== 最终结果 ===")
    print(f"RMSNorm scale参数应该初始化为: {optimal_scale:.6f}")
    
    return results

if __name__ == "__main__":
    results = calculate_rmsnorm_scale_for_llama_alignment()
    
    # 示例用法
    print(f"\n=== 使用示例 ===")
    print(f"在PyTorch中初始化RMSNorm:")
    print(f"```python")
    print(f"import torch.nn as nn")
    print(f"")
    print(f"class RMSNorm(nn.Module):")
    print(f"    def __init__(self, hidden_dim, eps=1e-8):")
    print(f"        super().__init__()")
    print(f"        self.scale = nn.Parameter(torch.ones(hidden_dim) * {results['optimal_scale']:.6f})")
    print(f"        self.eps = eps")
    print(f"    ")
    print(f"    def forward(self, x):")
    print(f"        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)")
    print(f"        return x * self.scale / rms")
    print(f"```")