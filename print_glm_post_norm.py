
import os
from turtle import pos
from safetensors.torch import load_file

glm_path="/ytech_m2v5_hdd/workspace/kling_mm/Models/GLM-4.5V/model-00046-of-00046.safetensors"

state_dict=load_file(glm_path)

post_norm_weight=state_dict['model.visual.merger.post_projection_norm.weight']

post_nrom_bias=state_dict['model.visual.merger.post_projection_norm.bias']

def print_abs_mean_std(tensor,name):
    abs_mean=tensor.abs().mean().item()
    std=tensor.std().item()
    print(f"{name} abs mean: {abs_mean:.6f}, std: {std:.6f}")
    
def print_l2_norm_stats(tensor,name):
    l2_norm=tensor.norm(p=2).item()
    print(f"{name} L2 norm: {l2_norm:.6f}")
    
print_l2_norm_stats(post_norm_weight,"post_norm_weight")
print_abs_mean_std(post_norm_weight,"post_norm_weight")
print_l2_norm_stats(post_nrom_bias,"post_nrom_bias")
print_abs_mean_std(post_nrom_bias,"post_nrom_bias")
    
print("post_norm_weight[:100]:",post_norm_weight[:100])
print("post_nrom_bias[:100]:",post_nrom_bias[:100])

print("post_norm_weight min:",post_norm_weight.min().item())
print("post_norm_weight max:",post_norm_weight.max().item())    

print("post_nrom_bias min:",post_nrom_bias.min().item())
print("post_nrom_bias max:",post_nrom_bias.max().item())