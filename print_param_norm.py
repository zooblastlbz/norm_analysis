
import os
from turtle import pos
from safetensors.torch import load_file

glm_path="/ytech_m2v5_hdd/workspace/kling_mm/Models/gemma-3-4b-it/model-00001-of-00002.safetensors"

state_dict=load_file(glm_path)

norm_weight_vision=state_dict['multi_modal_projector.mm_input_projection_weight']

post_norm_vision=state_dict['vision_tower.vision_model.post_layernorm.weight']

post_norm_vision_bias=state_dict['vision_tower.vision_model.post_layernorm.bias']


input_norm_weight=state_dict['language_model.model.layers.0.input_layernorm.weight']

#post_nrom_bias=state_dict['model.visual.merger.post_projection_norm.bias']

def print_abs_mean_std(tensor,name):
    abs_mean=tensor.abs().mean().item()
    std=tensor.std().item()
    print(f"{name} abs mean: {abs_mean:.6f}, std: {std:.6f}")
    
def print_l2_norm_stats(tensor,name):
    l2_norm=tensor.norm(p=2).item()
    print(f"{name} L2 norm: {l2_norm:.6f}")
    
print_l2_norm_stats(norm_weight_vision,"norm_weight_vision")
print_abs_mean_std(norm_weight_vision,"norm_weight_vision")

print_l2_norm_stats(post_norm_vision,"post_norm_vision")
print_abs_mean_std(post_norm_vision,"post_norm_vision")

print_l2_norm_stats(post_norm_vision_bias,"post_norm_vision_bias")
print_abs_mean_std(post_norm_vision_bias,"post_norm_vision_bias")
    
print_l2_norm_stats(input_norm_weight,"input_norm_weight")
print_abs_mean_std(input_norm_weight,"input_norm_weight")