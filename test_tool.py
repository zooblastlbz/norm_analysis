#!/usr/bin/env python3
"""
多模态模型分析工具测试脚本
"""

import subprocess
import sys
import os

def test_model(model_name, model_type, output_file):
    """测试单个模型"""
    print(f"\n测试 {model_type} 模型: {model_name}")
    print("-" * 50)
    
    test_command = [
        sys.executable,
        "analyze_qwen_embeddings.py",
        "--model", model_name,
        "--device", "cpu",
        "--output", output_file
    ]
    
    try:
        print("正在运行测试命令...")
        print(" ".join(test_command))
        
        result = subprocess.run(
            test_command,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        if result.returncode == 0:
            print(f"✅ {model_type} 模型测试成功！")
            print("输出结果（前500字符）：")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        else:
            print(f"❌ {model_type} 模型测试失败！")
            print("错误信息：")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {model_type} 模型测试超时（5分钟）")
    except Exception as e:
        print(f"❌ {model_type} 模型测试过程中出错：{e}")

def main():
    """主测试函数"""
    print("开始测试多模态模型分析工具...")
    
    # 创建测试结果目录
    os.makedirs("test_results", exist_ok=True)
    
    # 测试用例 - 使用较小的模型进行快速测试
    test_cases = [
        {
            "model_name": "Qwen/Qwen2.5-0.5B",
            "model_type": "Qwen2.5",
            "output_file": "test_results/qwen_test.json"
        },
        {
            "model_name": "google/vit-base-patch16-224",
            "model_type": "ViT",
            "output_file": "test_results/vit_test.json"
        },
        {
            "model_name": "google/siglip-base-patch16-224",
            "model_type": "SigLIP",
            "output_file": "test_results/siglip_test.json"
        }
    ]
    
    # 运行测试
    success_count = 0
    total_tests = len(test_cases)
    
    for test_case in test_cases:
        try:
            test_model(
                test_case["model_name"], 
                test_case["model_type"], 
                test_case["output_file"]
            )
            success_count += 1
        except Exception as e:
            print(f"❌ {test_case['model_type']} 测试异常：{e}")
    
    # 测试总结
    print("\n" + "="*60)
    print(f"测试完成！成功: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 所有测试都通过了！")
    elif success_count > 0:
        print("⚠️  部分测试通过，请检查失败的测试")
    else:
        print("❌ 所有测试都失败了，请检查环境配置")

if __name__ == "__main__":
    main()