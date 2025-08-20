#!/usr/bin/env python3
"""
简单测试脚本：使用较小的模型进行测试
"""

import subprocess
import sys
import os

def test_with_small_model():
    """使用较小的模型进行测试"""
    print("开始测试embedding分析工具...")
    
    # 使用Qwen2.5-0.5B进行测试（较小的模型）
    test_command = [
        sys.executable,
        "analyze_qwen_embeddings.py",
        "--model", "Qwen/Qwen2.5-0.5B",
        "--device", "cpu",
        "--output", "test_results.json"
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
            print("✅ 测试成功！")
            print("输出结果：")
            print(result.stdout)
        else:
            print("❌ 测试失败！")
            print("错误信息：")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ 测试超时（5分钟）")
    except Exception as e:
        print(f"❌ 测试过程中出错：{e}")

if __name__ == "__main__":
    test_with_small_model()