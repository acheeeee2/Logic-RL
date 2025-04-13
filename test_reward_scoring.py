#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
这个脚本演示了Logic-RL项目的奖励评分功能。
它展示了如何对模型的逻辑推理回答进行评分。
"""

import pandas as pd
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入奖励评分模块
from verl.utils.reward_score.kk import compute_score, parse_solution_text_format

def simulate_model_response(data_sample, correct=True):
    """
    根据数据样本生成模拟的模型回答
    
    Args:
        data_sample: 数据样本
        correct: 是否生成正确答案
    
    Returns:
        模拟的模型回答字符串
    """
    names = data_sample['names']
    solution = data_sample['solution']
    
    # 思考部分
    thinking = f"<think>\n我需要确定谁是骑士谁是无赖。\n\n"
    for step in data_sample['cot_repeat_steps']:
        thinking += f"{step}\n"
    thinking += "\n</think>"
    
    # 答案部分 - 根据correct参数决定是否生成正确答案
    answer = "<answer>\n"
    for i, name in enumerate(names):
        # 如果correct=True，使用正确答案；否则生成错误答案
        is_knight = solution[i] if correct else not solution[i]
        role = "knight" if is_knight else "knave"
        answer += f"{name} is a {role}.\n"
    answer += "</answer>"
    
    # 完整回复
    full_response = f"<|im_start|>assistant\n{thinking}\n{answer}\n<|im_end|>"
    return full_response

def main():
    print("="*80)
    print("Logic-RL 奖励评分演示".center(80, '='))
    print("="*80)
    
    # 加载数据集
    data_path = "data/kk/instruct/3ppl/test.parquet"
    print(f"正在加载数据集: {data_path}")
    df = pd.read_parquet(data_path)
    
    # 选择一个样本
    sample_idx = 0
    sample = df.iloc[sample_idx]
    print(f"选择样本 #{sample_idx}:")
    print(f"谜题: {sample['quiz'][:100]}...")
    print(f"参与者: {', '.join(sample['names'])}")
    
    # 准备地面真相数据
    ground_truth = {
        'solution_text_format': sample['solution_text_format']
    }
    
    print("\n\n" + "="*80)
    print("案例1: 正确格式和答案的响应".center(80, '='))
    correct_response = simulate_model_response(sample, correct=True)
    compute_score(correct_response, ground_truth)
    
    print("\n\n" + "="*80)
    print("案例2: 正确格式但错误答案的响应".center(80, '='))
    incorrect_response = simulate_model_response(sample, correct=False)
    compute_score(incorrect_response, ground_truth)
    
    print("\n\n" + "="*80)
    print("案例3: 格式错误的响应".center(80, '='))
    malformed_response = correct_response.replace("<think>", "我思考:").replace("</think>", "结束思考")
    compute_score(malformed_response, ground_truth)
    
    print("\n\n" + "="*80)
    print("Logic-RL工作流程".center(80, '='))
    print("""
1. 模型接收逻辑谜题问题（Knights and Knaves）
2. 模型需要按照特定格式回答:
   a. 包含<think>...</think>标签用于思考过程
   b. 包含<answer>...</answer>标签用于最终答案
3. 奖励函数评估:
   a. 检查格式是否正确（格式奖励）
   b. 验证答案是否正确（内容奖励）
4. 通过基于规则的强化学习优化模型能力
    """)
    print("="*80)

if __name__ == "__main__":
    main() 