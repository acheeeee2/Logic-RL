#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
这个脚本展示如何加载和处理Knights and Knaves逻辑谜题数据集。
"""

import pandas as pd
import json
from pathlib import Path

def load_dataset(file_path):
    """加载数据集"""
    df = pd.read_parquet(file_path)
    return df

def print_sample(df, idx=0):
    """打印样本数据"""
    sample = df.iloc[idx]
    
    print("="*50)
    print("谜题内容:")
    print("="*50)
    print(sample['quiz'])
    print("\n" + "="*50)
    print("参与者姓名:", ", ".join(sample['names']))
    print("="*50)
    
    print("\n解决方案:")
    print("="*50)
    solution_map = ["骑士" if status else "无赖" for status in sample['solution']]
    for name, solution in zip(sample['names'], solution_map):
        print(f"{name} 是 {solution}")
    print("="*50)
    
    print("\n思考过程:")
    print("="*50)
    print(sample['cot_head'])
    for step in sample['cot_repeat_steps']:
        print(f"- {step}")
    print(sample['cot_foot'])
    print("="*50)

def main():
    # 数据路径
    data_dir = Path("data/kk/instruct/3ppl")
    train_file = data_dir / "train.parquet"
    test_file = data_dir / "test.parquet"
    
    # 加载训练数据
    print("正在加载训练数据...")
    train_df = load_dataset(train_file)
    print(f"训练数据集包含 {len(train_df)} 个谜题")
    
    # 加载测试数据
    print("正在加载测试数据...")
    test_df = load_dataset(test_file)
    print(f"测试数据集包含 {len(test_df)} 个谜题")
    
    # 显示一个训练样本
    print("\n\n展示一个训练样本:\n")
    print_sample(train_df, 0)
    
    # 显示一个测试样本
    print("\n\n展示一个测试样本:\n")
    print_sample(test_df, 0)
    
    # 输出数据结构
    print("\n数据集结构:")
    print("列名:", list(train_df.columns))
    
    # 统计分析
    for i, df in enumerate([train_df, test_df]):
        dataset_name = "训练集" if i == 0 else "测试集"
        print(f"\n{dataset_name}平均参与者数量: {df['names'].apply(len).mean():.2f}")
        
        # 统计名字出现频率
        all_names = [name for names in df['names'] for name in names]
        top_names = pd.Series(all_names).value_counts().head(5)
        print(f"\n{dataset_name}中最常见的5个名字:")
        for name, count in top_names.items():
            print(f"- {name}: {count}次")

if __name__ == "__main__":
    main() 