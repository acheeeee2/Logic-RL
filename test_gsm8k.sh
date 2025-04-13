#!/bin/bash
set -x

# 设置环境
conda activate logic
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 查看数据
python -c "
import pandas as pd
import os

# 查看训练数据样例
train_file = 'data/kk/instruct/3ppl/train.parquet'
df = pd.read_parquet(train_file)
print('Data structure:')
print(df.columns)
print('\\nSample data:')
print(df.iloc[0])
" 