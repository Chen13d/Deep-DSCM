import os, sys
 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

from utils import *



import pandas as pd
import re
from pathlib import Path

# ==== 需要你改的部分（或直接用默认）====
in_csv = r"C:\Users\18923\Desktop\DSRM_paper_on_submission_material\DSRM paper\synthetic_data_eval\Mito_inner_deconv_Micro\rFRC.csv"  # 原始表格路径
# =======================================

# 输出文件名：原名后加_sorted
out_csv = Path(in_csv).with_name(Path(in_csv).stem + "_sorted.csv")

# 读入
df = pd.read_csv(in_csv)

# 找到所有 rFRC_value 列
value_cols = [c for c in df.columns if "rFRC_value" in c]

print(df.columns)

# 根据 rFRC_value 列，推断对应的 mean_rFRC 列名
pairs = []
for vcol in value_cols:
    # 假设只是把字符串里的 rFRC_value 换成 mean_rFRC
    mcol = vcol.replace("rFRC_value", "mean_FRC")
    print(mcol)
    if mcol in df.columns:
        pairs.append((mcol, vcol))
    else:
        print(f"[跳过] 找不到与 {vcol} 对应的 mean_rFRC 列：{mcol}")

# 为每一对列排序，并横向拼接
sorted_parts = []
for mcol, vcol in pairs:
    part = df[[mcol, vcol]].sort_values(by=vcol, ascending=True, na_position="last").reset_index(drop=True)
    sorted_parts.append(part)

# 如果还想保留未参与排序的其它列，可以把它们拼在后面
other_cols = [c for c in df.columns if c not in {c for p in pairs for c in p}]
if other_cols:
    # 其它列保持原顺序，不排序（或自行决定怎么处理）
    others = df[other_cols].reset_index(drop=True)
    sorted_parts.append(others)

# 合并并保存
out_df = pd.concat(sorted_parts, axis=1)
out_df.to_csv(out_csv, index=False)

print(f"已保存到: {out_csv}")