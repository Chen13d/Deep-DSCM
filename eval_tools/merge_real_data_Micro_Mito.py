import pandas as pd
import os
from glob import glob

# RSP RSE
if 0:
    # 设置文件夹路径和输出文件路径
    folder_path = r'D:\CQL\DSCM\real_data_eval\Micro_Mito_eval\results\RSP_RSE_value'
    output_path = r'D:\CQL\DSCM\real_data_eval\Micro_Mito_eval\results\merged_by_column.csv'

    # 获取所有CSV文件
    csv_files = glob(os.path.join(folder_path, '*.csv'))

    # 保存每个DataFrame，并以文件名前缀作为前缀重命名列
    merged_df = None

    for file in csv_files:
        try:
            filename = os.path.basename(file)
            prefix = os.path.splitext(filename)[0]  # 去除扩展名
            
            df = pd.read_csv(file)

            # 重命名列，加上前缀
            df = df.add_prefix(f'{prefix}_')
            
            # 合并（按列横向拼接）
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.concat([merged_df, df], axis=1)
        except Exception as e:
            print(f"读取失败: {file}，错误信息: {e}")

    # 保存为CSV
    if merged_df is not None:
        merged_df.to_csv(output_path, index=False)
        print(f"成功保存为：{output_path}")
    else:
        print("没有可合并的文件")

# Resolution
if 1:
    # 设置文件夹路径和输出文件路径
    folder_path = r'D:\CQL\DSCM\real_data_eval\Micro_Mito_eval\results\Resolution_csv'
    output_path = r'D:\CQL\DSCM\real_data_eval\Micro_Mito_eval\results\merged_by_column_resolution.csv'

    # 获取所有CSV文件
    csv_files = glob(os.path.join(folder_path, '*.csv'))

    # 保存每个DataFrame，并以文件名前缀作为前缀重命名列
    merged_df = None

    for file in csv_files:
        try:
            filename = os.path.basename(file)
            prefix = os.path.splitext(filename)[0]  # 去除扩展名
            
            df = pd.read_csv(file)

            # 重命名列，加上前缀
            df = df.add_prefix(f'{prefix}_')
            
            # 合并（按列横向拼接）
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.concat([merged_df, df], axis=1)
        except Exception as e:
            print(f"读取失败: {file}，错误信息: {e}")

    # 保存为CSV
    if merged_df is not None:
        merged_df.to_csv(output_path, index=False)
        print(f"成功保存为：{output_path}")
    else:
        print("没有可合并的文件")