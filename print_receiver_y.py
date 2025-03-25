"""
此脚本用于读取SEGY文件中所有检波点的Y坐标并去重打印。

主要步骤：
1. 使用内存映射和批处理方式读取SEGY文件
2. 提取所有检波点的Y坐标
3. 对Y坐标进行去重和排序
4. 以数组形式打印所有唯一的Y坐标值

输入：
- segy_file: SEGY文件路径

输出：
- 去重后的检波点Y坐标数组
"""

import segyio
import numpy as np
from tqdm import tqdm

def print_unique_receiver_y(segy_file, batch_size=100000):
    """读取并打印去重后的检波点Y坐标"""
    with segyio.open(segy_file, "r", ignore_geometry=True) as segyfile:
        if segyfile.mmap():
            print(f"\n文件 {segy_file} 已成功进行内存映射")
        else:
            print(f"\n文件 {segy_file} 内存映射失败！")
        
        # 获取文件信息
        num_traces = segyfile.tracecount
        print(f"\n文件信息:")
        print(f"总道数: {num_traces}")
        
        # 初始化数组
        print("\n读取Y坐标...")
        groupY = np.zeros(num_traces)
        
        # 分批读取Y坐标
        for batch_start in tqdm(range(0, num_traces, batch_size), desc="读取坐标"):
            batch_end = min(batch_start + batch_size, num_traces)
            groupY[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.GroupY)[batch_start:batch_end]
        
        # 获取唯一的Y坐标并排序
        unique_y = np.unique(groupY)
        
        print(f"\n去重后的Y坐标数组 ({len(unique_y)} 个):")
        print(f"y_coords = [{', '.join(map(str, unique_y.astype(int)))}]")

if __name__ == "__main__":
    segy_file = "../result_0118/G_shot_match.SEGY"
    print_unique_receiver_y(segy_file)