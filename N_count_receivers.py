"""
输入：
- filename: 输入的SEGY文件路径

描述：
此脚本用于统计SEGY文件中每个炮点的检波点数量，并输出统计信息。
使用内存映射和批处理方式处理大型SEGY文件。

输出：
- 每个炮点的检波点数量
- 检波点数量的统计范围（最小值、最大值、平均值）
"""

import segyio
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def count_receivers_per_shot(filename, batch_size=100000):
    """统计每个炮点的检波点数量"""
    with segyio.open(filename, "r", ignore_geometry=True) as segyfile:
        if segyfile.mmap():
            print(f"\n文件 {filename} 已成功进行内存映射")
        else:
            print(f"\n文件 {filename} 内存映射失败！")
        
        # 获取道数
        num_traces = segyfile.tracecount
        
        # 使用字典来存储每个炮点的检波点数量
        shot_receiver_count = defaultdict(int)
        
        # 分批读取所有坐标
        num_batches = (num_traces + batch_size - 1) // batch_size
        for batch in tqdm(range(num_batches), desc="统计检波点数量"):
            start = batch * batch_size
            end = min((batch + 1) * batch_size, num_traces)
            
            # 读取这个批次的炮点坐标
            sourceX = segyfile.attributes(segyio.TraceField.SourceX)[start:end]
            sourceY = segyfile.attributes(segyio.TraceField.SourceY)[start:end]
            
            # 将炮点坐标转换为复数作为唯一标识
            shot_points = sourceX + 1j * sourceY
            
            # 统计每个炮点的检波点数量
            for shot in shot_points:
                shot_receiver_count[shot] += 1
        
        # 计算统计信息
        receiver_counts = np.array(list(shot_receiver_count.values()))
        min_receivers = np.min(receiver_counts)
        max_receivers = np.max(receiver_counts)
        avg_receivers = np.mean(receiver_counts)
        
        # 输出统计结果
        print(f"\n统计结果:")
        print(f"总炮点数量: {len(shot_receiver_count)}")
        print(f"检波点数量范围: {min_receivers} - {max_receivers}")
        print(f"平均检波点数量: {avg_receivers:.2f}")
        
        # 输出每个炮点的详细信息
        # print("\n每个炮点的检波点数量:")
        # for shot, count in shot_receiver_count.items():
        #     real_x = int(shot.real)
        #     real_y = int(shot.imag)
        #     print(f"炮点坐标 ({real_x}, {real_y}): {count} 个检波点")

if __name__ == "__main__":
    filename = "../result_0118/O_final_shot.SEGY"
    count_receivers_per_shot(filename) 