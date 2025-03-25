"""
输入:
    多个SEGY文件路径
描述:
    使用批处理和内存映射扫描SEGY文件，统计每个炮点的检波点数量
    将检波点数不满足条件的炮点坐标保存为numpy二进制文件
输出:
    包含异常炮点坐标的.npy文件
"""

import segyio
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def scan_unfull_shots(filenames, receiver_count_filter, batch_size=100000):
    """
    输入:
        filenames: SEGY文件路径列表
        receiver_count_filter: 检波点数过滤函数,输入为count,返回True表示异常
        batch_size: 批处理大小
    描述:
        使用批处理和内存映射统计每个炮点的检波点数
    输出:
        异常炮点坐标数组
    """
    shot_receiver_count = {}
    
    # 统计每个炮点的检波点数
    for filename in filenames:
        with segyio.open(filename, "r", ignore_geometry=True) as segyfile:
            # 使用内存映射
            if segyfile.mmap():
                print(f"\nFile {filename} is memory mapped.")
            else:
                print(f"\nMemory mapping failed for {filename}!")    
            num_traces = segyfile.tracecount
            print(f"总道数: {num_traces}")
            
            num_batches = (num_traces + batch_size - 1) // batch_size
            
            for batch in tqdm(range(num_batches), desc="统计检波点数"):
                start = batch * batch_size
                end = min((batch + 1) * batch_size, num_traces)
                
                # 批量读取炮点坐标
                sourceX = np.array(segyfile.attributes(segyio.TraceField.SourceX)[start:end])
                sourceY = np.array(segyfile.attributes(segyio.TraceField.SourceY)[start:end])
                
                # 统计每个炮点的检波点数
                for i in range(len(sourceX)):
                    shot_key = (sourceX[i], sourceY[i])
                    shot_receiver_count[shot_key] = shot_receiver_count.get(shot_key, 0) + 1
    
    # 找出检波点数不满足条件的炮点
    bad_shots = [(x, y) for (x, y), count in shot_receiver_count.items() 
                   if receiver_count_filter(count)]
    
    # 同时输出每个异常炮点的实际检波点数，便于分析
    print("\n异常炮点统计:")
    for x, y in bad_shots:
        print(f"炮点({x}, {y}) 的检波点数: {shot_receiver_count[(x, y)]}")
    
    # 打印共发现多少炮点数
    print(f"共发现 {len(shot_receiver_count)} 个炮点")
    
    # 绘制检波点数量分布直方图
    counts = list(shot_receiver_count.values())
    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=50, edgecolor='black')
    plt.title('shots-receivers count')
    plt.xlabel('receivers count')
    plt.ylabel('shots count')
    plt.grid(True)
    plt.savefig('../fig_0118/test.png')
    plt.close()
    print(f"检波点数量分布直方图已保存至: ../fig_0118/receiver_count_histogram.png")
    
    return np.array(bad_shots)

def save_unfull_shots(bad_shots, output_file):
    """
    输入:
        bad_shots: 异常炮点坐标数组
        output_file: 输出文件路径(.npy)
    描述:
        将异常炮点坐标保存为numpy二进制文件
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, bad_shots)
    
    print(f"\n检波点数异常的炮点坐标已保存至: {output_file}")
    print(f"共发现 {len(bad_shots)} 个异常炮点")
    print(f"文件大小: {os.path.getsize(output_file) / 1024:.2f} KB")

if __name__ == "__main__":
    # 文件名设置
    filenames = [
        "../result_0118/K_final_shot.SEGY",
    ]
    output_file = "../result_0118/not_4096_shots.npy"
    
    # 定义检波点数过滤函数
    receiver_count_filter = lambda count: count != 4096
    
    # 扫描并保存异常炮点坐标
    bad_shots = scan_unfull_shots(filenames, receiver_count_filter, batch_size=100000)
    save_unfull_shots(bad_shots, output_file) 