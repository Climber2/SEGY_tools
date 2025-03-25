"""
此脚本用于可视化 SEGY 格式炮集数据中炮点和检波点坐标,分析和理解地震数据采集的空间分布。

主要功能：
1. 从多个SEGY文件中读取炮点和检波点的X和Y坐标。
2. 利用 matplotlib 库绘制这些坐标,其中检波点的颜色表示叠加道数,提供了关于数据厚度的视觉信息。
3. 炮点以红色标记,检波点根据其叠加道数使用色谱进行着色。
4. 图表包含标题、坐标轴标签、图例和网格,以增强信息的清晰度和可读性。
5. 将绘制的图形保存为 PNG 文件,并打印出有关文件的统计信息,如总道数和采样点数量等。

参数：
- filenames: SEGY文件路径列表。
- batch_size: 每批次处理的道数,默认为100000。

输出：
- 图形文件保存在指定位置。
- 控制台输出包括每个文件的地震道数量、采样点数量等统计信息。
"""

import segyio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def process_segy_file(filenames, batch_size=100000):
    # 访问非结构化数据
    all_unique_source_coords = set()
    all_unique_group_coords = set()
    
    for filename in filenames:
        unique_source_coords = set()
        unique_group_coords = set()
        
        with segyio.open(filename, ignore_geometry=True) as segyfile:
            # 使用内存映射
            if segyfile.mmap():
                print(f"\nFile {filename} is memory mapped.")
            else:
                print(f"\nMemory mapping failed for {filename}!")
            
            num_traces = segyfile.tracecount
            num_batches = (num_traces + batch_size - 1) // batch_size
            
            for batch in tqdm(range(num_batches), desc="Processing Traces"):
                start = batch * batch_size
                end = min((batch + 1) * batch_size, num_traces)
                
                # 提取当前批次的炮点坐标并去重
                sourceX = segyfile.attributes(segyio.TraceField.SourceX)[start:end]
                sourceY = segyfile.attributes(segyio.TraceField.SourceY)[start:end]
                unique_source_coords.update(set(zip(sourceX, sourceY)))
                
                # 提取当前批次的检波器坐标并去重
                groupX = segyfile.attributes(segyio.TraceField.GroupX)[start:end]
                groupY = segyfile.attributes(segyio.TraceField.GroupY)[start:end]
                unique_group_coords.update(set(zip(groupX, groupY)))
            
            # 输出当前文件信息
            print(f"\n{filename} 统计信息:")
            print("总地震道数量:", num_traces)
            print("采样时间点数量:", len(segyfile.samples))
            print("每炮的地震道数量:", segyfile.header[-1][segyio.TraceField.TraceNumber])
            print("炮的数量:", num_traces / segyfile.header[-1][segyio.TraceField.TraceNumber])
        
        # 更新总的坐标集合
        all_unique_source_coords.update(unique_source_coords)
        all_unique_group_coords.update(unique_group_coords)
    
    # 将所有文件的去重后的坐标转换为NumPy数组
    sourceX, sourceY = np.array(list(all_unique_source_coords)).T
    groupX, groupY = np.array(list(all_unique_group_coords)).T
    
    # 创建一个图形和轴对象
    fig, ax = plt.subplots(figsize=(16, 13))
    
    # 绘制所有检波点
    ax.scatter(groupX, groupY, color='blue', label='All Receivers', alpha=1)
    # 绘制所有炮点
    ax.scatter(sourceX, sourceY, color='green', label='All Sources', alpha=1)
    
    # 设置图形的标签和标题
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title("Combined Seismic Source and Receiver Coordinates")
    ax.legend()
    ax.grid(True)
    
    # 保存图形
    plt.savefig("../test.png", dpi=300, bbox_inches='tight')
    plt.close()

# 使用示例
filenames = [
    "../result_0118/G_vel_shot_match.segy",
]

process_segy_file(filenames)