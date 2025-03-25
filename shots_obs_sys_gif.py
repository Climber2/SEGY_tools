"""
此脚本用于动态展示 SEGY 格式炮集数据中每个炮点及其对应的检波点的空间分布。通过动画,可以观察每个炮点和其检波点的关系,有助于理解地震数据的采集布局。

主要功能：
1. 从SEGY文件中读取炮点(SourceX, SourceY)和检波点(GroupX, GroupY)的坐标。
2. 将坐标存储在Pandas DataFrame中,便于处理和过滤。
3. 提取所有独特的炮点,并按Y和X坐标排序,确保动画展示的顺序从左上角到右下角。
4. 使用matplotlib和matplotlib.animation创建动画,展示每个炮点及其相应的检波点。
5. 动画中,所有炮点和检波点以不同颜色标记,当前活跃的炮点和检波点以特殊颜色和符号突出显示。
6. 动画保存为GIF文件,方便展示和分享。

参数说明：
- filename: 要处理的SEGY文件的路径。
- batch_size: 每批次处理的道数,默认为100000。

输出：
- 动画GIF文件,展示了每个炮点及其检波点的变化过程。
"""

import segyio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from tqdm import tqdm

def process_segy_file(filenames, batch_size=100000):
    unique_source_coords = set()
    unique_group_coords = set()
    df_list = []  # 用于存储每个批次的DataFrame

    # 读取SEGY文件并提取坐标
    for filename in filenames:
        with segyio.open(filename, ignore_geometry=True) as segyfile:
            # 使用内存映射
            if segyfile.mmap():
                print(f"\nFile {filename} is memory mapped.")
            else:
                print(f"\nMemory mapping failed for {filename}!")

            num_traces = segyfile.tracecount
            num_batches = (num_traces + batch_size - 1) // batch_size
            
            for batch in tqdm(range(num_batches), desc="Processing batches"):
                start = batch * batch_size
                end = min((batch + 1) * batch_size, num_traces)
                
                # 提取当前批次的炮点坐标并去重
                sourceX = segyfile.attributes(segyio.TraceField.SourceX)[start:end]
                sourceY = segyfile.attributes(segyio.TraceField.SourceY)[start:end]
                unique_source_coords.update(set(zip(sourceX, sourceY)))
                
                # 提取当前批次的检波器坐标
                groupX = segyfile.attributes(segyio.TraceField.GroupX)[start:end]
                groupY = segyfile.attributes(segyio.TraceField.GroupY)[start:end]
                unique_group_coords.update(set(zip(groupX, groupY)))
                
                # 创建当前批次的DataFrame并添加到列表中
                batch_df = pd.DataFrame({
                    'sx': sourceX,
                    'sy': sourceY,
                    'gx': groupX,
                    'gy': groupY
                })
                df_list.append(batch_df)
        
    # 合并所有批次的DataFrame
    all_gx_gy = pd.concat(df_list, ignore_index=True)
    
    # 将炮点坐标转换为DataFrame并排序
    unique_sx_sy = pd.DataFrame(list(unique_source_coords), columns=['sx', 'sy'])
    unique_sx_sy = unique_sx_sy.sort_values(by=['sy', 'sx'], ascending=[False, True]).reset_index(drop=True)
    unique_gx_gy = pd.DataFrame(list(unique_group_coords), columns=['gx', 'gy'])
    
    # 创建一个图形和轴对象
    fig, ax = plt.subplots(figsize=(16, 13))
    # 设置步长
    step = 3 # 每隔step个炮点选择一个进行动画展示
    # 创建动画对象
    frames = range(0, len(unique_sx_sy), step)
    pbar = tqdm(total=len(frames), desc="Creating animation")  # 创建进度条
    
    # 动画更新函数
    def update(frame):
        # 获取当前 (sx, sy) 点
        sx, sy = unique_sx_sy.iloc[frame]
        # 提取与当前 (sx, sy) 对应的所有 (gx, gy) 点
        filtered_gx_gy = all_gx_gy[(all_gx_gy['sx'] == sx) & (all_gx_gy['sy'] == sy)]
        # 清空轴,以便绘制新的图形
        ax.clear()
        # 绘制当前 (sx, sy) 和对应的 (gx, gy) 点
        ax.scatter(unique_gx_gy['gx'], unique_gx_gy['gy'], color='blue', label='All Receivers', alpha=1)  # 显示所有检波点
        ax.scatter(filtered_gx_gy['gx'], filtered_gx_gy['gy'], marker='*', color='yellow', label='Current Receivers')  # 显示当前炮点对应的检波点
        ax.scatter(unique_sx_sy['sx'], unique_sx_sy['sy'], color='green', label='All Sources', alpha=1)  # 显示所有炮点
        ax.scatter(sx, sy, color='red', marker='o', label='Current Source')  # 显示当前活跃炮点
        # 设置图形的标签和标题
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f"Source Point: ({sx}, {sy}) and its Receivers")
        ax.legend()
        # 保证xy轴单位长度相等
        ax.set_aspect('equal')
        # 更新进度条
        pbar.update(1)
    
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000, repeat=False)
    # 保存动画为GIF
    ani.save('../fig_0118/M_shot_test.SEGY.gif', writer='pillow', fps=1)

# 文件名设置
filenames = [
    "../result_0118/M_shot_test.SEGY",
]

process_segy_file(filenames, batch_size=100000)
