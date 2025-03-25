"""
此脚本用于展示SEGY格式的炮集数据与速度模型的空间分布对比。

主要步骤：
1. 使用内存映射和批处理方式读取多个炮集和速度模型SEGY文件
2. 提取并去重炮点、检波点和速度点坐标
3. 使用matplotlib绘制三种点的分布对比图
4. 输出统计信息和可视化结果

输入：
- shots_files: 炮集SEGY文件路径列表
- vel_file: 速度模型SEGY文件路径
- output_fig_path: 输出图像的路径

输出：
- 包含炮点、检波点和速度点分布的PNG图像
- 数据统计信息
"""

import segyio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_distribution(receivers, sources, vel_points, output_path):
    """绘制检波点、炮点和速度点的分布图"""
    plt.figure(figsize=(20, 16))
    ax = plt.gca()

    # 先去重再画图,提高绘图效率
    unique_receivers = np.unique(receivers, axis=0)
    unique_sources = np.unique(sources, axis=0)
    unique_vel_points = np.unique(vel_points, axis=0)

    # 绘制速度点
    ax.scatter(unique_vel_points[:, 0], unique_vel_points[:, 1], 
              color='blue', marker='o', s=10, 
              label=f"Velocity Points ({len(unique_vel_points)})", alpha=1)
    
    # 绘制检波点
    ax.scatter(unique_receivers[:, 0], unique_receivers[:, 1], 
              color='yellow', marker='*', s=10, 
              label=f"Receivers ({len(unique_receivers)})", alpha=1)
    
    # 绘制炮点
    ax.scatter(unique_sources[:, 0], unique_sources[:, 1], 
              color='red', marker='o', s=10, 
              label=f"Sources ({len(unique_sources)})", alpha=1)

    # 添加标题和标签
    ax.set_title("Distribution of Sources, Receivers and Velocity Points", fontsize=16)
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)

    # 设置比例尺一致
    ax.set_aspect('equal', adjustable='box')

    # 显示图例
    ax.legend(fontsize=12)

    # 添加网格
    ax.grid(True)

    # 保存图形
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n分布图已保存为: {output_path}")

def read_coordinates(segyfile, batch_size):
    """从SEGY文件中批量读取坐标"""
    num_traces = segyfile.tracecount
    sourceX = np.zeros(num_traces)
    sourceY = np.zeros(num_traces)
    groupX = np.zeros(num_traces)
    groupY = np.zeros(num_traces)
    
    for batch_start in tqdm(range(0, num_traces, batch_size), desc="读取坐标"):
        batch_end = min(batch_start + batch_size, num_traces)
        sourceX[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.SourceX)[batch_start:batch_end]
        sourceY[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.SourceY)[batch_start:batch_end]
        groupX[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.GroupX)[batch_start:batch_end]
        groupY[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.GroupY)[batch_start:batch_end]
    
    return sourceX, sourceY, groupX, groupY

def display_geometry(shots_files, vel_file, output_fig_path, batch_size=100000):
    """处理多个炮集和速度模型SEGY文件并显示几何分布"""
    # 初始化存储所有炮集坐标的数组
    all_shots_sourceX = []
    all_shots_sourceY = []
    all_shots_groupX = []
    all_shots_groupY = []
    
    # 读取所有炮集文件
    print("\n读取炮集文件...")
    for shots_file in shots_files:
        with segyio.open(shots_file, "r", ignore_geometry=True) as segyfile:
            if segyfile.mmap():
                print(f"文件 {shots_file} 已成功进行内存映射")
            else:
                print(f"文件 {shots_file} 内存映射失败！")
            
            print(f"\n炮集文件 {shots_file} 信息:")
            print(f"总道数: {segyfile.tracecount}")
            print(f"采样点数: {len(segyfile.samples)}")
            
            # 读取炮集坐标
            sourceX, sourceY, groupX, groupY = read_coordinates(segyfile, batch_size)
            all_shots_sourceX.extend(sourceX)
            all_shots_sourceY.extend(sourceY)
            all_shots_groupX.extend(groupX)
            all_shots_groupY.extend(groupY)
    
    # 转换为numpy数组
    all_shots_sourceX = np.array(all_shots_sourceX)
    all_shots_sourceY = np.array(all_shots_sourceY)
    all_shots_groupX = np.array(all_shots_groupX)
    all_shots_groupY = np.array(all_shots_groupY)
    
    # 读取速度模型文件
    print("\n读取速度模型文件...")
    with segyio.open(vel_file, "r", ignore_geometry=True) as segyfile:
        if segyfile.mmap():
            print(f"文件 {vel_file} 已成功进行内存映射")
        else:
            print(f"文件 {vel_file} 内存映射失败！")
        
        print(f"\n速度模型文件信息:")
        print(f"总道数: {segyfile.tracecount}")
        print(f"采样点数: {len(segyfile.samples)}")
        
        # 读取速度模型坐标
        vel_sourceX, vel_sourceY, vel_groupX, vel_groupY = read_coordinates(segyfile, batch_size)
    
    # 获取唯一的坐标点
    print("\n处理坐标...")
    unique_receivers = np.unique(np.column_stack((all_shots_groupX, all_shots_groupY)), axis=0)
    unique_sources = np.unique(np.column_stack((all_shots_sourceX, all_shots_sourceY)), axis=0)
    unique_vel_points = np.unique(np.column_stack((vel_groupX, vel_groupY)), axis=0)
    
    # 输出统计信息
    print(f"\n统计信息:")
    print(f"检波点数量: {len(all_shots_groupX)} -> 去重后: {len(unique_receivers)}")
    print(f"炮点数量: {len(all_shots_sourceX)} -> 去重后: {len(unique_sources)}")
    print(f"速度点数量: {len(vel_groupX)} -> 去重后: {len(unique_vel_points)}")
    
    # 计算坐标范围
    print(f"\n坐标范围:")
    print(f"检波点 X: [{np.min(all_shots_groupX)}, {np.max(all_shots_groupX)}]")
    print(f"检波点 Y: [{np.min(all_shots_groupY)}, {np.max(all_shots_groupY)}]")
    print(f"炮点 X: [{np.min(all_shots_sourceX)}, {np.max(all_shots_sourceX)}]")
    print(f"炮点 Y: [{np.min(all_shots_sourceY)}, {np.max(all_shots_sourceY)}]")
    print(f"速度点 X: [{np.min(vel_groupX)}, {np.max(vel_groupX)}]")
    print(f"速度点 Y: [{np.min(vel_groupY)}, {np.max(vel_groupY)}]")
    
    # 绘制分布图
    print("\n生成分布图...")
    plot_distribution(unique_receivers, unique_sources, unique_vel_points, output_fig_path)

if __name__ == "__main__":
    shots_files = [
        "../result_0118/O_final_shot.SEGY",
    ]
    vel_file = "../result_0118/K_final_vel_full_new.segy"
    output_fig_path = "../fig_0118/test3.png"
    
    display_geometry(shots_files, vel_file, output_fig_path)