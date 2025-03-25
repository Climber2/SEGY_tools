"""
此脚本用于将检波点和炮点坐标匹配到规则网格点上。

主要步骤：
1. 读取规则化的网格点坐标
2. 读取SEGY文件中的检波点和炮点坐标
3. 计算检波点分布与网格点分布的整体偏移量
4. 对检波点和炮点坐标进行整体偏移
5. 将偏移后的检波点匹配到最近的网格点
6. 创建新的SEGY文件保存结果

输入：
- grid_file: 规则网格点坐标文件(.npy格式)
- segy_file: 输入SEGY文件路径
- new_filename: 输出SEGY文件路径
- output_fig_path: 匹配结果对比图输出路径

输出：
- 匹配后的新SEGY文件
- 匹配统计信息和对比图
"""

import segyio
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def plot_match_result(orig_coords, shifted_coords, grid_points, matched_coords, 
                     orig_source_coords, shifted_source_coords, output_path):
    """绘制匹配过程的对比图"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 30))
    
    # 绘制原始分布
    ax1.scatter(grid_points[:, 0], grid_points[:, 1], 
               color='red', alpha=0.5, s=10, label='Grid Points')
    ax1.scatter(orig_coords[:, 0], orig_coords[:, 1], 
               color='blue', alpha=0.5, s=10, label='Original Receivers')
    ax1.scatter(orig_source_coords[:, 0], orig_source_coords[:, 1],
               color='green', alpha=0.5, s=10, label='Original Sources')
    ax1.set_title("Before Shift")
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.legend()
    ax1.grid(True)
    ax1.set_aspect('equal')

    # 绘制偏移后的分布
    ax2.scatter(grid_points[:, 0], grid_points[:, 1], 
               color='red', alpha=0.5, s=10, label='Grid Points')
    ax2.scatter(shifted_coords[:, 0], shifted_coords[:, 1], 
               color='blue', alpha=0.5, s=10, label='Shifted Receivers')
    ax2.scatter(shifted_source_coords[:, 0], shifted_source_coords[:, 1],
               color='green', alpha=0.5, s=10, label='Shifted Sources')
    ax2.set_title("After Shift")
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.legend()
    ax2.grid(True)
    ax2.set_aspect('equal')

    # 绘制匹配后的分布
    ax3.scatter(grid_points[:, 0], grid_points[:, 1], 
               color='red', alpha=0.5, s=10, label='Grid Points')
    ax3.scatter(matched_coords[:, 0], matched_coords[:, 1], 
               color='blue', alpha=0.5, s=10, label='Matched Receivers')
    ax3.scatter(shifted_source_coords[:, 0], shifted_source_coords[:, 1],
               color='green', alpha=0.5, s=10, label='Shifted Sources')
    ax3.set_title("After Matching")
    ax3.set_xlabel('X Coordinate')
    ax3.set_ylabel('Y Coordinate')
    ax3.legend()
    ax3.grid(True)
    ax3.set_aspect('equal')

    plt.suptitle("Receiver Points Matching Process", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"匹配过程对比图已保存为: {output_path}")

def process_segy_file(grid_file, segy_file, new_filename, output_fig_path, batch_size=100000):
    """处理SEGY文件，将检波点匹配到网格点上"""
    # 加载网格点坐标
    print("\n加载网格点坐标...")
    grid_points = np.load(grid_file)
    grid_center = np.mean(grid_points, axis=0)
    print(f"网格点数量: {len(grid_points)}")
    print(f"网格中心点: ({grid_center[0]:.2f}, {grid_center[1]:.2f})")
    
    # 读取SEGY文件
    with segyio.open(segy_file, "r", ignore_geometry=True) as segyfile:
        if segyfile.mmap():
            print(f"\n文件 {segy_file} 已成功进行内存映射")
        else:
            print(f"\n文件 {segy_file} 内存映射失败！")
        
        # 获取文件信息
        num_traces = segyfile.tracecount
        spec = segyio.spec()
        spec.samples = segyfile.samples
        spec.format = segyfile.format
        spec.sorting = segyfile.sorting
        spec.tracecount = num_traces
        
        # 读取所有坐标
        print("\n读取所有坐标...")
        groupX = np.zeros(num_traces)
        groupY = np.zeros(num_traces)
        sourceX = np.zeros(num_traces)
        sourceY = np.zeros(num_traces)
        
        for batch_start in tqdm(range(0, num_traces, batch_size), desc="读取坐标"):
            batch_end = min(batch_start + batch_size, num_traces)
            groupX[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.GroupX)[batch_start:batch_end]
            groupY[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.GroupY)[batch_start:batch_end]
            sourceX[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.SourceX)[batch_start:batch_end]
            sourceY[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.SourceY)[batch_start:batch_end]
        
        # 获取唯一的检波点和炮点坐标
        orig_coords = np.unique(np.column_stack((groupX, groupY)), axis=0)
        orig_source_coords = np.unique(np.column_stack((sourceX, sourceY)), axis=0)
        orig_center = np.mean(orig_coords, axis=0)
        print(f"\n原始唯一检波点数量: {len(orig_coords)}")
        print(f"原始唯一炮点数量: {len(orig_source_coords)}")
        print(f"检波点中心点: ({orig_center[0]:.2f}, {orig_center[1]:.2f})")
        
        # 计算偏移量（取整）
        offset = np.round(grid_center - orig_center).astype(int)
        print(f"\n计算得到的偏移量: ({offset[0]}, {offset[1]})")
        
        # 对所有检波点和炮点坐标进行偏移
        shifted_groupX = groupX + offset[0]
        shifted_groupY = groupY + offset[1]
        shifted_sourceX = sourceX #+ offset[0]
        shifted_sourceY = sourceY #+ offset[1]
        
        shifted_coords = np.unique(np.column_stack((shifted_groupX, shifted_groupY)), axis=0)
        shifted_source_coords = np.unique(np.column_stack((shifted_sourceX, shifted_sourceY)), axis=0)
        
        # 使用KD树进行最近点匹配
        print("\n进行检波点匹配...")
        tree = cKDTree(grid_points)
        distances, indices = tree.query(shifted_coords)
        
        # 统计匹配结果
        max_dist = np.max(distances)
        mean_dist = np.mean(distances)
        print(f"最大匹配距离: {max_dist:.2f}")
        print(f"平均匹配距离: {mean_dist:.2f}")
        
        # 获取匹配后的坐标
        matched_coords = grid_points[indices]
        
        # 创建检波点到网格点的映射
        receiver_to_grid = {tuple(shifted): tuple(matched) 
                          for shifted, matched in zip(shifted_coords, matched_coords)}
        
        # 绘制对比图
        plot_match_result(orig_coords, shifted_coords, grid_points, matched_coords, 
                         orig_source_coords, shifted_source_coords, output_fig_path)
        
        # 创建新的SEGY文件
        print("\n创建新的SEGY文件...")
        with segyio.create(new_filename, spec) as new_segy:
            new_segy.text[0] = segyfile.text[0]
            new_segy.bin = segyfile.bin
            
            # 批量处理
            for batch_start in tqdm(range(0, num_traces, batch_size), desc="写入数据"):
                batch_end = min(batch_start + batch_size, num_traces)
                
                # 批量读取这个批次的header属性
                batch_headers = {
                    'source_depth': segyfile.attributes(segyio.TraceField.SourceDepth)[batch_start:batch_end],
                    'trace_number': segyfile.attributes(segyio.TraceField.TraceNumber)[batch_start:batch_end],
                    'sample_count': segyfile.attributes(segyio.TraceField.TRACE_SAMPLE_COUNT)[batch_start:batch_end],
                    'sample_interval': segyfile.attributes(segyio.TraceField.TRACE_SAMPLE_INTERVAL)[batch_start:batch_end]
                }
                
                # 处理每个道
                for i in range(batch_end - batch_start):
                    trace_index = batch_start + i
                    shifted_coord = (shifted_groupX[trace_index], shifted_groupY[trace_index])
                    matched_coord = receiver_to_grid[shifted_coord]
                    
                    # 写入道数据
                    new_segy.trace.raw[trace_index] = segyfile.trace.raw[trace_index]
                    
                    # 写入header
                    new_segy.header[trace_index].update({
                        segyio.TraceField.SourceDepth: batch_headers['source_depth'][i],
                        segyio.TraceField.TraceNumber: batch_headers['trace_number'][i],
                        segyio.TraceField.TRACE_SAMPLE_COUNT: batch_headers['sample_count'][i],
                        segyio.TraceField.TRACE_SAMPLE_INTERVAL: batch_headers['sample_interval'][i],
                        segyio.TraceField.GroupX: int(matched_coord[0]),
                        segyio.TraceField.GroupY: int(matched_coord[1]),
                        segyio.TraceField.SourceX: int(shifted_sourceX[trace_index]),
                        segyio.TraceField.SourceY: int(shifted_sourceY[trace_index])
                    })

        print(f"\n新的SEGY文件已保存为: {new_filename}")

if __name__ == "__main__":
    # 输入参数
    grid_file = "../result_0118/F_regularized_vel_receivers_new.npy"
    segy_file = "../result_0118/C_vel_shot_filter_new.segy"
    new_filename = "../result_0118/H_vel_receiver_match_new.segy"
    output_fig_path = "../fig_0118/H_vel_receiver_match_result.png"
    batch_size = 100000  # 批处理大小
    
    # 执行处理
    process_segy_file(grid_file, segy_file, new_filename, output_fig_path, batch_size) 