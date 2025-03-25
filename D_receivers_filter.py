"""
此脚本用于根据炮点坐标的矩形包围边界过滤检波点数据。

主要步骤：
1. 读取SEGY文件中的炮点和检波点坐标
2. 计算炮点坐标的矩形边界
3. 根据边界过滤检波点
4. 创建新的SEGY文件，只保留边界内的检波点数据

输入：
- filename: 原始SEGY文件路径
- new_filename: 新SEGY文件路径

输出：
- 过滤后的新SEGY文件
- 过滤前后的观测系统对比图
"""

import segyio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_filter_result(groupX, groupY, sourceX, sourceY, 
                      keep_indices, boundary, output_path):
    """
    绘制过滤前后的观测系统对比图，使用去重后的坐标点进行绘制
    """
    # 去重处理原始坐标
    unique_orig_coords = np.unique(np.column_stack((groupX, groupY)), axis=0)
    unique_orig_source = np.unique(np.column_stack((sourceX, sourceY)), axis=0)
    
    # 获取过滤后的坐标并去重
    filtered_groupX = groupX[keep_indices]
    filtered_groupY = groupY[keep_indices]
    filtered_sourceX = sourceX[keep_indices]
    filtered_sourceY = sourceY[keep_indices]
    unique_filtered_coords = np.unique(np.column_stack((filtered_groupX, filtered_groupY)), axis=0)
    unique_filtered_source = np.unique(np.column_stack((filtered_sourceX, filtered_sourceY)), axis=0)
    
    print(f"\n原始检波点数量: {len(groupX)} -> 去重后: {len(unique_orig_coords)}")
    print(f"原始炮点数量: {len(sourceX)} -> 去重后: {len(unique_orig_source)}")
    print(f"过滤后检波点数量: {len(filtered_groupX)} -> 去重后: {len(unique_filtered_coords)}")
    print(f"过滤后炮点数量: {len(filtered_sourceX)} -> 去重后: {len(unique_filtered_source)}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 绘制过滤前的坐标
    ax1.scatter(unique_orig_coords[:, 0], unique_orig_coords[:, 1], 
               color='blue', label=f'Receivers ({len(unique_orig_coords)})', 
               alpha=0.5, s=10)
    ax1.scatter(unique_orig_source[:, 0], unique_orig_source[:, 1], 
               color='green', label=f'Sources ({len(unique_orig_source)})', 
               alpha=0.8, s=20)
    # 绘制边界框
    ax1.plot([boundary[0], boundary[1], boundary[1], boundary[0], boundary[0]],
             [boundary[2], boundary[2], boundary[3], boundary[3], boundary[2]],
             'r--', label='Source Boundary')
    ax1.set_title("Before Filtering")
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制过滤后的坐标
    ax2.scatter(unique_filtered_coords[:, 0], unique_filtered_coords[:, 1], 
               color='blue', label=f'Receivers ({len(unique_filtered_coords)})', 
               alpha=0.5, s=10)
    ax2.scatter(unique_filtered_source[:, 0], unique_filtered_source[:, 1], 
               color='green', label=f'Sources ({len(unique_filtered_source)})', 
               alpha=0.8, s=20)
    # 绘制边界框
    ax2.plot([boundary[0], boundary[1], boundary[1], boundary[0], boundary[0]],
             [boundary[2], boundary[2], boundary[3], boundary[3], boundary[2]],
             'r--', label='Source Boundary')
    ax2.set_title("After Filtering")
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.legend()
    ax2.grid(True)
    
    # 设置相同的比例
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    
    # 添加总标题
    plt.suptitle("Observation System Comparison", fontsize=16)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"过滤对比图已保存为: {output_path}")

def filter_receivers_by_source_boundary(filename, new_filename, batch_size=100000):
    with segyio.open(filename, "r", ignore_geometry=True) as segyfile:
        if segyfile.mmap():
            print(f"\nFile {filename} is memory mapped.")
        else:
            print(f"\nMemory mapping failed for {filename}!")
            
        spec = segyio.spec()
        spec.samples = segyfile.samples
        spec.format = segyfile.format
        spec.sorting = segyfile.sorting

        # 初始化数组存储所有坐标
        num_traces = segyfile.tracecount
        sourceX_all = np.zeros(num_traces)
        sourceY_all = np.zeros(num_traces)
        groupX_all = np.zeros(num_traces)
        groupY_all = np.zeros(num_traces)
        
        # 分批读取所有坐标
        for batch_start in tqdm(range(0, num_traces, batch_size), desc="读取坐标"):
            batch_end = min(batch_start + batch_size, num_traces)
            sourceX_all[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.SourceX)[batch_start:batch_end]
            sourceY_all[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.SourceY)[batch_start:batch_end]
            groupX_all[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.GroupX)[batch_start:batch_end]
            groupY_all[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.GroupY)[batch_start:batch_end]

        # 计算炮点坐标的边界
        xmin, xmax = np.min(sourceX_all), np.max(sourceX_all)
        ymin, ymax = np.min(sourceY_all), np.max(sourceY_all)
        boundary = [xmin, xmax, ymin, ymax]
        print(f"\n炮点坐标边界:")
        print(f"X范围: {xmin} - {xmax}")
        print(f"Y范围: {ymin} - {ymax}")

        # 找出边界内的检波点
        keep_indices = np.where(
            (groupX_all >= xmin) & (groupX_all <= xmax) &
            (groupY_all >= ymin) & (groupY_all <= ymax)
        )[0]

        spec.tracecount = len(keep_indices)
        
        # 绘制过滤前后的对比图
        plot_filter_result(groupX_all, groupY_all, sourceX_all, sourceY_all, 
                         keep_indices, boundary, "../fig_0118/D_receiver_filter_result.png")

        # 创建新的SEGY文件并写入数据
        with segyio.create(new_filename, spec) as new_segy:
            new_segy.text[0] = segyfile.text[0]
            new_segy.bin = segyfile.bin

            # 批量处理
            for batch_start in tqdm(range(0, len(keep_indices), batch_size), desc="写入数据"):
                batch_end = min(batch_start + batch_size, len(keep_indices))
                batch_indices = keep_indices[batch_start:batch_end]
                
                # 批量读取这个批次的header属性
                batch_headers = {
                    'source_depth': segyfile.attributes(segyio.TraceField.SourceDepth)[batch_indices],
                    'trace_number': segyfile.attributes(segyio.TraceField.TraceNumber)[batch_indices],
                    'sample_count': segyfile.attributes(segyio.TraceField.TRACE_SAMPLE_COUNT)[batch_indices],
                    'sample_interval': segyfile.attributes(segyio.TraceField.TRACE_SAMPLE_INTERVAL)[batch_indices]
                }
                
                # 一次性更新这个批次的所有header
                for i in range(len(batch_indices)):
                    trace_index = batch_start + i
                    orig_index = batch_indices[i]
                    
                    # 读取并写入道数据
                    new_segy.trace.raw[trace_index] = segyfile.trace.raw[orig_index]
                    
                    # 写入header
                    new_segy.header[trace_index].update({
                        segyio.TraceField.SourceDepth: batch_headers['source_depth'][i],
                        segyio.TraceField.TraceNumber: batch_headers['trace_number'][i],
                        segyio.TraceField.TRACE_SAMPLE_COUNT: batch_headers['sample_count'][i],
                        segyio.TraceField.TRACE_SAMPLE_INTERVAL: batch_headers['sample_interval'][i],
                        segyio.TraceField.GroupX: int(groupX_all[orig_index]),
                        segyio.TraceField.GroupY: int(groupY_all[orig_index]),
                        segyio.TraceField.SourceX: int(sourceX_all[orig_index]),
                        segyio.TraceField.SourceY: int(sourceY_all[orig_index])
                    })

        print(f"\n新的SEGY文件已保存为 {new_filename}")

if __name__ == "__main__":
    filename = "../result_0118/D_receiver_filter.SEGY"
    new_filename = "../result_0118/test.SEGY"
    filter_receivers_by_source_boundary(filename, new_filename) 