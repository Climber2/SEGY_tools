"""
此脚本用于筛选SEGY文件中的炮点，删除不在其对应检波点覆盖范围内的炮点。

主要步骤：
1. 使用内存映射和批处理方式读取SEGY文件
2. 对每个炮点，检查其是否在对应检波点的覆盖范围内
3. 保留在覆盖范围内的炮点，删除范围外的炮点
4. 创建新的SEGY文件保存结果

输入：
- filename: 输入SEGY文件路径
- new_filename: 输出SEGY文件路径

输出：
- 筛选后的新SEGY文件
- 筛选前后的对比图
"""

import segyio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_filter_result(groupX, groupY, sourceX, sourceY, 
                      keep_indices, output_path):
    """绘制过滤前后的观测系统对比图"""
    # 去重处理原始坐标
    unique_orig_coords = np.unique(np.column_stack((groupX, groupY)), axis=0)
    unique_orig_source = np.unique(np.column_stack((sourceX, sourceY)), axis=0)
    
    # 获取过滤后的坐标并去重
    filtered_sourceX = sourceX[keep_indices]
    filtered_sourceY = sourceY[keep_indices]
    filtered_groupX = groupX[keep_indices]
    filtered_groupY = groupY[keep_indices]
    unique_filtered_coords = np.unique(np.column_stack((filtered_groupX, filtered_groupY)), axis=0)
    unique_filtered_source = np.unique(np.column_stack((filtered_sourceX, filtered_sourceY)), axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 绘制过滤前的坐标
    ax1.scatter(unique_orig_coords[:, 0], unique_orig_coords[:, 1], 
               color='blue', label=f'Receivers ({len(unique_orig_coords)})', 
               alpha=0.5, s=10)
    ax1.scatter(unique_orig_source[:, 0], unique_orig_source[:, 1], 
               color='red', label=f'Sources ({len(unique_orig_source)})', 
               alpha=0.8, s=20)
    ax1.set_title("Before Filtering")
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.legend()
    ax1.grid(True)
    ax1.set_aspect('equal')
    
    # 绘制过滤后的坐标
    ax2.scatter(unique_filtered_coords[:, 0], unique_filtered_coords[:, 1], 
               color='blue', label=f'Receivers ({len(unique_filtered_coords)})', 
               alpha=0.5, s=10)
    ax2.scatter(unique_filtered_source[:, 0], unique_filtered_source[:, 1], 
               color='red', label=f'Sources ({len(unique_filtered_source)})', 
               alpha=0.8, s=20)
    ax2.set_title("After Filtering")
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.legend()
    ax2.grid(True)
    ax2.set_aspect('equal')
    
    plt.suptitle("Source Coverage Filter Result", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n过滤前后统计:")
    print(f"检波点数量: {len(unique_orig_coords)} -> {len(unique_filtered_coords)}")
    print(f"炮点数量: {len(unique_orig_source)} -> {len(unique_filtered_source)}")
    print(f"对比图已保存为: {output_path}")

def filter_sources_by_coverage(filename, new_filename, batch_size=100000):
    """根据检波点覆盖范围过滤炮点"""
    with segyio.open(filename, "r", ignore_geometry=True) as segyfile:
        if segyfile.mmap():
            print(f"\n文件 {filename} 已成功进行内存映射")
        else:
            print(f"\n文件 {filename} 内存映射失败！")
            
        spec = segyio.spec()
        spec.samples = segyfile.samples
        spec.format = segyfile.format
        spec.sorting = segyfile.sorting

        # 初始化数组
        num_traces = segyfile.tracecount
        sourceX = np.zeros(num_traces)
        sourceY = np.zeros(num_traces)
        groupX = np.zeros(num_traces)
        groupY = np.zeros(num_traces)
        
        # 分批读取坐标
        print("\n读取坐标...")
        for batch_start in tqdm(range(0, num_traces, batch_size), desc="读取坐标"):
            batch_end = min(batch_start + batch_size, num_traces)
            sourceX[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.SourceX)[batch_start:batch_end]
            sourceY[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.SourceY)[batch_start:batch_end]
            groupX[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.GroupX)[batch_start:batch_end]
            groupY[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.GroupY)[batch_start:batch_end]

        # 获取唯一的炮点
        unique_sources = np.unique(np.column_stack((sourceX, sourceY)), axis=0)
        print(f"\n开始检查 {len(unique_sources)} 个炮点...")
        
        # 找出要保留的道的索引
        keep_indices = []
        for source in tqdm(unique_sources, desc="处理炮点"):
            # 找出该炮点对应的所有检波点
            source_mask = (sourceX == source[0]) & (sourceY == source[1])
            source_traces = np.where(source_mask)[0]
            receivers_x = groupX[source_mask]
            receivers_y = groupY[source_mask]
            
            # 计算检波点的覆盖范围
            x_min, x_max = np.min(receivers_x), np.max(receivers_x)
            y_min, y_max = np.min(receivers_y), np.max(receivers_y)
            
            # 如果炮点在覆盖范围内，保留其所有道
            if x_min <= source[0] <= x_max and y_min <= source[1] <= y_max:
                keep_indices.extend(source_traces)
        
        keep_indices = np.array(keep_indices)
        spec.tracecount = len(keep_indices)
        
        # 绘制过滤前后的对比图
        plot_filter_result(groupX, groupY, sourceX, sourceY, 
                         keep_indices, "../fig_0118/J_source_coverage_filter_result.png")

        # 创建新的SEGY文件
        print("\n创建新的SEGY文件...")
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
                
                # 处理每个道
                for i in range(len(batch_indices)):
                    trace_index = batch_start + i
                    orig_index = batch_indices[i]
                    
                    # 写入道数据
                    new_segy.trace.raw[trace_index] = segyfile.trace.raw[orig_index]
                    
                    # 写入header
                    new_segy.header[trace_index].update({
                        segyio.TraceField.SourceDepth: batch_headers['source_depth'][i],
                        segyio.TraceField.TraceNumber: batch_headers['trace_number'][i],
                        segyio.TraceField.TRACE_SAMPLE_COUNT: batch_headers['sample_count'][i],
                        segyio.TraceField.TRACE_SAMPLE_INTERVAL: batch_headers['sample_interval'][i],
                        segyio.TraceField.GroupX: int(groupX[orig_index]),
                        segyio.TraceField.GroupY: int(groupY[orig_index]),
                        segyio.TraceField.SourceX: int(sourceX[orig_index]),
                        segyio.TraceField.SourceY: int(sourceY[orig_index])
                    })

        print(f"\n新的SEGY文件已保存为: {new_filename}")

if __name__ == "__main__":
    filename = "../result_0118/G_shot_match.SEGY"
    new_filename = "../result_0118/J_final_shot.SEGY"
    filter_sources_by_coverage(filename, new_filename) 