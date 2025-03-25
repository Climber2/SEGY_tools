"""
此脚本用于处理 SEGY 格式的炮集数据或三维速度模型，允许用户根据自定义的过滤函数删除含特定炮点的地震道。

主要步骤：
1. 读取原始SEGY文件中的炮点坐标。
2. 应用过滤函数，识别出需要保留的炮点。
3. 创建一个新的SEGY文件，只包含经过过滤后的炮点数据。

函数：
- delete_sources_by_coordinates: 根据提供的过滤函数，处理和保存新的SEGY文件。

参数：
- filename: 原始的SEGY文件路径。
- new_filename: 保存处理后数据的新SEGY文件路径。
- filter_function: lambda过滤函数，用于确定哪些炮点需要被删除。
"""

import segyio
import numpy as np
from tqdm import tqdm  # 导入进度条库
import matplotlib.pyplot as plt

def plot_filter_result(groupX, groupY, sourceX, sourceY, 
                      keep_indices, output_path):
    """
    绘制过滤前后的观测系统对比图，使用去重后的坐标点进行绘制
    """
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
               color='green', label=f'Sources ({len(unique_orig_source)})', 
               alpha=0.8, s=20)
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

def delete_sources_by_coordinates(filename, new_filename, filter_function, batch_size=100000):
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
        keep_indices = []
        
        # 分批读取所有坐标
        num_batches = (num_traces + batch_size - 1) // batch_size
        for batch in tqdm(range(num_batches), desc="读取和处理坐标"):
            start = batch * batch_size
            end = min((batch + 1) * batch_size, num_traces)

            # 读取这个批次的坐标
            sourceX = segyfile.attributes(segyio.TraceField.SourceX)[start:end]
            sourceY = segyfile.attributes(segyio.TraceField.SourceY)[start:end]
            groupX = segyfile.attributes(segyio.TraceField.GroupX)[start:end]
            groupY = segyfile.attributes(segyio.TraceField.GroupY)[start:end]
            
            # 存储坐标
            sourceX_all[start:end] = sourceX
            sourceY_all[start:end] = sourceY
            groupX_all[start:end] = groupX
            groupY_all[start:end] = groupY

            # 使用numpy的向量化操作来加速过滤
            filter_results = filter_function(sourceX, sourceY)
            keep_indices.extend(np.where(~filter_results)[0] + start)

        spec.tracecount = len(keep_indices)
        
        # 绘制过滤前后的对比图
        plot_filter_result(groupX_all, groupY_all, sourceX_all, sourceY_all, 
                         keep_indices, "../fig_0118/C_vel_shot_filter_result.png")

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

        print(f"新的 SEGY 文件已保存为 {new_filename}")

filename = "../result_0118/B_vel_rotate.segy"
new_filename = "../result_0118/C_vel_shot_filter_new.segy"

# 创建一个 lambda 函数来定义删除规则，x和y为每一道的sx和sy。

# 删除检波点范围之外的速度点
xmin, xmax, ymin, ymax = 20315600, 20327860, 3931190, 3936720
filter_function1 = lambda x, y: (x < xmin) | (x > xmax) | (y < ymin) | (y > ymax)

# # 加载需要去除的炮点坐标
# unfull_shots = np.load("../result_0118/not_4096_shots.npy")
# # 创建一个复数数组来表示坐标对，这样可以一次性比较
# unfull_complex = unfull_shots[:, 0] + 1j * unfull_shots[:, 1]
# filter_function2 = lambda x, y: np.in1d(x + 1j * y, unfull_complex)

# # 定义新的过滤函数，使用复数进行向量化比较并同时满足function1和function2的条件
# filter_function_combined = lambda x, y: ((x < xmin) | (x > xmax) | (y < ymin) | (y > ymax)) | np.in1d(x + 1j * y, unfull_complex)

# valid_y = np.array([3931644, 3931744, 3931844, 3931944, 3932044, 3932144, 3932244, 3932344, 3932444, 3932544, 3932644, 3932744, 3932844, 3932944, 3933044, 3933144, 3933244, 3933344, 3933444, 3933544, 3933644, 3933744, 3933844, 3933944, 3934044, 3934144, 3934244, 3934344, 3934444, 3934544, 3934644, 3934744, 3934844, 3934944, 3935044, 3935144, 3935244, 3935344, 3935444, 3935544, 3935644, 3935744, 3935844, 3935944, 3936044, 3936144, 3936244, 3936344, 3936444, 3936544, 3936644, 3936744, 3936844, 3936944, 3937044, 3937144])
# filter_function3 = lambda x, y: ~np.isin(y, valid_y)

# 调用函数删除特定炮点
delete_sources_by_coordinates(filename, new_filename, filter_function1)
