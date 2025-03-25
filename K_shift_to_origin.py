"""
此脚本用于将SEGY文件中的检波点和炮点坐标整体平移，使最小坐标值为0。

主要步骤：
1. 使用内存映射和批处理方式读取SEGY文件
2. 计算所有检波点坐标的最小值作为偏移量
3. 对所有检波点和炮点坐标进行平移
4. 创建新的SEGY文件保存结果

输入：
- filename: 输入SEGY文件路径
- new_filename: 输出SEGY文件路径

输出：
- 平移后的新SEGY文件
- 平移前后的对比图
"""

import segyio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_shift_result(orig_groupX, orig_groupY, orig_sourceX, orig_sourceY,
                     shifted_groupX, shifted_groupY, shifted_sourceX, shifted_sourceY,
                     output_path):
    """绘制平移前后的对比图"""
    # 去重处理坐标
    unique_orig_coords = np.unique(np.column_stack((orig_groupX, orig_groupY)), axis=0)
    unique_orig_source = np.unique(np.column_stack((orig_sourceX, orig_sourceY)), axis=0)
    unique_shifted_coords = np.unique(np.column_stack((shifted_groupX, shifted_groupY)), axis=0)
    unique_shifted_source = np.unique(np.column_stack((shifted_sourceX, shifted_sourceY)), axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 绘制平移前的坐标
    ax1.scatter(unique_orig_coords[:, 0], unique_orig_coords[:, 1], 
               color='blue', label=f'Receivers ({len(unique_orig_coords)})', 
               alpha=0.5, s=10)
    ax1.scatter(unique_orig_source[:, 0], unique_orig_source[:, 1], 
               color='red', label=f'Sources ({len(unique_orig_source)})', 
               alpha=0.8, s=20)
    ax1.set_title("Before Shift")
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.legend()
    ax1.grid(True)
    ax1.set_aspect('equal')
    
    # 绘制平移后的坐标
    ax2.scatter(unique_shifted_coords[:, 0], unique_shifted_coords[:, 1], 
               color='blue', label=f'Receivers ({len(unique_shifted_coords)})', 
               alpha=0.5, s=10)
    ax2.scatter(unique_shifted_source[:, 0], unique_shifted_source[:, 1], 
               color='red', label=f'Sources ({len(unique_shifted_source)})', 
               alpha=0.8, s=20)
    ax2.set_title("After Shift")
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.legend()
    ax2.grid(True)
    ax2.set_aspect('equal')
    
    plt.suptitle("Coordinate Shift Result", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n坐标范围:")
    print(f"原始检波点 X: [{np.min(orig_groupX)}, {np.max(orig_groupX)}]")
    print(f"原始检波点 Y: [{np.min(orig_groupY)}, {np.max(orig_groupY)}]")
    print(f"平移后检波点 X: [{np.min(shifted_groupX)}, {np.max(shifted_groupX)}]")
    print(f"平移后检波点 Y: [{np.min(shifted_groupY)}, {np.max(shifted_groupY)}]")
    print(f"对比图已保存为: {output_path}")

def shift_coordinates(filename, new_filename, batch_size=100000):
    """将坐标平移到第一象限"""
    with segyio.open(filename, "r", ignore_geometry=True) as segyfile:
        if segyfile.mmap():
            print(f"\n文件 {filename} 已成功进行内存映射")
        else:
            print(f"\n文件 {filename} 内存映射失败！")
            
        spec = segyio.spec()
        spec.samples = segyfile.samples
        spec.format = segyfile.format
        spec.sorting = segyfile.sorting
        spec.tracecount = segyfile.tracecount

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

        # 计算偏移量
        x_min = np.min(groupX)
        y_min = np.min(groupY)
        print(f"\n计算得到的偏移量: ({x_min}, {y_min})")
        
        # 进行平移
        shifted_groupX = groupX - x_min
        shifted_groupY = groupY - y_min
        shifted_sourceX = sourceX - x_min
        shifted_sourceY = sourceY - y_min
        
        # 绘制平移前后的对比图
        plot_shift_result(groupX, groupY, sourceX, sourceY,
                         shifted_groupX, shifted_groupY, shifted_sourceX, shifted_sourceY,
                         "../fig_0118/K_shift_to_origin_result.png")

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
                    
                    # 写入道数据
                    new_segy.trace.raw[trace_index] = segyfile.trace.raw[trace_index]
                    
                    # 写入header
                    new_segy.header[trace_index].update({
                        segyio.TraceField.SourceDepth: batch_headers['source_depth'][i],
                        segyio.TraceField.TraceNumber: batch_headers['trace_number'][i],
                        segyio.TraceField.TRACE_SAMPLE_COUNT: 2500,
                        segyio.TraceField.TRACE_SAMPLE_INTERVAL: batch_headers['sample_interval'][i],
                        segyio.TraceField.GroupX: int(shifted_groupX[trace_index]),
                        segyio.TraceField.GroupY: int(shifted_groupY[trace_index]),
                        segyio.TraceField.SourceX: int(shifted_sourceX[trace_index]),
                        segyio.TraceField.SourceY: int(shifted_sourceY[trace_index])
                    })

        print(f"\n新的SEGY文件已保存为: {new_filename}")

if __name__ == "__main__":
    filename = "../result_0118/G_vel_shot_match_new.segy"
    new_filename = "../result_0118/K_final_vel_full_new.segy"
    shift_coordinates(filename, new_filename) 