"""
输入：
- filename: 输入的SEGY文件路径
- new_filename: 输出的SEGY文件路径
- max_receivers: 每个炮点允许的最大检波点数量

描述：
此脚本用于删除检波点数量超过指定阈值的炮点。
使用内存映射和批处理方式处理SEGY文件以提高效率。

输出：
- 新的SEGY文件，只包含检波点数量在阈值以下的炮点
- 删除前后的炮点分布对比图
- 统计信息（删除前后的炮点数量等）
"""

import segyio
import numpy as np
from tqdm import tqdm
from collections import defaultdict
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
               color='blue', label=f'检波点 ({len(unique_orig_coords)})', 
               alpha=0.5, s=10)
    ax1.scatter(unique_orig_source[:, 0], unique_orig_source[:, 1], 
               color='red', label=f'炮点 ({len(unique_orig_source)})', 
               alpha=0.8, s=20)
    ax1.set_title("过滤前")
    ax1.set_xlabel('X 坐标')
    ax1.set_ylabel('Y 坐标')
    ax1.legend()
    ax1.grid(True)
    ax1.set_aspect('equal')
    
    # 绘制过滤后的坐标
    ax2.scatter(unique_filtered_coords[:, 0], unique_filtered_coords[:, 1], 
               color='blue', label=f'检波点 ({len(unique_filtered_coords)})', 
               alpha=0.5, s=10)
    ax2.scatter(unique_filtered_source[:, 0], unique_filtered_source[:, 1], 
               color='red', label=f'炮点 ({len(unique_filtered_source)})', 
               alpha=0.8, s=20)
    ax2.set_title("过滤后")
    ax2.set_xlabel('X 坐标')
    ax2.set_ylabel('Y 坐标')
    ax2.legend()
    ax2.grid(True)
    ax2.set_aspect('equal')
    
    plt.suptitle(f"检波点数量过滤结果（阈值：{max_receivers}）", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n过滤前后统计:")
    print(f"检波点数量: {len(unique_orig_coords)} -> {len(unique_filtered_coords)}")
    print(f"炮点数量: {len(unique_orig_source)} -> {len(unique_filtered_source)}")
    print(f"对比图已保存为: {output_path}")

def delete_shots_by_receiver_count(filename, new_filename, max_receivers, batch_size=100000):
    """删除检波点数量超过阈值的炮点"""
    with segyio.open(filename, "r", ignore_geometry=True) as segyfile:
        if segyfile.mmap():
            print(f"\n文件 {filename} 已成功进行内存映射")
        else:
            print(f"\n文件 {filename} 内存映射失败！")
        
        # 获取道数
        num_traces = segyfile.tracecount
        
        # 初始化数组存储所有坐标
        sourceX_all = np.zeros(num_traces)
        sourceY_all = np.zeros(num_traces)
        groupX_all = np.zeros(num_traces)
        groupY_all = np.zeros(num_traces)
        
        # 使用字典来存储每个炮点的检波点数量
        shot_receiver_count = defaultdict(int)
        
        # 第一次遍历：统计每个炮点的检波点数量
        print("\n第一步：统计每个炮点的检波点数量")
        num_batches = (num_traces + batch_size - 1) // batch_size
        for batch in tqdm(range(num_batches), desc="读取坐标"):
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
            
            # 统计检波点数量
            shot_points = sourceX + 1j * sourceY
            for shot in shot_points:
                shot_receiver_count[shot] += 1
        
        # 找出需要保留的炮点
        valid_shots = {shot for shot, count in shot_receiver_count.items() 
                      if count <= max_receivers}
        
        # 第二次遍历：找出要保留的道
        print("\n第二步：筛选符合条件的道")
        keep_indices = []
        for i in range(num_traces):
            shot = sourceX_all[i] + 1j * sourceY_all[i]
            if shot in valid_shots:
                keep_indices.append(i)
        
        keep_indices = np.array(keep_indices)
        
        # 创建输出文件的规格
        spec = segyio.spec()
        spec.samples = segyfile.samples
        spec.format = segyfile.format
        spec.sorting = segyfile.sorting
        spec.tracecount = len(keep_indices)
        
        # 绘制过滤前后的对比图
        plot_filter_result(groupX_all, groupY_all, sourceX_all, sourceY_all, 
                         keep_indices, "../fig_0118/O_receiver_count_filter_result.png")
        
        # 创建新的SEGY文件并写入数据
        print("\n第三步：写入新文件")
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
        
        print(f"\n新的SEGY文件已保存为: {new_filename}")

if __name__ == "__main__":
    filename = "../result_0118/K_final_shot.SEGY"
    new_filename = "../result_0118/O_final_shot.SEGY"
    max_receivers = 4096  # 设置最大检波点数量阈值
    delete_shots_by_receiver_count(filename, new_filename, max_receivers) 