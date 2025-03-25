"""
此脚本用于随机删除SEGY文件中指定百分比的炮点。

主要步骤：
1. 使用内存映射和批处理方式读取SEGY文件
2. 获取唯一的炮点坐标
3. 随机选择指定百分比的炮点进行删除
4. 创建新的SEGY文件保存结果

输入：
- filename: 输入SEGY文件路径
- new_filename: 输出SEGY文件路径
- keep_percentage: 要保留的炮点百分比（0-100）

输出：
- 保留指定炮点后的新SEGY文件
- 删除前后的对比图
"""

import segyio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_filter_result(groupX, groupY, sourceX, sourceY, 
                      keep_indices, output_path):
    """绘制过滤前后的对比图"""
    # 去重处理原始坐标
    unique_orig_coords = np.unique(np.column_stack((groupX, groupY)), axis=0)
    unique_orig_source = np.unique(np.column_stack((sourceX, sourceY)), axis=0)
    
    # 获取保留的坐标并去重
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
    
    plt.suptitle("Random Shot Selection Result", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n过滤前后统计:")
    print(f"检波点数量: {len(unique_orig_coords)} -> {len(unique_filtered_coords)}")
    print(f"炮点数量: {len(unique_orig_source)} -> {len(unique_filtered_source)}")
    print(f"对比图已保存为: {output_path}")

def random_delete_shots(filename, new_filename, keep_percentage, batch_size=100000):
    """随机保留指定百分比的炮点"""
    with segyio.open(filename, "r", ignore_geometry=True) as segyfile:
        if segyfile.mmap():
            print(f"\n文件 {filename} 已成功进行内存映射")
        else:
            print(f"\n文件 {filename} 内存映射失败！")
            
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
        for batch in tqdm(range(num_batches), desc="读取坐标"):
            start = batch * batch_size
            end = min((batch + 1) * batch_size, num_traces)

            # 读取这个批次的坐标
            sourceX_all[start:end] = segyfile.attributes(segyio.TraceField.SourceX)[start:end]
            sourceY_all[start:end] = segyfile.attributes(segyio.TraceField.SourceY)[start:end]
            groupX_all[start:end] = segyfile.attributes(segyio.TraceField.GroupX)[start:end]
            groupY_all[start:end] = segyfile.attributes(segyio.TraceField.GroupY)[start:end]

        # 获取唯一的炮点
        unique_sources = np.unique(np.column_stack((sourceX_all, sourceY_all)), axis=0)
        num_sources = len(unique_sources)
        num_to_keep = int(num_sources * keep_percentage / 100)
        
        print(f"\n原始炮点数量: {num_sources}")
        print(f"将保留 {num_to_keep} 个炮点 ({keep_percentage}%)")
        
        # 随机选择要保留的炮点
        np.random.seed(42)  # 设置随机种子以保证结果可重复
        keep_source_indices = np.random.choice(num_sources, num_to_keep, replace=False)
        keep_sources = unique_sources[keep_source_indices]
        
        # 创建复数数组来表示要保留的炮点坐标对
        keep_complex = keep_sources[:, 0] + 1j * keep_sources[:, 1]
        
        # 使用向量化操作找出要保留的道
        source_complex = sourceX_all + 1j * sourceY_all
        keep_mask = np.isin(source_complex, keep_complex)
        keep_indices = np.where(keep_mask)[0]
        
        spec.tracecount = len(keep_indices)
        
        # 绘制过滤前后的对比图
        plot_filter_result(groupX_all, groupY_all, sourceX_all, sourceY_all, 
                         keep_indices, "../fig_0118/M_random_delete_shots_result.png")

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
                        segyio.TraceField.TRACE_SAMPLE_COUNT: 2500,
                        segyio.TraceField.TRACE_SAMPLE_INTERVAL: batch_headers['sample_interval'][i],
                        segyio.TraceField.GroupX: int(groupX_all[orig_index]),
                        segyio.TraceField.GroupY: int(groupY_all[orig_index]),
                        segyio.TraceField.SourceX: int(sourceX_all[orig_index]),
                        segyio.TraceField.SourceY: int(sourceY_all[orig_index])
                    })

        print(f"\n新的SEGY文件已保存为: {new_filename}")

if __name__ == "__main__":
    filename = "../result_0118/O_final_shot.SEGY"
    new_filename = "../result_0118/O_final_shot_test.SEGY"
    keep_percentage = 2  # 保留30%的炮点
    random_delete_shots(filename, new_filename, keep_percentage) 