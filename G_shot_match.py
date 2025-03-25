"""
此脚本用于将炮点坐标匹配到规则网格点上。

主要步骤：
1. 读取规则化的网格点坐标
2. 读取SEGY文件中的炮点和检波点坐标
3. 按照从右下到左上的顺序匹配炮点到网格点
4. 对超出距离阈值的炮点进行舍弃
5. 创建新的SEGY文件保存结果

输入：
- grid_file: 规则网格点坐标文件(.npy格式)
- segy_file: 输入SEGY文件路径
- new_filename: 输出SEGY文件路径
- max_distance: 匹配的最大距离阈值
- output_fig_path: 匹配结果对比图输出路径

输出：
- 匹配后的新SEGY文件
- 匹配统计信息
"""

import segyio
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def plot_match_result(orig_shots, matched_shots, grid_points, output_path):
    """绘制匹配前后的对比图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20))
    
    # 绘制原始炮点
    ax1.scatter(grid_points[:, 0], grid_points[:, 1], 
               color='gray', alpha=0.3, s=10, label='Grid Points')
    ax1.scatter(orig_shots[:, 0], orig_shots[:, 1], 
               color='red', alpha=0.8, s=10, label='Original Shots')
    ax1.set_title(f"Original Shots\n({len(orig_shots)} points)")
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.legend()
    ax1.grid(True)
    ax1.set_aspect('equal')

    # 绘制匹配后的炮点
    ax2.scatter(grid_points[:, 0], grid_points[:, 1], 
               color='gray', alpha=0.3, s=10, label='Grid Points')
    ax2.scatter(matched_shots[:, 0], matched_shots[:, 1], 
               color='red', alpha=0.8, s=5, label='Matched Shots')
    ax2.set_title(f"Matched Shots\n({len(matched_shots)} points)")
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.legend()
    ax2.grid(True)
    ax2.set_aspect('equal')

    plt.suptitle("Shot Points Matching Result", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"匹配结果对比图已保存为: {output_path}")

def sort_shots_from_right_bottom_to_left_top(shot_points):
    """从右下到左上排序炮点"""
    indices = np.lexsort((-shot_points[:, 0], -shot_points[:, 1]))
    return shot_points[indices]

def match_shots_to_grid(shot_points, grid_points, max_distance):
    """
    将炮点匹配到网格点上，超出距离阈值的炮点将被舍弃
    返回匹配字典和匹配后的炮点坐标数组
    """
    # 从右下到左上排序炮点
    sorted_shots = sort_shots_from_right_bottom_to_left_top(shot_points)
    
    # 初始化匹配结果
    shot_to_grid = {}
    matched_grid = np.zeros(len(grid_points), dtype=bool)
    matched_shots = []
    
    # 对每个炮点进行匹配
    for shot in sorted_shots:
        # 计算到所有未匹配网格点的距离
        distances = cdist([shot], grid_points[~matched_grid], 'euclidean')
        
        # 如果存在未匹配的网格点且最小距离在阈值内
        if len(distances[0]) > 0 and np.min(distances) <= max_distance:
            # 找到最近的未匹配网格点
            closest_index = np.argmin(distances)
            global_index = np.where(~matched_grid)[0][closest_index]
            
            # 记录匹配结果
            shot_tuple = tuple(shot)
            shot_to_grid[shot_tuple] = grid_points[global_index]
            matched_grid[global_index] = True
            matched_shots.append(grid_points[global_index])
    
    return shot_to_grid, np.array(matched_shots)

def process_segy_file(grid_file, segy_file, new_filename, max_distance, output_fig_path, batch_size=100000):
    """处理SEGY文件，将炮点匹配到网格点上"""
    # 加载网格点坐标
    print("\n加载网格点坐标...")
    grid_points = np.load(grid_file)
    
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
        
        # 读取所有坐标
        print("\n读取所有坐标...")
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
        
        # 获取唯一的炮点坐标
        unique_shots = np.unique(np.column_stack((sourceX, sourceY)), axis=0)
        print(f"\n原始唯一炮点数量: {len(unique_shots)}")
        
        # 匹配炮点到网格点
        print("\n进行炮点匹配...")
        shot_to_grid, matched_shots = match_shots_to_grid(unique_shots, grid_points, max_distance)
        print(f"成功匹配炮点数量: {len(matched_shots)}")
        
        # 绘制匹配结果对比图
        plot_match_result(unique_shots, matched_shots, grid_points, output_fig_path)
        
        # 找出要保留的道的索引
        keep_indices = []
        for i in range(num_traces):
            orig_shot = (sourceX[i], sourceY[i])
            if orig_shot in shot_to_grid:
                keep_indices.append(i)
        
        keep_indices = np.array(keep_indices)
        print(f"\n原始道数: {num_traces}")
        print(f"保留道数: {len(keep_indices)}")
        
        # 更新spec中的道数
        spec.tracecount = len(keep_indices)
        
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
                    orig_shot = (sourceX[orig_index], sourceY[orig_index])
                    new_shot = shot_to_grid[orig_shot]  # 这里一定能找到匹配，因为已经过滤过了
                    
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
                        segyio.TraceField.SourceX: int(new_shot[0]),
                        segyio.TraceField.SourceY: int(new_shot[1])
                    })

        print(f"\n新的SEGY文件已保存为: {new_filename}")

if __name__ == "__main__":
    # # 输入参数
    # grid_file = "../result_0118/F_regularized_vel_receivers.npy"
    # segy_file = "../result_0118/C_vel_shot_filter.segy"
    # new_filename = "../result_0118/G_vel_shot_match.segy"
    # max_distance = 30  # 最大匹配距离阈值
    # output_fig_path = "../fig_0118/G_vel_shot_match_result.png"
    # batch_size = 100000  # 批处理大小

    # 输入参数
    grid_file = "../result_0118/F_regularized_vel_receivers_new.npy"
    segy_file = "../result_0118/H_vel_receiver_match_new.segy"
    new_filename = "../result_0118/G_vel_shot_match_new.segy"
    max_distance = 30  # 最大匹配距离阈值
    output_fig_path = "../fig_0118/G_vel_shot_match_result.png"
    batch_size = 100000  # 批处理大小
    
    # 执行处理
    process_segy_file(grid_file, segy_file, new_filename, max_distance, output_fig_path, batch_size)
