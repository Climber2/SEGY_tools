"""
此脚本用于检查大型SEGY格式炮集数据或导出的坐标文件的检波点坐标矩阵间距一致性。

主要步骤：
1. 根据文件类型选择读取方式:
   - SEGY文件: 使用内存映射和批处理方式读取
   - NPY文件: 直接加载坐标数组
2. 对检波点坐标进行去重处理
3. 检查去重后的坐标矩阵的行列间距一致性
4. 输出详细的检查结果和统计信息
5. 导出去重后的检波点坐标文件

输入：
- filename: 文件路径，支持SEGY或NPY格式
- batch_size: 批处理大小，默认100000（仅SEGY格式有效）

输出：
- 检波点坐标的行列间距统计信息
- 间距一致性检查结果
- 坐标分布可视化图
- 去重后的检波点坐标文件(npy格式)
"""

import segyio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def plot_receiver_grid(unique_coords, x_diffs, y_diffs, output_path):
    """绘制检波点网格分布图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 绘制检波点分布
    ax1.scatter(unique_coords[:, 0], unique_coords[:, 1], 
               color='blue', alpha=0.5, s=10)
    ax1.set_title("Receiver Positions")
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.grid(True)
    ax1.set_aspect('equal')

    # 绘制间距分布直方图
    ax2.hist(x_diffs, bins=50, alpha=0.5, label='X spacing', color='blue')
    ax2.hist(y_diffs, bins=50, alpha=0.5, label='Y spacing', color='red')
    ax2.set_title("Spacing Distribution")
    ax2.set_xlabel('Spacing')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"网格分布图已保存为: {output_path}")

def check_geometry(filename, output_coords_path, output_fig_path, batch_size=100000):
    """检查检波点坐标的网格间距一致性"""
    # 根据文件扩展名判断文件类型
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext == '.npy':
        # 直接从NPY文件加载坐标
        print(f"\n从NPY文件加载坐标...")
        coords = np.load(filename)
        groupX = coords[:, 0]
        groupY = coords[:, 1]
        print(f"已加载坐标点数量: {len(coords)}")
        
    elif file_ext in ['.segy', '.sgy','.SEGY','.SGY']:
        # 从SEGY文件读取坐标
        with segyio.open(filename, "r", ignore_geometry=True) as segyfile:
            if segyfile.mmap():
                print(f"\n文件 {filename} 已成功进行内存映射")
            else:
                print(f"\n文件 {filename} 内存映射失败！")
            
            num_traces = segyfile.tracecount
            print(f"总道数: {num_traces}")
            
            # 分批读取所有检波点坐标
            groupX = np.zeros(num_traces)
            groupY = np.zeros(num_traces)
            
            for batch_start in tqdm(range(0, num_traces, batch_size), desc="读取坐标"):
                batch_end = min(batch_start + batch_size, num_traces)
                groupX[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.GroupX)[batch_start:batch_end]
                groupY[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.GroupY)[batch_start:batch_end]
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")

    # 对坐标进行去重
    print("\n进行坐标去重...")
    unique_coords = np.unique(np.column_stack((groupX, groupY)), axis=0)
    print(f"检波点数量: {len(groupX)} -> 去重后: {len(unique_coords)}")
    
    # 导出去重后的检波点坐标
    np.save(output_coords_path, unique_coords)
    print(f"去重后的检波点坐标已保存为: {output_coords_path}")
    
    # 获取唯一的X和Y坐标
    unique_x = np.unique(unique_coords[:, 0])
    unique_y = np.unique(unique_coords[:, 1])
    
    print(f"\nX坐标数量: {len(unique_x)}")
    print(f"Y坐标数量: {len(unique_y)}")
    
    # 计算相邻点间距
    x_diffs = np.diff(unique_x)
    y_diffs = np.diff(unique_y)
    
    # 分析间距统计特征
    x_spacing_stats = {
        'min': np.min(x_diffs),
        'max': np.max(x_diffs),
        'mean': np.mean(x_diffs),
        'std': np.std(x_diffs),
        'unique_values': np.unique(x_diffs)
    }
    
    y_spacing_stats = {
        'min': np.min(y_diffs),
        'max': np.max(y_diffs),
        'mean': np.mean(y_diffs),
        'std': np.std(y_diffs),
        'unique_values': np.unique(y_diffs)
    }
    
    # 判断间距一致性
    x_consistent = len(np.unique(x_diffs)) == 1
    y_consistent = len(np.unique(y_diffs)) == 1
    
    # 输出结果
    print("\nX方向间距统计:")
    print(f"最小值: {x_spacing_stats['min']}")
    print(f"最大值: {x_spacing_stats['max']}")
    print(f"平均值: {x_spacing_stats['mean']:.2f}")
    print(f"标准差: {x_spacing_stats['std']:.2f}")
    print(f"唯一值: {x_spacing_stats['unique_values']}")
    print(f"间距一致性: {x_consistent}")
    
    print("\nY方向间距统计:")
    print(f"最小值: {y_spacing_stats['min']}")
    print(f"最大值: {y_spacing_stats['max']}")
    print(f"平均值: {y_spacing_stats['mean']:.2f}")
    print(f"标准差: {y_spacing_stats['std']:.2f}")
    print(f"唯一值: {y_spacing_stats['unique_values']}")
    print(f"间距一致性: {y_consistent}")
    
    # 绘制检波点分布和间距直方图
    plot_receiver_grid(unique_coords, x_diffs, y_diffs, output_fig_path)
    
    return x_consistent, y_consistent, x_spacing_stats, y_spacing_stats

if __name__ == "__main__":
    filename = "../result_0118/H_vel_receiver_match_new.segy"
    output_coords_path = "../result_0118/C_vel_receivers_new.npy"
    output_fig_path = "../fig_0118/C_vel_geometry_check.png"
    check_geometry(filename, output_coords_path, output_fig_path)