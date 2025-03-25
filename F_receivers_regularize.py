"""
此脚本用于将不规则的检波点坐标规则化。

主要步骤：
1. 读取检波点坐标文件
2. 根据坐标范围和间距自动计算行列数
3. 将不规则散点映射到规则网格上
4. 导出规则化后的坐标文件
5. 生成对比图显示规则化效果

输入：
- coords_file: 检波点坐标文件路径(.npy格式)
- row_spacing: 期望的行间距
- col_spacing: 期望的列间距
- output_coords_path: 规则化后的坐标输出路径
- output_fig_path: 规则化对比图输出路径

输出：
- 规则化后的检波点坐标文件
- 规则化前后的对比图
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import cKDTree

def plot_regularization_result(orig_coords, reg_coords, output_path):
    """绘制规则化前后的对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 绘制原始坐标
    ax1.scatter(orig_coords[:, 0], orig_coords[:, 1], 
               color='blue', alpha=0.5, s=10)
    ax1.set_title(f"Original Receivers\n({len(orig_coords)} points)")
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.grid(True)
    ax1.set_aspect('equal')

    # 绘制规则化后的坐标
    ax2.scatter(reg_coords[:, 0], reg_coords[:, 1], 
               color='red', alpha=0.5, s=10)
    ax2.set_title(f"Regularized Receivers\n({len(reg_coords)} points)")
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.grid(True)
    ax2.set_aspect('equal')

    plt.suptitle("Receiver Coordinates Regularization", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"规则化对比图已保存为: {output_path}")

def regularize_coordinates(coords_file, row_spacing, col_spacing, output_coords_path, output_fig_path):
    """将检波点坐标规则化"""
    # 加载原始坐标
    print(f"\n加载原始坐标...")
    orig_coords = np.load(coords_file)
    print(f"原始坐标点数量: {len(orig_coords)}")
    
    # 计算坐标范围
    x_min, x_max = np.min(orig_coords[:, 0]), np.max(orig_coords[:, 0])
    y_min, y_max = np.min(orig_coords[:, 1]), np.max(orig_coords[:, 1])
    
    # 根据范围和间距计算行列数
    cols = int(round((x_max - x_min) / col_spacing)) + 1
    rows = int(round((y_max - y_min) / row_spacing)) + 1
    
    print(f"\n根据坐标范围计算得到:")
    print(f"X范围: {x_min} - {x_max}")
    print(f"Y范围: {y_min} - {y_max}")
    print(f"列数(X方向): {cols}")
    print(f"行数(Y方向): {rows}")
    
    # 计算网格的总宽度和高度
    total_width = (cols - 1) * col_spacing
    total_height = (rows - 1) * row_spacing
    
    # 计算网格的起始点（确保为整数）
    start_x = int(x_min)
    start_y = int(y_min)
    
    # 创建规则网格（使用整数运算）
    x = np.array([start_x + i * col_spacing for i in range(cols)], dtype=np.int32)
    y = np.array([start_y + i * row_spacing for i in range(rows)], dtype=np.int32)
    X, Y = np.meshgrid(x, y)
    
    # 创建规则化的坐标数组
    reg_coords = np.column_stack((X.ravel(), Y.ravel()))
    
    # 验证间距的一致性
    print(f"\n规则化网格信息:")
    print(f"网格范围: X[{start_x} - {start_x + total_width}]")
    print(f"          Y[{start_y} - {start_y + total_height}]")
    print(f"规则化后坐标点数量: {len(reg_coords)}")
    
    # 验证所有坐标都是整数
    if not np.all(reg_coords == reg_coords.astype(np.int32)):
        raise ValueError("存在非整数坐标！")
    
    # 使用KD树找到每个原始点最近的网格点
    tree = cKDTree(reg_coords)
    distances, _ = tree.query(orig_coords)
    
    print(f"\n映射统计:")
    print(f"最大映射距离: {np.max(distances):.2f}")
    print(f"平均映射距离: {np.mean(distances):.2f}")
    print(f"标准差: {np.std(distances):.2f}")
    
    # 导出规则化后的坐标
    np.save(output_coords_path, reg_coords.astype(np.int32))
    print(f"\n规则化后的坐标已保存为: {output_coords_path}")
    
    # 绘制对比图
    plot_regularization_result(orig_coords, reg_coords, output_fig_path)
    
    return reg_coords

if __name__ == "__main__":
    # 输入参数
    coords_file = "../result_0118/C_vel_receivers_new.npy"
    row_spacing = 50  # 期望的行间距
    col_spacing = 50  # 期望的列间距
    output_coords_path = "../result_0118/F_regularized_vel_receivers_new.npy"
    output_fig_path = "../fig_0118/F_vel_regularization.png"
    
    # 执行规则化
    regularize_coordinates(coords_file, row_spacing, col_spacing, output_coords_path, output_fig_path)