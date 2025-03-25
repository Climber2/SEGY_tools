"""
输入：
- filename: 原始SEGY文件路径，需要进行坐标旋转的文件
- new_filename: 新SEGY文件路径，保存旋转后的结果
- center_file: 用于计算旋转中心和角度的参考SEGY文件

描述：
此脚本用于处理SEGY格式数据的坐标旋转：
1. 使用内存映射方式读取SEGY文件
2. 从center_file计算旋转参数
3. 对filename中的坐标进行旋转变换
4. 使用numpy向量化操作进行坐标转换
5. 优化文件写入操作

输出：
- 包含旋转后坐标的新SEGY文件
- 处理进度信息
"""

import segyio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def calculate_rotation_params(center_file, batch_size=100000):
    """从参考文件计算旋转参数"""
    print("从参考文件计算旋转参数...")
    with segyio.open(center_file, "r", ignore_geometry=True) as center:
        if center.mmap():
            print(f"参考文件 {center_file} 已成功进行内存映射")
        else:
            print(f"参考文件 {center_file} 内存映射失败！")
        
        num_traces = center.tracecount
        x_all = np.zeros(num_traces)
        y_all = np.zeros(num_traces)
        
        # 分批读取所有坐标
        for batch_start in tqdm(range(0, num_traces, batch_size), desc="读取参考文件坐标"):
            batch_end = min(batch_start + batch_size, num_traces)
            x_all[batch_start:batch_end] = center.attributes(segyio.TraceField.GroupX)[batch_start:batch_end]
            y_all[batch_start:batch_end] = center.attributes(segyio.TraceField.GroupY)[batch_start:batch_end]
        # 对检波点坐标进行去重, 计算中心点, 并中心化坐标
        unique_coords = np.unique(np.column_stack((x_all, y_all)), axis=0)
        group_centerX = np.mean(unique_coords[:, 0])
        group_centerY = np.mean(unique_coords[:, 1])
        x_centered = unique_coords[:, 0] - group_centerX
        y_centered = unique_coords[:, 1] - group_centerY
        # 计算协方差矩阵, 计算旋转角度, 并构造旋转矩阵
        coords = np.vstack((x_centered, y_centered))
        cov_matrix = coords @ coords.T / len(unique_coords)
        _, eigenvectors = np.linalg.eig(cov_matrix)
        principal_vector = eigenvectors[:, 0]
        angle = np.arctan2(principal_vector[1], principal_vector[0])
        rotation_matrix = np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle),  np.cos(-angle)]
        ])
        # 输出计算出来的旋转角度
        print(f"计算得到的旋转角度: {np.degrees(angle)} 度")
        return group_centerX, group_centerY, rotation_matrix

def plot_rotation_result(groupX, groupY, sourceX, sourceY, 
                        final_groupX, final_groupY, final_sourceX, final_sourceY,
                        output_path):
    """
    绘制旋转前后的对比图，使用去重后的坐标点进行绘制
    """
    # 去重处理
    unique_orig_coords = np.unique(np.column_stack((groupX, groupY)), axis=0)
    unique_orig_source = np.unique(np.column_stack((sourceX, sourceY)), axis=0)
    unique_final_coords = np.unique(np.column_stack((final_groupX, final_groupY)), axis=0)
    unique_final_source = np.unique(np.column_stack((final_sourceX, final_sourceY)), axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 绘制旋转前的坐标（使用去重后的数据）
    ax1.scatter(unique_orig_coords[:, 0], unique_orig_coords[:, 1], 
               color='blue', label=f'Receivers ({len(unique_orig_coords)})', 
               alpha=0.5, s=10)
    ax1.scatter(unique_orig_source[:, 0], unique_orig_source[:, 1], 
               color='green', label=f'Sources ({len(unique_orig_source)})', 
               alpha=0.8, s=20)
    ax1.set_title("Before Rotation")
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制旋转后的坐标（使用去重后的数据）
    ax2.scatter(unique_final_coords[:, 0], unique_final_coords[:, 1], 
               color='blue', label=f'Receivers ({len(unique_final_coords)})', 
               alpha=0.5, s=10)
    ax2.scatter(unique_final_source[:, 0], unique_final_source[:, 1], 
               color='green', label=f'Sources ({len(unique_final_source)})', 
               alpha=0.8, s=20)
    ax2.set_title("After Rotation")
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.legend()
    ax2.grid(True)
    
    # 设置相同的比例
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    
    # 添加总标题
    plt.suptitle("Coordinate Rotation Comparison", fontsize=16)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"旋转对比图已保存为: {output_path}")

def process_segy(filename, new_filename, center_file, batch_size=100000):
    # 从参考文件计算旋转参数
    group_centerX, group_centerY, rotation_matrix = calculate_rotation_params(center_file, batch_size)

    # 打开需要旋转的SEGY文件
    with segyio.open(filename, "r", ignore_geometry=True) as segyfile:
        if segyfile.mmap():
            print(f"\n文件 {filename} 已成功进行内存映射")
        else:
            print(f"\n文件 {filename} 内存映射失败！")

        num_traces = segyfile.tracecount
        
        # 读取需要旋转的文件的坐标
        groupX = np.zeros(num_traces)
        groupY = np.zeros(num_traces)
        sourceX = np.zeros(num_traces)
        sourceY = np.zeros(num_traces)
        
        for batch_start in tqdm(range(0, num_traces, batch_size), desc="读取待旋转文件坐标"):
            batch_end = min(batch_start + batch_size, num_traces)
            groupX[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.GroupX)[batch_start:batch_end]
            groupY[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.GroupY)[batch_start:batch_end]
            sourceX[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.SourceX)[batch_start:batch_end]
            sourceY[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.SourceY)[batch_start:batch_end]

        # 向量化处理所有坐标
        # 检波点
        group_coords = np.vstack((groupX - group_centerX, groupY - group_centerY)).T
        rotated_group = group_coords @ rotation_matrix.T
        final_groupX = rotated_group[:, 0] + group_centerX
        final_groupY = rotated_group[:, 1] + group_centerY

        # 炮点
        source_coords = np.vstack((sourceX - group_centerX, sourceY - group_centerY)).T
        rotated_source = source_coords @ rotation_matrix.T
        final_sourceX = rotated_source[:, 0] + group_centerX
        final_sourceY = rotated_source[:, 1] + group_centerY

        # 在计算完旋转坐标后，添加绘图调用
        plot_rotation_result(
            groupX, groupY, sourceX, sourceY,
            final_groupX, final_groupY, final_sourceX, final_sourceY,
            "../fig_0118/B_vel_rotate.png"
        )

        spec = segyio.spec()
        spec.samples = segyfile.samples
        spec.format = segyfile.format
        spec.sorting = segyfile.sorting
        spec.tracecount = num_traces

        # 创建新文件并写入数据
        with segyio.create(new_filename, spec) as new_segy:
            new_segy.text[0] = segyfile.text[0]
            new_segy.bin = segyfile.bin

            # 分批处理
            for batch_start in tqdm(range(0, num_traces, batch_size), desc="处理数据"):
                batch_end = min(batch_start + batch_size, num_traces)
                batch_size_current = batch_end - batch_start

                # 批量写入道数据
                new_segy.trace.raw[batch_start:batch_end] = segyfile.trace.raw[batch_start:batch_end]

                # 批量处理header属性
                headers = np.zeros(batch_size_current, dtype=np.dtype([
                    ('source_depth', 'i4'),
                    ('trace_number', 'i4'), 
                    ('sample_count', 'i4'),
                    ('sample_interval', 'i4')
                ]))
                
                # 使用numpy向量化操作填充header数组
                headers['source_depth'] = segyfile.attributes(segyio.TraceField.SourceDepth)[batch_start:batch_end]
                headers['trace_number'] = segyfile.attributes(segyio.TraceField.TraceNumber)[batch_start:batch_end]
                headers['sample_count'] = segyfile.attributes(segyio.TraceField.TRACE_SAMPLE_COUNT)[batch_start:batch_end]
                headers['sample_interval'] = segyfile.attributes(segyio.TraceField.TRACE_SAMPLE_INTERVAL)[batch_start:batch_end]

                # 逐道写入header
                for i in range(batch_size_current):
                    trace_index = batch_start + i
                    header = headers[i]
                    new_segy.header[trace_index].update({
                        segyio.TraceField.SourceDepth: header['source_depth'],
                        segyio.TraceField.TraceNumber: header['trace_number'],
                        segyio.TraceField.TRACE_SAMPLE_COUNT: header['sample_count'],
                        segyio.TraceField.TRACE_SAMPLE_INTERVAL: header['sample_interval'],
                        segyio.TraceField.GroupX: int(final_groupX[trace_index]),
                        segyio.TraceField.GroupY: int(final_groupY[trace_index]),
                        segyio.TraceField.SourceX: int(final_sourceX[trace_index]),
                        segyio.TraceField.SourceY: int(final_sourceY[trace_index])
                    })

        print(f"\n新的SEGY文件已保存为 {new_filename}")

if __name__ == "__main__":
    filename = "../result_0118/A_merge.SEGY"
    new_filename = "../result_0118/B_rotate.SEGY"
    center_file = "../result_0118/A_merge.SEGY"
    process_segy(filename, new_filename, center_file)
