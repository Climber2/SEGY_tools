"""
速度模型插值脚本

输入：
- 原始SEGY文件路径：包含深度间隔为50m的速度模型
- 输出SEGY文件路径：将保存深度间隔为10m的插值后速度模型

描述：
此脚本将原始速度模型（深度间隔为50m，z方向有201个点）插值为更高分辨率的模型
（深度间隔为10m，z方向有1001个点）。插值过程保持xy坐标和其他属性不变，
只对每一道的速度值（raw数组）进行插值处理。

输出：
- 新的SEGY文件，包含插值后的速度模型
"""

import segyio
import numpy as np
from tqdm import tqdm
from scipy import interpolate

def interpolate_velocity_model(input_filename, output_filename, batch_size=100):
    """
    将速度模型从50m间隔插值到10m间隔
    
    参数：
    - input_filename: 输入SEGY文件路径，包含50m间隔的速度模型
    - output_filename: 输出SEGY文件路径，将保存10m间隔的速度模型
    - batch_size: 批处理的道数，用于优化内存使用
    
    返回：
    - None，结果保存到output_filename指定的文件
    """
    # 打开输入文件
    with segyio.open(input_filename, "r", ignore_geometry=True) as segyfile:
        # 尝试内存映射以提高性能
        if segyfile.mmap():
            print(f"\n文件 {input_filename} 已成功内存映射。")
        else:
            print(f"\n文件 {input_filename} 内存映射失败！")
        
        # 获取文件规格
        spec = segyio.spec()
        spec.format = segyfile.format
        spec.sorting = segyfile.sorting
        
        # 获取原始采样点和道数
        orig_samples = segyfile.samples
        num_traces = segyfile.tracecount
        
        # 原始深度采样点（201个点，间隔50m）
        orig_depth = orig_samples
        
        # 新的深度采样点（1001个点，间隔10m）
        # 确保新的深度范围与原始深度范围相同
        start_depth = orig_depth[0]
        end_depth = orig_depth[-1]
        new_samples = np.linspace(start_depth, end_depth, 1001)
        
        # 更新规格中的采样点
        spec.samples = new_samples
        spec.tracecount = num_traces
        
        print(f"原始采样点数: {len(orig_samples)}")
        print(f"插值后采样点数: {len(new_samples)}")
        
        # 创建新的SEGY文件
        with segyio.create(output_filename, spec) as new_segy:
            # 复制文本头和二进制头
            new_segy.text[0] = segyfile.text[0]
            new_segy.bin = segyfile.bin
            
            # 更新二进制头中的采样点数和采样间隔
            new_segy.bin.update({
                segyio.BinField.Samples: len(new_samples),
                segyio.BinField.Interval: 5000  # 10m = 5000微秒
            })
            
            # 分批处理所有道
            num_batches = (num_traces + batch_size - 1) // batch_size
            for batch in tqdm(range(num_batches), desc="插值处理中"):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, num_traces)
                
                # 处理这个批次的每一道
                for i in range(start_idx, end_idx):
                    # 读取原始道数据
                    orig_trace = segyfile.trace.raw[i]
                    
                    # 创建插值函数
                    f = interpolate.interp1d(orig_depth, orig_trace, kind='cubic', bounds_error=False, fill_value="extrapolate")
                    
                    # 应用插值函数生成新的道数据
                    new_trace = f(new_samples)
                    
                    # 写入新的道数据
                    new_segy.trace.raw[i] = new_trace
                    
                    # 复制原始道头信息
                    trace_header = segyfile.header[i]
                    new_segy.header[i] = trace_header
                    
                    # 更新采样点数和采样间隔
                    new_segy.header[i].update({
                        segyio.TraceField.TRACE_SAMPLE_COUNT: len(new_samples),
                        segyio.TraceField.TRACE_SAMPLE_INTERVAL: 5000  # 10m = 5000微秒
                    })
            
            print(f"插值完成，新的SEGY文件已保存为 {output_filename}")

if __name__ == "__main__":
    # 设置输入和输出文件路径
    input_filename = "../result_0118/K_final_vel_full_new.segy"  # 请替换为实际的输入文件路径
    output_filename = "../result_0118/K_final_vel_full_10m_new.segy"  # 请替换为实际的输出文件路径
    
    # 执行插值
    interpolate_velocity_model(input_filename, output_filename) 