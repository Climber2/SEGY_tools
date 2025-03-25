"""
此脚本用于检查SEGY文件的采样点数信息，对比文件头和道头中的采样点数是否一致，并检查每一道的实际数据长度。

主要步骤：
1. 读取SEGY文件头中的采样点数
2. 读取每一道道头中的采样点数
3. 检查每一道的实际二进制数据长度
4. 检查三者是否存在不一致
5. 输出统计信息和异常道信息

输入：
- filename: 输入SEGY文件路径

输出：
- 采样点数统计信息
- 不一致情况的详细报告
- 异常道的具体信息
"""

import segyio
import numpy as np
from tqdm import tqdm

def check_samples(filename, batch_size=100000):
    """检查SEGY文件的采样点数信息和实际数据长度"""
    with segyio.open(filename, "r", ignore_geometry=True) as segyfile:
        if segyfile.mmap():
            print(f"\n文件 {filename} 已成功进行内存映射")
        else:
            print(f"\n文件 {filename} 内存映射失败！")
        
        # 获取文件头中的采样点数
        file_samples = segyfile.samples
        print(f"\n文件头信息:")
        print(f"采样点数: {len(file_samples)}")
        print(f"采样点数组: {file_samples}")
        
        # 获取文件信息
        num_traces = segyfile.tracecount
        print(f"总道数: {num_traces}")
        
        # 分批读取每道的采样点数和实际数据
        print("\n读取道头采样点数和实际数据...")
        trace_samples = np.zeros(num_traces, dtype=np.int32)
        actual_lengths = np.zeros(num_traces, dtype=np.int32)
        
        for batch_start in tqdm(range(0, num_traces, batch_size), desc="读取数据"):
            batch_end = min(batch_start + batch_size, num_traces)
            # 读取道头中的采样点数
            trace_samples[batch_start:batch_end] = segyfile.attributes(segyio.TraceField.TRACE_SAMPLE_COUNT)[batch_start:batch_end]
            
            # 读取实际数据长度
            for i in range(batch_start, batch_end):
                actual_lengths[i] = len(segyfile.trace.raw[i])
        
        # 统计分析
        unique_header_samples = np.unique(trace_samples)
        unique_actual_lengths = np.unique(actual_lengths)
        print(f"\n统计信息:")
        print(f"文件头采样点数: {len(file_samples)}")
        print(f"道头中出现的采样点数: {unique_header_samples}")
        print(f"实际数据长度: {unique_actual_lengths}")
        
        # 检查不一致 - 使用numpy的比较操作
        header_mismatch = np.where(trace_samples != len(file_samples))[0]
        data_mismatch = np.where(actual_lengths != len(file_samples))[0]
        header_data_mismatch = np.where(trace_samples != actual_lengths)[0]
        
        # 输出不一致信息
        if len(header_mismatch) > 0:
            print(f"\n发现 {len(header_mismatch)} 道的道头采样点数与文件头不一致!")
            print("\n前10个不一致的道:")
            for idx in header_mismatch[:10]:
                print(f"道 {idx + 1}: 道头采样点数 = {trace_samples[idx]}, 文件头采样点数 = {len(file_samples)}")
        
        if len(data_mismatch) > 0:
            print(f"\n发现 {len(data_mismatch)} 道的实际数据长度与文件头不一致!")
            print("\n前10个不一致的道:")
            for idx in data_mismatch[:10]:
                print(f"道 {idx + 1}: 实际数据长度 = {actual_lengths[idx]}, 文件头采样点数 = {len(file_samples)}")
        
        if len(header_data_mismatch) > 0:
            print(f"\n发现 {len(header_data_mismatch)} 道的道头采样点数与实际数据长度不一致!")
            print("\n前10个不一致的道:")
            for idx in header_data_mismatch[:10]:
                print(f"道 {idx + 1}: 道头采样点数 = {trace_samples[idx]}, 实际数据长度 = {actual_lengths[idx]}")
        
        if len(header_mismatch) == 0 and len(data_mismatch) == 0 and len(header_data_mismatch) == 0:
            print("\n所有道的采样点数和实际数据长度都一致")
        
        # 输出分布统计
        print("\n采样点数和数据长度分布:")
        print("\n道头采样点数分布:")
        for sample_count in unique_header_samples:
            count = np.sum(trace_samples == sample_count)
            print(f"采样点数 {sample_count}: {count} 道 ({count/num_traces*100:.2f}%)")
        
        print("\n实际数据长度分布:")
        for length in unique_actual_lengths:
            count = np.sum(actual_lengths == length)
            print(f"数据长度 {length}: {count} 道 ({count/num_traces*100:.2f}%)")

if __name__ == "__main__":
    filename = "../result_0118/K_final_shot.SEGY"
    check_samples(filename)