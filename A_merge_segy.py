"""
输入：
- 多个SEGY格式的地震数据文件路径列表
- 输出文件路径

描述：
此脚本用于将多个SEGY文件合并为一个新的SEGY文件：
1. 使用内存映射方式读取SEGY文件
2. 批量读取和处理道数据
3. 保持原始文件的采样率、格式等参数
4. 正确合并所有道数据和头信息
5. 只保留必要的header属性
6. 使用numpy向量化操作优化header写入速度

输出：
- 合并后的新SEGY文件
- 处理进度信息
"""

import segyio
import numpy as np
from tqdm import tqdm

def merge_segy_files(input_files, output_file, batch_size=100000):
    """
    合并多个SEGY文件为一个新的SEGY文件,并输出header属性
    
    参数：
    input_files: list, SEGY文件路径列表
    output_file: str, 输出文件路径
    batch_size: int, 每批次处理的道数
    """
    
    # 首先统计总道数和验证文件兼容性
    total_traces = 0
    first_file = None
    
    print("检查文件兼容性...")
    for filename in input_files:
        with segyio.open(filename, "r", ignore_geometry=True) as segy:
            if first_file is None:
                first_file = {
                    'samples': segy.samples,
                    'format': segy.format,
                    'sorting': segy.sorting,
                }
            else:
                # 验证文件兼容性
                if (segy.samples.size != first_file['samples'].size or
                    str(segy.format) != str(first_file['format'])):
                    raise ValueError(f"文件 {filename} 与第一个文件的格式不兼容")
            
            total_traces += segy.tracecount
            
    print(f"\n总道数: {total_traces}")
    
    # 创建输出文件的规格
    spec = segyio.spec()
    spec.samples = first_file['samples']
    spec.format = first_file['format']
    spec.sorting = first_file['sorting']
    spec.tracecount = total_traces
    
    # 创建新文件并写入数据
    with segyio.create(output_file, spec) as new_segy:
        # 复制第一个文件的文本头和二进制头
        with segyio.open(input_files[0], "r", ignore_geometry=True) as first_segy:
            new_segy.text[0] = first_segy.text[0]
            new_segy.bin = first_segy.bin
        current_trace = 0
        # 处理每个输入文件
        for filename in input_files:
            with segyio.open(filename, "r", ignore_geometry=True) as segy:
                if segy.mmap():
                    print(f"文件 {filename} 已成功进行内存映射")
                else:
                    print(f"文件 {filename} 内存映射失败！")
                num_traces = segy.tracecount
                num_batches = (num_traces + batch_size - 1) // batch_size
                print(f"\n处理文件: {filename}")
                for batch in tqdm(range(num_batches), desc="合并道数据"):
                    start = batch * batch_size
                    end = min((batch + 1) * batch_size, num_traces)
                    batch_size_current = end - start
                    # 批量读写道数据
                    traces = segy.trace[start:end]
                    new_segy.trace[current_trace:current_trace + batch_size_current] = traces
                    # 批量读取道头
                    headers = np.zeros(batch_size_current, dtype=np.dtype([
                        ('trace_number', 'i4'), 
                        ('sample_count', 'i4'), ('sample_interval', 'i4'),
                        ('group_x', 'i4'),      ('group_y', 'i4'),
                        ('source_x', 'i4'),     ('source_y', 'i4')
                    ]))
                    headers['trace_number'] = segy.attributes(segyio.TraceField.TraceNumber)[start:end]
                    headers['sample_count'] = segy.attributes(segyio.TraceField.TRACE_SAMPLE_COUNT)[start:end]
                    headers['sample_interval'] = segy.attributes(segyio.TraceField.TRACE_SAMPLE_INTERVAL)[start:end]
                    headers['group_x'] = segy.attributes(segyio.TraceField.GroupX)[start:end]
                    headers['group_y'] = segy.attributes(segyio.TraceField.GroupY)[start:end]
                    headers['source_x'] = segy.attributes(segyio.TraceField.SourceX)[start:end]
                    headers['source_y'] = segy.attributes(segyio.TraceField.SourceY)[start:end]
                    # 逐道写入道头
                    for i in range(batch_size_current):
                        trace_index = current_trace + i
                        new_segy.header[trace_index].update({
                            segyio.TraceField.TraceNumber: headers[i]['trace_number'],
                            segyio.TraceField.TRACE_SAMPLE_COUNT: headers[i]['sample_count'],
                            segyio.TraceField.TRACE_SAMPLE_INTERVAL: headers[i]['sample_interval'],
                            segyio.TraceField.GroupX: headers[i]['group_x'],
                            segyio.TraceField.GroupY: headers[i]['group_y'],
                            segyio.TraceField.SourceX: headers[i]['source_x'],
                            segyio.TraceField.SourceY: headers[i]['source_y']
                        })
                    current_trace += batch_size_current
                        
    print(f"\n合并完成！新文件已保存为: {output_file}")
    print(f"总道数: {total_traces}")

if __name__ == "__main__":
    # 要合并的文件列表
    input_files = [
        "../data/SP222_MUTE.SEGY",
        "../data/SP220_MUTE.SEGY",
    ]
    
    # 输出文件路径
    output_file = "../result_0118/A_merge.SEGY"
    
    # 执行合并
    merge_segy_files(input_files, output_file, 100000)