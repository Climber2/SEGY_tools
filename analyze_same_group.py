"""
此脚本用于分析 SEGY 格式炮集数据中炮点和检波点的关系。通过统计每个检波点对应的炮点数量，找出对应炮点数最多的检波点集合，并展示相关信息。

主要功能：
1. 从多个SEGY文件中读取炮点(SourceX, SourceY)和检波点(GroupX, GroupY)的坐标。
2. 去重并显示所有检波点。
3. 统计每个检波点集合对应的炮点数量，找出对应炮点数最多的检波点集合。
4. 显示所有去重后的炮点。
5. 显示对应炮点数最多的检波点集合及其对应的炮点。
6. 绘制所有去重后的检波点和炮点。

参数说明：
- filenames: 要处理的SEGY文件的路径列表。

输出：
- 控制台输出包括所有检波点、炮点、对应炮点数最多的检波点集合及其对应的炮点。
"""

import segyio
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyze_segy_files(filenames, batch_size=100000):
    all_source_coords = []
    all_group_coords = []
    
    # 读取SEGY文件并提取坐标
    for filename in filenames:
        with segyio.open(filename, ignore_geometry=True) as segyfile:
            sourceX = segyfile.attributes(segyio.TraceField.SourceX)
            sourceY = segyfile.attributes(segyio.TraceField.SourceY)
            groupX = segyfile.attributes(segyio.TraceField.GroupX)  
            groupY = segyfile.attributes(segyio.TraceField.GroupY)
            
            num_traces = segyfile.tracecount
            num_batches = (num_traces + batch_size - 1) // batch_size
            
            for batch in tqdm(range(num_batches), desc=f"Processing {filename}"):
                start = batch * batch_size
                end = min((batch + 1) * batch_size, num_traces)
                
                all_source_coords.extend(zip(sourceX[start:end], sourceY[start:end]))
                all_group_coords.extend(zip(groupX[start:end], groupY[start:end]))
                
    print("for loop end")
    
    # 创建DataFrame  
    df = pd.DataFrame({
        'sx': [coord[0] for coord in all_source_coords],
        'sy': [coord[1] for coord in all_source_coords],
        'gx': [coord[0] for coord in all_group_coords],
        'gy': [coord[1] for coord in all_group_coords]
    })
    print("df created")
    
    # 去重
    unique_group_coords = df[['gx', 'gy']].drop_duplicates()
    unique_source_coords = df[['sx', 'sy']].drop_duplicates()
    print("unique_group_coords created")
    # 创建字典以存储检波点集合和对应的炮点集合
    group_to_sources = {}

    # 遍历去重后的炮点集合
    for source in tqdm(unique_source_coords.itertuples(index=False), total=len(unique_source_coords)):
        sx, sy = source.sx, source.sy
        # 获取当前炮点对应的所有检波点
        corresponding_groups = df[(df['sx'] == sx) & (df['sy'] == sy)][['gx', 'gy']].drop_duplicates()
        group_set = frozenset(map(tuple, corresponding_groups.values))  # 使用 frozenset 作为字典的键
        if group_set not in group_to_sources:
            group_to_sources[group_set] = set()
        group_to_sources[group_set].add((sx, sy))  # 将炮点添加到对应的检波点集合中
    print("group_to_sources created")
    # 找出对应炮点数最多的检波点集合
    max_group = max(group_to_sources.items(), key=lambda item: len(item[1]))
    max_group_key, corresponding_sources = max_group
    print("max_group created")
    # 绘制所有去重后的检波点和炮点
    plt.figure(figsize=(10, 8))
    
    # 绘制去重后的检波点
    plt.scatter(unique_group_coords['gx'], unique_group_coords['gy'], color='blue', label='Unique Receivers', alpha=0.6)
    
    # 绘制对应炮点数最多的检波点
    max_group_coords = pd.DataFrame(list(max_group_key), columns=['gx', 'gy'])
    plt.scatter(max_group_coords['gx'], max_group_coords['gy'], color='yellow', marker='*', s=100, label='Max Count Receivers')
    
    # 绘制去重后的炮点
    plt.scatter(unique_source_coords['sx'], unique_source_coords['sy'], color='green', label='Unique Sources', alpha=0.6)
    
    # 绘制对应炮点数最多的检波点集合所对应的所有炮点
    corresponding_sources_df = pd.DataFrame(list(corresponding_sources), columns=['sx', 'sy'])
    plt.scatter(corresponding_sources_df['sx'], corresponding_sources_df['sy'], color='red', marker='o', s=50, label='Corresponding Sources')
    print(corresponding_sources_df)
    plt.title("Seismic Source and Receiver Coordinates")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.savefig("../seismic_analysis.png")
    plt.close()

# 文件名设置
filenames = [
    '../data/SP220_MUTE.SEGY',
    '../data/SP222_MUTE.SEGY',
    # '../data/SHOT_200065.SGY',
]

analyze_segy_files(filenames,batch_size=32) 