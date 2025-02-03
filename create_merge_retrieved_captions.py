import json
from collections import defaultdict

# 读取两个JSON文件
with open('data/LEVIR_CC/TRAIN_retrived_caps.json', 'r') as f1, open('data/LEVIR_CC/v1/TEST_retrived_caps_v2.json', 'r') as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

# 使用 defaultdict 初始化合并后的字典
merged_data = defaultdict(list)

# 合并第一个文件的数据
for key, value in data1.items():
    merged_data[key].extend(value)

# 合并第二个文件的数据
for key, value in data2.items():
    merged_data[key].extend(value)

# 将 defaultdict 转换为普通字典
merged_data = dict(merged_data)

# 将合并后的数据写入新文件
with open('data/LEVIR_CC/merged_retrieved_captions.json', 'w') as outfile:
    json.dump(merged_data, outfile, indent=2)

# 输出合并后的结构示例（仅用于演示）
print("Merged JSON Structure Example:")
for key in merged_data:
    print(f'"{key}": {len(merged_data[key])} entries')