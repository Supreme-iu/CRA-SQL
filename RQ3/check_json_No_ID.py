import json


def find_missing_ids(json_data):
    # 提取所有键并转换为整数
    ids = [int(key) for key in json_data.keys()]

    if not ids:  # 如果JSON为空
        return []

    # 获取最小和最大ID
    min_id = min(ids)
    max_id = max(ids)

    # 生成完整的ID范围
    full_range = set(range(min_id, max_id + 1))

    # 获取现有的ID集合
    existing_ids = set(ids)

    # 找出缺失的ID
    missing_ids = sorted(full_range - existing_ids)

    return missing_ids


# 示例使用
if __name__ == "__main__":
    # 从文件读取JSON数据
    with open('临时.json', 'r') as f:
        data = json.load(f)

    missing_ids = find_missing_ids(data)
    print("缺失的ID:", missing_ids)