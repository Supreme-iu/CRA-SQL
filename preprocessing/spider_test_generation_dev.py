import json
import os

import openai  # 或其他大模型API
from tqdm import tqdm

# 配置大模型API（以OpenAI为例）
openai.api_key = "sk-xxxxxxxx"
openai.api_base = "openai"


def generate_evidence(question, sql):
    """调用大模型生成evidence"""
    prompt = f"""
    给定一个Text2SQL任务的问题和对应的SQL查询：
    问题：{question}
    SQL：{sql}

    请生成简明扼要的提示信息，说清楚关键逻辑关系。
    生成要求：
    1. 说明关键列/值和涉及的计算关系
    2. 用英文描述，不要包含其他无关输出，总共不超过3句话
    3. 不要包含SQL语法解释

    优秀案例：
        例1:
        问题: "What is the highest eligible free rate for K-12 students in the schools in Alameda County?",
        SQL: "SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1",
        输出: "Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`",
        例2:
        问题: "In which city can you find the school in the state of California with the lowest latitude coordinates and what is its lowest grade? Indicate the school name.",
        SQL: "SELECT T2.City, T1.`Low Grade`, T1.`School Name` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.State = 'CA' ORDER BY T2.Latitude ASC LIMIT 1",
        输出: "State of California refers to state = 'CA'"
        例3:
        问题: "Provide the name of superhero with superhero ID 294.",
        SQL: "SELECT superhero_name FROM superhero WHERE id = 294",
        输出: "name of superhero refers to superhero_name; superhero ID 294 refers to superhero.id = 294;"
    """

    try:
        response = openai.ChatCompletion.create(
            model="Pro/deepseek-ai/DeepSeek-V3",
            messages=[
                {"role": "system", "content": "你是一个专业的Text2SQL专家"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip().strip('"\'')
    except Exception as e:
        print(f"生成evidence失败: {e}")
        return "自动生成evidence失败，请手动补充"


def process_and_save_incrementally(input_path, output_path):
    """实时处理并保存每个结果"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 初始化existing_data
    existing_data = []
    start_idx = 0

    # 智能处理输出文件
    if os.path.exists(output_path):
        if os.path.getsize(output_path) > 0:
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                start_idx = len(existing_data)
                print(f"检测到已有{start_idx}条记录，将从第{start_idx + 1}条开始处理")
            except json.JSONDecodeError:
                print("警告：输出文件格式无效，将创建新文件")
                existing_data = []
        else:
            print("输出文件为空，将创建新文件")
    else:
        print("输出文件不存在，将创建新文件")

    # 处理新数据
    for idx, item in enumerate(tqdm(data[start_idx:], desc="Processing", initial=start_idx)):
        new_item = {
            "question_id": start_idx + idx,
            "db_id": item["db_id"],
            "question": item["question"],
            "SQL": item["query"],
            "difficulty": item.get("difficulty", "unknown")
        }

        # 生成并打印evidence
        evidence = generate_evidence(item["question"], item["query"])
        new_item["evidence"] = evidence

        # 打印当前生成结果
        print("\n" + "=" * 50)
        print(f"Question {new_item['question_id']}:")
        print(f"Question: {new_item['question']}")
        print(f"SQL: {new_item['SQL']}")
        print(f"Evidence: {evidence}")
        print("=" * 50 + "\n")

        # 添加到数据列表并实时保存
        existing_data.append(new_item)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    input_json = "spider_test.json"  # 替换为你的实际输入文件路径
    output_json = "spider_test_with_evidence.json"  # 输出文件路径

    print("开始处理JSON文件...")
    try:
        process_and_save_incrementally(input_json, output_json)
        print("处理完成！结果已保存到", output_json)
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        print("部分结果可能已保存，请检查输出文件")
