import os
import json
import tqdm
import argparse
from src.modules import TASL, EnhancedTALOG
from src.rag import RAGModule


def generate_sql(tasl, talog, output_path):
    question_json = tasl.question_json
    output_dic = {}
    start_idx = 0
    # 先尝试读取已有数据（如果文件存在）
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            try:
                output_dic = json.load(f)  # 加载已有数据
            except json.JSONDecodeError:
                output_dic = {}  # 文件内容无效时重置
    # 找到最大的 key（索引），并确定下一个开始的 i
    if output_dic:
        max_id = max(int(k) for k in output_dic.keys())
        start_idx = max_id + 1

    print(f"将从索引 {start_idx} 开始继续生成 SQL...")

    for i in tqdm.tqdm(range(start_idx, len(question_json))):
        db_id = question_json[i]['db_id']
        try:
            sl_schemas = tasl.get_schema(i)
            result = talog.sr2sql(i, sl_schemas)

            if result is None:
                # 生成默认错误SQL
                sql = "SELECT * FROM " + db_id.split('/')[-1].split('.')[0] + "_error"
                print(f"Warning: Generated default SQL for index {i}")
            else:
                _, sql = result

            # 统一格式化处理
            sql = (sql.replace('\"', '')
                   .replace('\\\n', ' ')
                   .replace('\n', ' ')
                   .strip())

            # 确保最终格式正确
            sql = sql + '\t----- bird -----\t' + db_id

        except Exception as e:
            # 异常情况下的兜底处理
            sql = f"SELECT 'ERROR' AS error_message\t----- bird -----\t{db_id}"
            print(f"Error processing index {i}: {str(e)}")

        print("\n" + "=" * 50)
        print("Final SQL:\n" + sql)
        print("=" * 50 + "\n")

        # 更新字典
        output_dic[str(i)] = sql

        # 实时写入文件
        with open(output_path, 'w') as f:
            json.dump(output_dic, f, indent=4)


def parser():
    parser = argparse.ArgumentParser("Text-to-SQL with RAG")
    parser.add_argument('--db_root_path', type=str, default="./data/dev_databases")
    parser.add_argument('--column_meaning_path', type=str, default="./outputs/column_meaning.json")
    parser.add_argument('--example_db', default="./question.json")  # 新增参数
    parser.add_argument('--mode', type=str, default='dev')
    parser.add_argument('--output_path', type=str, default=f"./outputs/predict_dev.json")
    opt = parser.parse_args()
    return opt


def main(opt):
    db_root_path = opt.db_root_path
    print("\n" + "=" * 50)
    print("db_root_path:" + db_root_path)
    print("=" * 50 + "\n")

    column_meaning_path = opt.column_meaning_path
    mode = opt.mode
    output_path = opt.output_path
    example_db = opt.example_db

    rag = RAGModule(example_db)
    tasl = TASL(db_root_path, mode, column_meaning_path)
    # talog = TALOG(db_root_path, mode, rag)
    # 启用RAG
    talog = EnhancedTALOG(db_root_path, mode, rag)
    # 不启用RAG
    #talog = EnhancedTALOG(db_root_path, mode)
    generate_sql(tasl, talog, output_path)


if __name__ == '__main__':
    opt = parser()
    main(opt)
