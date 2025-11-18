import os
import json
import random
import re
import time

import tqdm
import sqlite3
import csv
from src.prompt_bank import dummy_sql_prompt, sr_examples, generate_sr, sr2sql
from src.llm import collect_response
# from src.llm_local import get_response
from src.rag import RAGModule


class BaseModule():
    def __init__(self, db_root_path, mode):
        self.db_root_path = db_root_path
        self.mode = mode
        table_json_path = os.path.join(db_root_path, f'{mode}_tables.json')
        question_path = os.path.join(db_root_path, f'{mode}.json')
        self.table_json = json.load(open(table_json_path, 'r'))
        self.question_json = json.load(open(question_path, 'r'))
        # self.csv_info, self.value_prompts = self._get_info_from_csv()

    def _get_info_from_csv(self):
        """从CSV文件读取数据库列的详细信息"""
        csv_info = {}  # 存储列信息：数据库+表 -> 列详情
        value_prompt = {}  # 存储列值示例提示
        for i in tqdm.tqdm(range(len(self.table_json))):
            table_info = self.table_json[i]
            db_id = table_info['db_id']
            db_path = os.path.join(self.db_root_path, db_id, f'{db_id}.sqlite')
            db_path = os.path.normpath(db_path)
            print("\n" + "=" * 50)
            print("初始化数据库：" + db_path)
            print("=" * 50 + "\n")
            time.sleep(1)
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            # 处理每个表的CSV描述文件
            csv_dir = os.path.join(self.db_root_path, db_id, 'database_description')
            otn_list = table_info['table_names_original']
            tn_list = table_info['table_names']
            # 遍历原始表名和标准表名
            for otn, tn in zip(otn_list, tn_list):
                # 确定CSV文件路径
                if os.path.exists(os.path.join(csv_dir, f"{tn}.csv")):
                    csv_path = os.path.join(os.path.join(csv_dir, f"{tn}.csv"))
                else:
                    csv_path = os.path.join(os.path.join(csv_dir, f"{otn}.csv"))
                csv_dict = csv.DictReader(open(csv_path, newline='', encoding="latin1"))
                column_info = {}
                # 解析CSV中的列信息
                for row in csv_dict:
                    headers = list(row.keys())
                    ocn_header = [h for h in headers if 'original_column_name' in h][0]  # remove BOM
                    ocn, cn = row[ocn_header].strip(), row['column_name']
                    column_description = row['column_description'].strip()
                    column_type = row['data_format'].strip()
                    column_name = cn if cn not in ['', ' '] else ocn
                    value_description = row['value_description'].strip()
                    # 构建列信息：名称、描述、类型、值示例
                    column_info[ocn] = [column_name, column_description, column_type, value_description]
                    # 获取列值示例用于提示
                    if column_type in ['text', 'date', 'datetime']:
                        sql = f'''SELECT DISTINCT "{ocn}" FROM `{otn}` where "{ocn}" IS NOT NULL ORDER BY RANDOM()'''
                        cursor.execute(sql)
                        values = cursor.fetchall()
                        if len(values) > 0 and len(values[0][0]) < 50:
                            if len(values) <= 10:
                                example_values = [v[0] for v in values]
                                value_prompt[f"{db_id}|{otn}|{ocn}"] = f"all possible values are {example_values}"
                                # value_prompt[f"{db_id}|{otn}|{ocn}"] = f"all possible values of the column are {', '.join(example_values)}."
                            else:
                                example_values = [v[0] for v in values[:3]]
                                value_prompt[f"{db_id}|{otn}|{ocn}"] = f"example values are {example_values}"
                                # value_prompt[f"{db_id}|{otn}|{ocn}"] = f"three example values of the column are {', '.join(example_values)}."

                csv_info[f"{db_id}|{otn}"] = column_info
            # pdb.set_trace()
        return csv_info, value_prompt

    def generate_pk_fk(self, question_id):
        """生成主键和外键信息"""
        question_info = self.question_json[question_id]
        db_id = question_info['db_id']
        table = [content for content in self.table_json if content['db_id'] == db_id][0]
        pk_dict = {}  # 主键字典：表 -> 主键列
        fk_dict = {}  # 外键字典：源列 -> 目标列
        table_names_original = table['table_names_original']
        column_names_original = table['column_names_original']
        primary_keys = table['primary_keys']
        foreign_keys = table['foreign_keys']

        for _, pk_idx in enumerate(primary_keys):
            if type(pk_idx) == int:
                pk_dict[str(table_names_original[column_names_original[pk_idx][0]])] = [
                    column_names_original[pk_idx][-1]]
            else:
                pk_dict[str(table_names_original[column_names_original[pk_idx[0]][0]])] = [
                    column_names_original[idx][-1] for idx in pk_idx]

        for cur_fk in foreign_keys:
            src_col_idx, tgt_col_idx = cur_fk
            src_col_name = str(table_names_original[column_names_original[src_col_idx][0]]) + '.' + str(
                column_names_original[src_col_idx][-1])
            tgt_col_name = str(table_names_original[column_names_original[tgt_col_idx][0]]) + '.' + str(
                column_names_original[tgt_col_idx][-1])
            fk_dict[src_col_name] = tgt_col_name
        return pk_dict, fk_dict


class TASL(BaseModule):
    """语义增强模块，处理数据库模式重构和虚拟SQL生成"""

    def __init__(self, db_root_path, mode, column_meaning_path, max_retries=2):
        super().__init__(db_root_path, mode)
        # 加载列语义描述
        self.column_meanings = json.load(open(column_meaning_path, 'r', encoding='utf-8'))
        self.mode = mode
        self.max_retries = max_retries  # 最大重试次数
        # 重构模式字典
        self.schema_item_dic = self._reconstruct_schema()

    def _reconstruct_schema(self):
        """重构数据库模式，整合列语义信息"""
        schema_item_dic = {}
        db_id_list = [content['db_id'] for content in self.table_json]

        schema_item_dic = {}
        for db_id in db_id_list:
            # 为每个数据库构建表结构
            content = [content for content in self.table_json if content['db_id'] == db_id][0]
            otn_list = content['table_names_original']
            schema_for_db = dict(zip(otn_list, [{} for _ in range(len(otn_list))]))
            schema_item_dic[db_id] = schema_for_db
        # 填充列语义信息
        for key, value in self.column_meanings.items():
            db_id, otn, ocn = key.split('|')
            value = value.replace('#', '')
            value = value.replace('\n', ',  ')
            schema_item_dic[db_id][otn][ocn] = value
        return schema_item_dic

    def _validate_sql(self, sql: str, db_id: str) -> bool:
        """改进的SQL验证方法，支持表别名和更精确的验证"""
        try:
            # 获取数据库schema信息
            db_info = [content for content in self.table_json if content['db_id'] == db_id][0]
            valid_tables = set(db_info["table_names_original"])

            # 构建列名映射：{表名: {列名}}
            column_map = {}
            for table_id, col_name in db_info["column_names_original"][1:]:
                table_name = db_info["table_names_original"][table_id]
                if table_name not in column_map:
                    column_map[table_name] = set()
                column_map[table_name].add(col_name)

            # 1. 识别表别名（包括带AS和不带AS的情况）
            alias_pattern = r'(?:FROM|JOIN)\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?(?=\s|$|,)'
            table_aliases = {}
            for match in re.finditer(alias_pattern, sql, re.IGNORECASE):
                table_name, alias = match.groups()
                if table_name in valid_tables:
                    table_aliases[alias or table_name] = table_name  # 保存别名到原名的映射

            # 如果没有识别到任何有效表，直接返回False
            if not table_aliases:
                return False

            # 2. 列引用验证（支持三种格式：表.列、别名.列、和无前缀列名）
            column_ref_pattern = r'(?:\b([a-zA-Z_]\w*)\.)?`?([a-zA-Z_]\w*)`?\b'
            has_valid_columns = False

            for match in re.finditer(column_ref_pattern, sql):
                qualifier, col_name = match.groups()

                # 情况1：带限定符的列（表.列 或 别名.列）
                if qualifier:
                    # 先检查是否是表别名
                    actual_table = table_aliases.get(qualifier,
                                                     qualifier if qualifier in valid_tables else None)
                    if actual_table and col_name.lower() in {c.lower() for c in column_map.get(actual_table, set())}:
                        has_valid_columns = True
                    else:
                        return False  # 包含无效的限定列引用

                # 情况2：无前缀列名（仅在查询只涉及单表时可靠）
                elif len(table_aliases) == 1:
                    actual_table = list(table_aliases.values())[0]
                    if col_name.lower() in {c.lower() for c in column_map.get(actual_table, set())}:
                        has_valid_columns = True
                    else:
                        return False  # 无效的无前缀列名

            # 确保至少有一个有效列引用
            return has_valid_columns

        except Exception as e:
            print(f"SQL验证时出错: {e}")
            return False

    def _generate_database_schema(self, schema_for_db):
        schema_prompt = '{\n '
        for table_name, cn_prompt in schema_for_db.items():
            schema_prompt += f'{table_name}:\n  ' + '{\n\t'
            for cn, prompt in cn_prompt.items():
                schema_prompt += f"{cn}: {prompt}" + '\n\t'
                schema_prompt += '\n\t'
            schema_prompt += '}\n '
        schema_prompt += '}'
        return schema_prompt

    def generate_dummy_sql(self, question_id):
        """生成虚拟SQL查询（使用LLM）"""
        output_dummy = {}  # 存储生成的dummy SQL,也是
        # 先尝试读取已有数据（如果文件存在）
        if os.path.exists("dummy_sql.json"):
            with open("dummy_sql.json", 'r') as f:
                try:
                    output_dummy = json.load(f)
                except json.JSONDecodeError:
                    output_dummy = {}
        question = self.question_json[question_id]
        db_id = question['db_id']
        q = question['question']
        evidence = question['evidence']
        pk_dict, fk_dict = self.generate_pk_fk(question_id)
        db_prompt_dic = self._reconstruct_schema()
        db_prompt = db_prompt_dic[db_id]
        database_schema = self._generate_database_schema(db_prompt)
        # print("\n" + "=" * 50)
        # print("database_schema:\n" + database_schema)
        # print("=" * 50 + "\n")

        # 构建模式提示，调用LLM生成SQL
        prompt = dummy_sql_prompt.format(database_schema=database_schema,
                                         primary_key_dic=pk_dict,
                                         foreign_key_dic=fk_dict,
                                         question_prompt=q,
                                         evidence=evidence)

        # # API调用
        # dummy_sql = collect_response(prompt, stop='return SQL')
        # # 本地LLM调用
        # # dummy_sql = get_response(prompt, stop='return SQL')
        # print("\n" + "=" * 50)
        # print("dummy_sql:\n" + dummy_sql)
        # print("=" * 50 + "\n")
        # return prompt, dummy_sql
        generated_sqls = []  # 存储所有生成的SQL
        for attempt in range(self.max_retries):
            print(f"\n{'=' * 50}")
            print(f"尝试第 {attempt + 1} 次生成SQL")

            # 生成SQL
            dummy_sql = collect_response(prompt, stop='return SQL')
            sql_block_match = re.search(r"```sql\n(.*?)\n```", dummy_sql, re.DOTALL)
            if sql_block_match:
                dummy_sql = sql_block_match.group(1).strip()
            generated_sqls.append(dummy_sql)
            print("\n生成的SQL:\n" + dummy_sql)
            # 处理SQL
            processed_sql = (dummy_sql.replace('\"', '')
                             .replace('\\\n', ' ')
                             .replace('\n', ' ')
                             .strip())
            if attempt == 0:
                processed_sql = processed_sql + '\t----- bird -----\t' + db_id
                print("\n纯净模式生成的SQL:\n" + processed_sql)
                # 更新字典并写入文件
                output_dummy[str(question_id)] = processed_sql
                with open("dummy_sql.json", 'w') as f:
                    json.dump(output_dummy, f, indent=4)
            # 验证SQL
            if self._validate_sql(dummy_sql, db_id):
                print("SQL验证通过")
                break
            else:
                print("SQL验证失败: 包含无效表或列")
                # 更新提示以包含更多指导
                prompt += (
                    "\n\n注意：你上次生成的SQL包含了一些不在database_schema中的表或列。"
                    "请严格基于database_schema给出的数据表和列生成SQL。"
                )

        print("=" * 50 + "\n")
        print("所有生成的SQL:")
        for i, sql in enumerate(generated_sqls, 1):
            print(f"\n第 {i} 次生成的SQL:\n{sql}")

        # 返回prompt和所有生成的SQL（最多2个）
        return prompt, tuple(generated_sqls)

    def get_schema(self, question_id):
        question_info = self.question_json[question_id]
        db_id = question_info['db_id']
        _, dummy_sqls = self.generate_dummy_sql(question_id)

        # 获取数据库的表和列信息
        table_info = [content for content in self.table_json if content['db_id'] == db_id][0]
        table_names_list = table_info["table_names_original"]
        column_names_list = [
            [table_names_list[int(content[0])], content[1]]
            for content in table_info['column_names_original'][1:]
        ]

        schemas = []

        # 遍历每个SQL
        for dummy_sql in dummy_sqls:
            if not isinstance(dummy_sql, str):
                continue  # 跳过非字符串（如None或无效SQL）

            # 清理SQL（去掉引号、统一大小写）
            dummy_sql_clean = dummy_sql.replace('"', '').replace("'", '').lower()

            # 检查表名
            filtered_tables = []
            for table in table_names_list:
                if table.lower() in dummy_sql_clean:
                    filtered_tables.append(table)

            # 检查列名
            for col_info in column_names_list:
                table, column = col_info[0], col_info[1]
                if column.lower() in dummy_sql_clean and table in filtered_tables:
                    schemas.append((table, column))  # 存储 (表名, 列名)

        # 去重（避免重复的表和列）
        schemas = list(set(schemas))
        print("所有SQL涉及的表和列:", schemas)
        return schemas


class EnhancedTALOG(BaseModule):
    """增强版TALOG，集成RAG功能"""

    def __init__(self, db_root_path, mode, rag_module=None):
        super().__init__(db_root_path, mode)
        """
        Args:
            db_root_path: 数据库根路径
            mode: 模式（dev/test）
            rag_module: 可选，传入RAG模块则启用增强功能
        """
        self.rag = rag_module
        self.csv_info, self.value_prompts = self._get_info_from_csv()
        # print("\n" + "=" * 50)
        # print(f"self.value_prompts content: {self.value_prompts}")
        # print("=" * 50 + "\n")

    def generate_schema_prompt(self, question_id, sl_schemas):
        """生成详细的模式提示"""
        question_info = self.question_json[question_id]
        db_id = question_info['db_id']
        schema_item_dic = {}

        for otn, ocn in sl_schemas:
            column_name, column_description, column_type, value_description = self.csv_info[f"{db_id}|{otn}"][ocn]
            value_prompt = self.value_prompts.get(f"{db_id}|{otn}|{ocn}")
            tmp_prompt = f"{column_type}, the full column name is {column_name}"
            if column_description not in ['', ' ', None]:
                column_description = column_description.replace('\n', ' ')
                tmp_prompt += f', column description is {column_description}'
            if value_description not in ['', ' ', None]:
                value_description = value_description.replace('\n', ' ')
                tmp_prompt += f", value description is {value_description}"
            if value_prompt:
                tmp_prompt += f", {value_prompt}"
            if ' ' in otn: otn = f"`{otn}`"
            if ' ' in ocn: ocn = f"`{ocn}`"
            schema_item_dic[f"{otn}.{ocn}"] = tmp_prompt
        # 构建包含列类型、描述、示例值的提示
        schema_prompt = '{\n\t'
        for otn_ocn, cn_prompt in schema_item_dic.items():
            schema_prompt += f'{otn_ocn}: {cn_prompt}\n'
            schema_prompt += '\n\t'
        schema_prompt += '}'
        return schema_prompt

    def generate_sr(self, question_id, sl_schemas):
        """集成RAG的SR生成方法"""
        question = self.question_json[question_id]
        query = f"{question['question']} {question['evidence']}"
        print("\n" + "=" * 50)
        print(f"\nUser query: {query}")
        example_texts = []
        if self.rag:
            # 使用 RAG 检索相似示例
            retrieved_results = self.rag.retrieve(query)
            print("\n" + "=" * 50)
            print("RAG 检索到的示例:")

            for result in retrieved_results:
                print(f"\n最佳匹配（相似度: {result['similarity']:.2f}）:")
                print(f"原始问题: {result['original_question']}")
                print(f"证据: {result['evidence']}")
                print(f"数据库: {result['db_id']}")

                sql_examples = result['sql_examples']

                # 随机抽取2个示例
                import random
                random.shuffle(sql_examples)
                top_examples = sql_examples[:2]

                print("\n随机选择的2个示例:")
                for i, example in enumerate(top_examples, 1):
                    print("-" * 50 + "\n")
                    print(f"\n示例 {i}:")
                    print(example['full_example'])
                    print(f"SQL: {example['sql']}")
                    print("-" * 50 + "\n")

                example_texts.append(
                    "\n".join(
                        f"示例 {i}:\n{example['full_example']}\n#SQL: {example['sql']}\n"
                        for i, example in enumerate(top_examples, 1)
                    )
                )
            print("=" * 50 + "\n")
            #     # 筛选出标签后有内容且内容有效的完整示例
            #     def has_valid_content(full_example, tags, invalid_values={" None", ""}):
            #         for tag in tags:
            #             index = full_example.find(tag)
            #             if index == -1:
            #                 return False
            #             # 提取标签后面的内容
            #             content = full_example[index + len(tag):].strip()
            #             # 检查内容有效性
            #             if not content or content in invalid_values:
            #                 return False
            #         return True
            #
            #     content_tags = ['#reason:', '#columns:', '#values:', '#SELECT:', '#SQL-like:']
            #     sql_examples = result['sql_examples']
            #
            #     # 使用 evaluate_examples_similarity 方法计算每个示例的相似度
            #     sorted_examples = self.rag.evaluate_examples_similarity(query, sql_examples)
            #     print("\n排序后的示例相似度:")
            #     for example in sorted_examples:
            #         print(
            #             f"相似度: {example['similarity']:.2f}, 内容完整: {has_valid_content(example['example']['full_example'], content_tags)}, 问题部分:{example['example']['question_part']}")
            #
            #     top_examples = []
            #     selected_indices = set()
            #     # 提取最高相似度示例
            #     top_similarity = sorted_examples[0]['similarity']
            #     print(f"\n选择相似度最高的示例: {sorted_examples[0]['similarity']:.2f}")
            #
            #     # 找到符合条件的完整示例
            #     for example in sorted_examples:
            #         example_data = example['example']
            #         if example['similarity'] >= top_similarity - 0.1:
            #             if has_valid_content(example_data['full_example'], content_tags):
            #                 top_examples.append(example)
            #                 selected_indices.add(sql_examples.index(example_data))
            #                 print(f"选择相似度接近的完整示例: {example['similarity']:.2f}")
            #
            #         if len(top_examples) >= 2:
            #             break
            #
            #     # 如果没有足够的完整示例，继续选择相似度高的示例（不重复选择）
            #     for example in sorted_examples:
            #         if len(top_examples) >= 2:
            #             break
            #         example_data = example['example']
            #         index = sql_examples.index(example_data)
            #         if index not in selected_indices:
            #             top_examples.append(example)
            #             selected_indices.add(index)
            #             print(f"补充选择高相似度示例: {example['similarity']:.2f}")
            #
            #     for i, example in enumerate(top_examples, 1):
            #         sql_example = example['example']
            #         print("-" * 50 + "\n")
            #         print(f"\n示例 {i}:")
            #         print(sql_example['full_example'])
            #         print(f"SQL: {sql_example['sql']}")
            #         print("-" * 50 + "\n")
            #
            #     example_texts.append(
            #         "\n".join(
            #             f"示例 {i}:\n{sql['example']['full_example']}\n#SQL: {sql['example']['sql']}\n"
            #             for i, sql in enumerate(top_examples, 1)
            #         )
            #     )
            # print("=" * 50 + "\n")

        # 构建增强提示
        processed_schema = [f"{t}.{c}" for t, c in sl_schemas]
        enhance_sr_prompt = generate_sr.format(
            sr_example=sr_examples,
            question=question['question'],
            schema=str(processed_schema),
            column_description=self.generate_schema_prompt(question_id, sl_schemas),
            evidence=question['evidence']
        )
        # API调用
        enhance_sr = collect_response(enhance_sr_prompt, max_tokens=800)
        # enhance_sr = ''
        # 本地LLM调用
        # enhance_sr = get_response(enhance_sr_prompt, max_tokens=800)
        return enhance_sr_prompt, enhance_sr, example_texts

    def sr2sql(self, question_id, sl_schemas):
        """将语义表示转换为SQL查询（内置多模式提取）"""
        question = self.question_json[question_id]
        q = question['question']
        e = question['evidence']
        schema = ['.'.join(t) for t in sl_schemas] if sl_schemas else []
        _, sr, examples = self.generate_sr(question_id, sl_schemas)

        print("\n" + "=" * 50)
        print("Final sr：" + sr)
        print("=" * 50 + "\n")

        sr = sr.replace('\"', '')
        database_schema = self.generate_schema_prompt(question_id, sl_schemas)
        _, fk = self.generate_pk_fk(question_id)

        sr2sql_prompt = sr2sql.format(
            question=q,
            schema=schema,
            evidence=e,
            column_description=database_schema,
            SR=sr,
            examples="\n".join(examples),
            foreign_key_dic=fk
        )

        print("\n" + "=" * 50)
        print("Final text2sql prompt：" + sr2sql_prompt)
        print("=" * 50 + "\n")

        # API调用
        tmp_sql = collect_response(sr2sql_prompt.strip('\n'))
        # 本地LLM调用
        # tmp_sql = get_response(sr2sql_prompt.strip('\n'))

        # 四种SQL提取模式（按优先级尝试）
        extracted_sql = None

        # 模式1：```sql代码块
        sql_block_match = re.search(r"```sql\n(.*?)\n```", tmp_sql, re.DOTALL)
        if sql_block_match:
            extracted_sql = sql_block_match.group(1).strip()

        # # 模式2：<!-- SQL -->注释块
        # if not extracted_sql:
        #     xml_match = re.search(r"<!--\s*SQL\s*-->\n(.*?)(?=\n<!--|$)", tmp_sql, re.DOTALL)
        #     if xml_match:
        #         extracted_sql = xml_match.group(1).strip()
        #
        # # 模式3："SQL: ..."引号格式
        # if not extracted_sql:
        #     quoted_match = re.search(r"[\"']SQL:\s*(.*?)[\"']", tmp_sql, re.DOTALL)
        #     if quoted_match:
        #         extracted_sql = quoted_match.group(1).strip()
        #
        # # 模式4：纯净SELECT语句
        # if not extracted_sql:
        #     select_match = re.search(r"(SELECT\s+.+?;)(?=\s|$)", tmp_sql, re.DOTALL | re.IGNORECASE)
        #     if select_match:
        #         extracted_sql = select_match.group(1).strip()

        # 提取失败处理
        if not extracted_sql:
            print("\n" + "=" * 50)
            print("SQL提取失败，原始输出：\n" + tmp_sql)
            print("=" * 50 + "\n")
            return None

        # 基础清理
        final_sql = extracted_sql.replace('\"', '').replace('\n', ' ').strip()
        if not final_sql.endswith(';'):
            final_sql += ';'

        print("\n" + "=" * 50)
        print("原始输出：\n" + tmp_sql)
        print("\n提取结果：\n" + final_sql)
        print("=" * 50 + "\n")

        return sr, final_sql


if __name__ == '__main__':
    db_root_path = './data/dev_databases'
    column_meaning_path = './outputs/column_meaning.json'
    example_db_path = os.path.abspath('./question.json')  # 新增RAG示例库路径
    test_module = BaseModule(db_root_path, mode='test')
    # pk_dict,_ = test_module.generate_pk_fk(0)
    # print(pk_dict)
    question_id = 0
    tasl = TASL(db_root_path, 'dev', column_meaning_path)
    rag = RAGModule(example_db_path)
    talog = EnhancedTALOG(db_root_path, 'dev', rag)

    # talog = TALOG(db_root_path)
    sl_schemas = tasl.get_schema(question_id)
    sql = talog.sr2sql(question_id, sl_schemas)
