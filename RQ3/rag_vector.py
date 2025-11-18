import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from typing import List, Dict


class RAGModule:
    def __init__(self, example_db_path='question.json'):
        """初始化RAG模块"""
        self.example_db = self._load_examples(example_db_path)
        self.model = SentenceTransformer('/model/LLM/bge-large')
        self._preprocess_embeddings()

    def _load_examples(self, path: str) -> List[Dict]:
        """加载示例数据库"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('questions', [])

    def _preprocess_embeddings(self):
        """预处理嵌入向量"""
        self.example_texts = [
            f"{ex['question']} {ex['evidence']}"
            for ex in self.example_db
        ]
        self.example_embeddings = self.model.encode(
            self.example_texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )

    def _extract_full_examples(self, prompt: str) -> List[Dict]:
        """从prompt中提取完整的SQL示例片段"""
        examples = []
        # 分割每个完整的示例
        example_blocks = prompt.split('/* Answer the following:')[1:]

        for block in example_blocks:
            if '#SQL:' in block:
                # 提取问题描述
                question_part = block.split('*/')[0].strip()
                # 提取完整示例内容
                full_example = '/* Answer the following:' + block.split('#SQL:')[0].strip()
                # 提取SQL语句
                sql = block.split('#SQL:')[1].strip()

                examples.append({
                    'question_part': question_part,
                    'full_example': full_example,
                    'sql': sql
                })
        return examples

    @lru_cache(maxsize=1000)
    def retrieve(self, query: str, top_k: int = 1, min_similarity: float = 0.4) -> List[Dict]:
        """
        检索相似示例并返回完整的SQL示例片段

        Args:
            query: 查询文本
            top_k: 返回数量
            min_similarity: 最小相似度阈值

        Returns:
            包含完整SQL示例片段的检索结果列表
        """
        self._print_header(f"RAG query: {query}")

        query_embedding = self.model.encode(query, convert_to_tensor=True)
        sims = cosine_similarity(
            query_embedding.cpu().numpy().reshape(1, -1),
            self.example_embeddings.cpu().numpy()
        )[0]

        top_indices = np.argsort(sims)[-top_k:][::-1]
        results = []

        print("\nTOP MATCHING EXAMPLES:")
        for i in top_indices:
            if sims[i] > min_similarity:
                example = self.example_db[i]
                # 提取该prompt中的所有完整SQL示例
                sql_examples = self._extract_full_examples(example.get('prompt', ''))

                if sql_examples:
                    result = {
                        'similarity': round(sims[i], 4),
                        'original_question': example.get('question', ''),
                        'evidence': example.get('evidence', ''),
                        'db_id': example.get('db_id', ''),
                        'sql_examples': sql_examples  # 包含所有完整SQL示例
                    }
                    self._print_result(result)
                    results.append(result)

        self._print_footer()
        return results

    def evaluate_examples_similarity(self, query: str, examples: List[Dict]) -> List[Dict]:
        """
        计算每个示例的问题与查询之间的相似度，并返回包含所有示例的结果列表

        Args:
            query: 查询文本
            examples: 待评估的示例列表

        Returns:
            每个示例与查询的相似度结果列表
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        example_embeddings = self.model.encode(
            [ex['question_part'] for ex in examples],
            convert_to_tensor=True
        )

        # 计算相似度
        sims = cosine_similarity(
            query_embedding.cpu().numpy().reshape(1, -1),
            example_embeddings.cpu().numpy()
        )[0]

        # 包含每个示例和其相似度的新结构
        examples_with_similarity = [{'example': ex, 'similarity': sim} for ex, sim in zip(examples, sims)]

        # 按相似度排序
        examples_with_similarity.sort(key=lambda x: x['similarity'], reverse=True)

        return examples_with_similarity

    def _print_header(self, title: str):
        """打印输出头部"""
        print("\n" + "=" * 80)
        print(title.center(80))
        print("=" * 80)

    def _print_result(self, result: Dict):
        """打印单个检索结果及其SQL示例"""
        print(f"\n[相似度: {result['similarity']:.2f}]")
        print(f"[原始问题]: {result['original_question']}")
        print(f"[证据]: {result['evidence']}")
        print(f"[数据库]: {result['db_id']}")
        print("=" * 80)
        # print("\n[相关SQL示例]:")
        # for i, example in enumerate(result['sql_examples'], 1):
        #     print(f"\n示例 {i}:")
        #     print("-" * 60)
        #     print(example['full_example'])
        #     print("#SQL:", example['sql'])
        #     print("-" * 60)

    def _print_footer(self):
        """打印输出尾部"""
        print("\n" + "=" * 80 + "\n")


