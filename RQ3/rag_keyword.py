import json
import numpy as np
from typing import List, Dict
import re
from rank_bm25 import BM25Okapi
from functools import lru_cache

class RAGModule:
    def __init__(self, example_db_path='question.json'):
        """初始化RAG模块（关键词检索版）"""
        self.example_db = self._load_examples(example_db_path)
        self._preprocess_keywords()

    def _load_examples(self, path: str) -> List[Dict]:
        """加载示例数据库"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('questions', [])

    def _sql_tokenizer(self, text: str) -> List[str]:
        """Specialized tokenizer for SQL-related text"""
        if not isinstance(text, str):
            return []
        # Tokenize while preserving SQL keywords and symbols
        tokens = re.findall(r"\b\w+\b|!=|>=|<=|==|=|>|<|\+|-|\*|/|\(|\)", text.lower())
        return [t for t in tokens if len(t) > 1 or t in {'=', '>', '<', '(', ')'}]

    def _preprocess_keywords(self):
        """预处理关键词索引"""
        # 准备检索文本 (问题+证据)
        self.example_texts = [
            f"{ex.get('question', '')} {ex.get('evidence', '')}"
            for ex in self.example_db
        ]
        
        # BM25初始化
        tokenized_corpus = [self._sql_tokenizer(text) for text in self.example_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _extract_full_examples(self, prompt: str) -> List[Dict]:
        """从prompt中提取完整的SQL示例片段（保持原样）"""
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
        检索相似示例并返回完整的SQL示例片段（接口保持不变）
        
        Args:
            query: 查询文本
            top_k: 返回数量
            min_similarity: 最小相似度阈值
            
        Returns:
            包含完整SQL示例片段的检索结果列表（格式与原来完全相同）
        """
        self._print_header(f"RAG查询: {query}")

        # 中文分词处理
        tokenized_query = self._sql_tokenizer(query)
        
        if not tokenized_query:
            return []

        # 计算BM25分数
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # 归一化到0-1范围以保持与原来相似度的一致性
        max_score = max(doc_scores) if max(doc_scores) > 0 else 1
        normalized_scores = doc_scores / max_score

        top_indices = np.argsort(doc_scores)[-top_k:][::-1]
        results = []

        print("\n最佳匹配示例:")
        for i in top_indices:
            if normalized_scores[i] > min_similarity:
                example = self.example_db[i]
                # 提取该prompt中的所有完整SQL示例（保持原样）
                sql_examples = self._extract_full_examples(example.get('prompt', ''))

                if sql_examples:
                    result = {
                        'similarity': round(float(normalized_scores[i]), 4),  # 保持4位小数
                        'original_question': example.get('question', ''),
                        'evidence': example.get('evidence', ''),
                        'db_id': example.get('db_id', ''),
                        'sql_examples': sql_examples  # 保持原数据结构
                    }
                    self._print_result(result)
                    results.append(result)

        self._print_footer()
        return results

    def evaluate_examples_similarity(self, query: str, examples: List[Dict]) -> List[Dict]:
        """
        计算每个示例的问题与查询之间的相似度（改用关键词相似度计算）
        
        Args:
            query: 查询文本
            examples: 待评估的示例列表
            
        Returns:
            每个示例与查询的相似度结果列表（格式与原来完全相同）
        """
        tokenized_query = self._sql_tokenizer(query)
        if not tokenized_query:
            return [{'example': ex, 'similarity': 0.0} for ex in examples]
            
        # 计算每个示例的BM25分数
        example_texts = [ex['question_part'] for ex in examples]
        tokenized_examples = [self._sql_tokenizer(text) for text in example_texts]
        
        # 临时创建BM25索引
        temp_bm25 = BM25Okapi(tokenized_examples)
        doc_scores = temp_bm25.get_scores(tokenized_query)
        
        # 归一化分数
        max_score = max(doc_scores) if max(doc_scores) > 0 else 1
        normalized_scores = doc_scores / max_score
        
        # 保持与原结构一致
        examples_with_similarity = [{
            'example': ex, 
            'similarity': float(score)
        } for ex, score in zip(examples, normalized_scores)]
        
        # 按相似度排序
        examples_with_similarity.sort(key=lambda x: x['similarity'], reverse=True)
        
        return examples_with_similarity

    # 以下打印方法保持完全不变
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

    def _print_footer(self):
        """打印输出尾部"""
        print("\n" + "=" * 80 + "\n")