import os
import sys
import json
import numpy as np
import argparse
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut
import time
import math
from contextlib import closing
from tqdm import tqdm  # 进度条支持

# 全局变量改为类封装
class QueryEvaluator:
    def __init__(self):
        self.exec_result = []
        self.lock = mp.Lock()
    
    def result_callback(self, result):
        with self.lock:
            self.exec_result.append(result)

def clean_abnormal(input):
    """剔除3σ以外的异常值"""
    input = np.asarray(input)
    if len(input) == 0:
        return []
    mean = np.mean(input, axis=0)
    std = np.std(input, axis=0)
    return input[(input < mean + 3 * std) & (input > mean - 3 * std)].tolist()

def execute_sql(sql, db_path, max_retry=3):
    """带重试机制的SQL执行"""
    for _ in range(max_retry):
        try:
            with closing(sqlite3.connect(db_path)) as conn:
                cursor = conn.cursor()
                start_time = time.perf_counter_ns()  # 更精确的计时
                cursor.execute(sql)
                return time.perf_counter_ns() - start_time
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and _ < max_retry - 1:
                time.sleep(0.1)
                continue
            raise
    return 0

def iterated_execute_sql(predicted_sql, ground_truth, db_path, iterate_num=5):
    """带结果验证的迭代执行"""
    # 先验证结果正确性
    with closing(sqlite3.connect(db_path)) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(predicted_sql)
            predicted_res = cursor.fetchall()
            cursor.execute(ground_truth)
            ground_truth_res = cursor.fetchall()
            if set(predicted_res) != set(ground_truth_res):
                return 0.0
        except:
            return 0.0
    
    # 性能测试
    diff_list = []
    for _ in range(iterate_num):
        try:
            pred_time = execute_sql(predicted_sql, db_path)
            truth_time = execute_sql(ground_truth, db_path)
            if pred_time > 0 and truth_time > 0:
                diff_list.append(truth_time / pred_time)
        except:
            continue
    
    processed_diff = clean_abnormal(diff_list)
    return sum(processed_diff) / len(processed_diff) if processed_diff else 0.0

def execute_model(args):
    """适配多进程的封装函数"""
    predicted_sql, ground_truth, db_place, idx, iterate_num, single_timeout = args
    try:
        time_ratio = func_timeout(single_timeout, iterated_execute_sql,
                                args=(predicted_sql, ground_truth, db_place, iterate_num))
    except FunctionTimedOut:
        sys.stderr.write(f"\nTimeout on query {idx}: {predicted_sql[:50]}...\n")
        time_ratio = 0
    except Exception as e:
        sys.stderr.write(f"\nError on query {idx}: {str(e)}\n")
        time_ratio = 0
    return {'sql_idx': idx, 'time_ratio': time_ratio}

def package_sqls(sql_path, db_root_path, mode='gpt', data_mode='dev'):
    """SQL加载优化"""
    clean_sqls = []
    db_path_list = []
    
    if mode == 'gpt':
        with open(f"{sql_path}predict_{data_mode}.json") as f:
            sql_data = json.load(f)
        for idx, sql_str in sql_data.items():
            if isinstance(sql_str, str):
                sql, db_name = sql_str.split('\t----- bird -----\t')
            else:
                sql, db_name = " ", "financial"
            clean_sqls.append(sql)
            db_path_list.append(f"{db_root_path}{db_name}/{db_name}.sqlite")
    else:
        with open(f"{sql_path}{data_mode}_gold.sql") as f:
            for idx, line in enumerate(f):
                sql, db_name = line.strip().split('\t')
                clean_sqls.append(sql)
                db_path_list.append(f"{db_root_path}{db_name}/{db_name}.sqlite")
    
    return clean_sqls, db_path_list

def run_sqls_parallel(evaluator, sqls, db_places, num_cpus=4, iterate_num=5, meta_time_out=30.0):
    """优化后的并行执行"""
    ctx = mp.get_context('spawn')
    task_args = [
        (pred_sql, gt_sql, db_place, i, iterate_num, meta_time_out/iterate_num)
        for i, (pred_sql, gt_sql, db_place) in enumerate(zip(sqls[0], sqls[1], db_places))
    ]
    
    with ctx.Pool(processes=num_cpus) as pool:
        results = list(tqdm(
            pool.imap(execute_model, task_args),
            total=len(task_args),
            desc="Evaluating queries"
        ))
        for r in results:
            evaluator.result_callback(r)

def compute_ves(results):
    """计算速度评分"""
    if not results:
        return 0.0
    return sum(math.sqrt(r['time_ratio']) for r in results) * 100 / len(results)

def compute_ves_by_diff(exec_results, diff_json_path):
    """分级评估"""
    with open(diff_json_path) as f:
        contents = json.load(f)
    
    # 确保数据对齐
    min_len = min(len(exec_results), len(contents))
    paired_data = zip(exec_results[:min_len], contents[:min_len])
    
    categories = {
        'simple': [],
        'moderate': [],
        'challenging': []
    }
    
    for result, content in paired_data:
        if content['difficulty'] in categories:
            categories[content['difficulty']].append(result)
    
    scores = {k: compute_ves(v) for k, v in categories.items()}
    scores['total'] = compute_ves(exec_results[:min_len])
    
    counts = {
        'simple': len(categories['simple']),
        'moderate': len(categories['moderate']),
        'challenging': len(categories['challenging']),
        'total': min_len
    }
    
    return scores, counts

def print_results(scores, counts):
    """美观的结果输出"""
    print("\n" + "="*80)
    print("{:<15} {:<15} {:<15} {:<15}".format(
        "Difficulty", "Count", "VES Score", "Progress"
    ))
    print("-"*80)
    
    for diff in ['simple', 'moderate', 'challenging', 'total']:
        print("{:<15} {:<15} {:<15.2f} {:<15}".format(
            diff.capitalize(),
            counts[diff],
            scores[diff],
            "✓" if counts[diff] > 0 else "✗"
        ))
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="SQL Velocity Evaluation System")
    parser.add_argument('--predicted_sql_path', type=str, required=True)
    parser.add_argument('--ground_truth_path', type=str, required=True)
    parser.add_argument('--data_mode', type=str, default='dev')
    parser.add_argument('--db_root_path', type=str, required=True)
    parser.add_argument('--num_cpus', type=int, default=max(1, mp.cpu_count()-1))
    parser.add_argument('--meta_time_out', type=float, default=30.0)
    parser.add_argument('--iterate_num', type=int, default=5)
    parser.add_argument('--mode_gt', type=str, default='gt')
    parser.add_argument('--mode_predict', type=str, default='gpt')
    parser.add_argument('--diff_json_path', type=str, required=True)
    args = parser.parse_args()
    
    evaluator = QueryEvaluator()
    
    print("Loading SQL queries...")
    pred_queries, db_paths = package_sqls(
        args.predicted_sql_path, args.db_root_path, 
        mode=args.mode_predict, data_mode=args.data_mode
    )
    gt_queries, _ = package_sqls(
        args.ground_truth_path, args.db_root_path,
        mode='gt', data_mode=args.data_mode
    )
    
    print(f"Evaluating {len(pred_queries)} queries with {args.num_cpus} cores...")
    run_sqls_parallel(
        evaluator,
        (pred_queries, gt_queries),
        db_paths,
        num_cpus=args.num_cpus,
        iterate_num=args.iterate_num,
        meta_time_out=args.meta_time_out
    )
    
    print("\nCalculating results...")
    scores, counts = compute_ves_by_diff(evaluator.exec_result, args.diff_json_path)
    print_results(scores, counts)

if __name__ == '__main__':
    main()