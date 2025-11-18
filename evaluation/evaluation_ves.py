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


def result_callback(result):
    exec_result.append(result)


def clean_abnormal(input):
    input = np.asarray(input)
    processed_list = []
    mean = np.mean(input, axis=0)
    std = np.std(input, axis=0)
    for x in input:
        if x < mean + 3 * std and x > mean - 3 * std:
            processed_list.append(x)
    return processed_list


def execute_sql(sql, db_path):
    # Connect to the database  
    conn = sqlite3.connect(db_path)
    # Create a cursor object  
    cursor = conn.cursor()
    start_time = time.process_time_ns()
    cursor.execute(sql)
    exec_time = time.process_time_ns() - start_time
    return exec_time


def iterated_execute_sql(predicted_sql, ground_truth, db_path, iterate_num):
    conn = sqlite3.connect(db_path)
    diff_list = []
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    time_ratio = 0
    if set(predicted_res) == set(ground_truth_res):
        for i in range(iterate_num):
            predicted_time = execute_sql(predicted_sql, db_path)
            ground_truth_time = execute_sql(ground_truth, db_path)
            diff_list.append(ground_truth_time / predicted_time)
        processed_diff_list = clean_abnormal(diff_list)
        time_ratio = sum(processed_diff_list) / len(processed_diff_list)
    return time_ratio


def execute_model(predicted_sql, ground_truth, db_place, idx, iterate_num, meta_time_out):
    try:
        time_ratio = func_timeout(meta_time_out * iterate_num, iterated_execute_sql,
                                  args=(predicted_sql, ground_truth, db_place, iterate_num))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        time_ratio = 0
    except Exception:
        time_ratio = 0
    result = {'sql_idx': idx, 'time_ratio': time_ratio}
    return result


def package_sqls(sql_path, db_root_path, mode='gpt', data_mode='dev'):
    clean_sqls = []
    db_path_list = []
    if mode == 'gpt':
        sql_data = json.load(open(sql_path + 'predict_' + data_mode + '.json', 'r'))
        for idx, sql_str in sql_data.items():
            if isinstance(sql_str, str):
                sql, db_name = sql_str.split('\t----- spider -----\t')
            else:
                sql, db_name = " ", "financial"
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')
    elif mode == 'gt':
        sqls = open(sql_path + data_mode + '_gold.sql')
        sql_txt = sqls.readlines()
        for idx, sql_str in enumerate(sql_txt):
            sql, db_name = sql_str.strip().split('\t')
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')
    return clean_sqls, db_path_list


def run_sqls_parallel(sqls, db_places, num_cpus=1, iterate_num=100, meta_time_out=30.0):
    pool = mp.Pool(processes=num_cpus)
    for i, sql_pair in enumerate(sqls):
        predicted_sql, ground_truth = sql_pair
        pool.apply_async(execute_model, args=(predicted_sql, ground_truth, db_places[i], i, iterate_num, meta_time_out),
                         callback=result_callback)
    pool.close()
    pool.join()


def sort_results(list_of_dicts):
    return sorted(list_of_dicts, key=lambda x: x['sql_idx'])


def compute_ves(exec_results):
    num_queries = len(exec_results)
    total_ratio = 0
    for i, result in enumerate(exec_results):
        if result['time_ratio'] != 0:
            total_ratio += math.sqrt(result['time_ratio']) * 100
    ves = (total_ratio / num_queries)
    return ves


# def load_json(dir):
#     with open(dir, 'r') as j:
#         contents = json.loads(j.read())
#     return contents
def load_json(dir):
    try:
        with open(dir, 'r', encoding='utf-8') as j:
            contents = json.load(j)  # 直接使用 json.load 更简洁
    except UnicodeDecodeError:
        # 如果 utf-8 失败，尝试其他编码（如 latin-1）
        with open(dir, 'r', encoding='latin-1') as j:
            contents = json.load(j)
    return contents


# def compute_ves_by_diff(exec_results, diff_json_path):
#     num_queries = len(exec_results)
#     contents = load_json(diff_json_path)
#     easy_results, medium_results, hard_results, extra_results = [], [], [], []
#     for i, content in enumerate(contents):
#         if content['difficulty'] == 'easy':
#             easy_results.append(exec_results[i])
#         elif content['difficulty'] == 'medium':
#             medium_results.append(exec_results[i])
#         elif content['difficulty'] == 'hard':
#             hard_results.append(exec_results[i])
#         elif content['difficulty'] == 'extra':
#             extra_results.append(exec_results[i])
#     easy_ves = compute_ves(easy_results)
#     medium_ves = compute_ves(medium_results)
#     hard_ves = compute_ves(hard_results)
#     extra_ves = compute_ves(extra_results)
#     all_ves = compute_ves(exec_results)
#     count_lists = [len(easy_results), len(medium_results), len(hard_results), len(extra_results), num_queries]
#     return easy_ves, medium_ves, hard_ves, extra_ves, all_ves, count_lists
def compute_ves_by_diff(exec_results, diff_json_path):
    contents = load_json(diff_json_path)
    easy_results, medium_results, hard_results, extra_results = [], [], [], []

    # 只遍历exec_results的索引范围
    for i in range(len(exec_results)):
        if i >= len(contents):  # 防止contents比exec_results短
            break
        content = contents[i]
        if content['difficulty'] == 'easy':
            easy_results.append(exec_results[i])
        elif content['difficulty'] == 'medium':
            medium_results.append(exec_results[i])
        elif content['difficulty'] == 'hard':
            hard_results.append(exec_results[i])
        elif content['difficulty'] == 'extra':
            extra_results.append(exec_results[i])

    # 计算VES（避免除零）
    easy_ves = compute_ves(easy_results) if easy_results else 0
    medium_ves = compute_ves(medium_results) if medium_results else 0
    hard_ves = compute_ves(hard_results) if hard_results else 0
    extra_ves = compute_ves(extra_results) if extra_results else 0
    all_ves = compute_ves(exec_results) if exec_results else 0

    count_lists = [len(easy_results), len(medium_results), len(hard_results), len(extra_results), len(exec_results)]
    return easy_ves, medium_ves, hard_ves, extra_ves, all_ves, count_lists


def print_data(score_lists, count_lists):
    levels = ['easy', 'medium', 'hard', 'extra', 'total']
    # print("{:20} {:20} {:20} {:20} {:20} {:20}".format("", *levels))
    # print("{:20} {:<20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))
    print('============================================= VES ===================================================')
    print("{:18} {:<18.2f} {:<18.2f} {:<18.2f} {:<18.2f} {:<18.2f}".format('ves', *score_lists))


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--predicted_sql_path', type=str, required=True, default='')
    args_parser.add_argument('--ground_truth_path', type=str, required=True, default='')
    args_parser.add_argument('--data_mode', type=str, required=True, default='dev')
    args_parser.add_argument('--db_root_path', type=str, required=True, default='')
    args_parser.add_argument('--num_cpus', type=int, default=1)
    args_parser.add_argument('--meta_time_out', type=float, default=30.0)
    args_parser.add_argument('--mode_gt', type=str, default='gt')
    args_parser.add_argument('--mode_predict', type=str, default='gpt')
    args_parser.add_argument('--diff_json_path', type=str, default='')
    args = args_parser.parse_args()
    exec_result = []

    pred_queries, db_paths = package_sqls(args.predicted_sql_path, args.db_root_path, mode=args.mode_predict,
                                          data_mode=args.data_mode)
    gt_queries, db_paths_gt = package_sqls(args.ground_truth_path, args.db_root_path, mode='gt',
                                           data_mode=args.data_mode)

    query_pairs = list(zip(pred_queries, gt_queries))
    run_sqls_parallel(query_pairs, db_places=db_paths, num_cpus=args.num_cpus, iterate_num=100,
                      meta_time_out=args.meta_time_out)
    exec_result = sort_results(exec_result)
    #print("exec_result 内容：", exec_result)
    easy_ves, medium_ves, hard_ves, extra_ves, ves, count_lists = compute_ves_by_diff(exec_result, args.diff_json_path)
    score_lists = [easy_ves, medium_ves, hard_ves, extra_ves, ves]
    print_data(score_lists, count_lists)
    print('======================================================================================================')
    print("Finished evaluation")
