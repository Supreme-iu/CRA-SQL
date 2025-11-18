import sys
import json
import argparse
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut


def load_json(dir):
    with open(dir, 'r', encoding='utf8') as j:
        contents = json.loads(j.read())
    return contents


def result_callback(result):
    exec_result.append(result)


def execute_sql(predicted_sql, ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res


def execute_model(predicted_sql, ground_truth, db_place, idx, meta_time_out):
    try:
        res = func_timeout(meta_time_out, execute_sql,
                           args=(predicted_sql, ground_truth, db_place))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f'timeout',)]
        res = 0
    except Exception as e:
        result = [(f'error',)]
        res = 0

    result = {'sql_idx': idx, 'res': res}
    return result


def package_sqls(sql_path, db_root_path, mode='gpt', data_mode='dev'):
    clean_sqls = []
    db_path_list = []
    if mode == 'gpt':
        sql_data = json.load(open(sql_path + 'predict_' + data_mode + '.json', 'r', encoding='utf8'))
        for idx, sql_str in sql_data.items():
            if isinstance(sql_str, str):
                sql, db_name = sql_str.split('\t----- spider -----\t')
            else:
                sql, db_name = " ", "financial"
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')
    elif mode == 'gt':
        sqls = open(sql_path + 'test' + '_gold.sql') if data_mode == 'test' else open(sql_path + 'dev' + '_gold.sql')
        sql_txt = sqls.readlines()
        for idx, sql_str in enumerate(sql_txt):
            sql, db_name = sql_str.strip().split('\t')
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    return clean_sqls, db_path_list


def run_sqls_parallel(sqls, db_places, num_cpus=1, meta_time_out=30.0):
    pool = mp.Pool(processes=num_cpus)
    for i, sql_pair in enumerate(sqls):
        predicted_sql, ground_truth = sql_pair
        pool.apply_async(execute_model, args=(predicted_sql, ground_truth, db_places[i], i, meta_time_out),
                         callback=result_callback)
    pool.close()
    pool.join()


def sort_results(list_of_dicts):
    return sorted(list_of_dicts, key=lambda x: x['sql_idx'])


# def compute_acc_by_diff(exec_results, diff_json_path):
#     num_queries = len(exec_results)
#     results = [res['res'] for res in exec_results]
#     contents = load_json(diff_json_path)
#     easy_results, medium_results, hard_results, extra_results = [], [], [], []
#
#     for i, content in enumerate(contents):
#         if content['difficulty'] == 'easy':
#             easy_results.append(exec_results[i])
#         elif content['difficulty'] == 'medium':
#             medium_results.append(exec_results[i])
#         elif content['difficulty'] == 'hard':
#             hard_results.append(exec_results[i])
#         elif content['difficulty'] == 'extra':
#             extra_results.append(exec_results[i])
#
#     easy_acc = sum([res['res'] for res in easy_results]) / len(easy_results) if easy_results else 0
#     medium_acc = sum([res['res'] for res in medium_results]) / len(medium_results) if medium_results else 0
#     hard_acc = sum([res['res'] for res in hard_results]) / len(hard_results) if hard_results else 0
#     extra_acc = sum([res['res'] for res in extra_results]) / len(extra_results) if extra_results else 0
#     all_acc = sum(results) / num_queries if num_queries else 0
#     count_lists = [len(easy_results), len(medium_results), len(hard_results), len(extra_results), num_queries]
#     return easy_acc * 100, medium_acc * 100, hard_acc * 100, extra_acc * 100, all_acc * 100, count_lists
def compute_acc_by_diff(exec_results, diff_json_path):
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

    # 计算准确率（避免除零）
    easy_acc = sum([res['res'] for res in easy_results]) / len(easy_results) if easy_results else 0
    medium_acc = sum([res['res'] for res in medium_results]) / len(medium_results) if medium_results else 0
    hard_acc = sum([res['res'] for res in hard_results]) / len(hard_results) if hard_results else 0
    extra_acc = sum([res['res'] for res in extra_results]) / len(extra_results) if extra_results else 0
    all_acc = sum([res['res'] for res in exec_results]) / len(exec_results) if exec_results else 0

    count_lists = [len(easy_results), len(medium_results), len(hard_results), len(extra_results), len(exec_results)]
    return easy_acc * 100, medium_acc * 100, hard_acc * 100, extra_acc * 100, all_acc * 100, count_lists


def print_data(score_lists, count_lists):
    levels = ['easy', 'medium', 'hard', 'extra', 'total']
    print("{:18} {:18} {:18} {:18} {:18} {:18}".format("", *levels))
    print("{:18} {:<18} {:<18} {:<18} {:<18} {:<18}".format('count', *count_lists))
    print('============================================ ACCURACY ===============================================')
    print("{:18} {:<18.2f} {:<18.2f} {:<18.2f} {:<18.2f} {:<18.2f}".format('accuracy', *score_lists))


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
    args_parser.add_argument('--difficulty', type=str, default='simple')
    args_parser.add_argument('--diff_json_path', type=str, default='')
    args = args_parser.parse_args()
    exec_result = []

    pred_queries, db_paths = package_sqls(args.predicted_sql_path, args.db_root_path, mode=args.mode_predict,
                                          data_mode=args.data_mode)
    gt_queries, db_paths_gt = package_sqls(args.ground_truth_path, args.db_root_path, mode='gt',
                                           data_mode=args.data_mode)

    query_pairs = list(zip(pred_queries, gt_queries))
    run_sqls_parallel(query_pairs, db_places=db_paths, num_cpus=args.num_cpus, meta_time_out=args.meta_time_out)
    exec_result = sort_results(exec_result)

    print('start calculate')
    easy_acc, medium_acc, hard_acc, extra_acc, acc, count_lists = compute_acc_by_diff(exec_result, args.diff_json_path)
    score_lists = [easy_acc, medium_acc, hard_acc, extra_acc, acc]
    print_data(score_lists, count_lists)
    print('=====================================================================================================')
    print("Finished evaluation")
