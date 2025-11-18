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
    try:
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
        return 1 if set(predicted_res) == set(ground_truth_res) else 0
    except Exception as e:
        print(f"SQL Error: {e}")
        return 0
    finally:
        conn.close()

def execute_model(predicted_sql, ground_truth, db_place, idx, meta_time_out):
    try:
        res = func_timeout(meta_time_out, execute_sql,
                         args=(predicted_sql, ground_truth, db_place))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        print(f"Timeout on SQL {idx}: {predicted_sql[:100]}...")
        res = 0
    except Exception as e:
        print(f"Error on SQL {idx}: {e}")
        res = 0
    return {'sql_idx': idx, 'res': res}

def package_sqls(sql_path, db_root_path, mode='gpt', data_mode='dev'):
    clean_sqls = []
    db_path_list = []
    if mode == 'gpt':
        with open(sql_path + 'predict_' + data_mode + '.json', 'r', encoding='utf8') as f:
            sql_data = json.load(f)
        for idx, sql_str in sql_data.items():
            if isinstance(sql_str, str):
                sql, db_name = sql_str.split('\t----- bird -----\t')
                clean_sqls.append(sql)
                db_path_list.append(f"{db_root_path}{db_name}/{db_name}.sqlite")
            else:
                clean_sqls.append(" ")
                db_path_list.append(f"{db_root_path}financial/financial.sqlite")
    elif mode == 'gt':
        filename = 'test_gold.sql' if data_mode == 'test' else 'dev_gold.sql'
        with open(sql_path + filename) as f:
            sql_txt = f.readlines()
        for idx, sql_str in enumerate(sql_txt):
            sql, db_name = sql_str.strip().split('\t')
            clean_sqls.append(sql)
            db_path_list.append(f"{db_root_path}{db_name}/{db_name}.sqlite")
    return clean_sqls, db_path_list

def run_sqls_parallel(sqls, db_places, num_cpus=1, meta_time_out=30.0):
    pool = mp.Pool(processes=num_cpus)
    results = []
    for i, sql_pair in enumerate(sqls):
        predicted_sql, ground_truth = sql_pair
        results.append(
            pool.apply_async(
                execute_model,
                args=(predicted_sql, ground_truth, db_places[i], i, meta_time_out),
                callback=result_callback
            )
        )
    pool.close()
    pool.join()
    return [r.get() for r in results]

def sort_results(list_of_dicts):
    return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

def compute_acc_by_diff(exec_results, diff_json_path):
    contents = load_json(diff_json_path)
    
    # Ensure lengths match
    min_len = min(len(exec_results), len(contents))
    exec_results = exec_results[:min_len]
    contents = contents[:min_len]
    
    simple_results, moderate_results, challenging_results = [], [], []
    
    for i in range(min_len):
        difficulty = contents[i]['difficulty']
        if difficulty == 'simple':
            simple_results.append(exec_results[i])
        elif difficulty == 'moderate':
            moderate_results.append(exec_results[i])
        elif difficulty == 'challenging':
            challenging_results.append(exec_results[i])
    
    def calc_acc(results):
        return sum(r['res'] for r in results) / len(results) * 100 if results else 0.0
    
    simple_acc = calc_acc(simple_results)
    moderate_acc = calc_acc(moderate_results)
    challenging_acc = calc_acc(challenging_results)
    all_acc = calc_acc(exec_results)
    
    count_lists = [
        len(simple_results),
        len(moderate_results),
        len(challenging_results),
        len(exec_results)
    ]
    
    return simple_acc, moderate_acc, challenging_acc, all_acc, count_lists

def print_data(score_lists, count_lists):
    levels = ['simple', 'moderate', 'challenging', 'total']
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))
    print('======================================    ACCURACY    =====================================')
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('accuracy', *score_lists))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicted_sql_path', type=str, required=True)
    parser.add_argument('--ground_truth_path', type=str, required=True)
    parser.add_argument('--data_mode', type=str, required=True, default='dev')
    parser.add_argument('--db_root_path', type=str, required=True)
    parser.add_argument('--num_cpus', type=int, default=1)
    parser.add_argument('--meta_time_out', type=float, default=30.0)
    parser.add_argument('--mode_gt', type=str, default='gt')
    parser.add_argument('--mode_predict', type=str, default='gpt')
    parser.add_argument('--difficulty', type=str, default='simple')
    parser.add_argument('--diff_json_path', type=str, required=True)
    args = parser.parse_args()
    
    exec_result = []
    
    pred_queries, db_paths = package_sqls(
        args.predicted_sql_path,
        args.db_root_path,
        mode=args.mode_predict,
        data_mode=args.data_mode
    )
    
    gt_queries, db_paths_gt = package_sqls(
        args.ground_truth_path,
        args.db_root_path,
        mode='gt',
        data_mode=args.data_mode
    )
    
    query_pairs = list(zip(pred_queries, gt_queries))
    exec_result = run_sqls_parallel(
        query_pairs,
        db_places=db_paths,
        num_cpus=args.num_cpus,
        meta_time_out=args.meta_time_out
    )
    exec_result = sort_results(exec_result)
    
    print('start calculate')
    simple_acc, moderate_acc, challenging_acc, acc, count_lists = compute_acc_by_diff(
        exec_result,
        args.diff_json_path
    )
    score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
    print_data(score_lists, count_lists)
    print('===========================================================================================')
    print("Finished evaluation")