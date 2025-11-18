import json  

def read_sql_file(file_path):  
    with open(file_path, 'r', encoding='utf-8') as file:  
        lines = file.readlines()  
        
    # 去除每行末尾的换行符和额外的空白  
    lines = [line.strip() for line in lines]  
    
    return lines  

def transform_data(lines):  
    json_dict = {}  
    for i, line in enumerate(lines):  
        sql_query, table = line.split('\t')  
        # 构造符合新格式的字符串  
        transformed_string = f"{sql_query}\t----- spider -----\t{table}"  
        json_dict[str(i)] = transformed_string  
    
    return json_dict  

def write_json(file_path, data):  
    with open(file_path, 'w', encoding='utf-8') as file:  
        json.dump(data, file, ensure_ascii=False, indent=4)  

if __name__ == "__main__":  
    input_file = 'spider_predict_dev.sql'  
    output_file = 'transforsmed_data.json'  
    
    # 读取和处理 SQL 文件内容  
    lines = read_sql_file(input_file)  
    transformed_data = transform_data(lines)  
    
    # 写入转换后的数据到文件  
    write_json(output_file, transformed_data)  
    
    print("数据转换完成，并已保存到", output_file)  