import os  
import sqlite3  
import json  
import pandas as pd  

db_root_path = './data/dev_databases'  

def get_table_info(cursor, table_name):  
    """Fetch columns information from sqlite table"""  
    cursor.execute(f"PRAGMA table_info({table_name})")  
    return cursor.fetchall()  

def process_sqlite_db(db_path, db_id):  
    """Generate table and CSV descriptions from an SQLite database."""  
    conn = sqlite3.connect(db_path)  
    cursor = conn.cursor()  

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")  
    table_names = [row[0] for row in cursor.fetchall()]  

    # JSON structure for table descriptions  
    table_json_data = {  
        "db_id": db_id,  
        "table_names_original": table_names,  
        "table_names": table_names  # Assuming no transformation required  
    }  

    # Ensure the description directory exists  
    description_dir = os.path.join(os.path.dirname(db_path), 'database_description')  
    os.makedirs(description_dir, exist_ok=True)  

    # Generate CSVs for each table in the SQLite database  
    for table_name in table_names:  
        columns_info = get_table_info(cursor, table_name)  
        
        rows = []  
        for column in columns_info:  
            col_name = column[1]  
            row = {  
                "original_column_name": col_name,  
                "column_name": col_name,  
                "column_description": f"This is a description for {col_name}",  
                "data_format": "text",  # or derive from column[2] for the data type  
                "value_description": f"Example description for {col_name}"  
            }  
            rows.append(row)  

        # Save as a CSV  
        df = pd.DataFrame(rows)  
        csv_path = os.path.join(description_dir, f'{table_name}.csv')  
        df.to_csv(csv_path, index=False, encoding='latin1')  

    conn.close()  
    return table_json_data  

# Process each SQLite database in the root path  
all_table_json_data = []  

for db_id in os.listdir(db_root_path):  
    db_dir = os.path.join(db_root_path, db_id)  
    db_path = os.path.join(db_dir, f'{db_id}.sqlite')  

    if os.path.isfile(db_path):  
        table_json_data = process_sqlite_db(db_path, db_id)  
        all_table_json_data.append(table_json_data)  

# Generate the consolidated JSON description file  
output_json_path = os.path.join(db_root_path, 'dev_tables.json')  
with open(output_json_path, 'w') as f:  
    json.dump(all_table_json_data, f, indent=4)  

print(f"Descriptions generated and saved to {output_json_path}.")