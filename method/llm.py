import json
import os
import time  
import openai  

# 设置 OpenAI API 配置  
openai.api_base = "https://api.siliconflow.cn/v1"
#openai.api_key = "sk-tnxnhlzczegkknjvkhjnwbliyumaszfryxqagiujhjugdomk"
openai.api_key = "sk-mcgsoianzbvfumicwugtijygemwyialhwajsdnczcfzrsfyn"

# GPT API
#openai.api_base = "https://api.chatanywhere.tech/v1"
#openai.api_key = "sk-Qj7Y8tbofL4ZifwzwJ9lguP6eUAlypvFwYFaXZxqgj30xlD6"

def connect_gpt4(message, prompt):
    print("Connecting to...")
    response = openai.ChatCompletion.create(
                    model="gpt-4-turbo-ca",  # 确认使用的模型名称正确
                    messages=[{"role": "system", "content": f"{message}"},
                              {"role": "user", "content": f"{prompt}"}],
                    temperature=0,
                    max_tokens=800,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None)
    return response['choices'][0]['message']['content']

def collect_response(prompt, max_tokens=800, stop=None):
    model_name = "Qwen/Qwen3-32B"
    while True:  
        try:  
            response = openai.ChatCompletion.create(  
                model=model_name,  # 确认使用的模型名称正确
                messages=[  
                    {"role": "system", "content": "You are an AI assistant that helps people find information."},  
                    {"role": "user", "content": f"{prompt}"}  
                ],  
                temperature=0,  
                max_tokens=max_tokens,  
                top_p=1,  
                frequency_penalty=0,  
                presence_penalty=0,  
                stop=stop,
                enable_thinking=False
            )
            print("\n" + "=" * 50)
            print("Connecting to model：" + model_name)
            print("=" * 50 + "\n")
            return response['choices'][0]['message']['content'].strip()

        except openai.error.OpenAIError as e:  
            print("OpenAI API error occurred.")  
            print(f"Exception message: {e}")  
            if hasattr(e, 'http_status'):  
                print(f"HTTP Status: {e.http_status}")  
            if hasattr(e, 'response') and isinstance(e.response, dict):  
                if 'error' in e.response:  
                    print("Error details:")  
                    print(json.dumps(e.response['error'], indent=2))  
            else:  
                print("No detailed error information available.")  
            print("Retrying in 1 second...")  
            time.sleep(1)  

        except Exception as e:  
            print(f"An unexpected error occurred: {e}")  
            time.sleep(1) 