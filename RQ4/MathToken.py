import tiktoken

text = """SELECT MAX(`Percent (%) Eligible Free (K-12)`)
FROM frpm
JOIN schools ON frpm.CDSCode = schools.CDSCode
WHERE schools.County = 'Alameda';

"""

# 选择编码（根据模型选择）
# 例如 GPT-4/3.5 通常用 "cl100k_base"
encoding = tiktoken.get_encoding("cl100k_base")

# 计算 token 数量
token_count = len(encoding.encode(text))
print("Token 数量:", token_count)
