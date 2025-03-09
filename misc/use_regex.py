import re

# 测试字符串
text = '{ "name": "John", "age": 30 } more text [1, 2, 3] and even more text.'

# 正则表达式B：使用贪婪匹配
pattern_b = r"\{.*\}|\[.*\]"

# 使用 re.findall() 来查找所有匹配的内容
matches_b = re.findall(pattern_b, text)

# 打印匹配结果
print("匹配结果（使用贪婪匹配B）:")
for match in matches_b:
    print(match)
