# 打开文件进行读取和写入
with open('/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/data/egoschema/example_summary.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# 替换所有的分号为换行符
modified_content = content.replace(';', '\n')

# 去除每行开头的空格
modified_content = '\n'.join([line.lstrip() for line in modified_content.split('\n')])

# 将修改后的内容写入到新的文件或覆盖原文件
with open('/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/data/egoschema/example_summary_new.txt', 'w', encoding='utf-8') as file:
    file.write(modified_content)

print("替换完成，修改后的内容已写入 output.txt")
