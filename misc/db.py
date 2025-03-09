import pandas as pd

# 读取 Parquet 文件
parquet_file = '/Users/sunqifan/Desktop/egoschema_videos/parquet/subset_test-00000-of-00001.parquet'
df = pd.read_parquet(parquet_file)

# 将 DataFrame 写入 Excel 文件
excel_file = '/Users/sunqifan/Desktop/egoschema_videos/parquet/subset_test.xlsx'
df.to_excel(excel_file, index=False)

print(f"Parquet file has been converted to Excel file: {excel_file}")