import pandas as pd

# 读取Excel文件
df = pd.read_excel('D:/大三上学期/大数据技术/大作业/FJ020.xlsx')

# 定义一个函数，用于填充每列中仅有一行的缺失值
def fill_single_nan(col):
    # 如果该列是数值类型
    if pd.api.types.is_numeric_dtype(col):
        # 遍历列中的每个值（从第二个到倒数第二个）
        for i in range(1, len(col) - 1):
            # 如果当前值是 NaN 且前后值均有效
            if pd.isna(col[i]) and not pd.isna(col[i - 1]) and not pd.isna(col[i + 1]):
                # 用前后值的平均数填充
                col[i] = (col[i - 1] + col[i + 1]) / 2
    return col

# 对所有列应用单行填充函数，生成填充后的DataFrame
df_filled = df.apply(fill_single_nan)

# 遍历每一行，检查 NaN 的数量
for index, row in df_filled.iterrows():
    # 计算当前行中所有列的 NaN 数量
    nan_count = row.isna().sum()

    # 如果当前行有 NaN 值
    if nan_count > 0:
        # 计算该行中其他列的 NaN 数量
        other_nan_count = nan_count - 1  # 当前行的一个 NaN
        non_nan_count = len(row) - nan_count  # 其他非 NaN 的数量

        # 如果其他列的 NaN 数量大于 5，则删除该行
        if other_nan_count > 5:
            df_filled.drop(index, inplace=True)
        else:
            # 否则将当前行的 NaN 值赋值为 0.0
            df_filled.loc[index] = row.fillna(0.0)

# 将清洗后的数据保存到新的Excel文件中，文件名为'cleaned_data.xlsx'
df_filled.to_excel('cleaned_data.xlsx', index=False)

# 输出清洗后的数据以便查看
print(df_filled)
