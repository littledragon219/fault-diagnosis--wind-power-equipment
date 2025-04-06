# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 19:47:47 2024

@author: Saidov Abdukhalil
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    path = r"C:\Users\Administrator\Desktop\FJ020.xlsx"
    df = pd.read_excel(path)

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
    
    # Set plot style
    plt.style.use('classic')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    numeric_df = df_filled.select_dtypes(include=['float64', 'int64'])

    # Create correlation matrix and heatmap
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(16, 16))
    sns.heatmap(correlation_matrix, 
                annot=True,
                cmap='coolwarm',
                linewidths=2,
                fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

    # Modify the plot_correlation_pair function
    def plot_correlation_pair(df, var1, var2, title):
        # Create a new figure for each pair
        plt.figure(figsize=(12, 10))
        
        # Create subplot grid
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Create index for time series if not present
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = range(len(df))
        
        # Time series plot for first variable
        ax1.plot(df.index, df[var1], color='blue')
        ax1.set_title(f'{var1} 时间序列')
        ax1.set_ylabel(var1)
        ax1.grid(True)
        
        # Time series plot for second variable
        ax2.plot(df.index, df[var2], color='red')
        ax2.set_title(f'{var2} 时间序列')
        ax2.set_ylabel(var2)
        ax2.grid(True)
        
        # Scatter plot
        ax3.scatter(df[var1], df[var2], color='blue', alpha=0.5)
        ax3.set_title(f'{var1} vs {var2} 散点图')
        ax3.set_xlabel(var1)
        ax3.set_ylabel(var2)
        ax3.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    # Print column names to verify
    print("Available columns:", df_filled.columns.tolist())

    plot_correlation_pair(df_filled, 
                         '机组齿轮箱油温[NX001W20A:MBA000A1AB002AF02]',
                         '风机齿轮箱轴温[NX001W20A:MBN000A1AC002AF02]',
                         '机组齿轮箱油温 VS 风机齿轮箱轴温')
    
    plot_correlation_pair(df_filled,
                         '机组风速[NX001W20A:QZB000A1HB001AF02]',
                         '机组有功功率[NX001W20A:MZZ000A1ED001AF02]',
                         '机组风速 VS 机组有功功率')
    
    plot_correlation_pair(df_filled,
                         '机组发电机转速[NX001W20A:MAZ000A1EC002AF02]',
                         '风轮转速[NX001W20A:MCZ000A1EC002AF02]',
                         '机组发电机转速 VS 风轮转速')

    # Save the last figure
    plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')

except FileNotFoundError:
    print(f"Error: The file at {path} was not found.")
except KeyError:
    print(f"Error: Column not found in the dataset. Available columns are:\n{df_filled.columns.tolist()}")
except Exception as e:
    print(f"An unexpected error occurred: {str(e)}")
