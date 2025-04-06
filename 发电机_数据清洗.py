import os
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm  
import matplotlib.pyplot as plt

# 定义自相关零点技术函数
def ac_cz(data):  
    n = len(data)
    ac = sm.tsa.acf(data, nlags=n)  # 计算自相关序列
    czn = 0
    for i in range(n - 1):
        if ac[i] * ac[i + 1] < 0:  # 过零点累计计数
            czn += 1
    czn = czn / n  # 计算百分比
    return (ac, czn)

# 定义特征提取函数
def feature_extract(data):
    mean = np.mean(data)
    std = np.std(data)
    max_val = np.max(data)
    min_val = np.min(data)
    rms = np.sqrt(np.mean(data**2))
    return np.array([mean, std, max_val, min_val, rms])

# 初始化需要的列表
Speed = []
Time = []
CZN = []
Feature = []
File_path = []

# 文件路径和文件名设置
TD_choose = './发电机_自由端_水平_51200'
Folder_path = os.path.join(TD_choose)
File_name = os.listdir(Folder_path)

# 遍历文件，提取时间、速度和其他数据
for n in File_name:
    _index = [i for i, j in enumerate(n) if j == "_"]
    time_str = n[_index[-1] + 1: -4]
    time_sec = datetime.datetime.strptime(time_str, "%Y%m%d%H%M%S")
    speed = int(n[_index[-2] + 1: _index[-1] - 3])
    file_path = os.path.join(Folder_path, n)
    Speed.append(speed)
    Time.append(time_sec)
    
    with open(file_path, 'r', encoding='UTF-8-sig') as x:
        data = x.read().split()
    data = np.array([float(i) for i in data])
    
    feature = feature_extract(data)  # 计算特征
    Feature.append(feature)
    ac, czn = ac_cz(data)  # 计算自相关和过零点比例
    CZN.append(czn)
    File_path.append(file_path)

# 将数据按时间排序
Z = list(zip(Time, Feature, Speed, CZN, File_path))
Z.sort()  # 按时间排序
Time, Feature, Speed, CZN, File_path = zip(*Z)
Time = np.array(Time)
Feature = np.array(Feature)
Speed = np.array(Speed)
CZN = np.array(CZN)

# 计算停机天数
delta_days = []
Re_Time = [0]
for i in range(len(Time) - 1):
    delta_days.append((Time[i + 1] - Time[i]).days)
    delta = (Time[i + 1] - Time[i]).total_seconds() / (24 * 60 * 60) - delta_days[i]
    Re_Time.append(Re_Time[i] + delta)
Re_Time = np.array(Re_Time)
Index_1 = [i for i, x in enumerate(delta_days) if x >= 1]

# 删除非工作转速的数据
Junk_1 = [i for i, x in enumerate(Speed) if x not in range(1000, 2000)]
Index_1 = list(set(range(len(Speed))) - set(Junk_1))
Time_1 = Re_Time[Index_1]
Feature_1 = Feature[Index_1]  # 保留特征
CZN_1 = CZN[Index_1]

# 删除 CZN 值小于 0.05 的数据
Junk_2 = [i for i, x in enumerate(CZN_1) if x < 0.05]
Junk = set(Junk_1) | set(Junk_2)
Index_2 = list(set(range(len(CZN_1))) - Junk)

# 提取清洗后的数据
Time_2 = Re_Time[Index_2]
Feature_2 = Feature_1[Index_2]

# 创建一个 DataFrame，包括所有需要的特征
output_df = pd.DataFrame({
    'Time': Time_2,
    'Mean': Feature_2[:, 0],
    'Std': Feature_2[:, 1],
    'Max': Feature_2[:, 2],
    'Min': Feature_2[:, 3],
    'RMS': Feature_2[:, 4],
    'Speed': Speed[Index_2],
    'CZN': CZN_1[Index_2]
})

# 指定 Excel 文件名
output_file = './清洗后的数据.xlsx'  # 确保有 .xlsx 扩展名

# 将 DataFrame 输出为 Excel 文件
output_df.to_excel(output_file, index=False)

print(f"数据已成功输出到 {output_file}")

# 设置图形大小
plt.figure(figsize=(12, 6))

# 选择要绘制的特征列（除了 'Time'）
features = ['Mean', 'Std', 'RMS', 'Speed', 'CZN']  

# 使用 for 循环绘制每个特征
for feature in features:
    plt.figure(figsize=(12, 6))
    plt.plot(Time_2, output_df[feature], marker='o')
    plt.title(f'{feature} Over Time', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.show()