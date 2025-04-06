import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score
from pmdarima import auto_arima

# 定义 Dickey-Fuller 测试函数
def ad_fuller(timeseries):
    print('Dickey-Fuller Test indicates:')
    df_test = adfuller(timeseries, regression='ct', autolag='AIC')
    output = pd.Series(df_test[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    print(output)

# 1. 导入清洗后的数据
file_path = './清洗后的数据.xlsx'  # 请根据实际路径修改
df = pd.read_excel(file_path)

# 2. 选择需要的特征
data = df['RMS'].values
time = df['Time'].values  # 使用 Time 列

# 3. 计算自相关系数和偏自相关系数
lag_acf = acf(data, nlags=20)  # 自相关系数
lag_pacf = pacf(data, nlags=20)  # 偏自相关系数

# 4. 绘制自相关和偏自相关图
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.stem(lag_acf)
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('ACF')

plt.subplot(122)
plt.stem(lag_pacf)
plt.title('Partial Autocorrelation Function (PACF)')
plt.xlabel('Lag')
plt.ylabel('PACF')

plt.tight_layout()
plt.show()

# 5. 检查平稳性（使用自定义的 ad_fuller 函数）
ad_fuller(data)

# 6. 设置训练数据长度
train_n = 400  # 训练数据长度
train_data = data[:train_n]
test_data = data[train_n:]

# 7. 使用 SARIMA 模型
# 自动选择 SARIMA 参数
auto_model = auto_arima(train_data, seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True)
print(auto_model.summary())

# 获取自动选择的参数
order = auto_model.order
seasonal_order = auto_model.seasonal_order

# 8. 使用自动选择的 SARIMA 参数拟合模型
sarima_model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
sarima_model_fit = sarima_model.fit(disp=False)

# 9. 进行预测
n_forecast = len(test_data)  # 预测测试集的长度
forecast = sarima_model_fit.get_forecast(steps=n_forecast)
forecast_values = forecast.predicted_mean
forecast_ci = forecast.conf_int()  # 获取置信区间

# 10. 计算均方误差和决定系数
mse = mean_squared_error(test_data, forecast_values)
r2 = r2_score(test_data, forecast_values)

print(f'Mean Squared Error: {mse:.4f}')
print(f'Coefficient of Determination (R^2) for Test Data: {r2:.4f}')

# 11. 可视化预测结果
plt.figure(figsize=(12, 6))
plt.plot(range(len(train_data)), train_data, label='Training Data', color='blue')
plt.plot(range(len(train_data), len(train_data) + len(forecast_values)), forecast_values, label='Predictions', color='red')

# 添加实际值
plt.plot(range(len(train_data), len(train_data) + len(test_data)), test_data, label='Actual Values', color='green', linestyle='dashed')

# 填充置信区间
# 修改此处，使用 NumPy 数组索引
plt.fill_between(range(len(train_data), len(train_data) + len(forecast_values)), 
                 forecast_ci[:, 0], forecast_ci[:, 1], color='pink', alpha=0.3)

plt.title('SARIMA Model Predictions with Confidence Interval')
plt.xlabel('Time Steps')
plt.ylabel('RMS Values')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()