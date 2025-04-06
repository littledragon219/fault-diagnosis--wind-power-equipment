import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据准备
df = pd.read_excel('./清洗后的数据.xlsx')  # 请根据实际路径修改
time = df['Time'].values
rms = df['RMS'].values

# 2. 设置 RMS 阈值
threshold = 40
failure_time = time[np.where(rms >= threshold)[0][0]] if any(rms >= threshold) else time[-1]
rul = failure_time - time  # 计算剩余寿命（RUL）

# 3. 初始状态定义 (x0)
x0 = rms[0]  # 假设初始健康状态等于初始的 RMS 值

# 4. 最大似然估计（MLE）函数
def MLE_est(t, dx):
    K = len(t)
    
    # 定义 lambda 函数
    a = lambda b: np.sum(dx * (np.exp(b * t) - 1)) / (np.sum((np.exp(b * t) - 1)**2) + 1e-8)

    sigma2 = lambda b: np.mean((dx - a(b) * (np.exp(b * t) - 1))**2)
    
    log_L = lambda b: K / 2 * np.log(2 * np.pi * sigma2(b)) + K / 2
    
    # 优化以获得最佳的 b 值
    opt_res = minimize(log_L, 0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    b_est = opt_res.x[0]
    sigma2_est = sigma2(b_est)
    a_est = a(b_est)
    
    return a_est, b_est, sigma2_est

# 使用 dx = rms - x0 进行估计
dx = rms - x0
a, b, sigma2 = MLE_est(time, dx)

# 5. 使用指数模型来拟合健康指标
def exp_model(t, a, b):
    return a * (np.exp(b * t) - 1)

# 6. 拟合指数模型
initial_params = [x0, 0.01]  # 假设初始参数
params, covariance = curve_fit(exp_model, time, rms, p0=initial_params, maxfev=10000)

# 7. 预测 RMS 的未来值
time_future = np.arange(time[-1], time[-1] + 100)  # 假设未来 100 个时间点
rms_pred = exp_model(time_future, *params)

# 8. 计算 RUL 的概率密度函数（PDF）
def rul_pdf(param, tau, D):
    x0, a, b, sigma2 = param
    phi = lambda z: np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
    pdf = phi((D - x0 - a * (np.exp(b * tau) - 1)) / np.sqrt(sigma2)) * a * b * np.exp(b * tau) / np.sqrt(sigma2)
    return pdf

# 9. 计算 RUL PDF
tau = np.arange(1, 120)  # 时间范围
param = [x0, a, b, sigma2]
D = threshold
pdf = rul_pdf(param, tau, D)

# 10. 可视化 RMS 预测和 RUL PDF
fig, ax1 = plt.subplots(figsize=(10, 6))

# RMS 预测
ax1.plot(time, rms, label="Actual RMS", color="blue")
ax1.plot(time_future, rms_pred, label="Predicted RMS", color="red", linestyle="dashed")
ax1.axhline(y=threshold, color="green", linestyle="--", label="Failure Threshold")
ax1.set_xlabel("Time")
ax1.set_ylabel("RMS")
ax1.legend(loc='upper left')
ax1.set_title("RMS Prediction with Exponential Model and RUL PDF")

# 创建第二个 y 轴
ax2 = ax1.twinx()
ax2.plot(tau, pdf, label="RUL PDF", color="purple")
ax2.set_ylabel("Probability Density")
ax2.legend(loc='upper right')

plt.grid()
plt.show()

# 11. 计算均方误差和 R²
mse = mean_squared_error(rms, exp_model(time, *params))
r2 = r2_score(rms, exp_model(time, *params))
print(f"Mean Squared Error: {mse:.4f}")
print(f"R²: {r2:.4f}")

# 12. 计算残差
residuals = rms - exp_model(time, *params)

# 13. 残差分布可视化
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=True, color='purple')
plt.title("Distribution of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid()
plt.show()

# 14. 寿命预测时间点
failure_index = np.where(rms_pred >= threshold)[0][0]  # 第一个超出阈值的位置
predicted_failure_time = time_future[failure_index]
predicted_rul = predicted_failure_time - time[-1]  # 剩余寿命

print(f"Predicted Failure Time: {predicted_failure_time}")
print(f"Predicted Remaining Useful Life (RUL): {predicted_rul}")