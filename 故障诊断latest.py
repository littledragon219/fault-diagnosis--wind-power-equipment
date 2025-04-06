import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

import torch
import torch.nn as nn

# 1. 使用 pandas 读取 Excel 文件
file_path = 'D:/大三上学期/大数据技术/大作业/清洗后的数据.xlsx'
data = pd.read_excel(file_path, sheet_name=None)
df = data[list(data.keys())[0]]

# 2. 提取时间、RMS、标准差和均值数据
time = df['Time'].values
rms_data = df['RMS'].values.reshape(-1, 1)
std_data = df['Std'].values.reshape(-1, 1)
mean_data = df['Mean'].values.reshape(-1, 1)

# 3. 将 RMS、标准差和均值数据结合
combined_data = np.hstack((rms_data, std_data, mean_data))  # 将所有特征合并为一个数组

# 4. 标准化数据
scaler = StandardScaler()
combined_data_scaled = scaler.fit_transform(combined_data)

# 5. 自注意力模型定义
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        attention_scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(q.size(-1))
        attention_weights = self.softmax(attention_scores)
        output = torch.bmm(attention_weights, v)
        return output

# 6. 转换数据为 PyTorch 格式
combined_tensor = torch.tensor(combined_data_scaled, dtype=torch.float32).unsqueeze(0)  # 添加 batch 维度

# 7. 应用自注意力
self_attention = SelfAttention(input_dim=3)  # 输入维度为 3（RMS, Std, Mean）
attention_output = self_attention(combined_tensor)  # 注意力输出
fused_features = attention_output.squeeze(0).detach().numpy()  # 去掉 batch 维度

# 8. 重塑数据以适应 LSTM 输入
time_steps = 20
X_lstm = []
for i in range(len(fused_features) - time_steps):
    X_lstm.append(fused_features[i:i + time_steps])
X_lstm = np.array(X_lstm)

# 9. 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 10. 训练 LSTM 模型
input_dim = 3  # 输入特征数量
hidden_dim = 5  # LSTM 隐藏层维度
output_dim = 3  # 输出特征数量（可以根据需要调整）

lstm_model = LSTMModel(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

# 转换训练数据为 PyTorch 格式
X_lstm_tensor = torch.tensor(X_lstm, dtype=torch.float32)

# 训练 LSTM 模型
num_epochs = 500
for epoch in range(num_epochs):
    lstm_model.train()
    optimizer.zero_grad()
    output = lstm_model(X_lstm_tensor)
    loss = criterion(output, torch.tensor(fused_features[time_steps:], dtype=torch.float32))
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 11. 提取 LSTM 输出特征
lstm_model.eval()
with torch.no_grad():
    lstm_output = lstm_model(X_lstm_tensor).numpy()

# 12. 确保 LSTM 输出的长度与原始数据匹配
rms_data_reduced = rms_data[time_steps:]  # 只选择与 LSTM 输出相对应的部分

# 13. 使用 LOF 识别异常值
contamination_ratio = 0.05  # 假设预期的异常比例为 5%
lof = LocalOutlierFactor(n_neighbors=60, contamination=contamination_ratio)
y_pred = lof.fit_predict(lstm_output)

# LOF 返回 -1 表示异常值，1 表示正常值
outliers = rms_data_reduced[y_pred == -1].flatten()
normal = rms_data_reduced[y_pred == 1].flatten()

# 14. 可视化 LSTM 输出特征的单独图表
plt.figure(figsize=(12, 6))

# 绘制 LSTM 输出特征
plt.plot(time[time_steps:], lstm_output, label='LSTM Output Features', color='orange')

# 绘制正常值
plt.scatter(time[time_steps:][y_pred == 1], normal, color='blue', label='Normal', s=50)
# 绘制异常值
plt.scatter(time[time_steps:][y_pred == -1], outliers, color='red', label='Outliers', s=50)

plt.title('LSTM Output Features with Anomaly Detection')
plt.xlabel('Time')
plt.ylabel('LSTM Output Features')
plt.legend()
plt.grid()
plt.show()

# 15. 生成另一个 LSTM 输出特征的图表
plt.figure(figsize=(12, 6))

# 绘制 LSTM 输出特征
plt.plot(time[time_steps:], lstm_output, label='LSTM Output Features', color='orange')

# 用不同颜色标明异常区间
for i in range(len(y_pred)):
    if y_pred[i] == -1:
        plt.axvspan(time[time_steps:][i], time[time_steps:][i + 1], color='red', alpha=0.3)

plt.title('LSTM Output Features with Highlighted Anomalies')
plt.xlabel('Time')
plt.ylabel('LSTM Output Features')
plt.legend()
plt.grid()
plt.show()

# 输出异常值
print("Detected Outliers (RMS values):", outliers)