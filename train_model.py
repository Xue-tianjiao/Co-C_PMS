import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt

# 读取数据
file_path = 'D:\\A-清华2024\\A-实验室\\AA-文章\\数据集\\运行0530.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1', engine='openpyxl')

# 将前六列视为分类变量，后续列（除最后一列）为连续变量
X_categorical = df.iloc[:, :2]  # 前四列
X_numerical = df.iloc[:, 2:-1]  # 后续列（不包括最后一列）

# 对连续变量进行标准化
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)

# 合并分类变量和标准化后的连续变量
X = np.hstack((X_categorical.values, X_numerical_scaled))
y = df.iloc[:, -1].values  # 最后一列为目标变量

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

# 设置参数范围
param_grid = {
    'n_estimators': [1600],
    'learning_rate': [0.02],  # 设置较低的学习率
    'max_depth': [2],
    'alpha': [0],
    'lambda': [3],
    'min_child_weight': [3],
    'gamma': [0],
}
# 初始化XGBoost回归器
xg_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=80)

# 使用GridSearchCV进行10倍交叉验证
grid_search = GridSearchCV(estimator=xg_regressor, param_grid=param_grid,
                           cv=5, scoring='r2', verbose=1)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 获取最佳参数
best_params = grid_search.best_params_
print("Best parameters found by GridSearchCV:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# 使用最佳参数训练最终模型
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# 预测训练集和测试集
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# 计算训练集和测试集的 RMSE 和 R²
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train RMSE: {train_rmse:.2f}, Train R²: {train_r2:.2f}")
print(f"Test RMSE: {test_rmse:.2f}, Test R²: {test_r2:.2f}")

# 绘制训练集和测试集的拟合图
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, color='blue', label='Train Data', alpha=0.6)
plt.scatter(y_test, y_test_pred, color='green', label='Test Data', alpha=0.6)

# 绘制理想预测线
min_value = min(min(y_train), min(y_test))
max_value = max(max(y_train), max(y_test))
plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--', label='Ideal Fit')

# 添加图例和标签
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Values (Train & Test)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import joblib

# 保存模型和标准化器
joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("模型和 scaler 已保存")