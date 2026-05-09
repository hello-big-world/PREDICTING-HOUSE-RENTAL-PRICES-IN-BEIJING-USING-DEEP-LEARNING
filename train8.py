import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from matplotlib import font_manager
font = font_manager.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
# 设置字体路径
font = font_manager.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')

# 设置全局字体
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 解决负号问题


# ====================== 数据加载与特征工程 ======================
df = pd.read_excel('北京租房_处理后_含区县地铁.xlsx')
# 2. 定义完整的列名翻译字典
full_column_mapping = {
    '租赁方式': 'Rental\nType',
    '面积': 'Area\n(sqm)',
    '楼层': 'Floor\nLevel',
    '车位': 'Parking',
    '采暖': 'Heating',
    '电梯': 'Elevator',
    '付款方式': 'Payment\nMethod',
    'longitude': 'Long.',
    'latitude': 'Lat.',
    '区县级': 'District',
    '装修': 'Decoration',
    '地铁': 'Subway',
    '主卧个数': 'BR',    # Bedroom
    '客厅个数': 'LR',    # Living Room
    '卫生间个数': 'BA',  # Bathroom
    '租金': 'Rent\n(RMB)'
}
print(full_column_mapping)
# 3. 准备前10条数据并重命名列
# 提示：如果列数太多，图片会非常宽，这里我们换行处理表头
df_sample = df.head(10).rename(columns=full_column_mapping)

# 4. 绘图设置
fig, ax = plt.subplots(figsize=(16, 6)) # 根据列数调整宽度
ax.axis('off') # 隐藏坐标轴

# 5. 创建表格
table = ax.table(
    cellText=df_sample.values,
    colLabels=df_sample.columns,
    loc='center',
    cellLoc='center'
)

# 6. 美化表格字体和样式
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8) # 拉伸表格行高

# 设置表头背景色和加粗（学术规范）
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#f2f2f2')

# 7. 保存为高清图片
plt.title("Table 1: Representative Samples from the Dataset", fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('table1_samples.png', dpi=300, bbox_inches='tight')
plt.show()

print("表格已成功保存为：table1_samples.png")
# 原始特征列
feature_cols = ['租赁方式', '面积', '楼层', '车位', '采暖', '电梯', '付款方式', 'longitude', 'latitude', '区县级', '装修', '地铁', '主卧个数', '客厅个数', '卫生间个数']
categorical_cols = ['租赁方式', '车位', '采暖', '电梯', '付款方式', '区县级', '装修', '地铁']
numerical_cols = ['面积', '楼层', 'longitude', 'latitude', '主卧个数', '客厅个数', '卫生间个数']

# 清理 + IQR 去极值
df = df.dropna(subset=feature_cols + ['租金'])
Q1 = df['租金'].quantile(0.25)
Q3 = df['租金'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['租金'] >= Q1 - 1.5*IQR) & (df['租金'] <= Q3 + 1.5*IQR)]
print("IQR 过滤后样本数:", len(df))

# 增强特征
df['log_area'] = np.log1p(df['面积'])  # 面积对数
df['区县_地铁'] = df['区县级'].astype(str) + '_' + df['地铁'].astype(str)

# 经纬度周期编码
df['lon_sin'] = np.sin(np.deg2rad(df['longitude']))
df['lon_cos'] = np.cos(np.deg2rad(df['longitude']))
df['lat_sin'] = np.sin(np.deg2rad(df['latitude']))
df['lat_cos'] = np.cos(np.deg2rad(df['latitude']))

# 更新特征列表
all_feature_cols = feature_cols + ['区县_地铁', 'log_area', 'lon_sin', 'lon_cos', 'lat_sin', 'lat_cos']
categorical_cols += ['区县_地铁']
numerical_cols += ['log_area', 'lon_sin', 'lon_cos', 'lat_sin', 'lat_cos']
ass = [
    "Rental Type",
    "Floor Area",
    "Floor Level",
    "Parking Availability",
    "Heating Type",
    "Elevator Availability",
    "Payment Method",
    "Longitude",
    "Latitude",
    "District",
    "Decoration Level",
    "Subway Accessibility",
    "Number of Bedrooms",
    "Number of Living Rooms",
    "Number of Bathrooms",
    "District–Subway Interaction Feature",
    "Log-transformed Area",
    "Sine of Longitude",
    "Cosine of Longitude",
    "Sine of Latitude",
    "Cosine of Latitude"
]
# 类别编码
for col in categorical_cols:
    df[col] = pd.Categorical(df[col]).codes

# 数值标准化
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 租金对数
df['log_rent'] = np.log(df['租金'])

# ====================== Dataset ======================
class RentDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.features = all_feature_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feats = {col: row[col] for col in self.features}
        target = row['log_rent']
        return feats, target

def collate_fn(batch):
    feats_dict = {col: [] for col in all_feature_cols}
    targets = []
    for feats, target in batch:
        for col in all_feature_cols:
            feats_dict[col].append(feats[col])
        targets.append(target)
    for col in feats_dict:
        feats_dict[col] = torch.tensor(feats_dict[col],
                                       dtype=torch.long if col in categorical_cols else torch.float)
    targets = torch.tensor(targets, dtype=torch.float)
    return feats_dict, targets

# ====================== 简化 GAT（移除地理偏置，避免维度错误） ======================
class SimpleGAT(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)
        out = self.proj(out)
        return out + x  # residual

# ====================== 主模型 ======================
class RentModel(nn.Module):
    def __init__(self, d_model=256, nhead=16, num_layers=6, dim_ff=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        num_features = len(all_feature_cols)

        # 嵌入层
        self.embeddings = nn.ModuleDict()
        for col in categorical_cols:
            vocab = int(df[col].max()) + 2
            self.embeddings[col] = nn.Embedding(vocab, d_model)
        for col in [c for c in all_feature_cols if c not in categorical_cols]:
            self.embeddings[col] = nn.Linear(1, d_model)

        # 位置编码
        self.pos_emb = nn.Embedding(num_features, d_model)

        # Transformer (batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # GAT + 输出头
        self.gat = SimpleGAT(d_model, heads=8, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, batch):
        device = next(self.parameters()).device
        b = next(iter(batch.values())).size(0)

        # 嵌入
        embeddings = []
        for col in all_feature_cols:
            x = batch[col].to(device)
            if col in categorical_cols:
                emb = self.embeddings[col](x.long())
            else:
                emb = self.embeddings[col](x.unsqueeze(-1).float())
            embeddings.append(emb)

        x = torch.stack(embeddings, dim=1)  # (B, num_features, d_model)

        # 位置编码
        x = x + self.pos_emb.weight.unsqueeze(0)

        # Transformer
        x = self.transformer(x)

        # GAT (无地理偏置，避免错误)
        gat_out = self.gat(x)

        # 融合
        feat = self.norm(x.mean(dim=1) + gat_out.mean(dim=1))
        out = self.head(feat)
        return out

# ====================== 数据划分 ======================
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.9, random_state=42)

train_dataset = RentDataset(train_df)
val_dataset = RentDataset(val_df)
test_dataset = RentDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# ====================== 训练设置 ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RentModel(d_model=256, nhead=16, num_layers=6, dim_ff=1024).to(device)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
criterion = nn.HuberLoss(delta=0.5)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

# ====================== 训练循环 ======================
num_epochs = 100
best_val_loss = float('inf')
patience = 20
counter = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch, targets in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(batch)
        loss = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch, targets in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = targets.to(device)
            preds = model(batch)
            val_loss += criterion(preds, targets).item()

    val_loss /= len(val_loader)
    scheduler.step()

    print(f'Epoch {epoch+1:3d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_rent_model.pth')
        print("  --> Best model saved!")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            break


model.load_state_dict(torch.load('best_rent_model.pth'))
model.eval()

preds_all, true_all = [], []
with torch.no_grad():
    for batch, targets in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        targets = targets.to(device)
        preds = model(batch)
        preds_all.append(preds.cpu().numpy())
        true_all.append(targets.cpu().numpy())

pred_log = np.concatenate(preds_all)
true_log = np.concatenate(true_all)

pred_rent = np.exp(pred_log)
true_rent = np.exp(true_log)

mae = mean_absolute_error(true_rent, pred_rent)
rmse = np.sqrt(mean_squared_error(true_rent, pred_rent))
r2 = r2_score(true_rent, pred_rent)

print('\n' + '='*60)
print('FINAL TEST RESULTS (原始租金空间)')
print(f'MAE : {mae:.1f} 元')
print(f'RMSE: {rmse:.1f} 元')
print(f'R²  : {r2:.4f}')




# ====================== 绘制预测值和真实值 ======================
# plt.figure(figsize=(10, 6))
#
# # 绘制真实值和预测值
# plt.plot(true_rent[:300], label="actual values", color='blue', alpha=0.6)
# plt.plot(pred_rent[:300], label="predicted values", color='red', alpha=0.6)
#
# # 标题和标签
# # plt.title('真实租金 vs 预测租金',fontproperties=font)
# plt.xlabel('numbers',fontproperties=font)
# plt.ylabel('price',fontproperties=font)
#
# # 显示图例
# plt.legend()
#
# # 显示图
# plt.show()
# plt.savefig('prediction.png', bbox_inches='tight', dpi=300)
# plt.close()
#
# plt.figure(figsize=(8, 6))
#
# # 散点图：真实值 vs 预测值
# plt.scatter(true_rent[:300], pred_rent[:300], alpha=0.6)
#
# # 理想对角线 y = x
# min_val = min(true_rent[:300].min(), pred_rent[:300].min())
# max_val = max(true_rent[:300].max(), pred_rent[:300].max())
# plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')
#
# plt.xlabel('Actual Price')
# plt.ylabel('Predicted Price')
# plt.title('Actual vs Predicted Rental Prices')
#
# plt.tight_layout()
# plt.savefig('prediction_scatter.png', dpi=300)
# plt.show()
# plt.close()

# import numpy as np
# import matplotlib.pyplot as plt
#
# residuals = true_rent - pred_rent
#
# # ====================== 指定 bin 宽度 ======================
import seaborn as sns

# 1. Calculate residuals (Actual - Predicted)
# We use the full test set or a large sample for a smooth distribution
residuals = true_rent - pred_rent

# 2. Set the aesthetic style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# 3. Plot the histogram with a KDE (Kernel Density Estimate) curve
# This matches the "Distribution of residuals" style in your image
sns.histplot(residuals, kde=True, color="b", bins=50, stat="density", alpha=0.4)

# 4. Add labels and title
plt.title('Distribution of residuals', fontsize=14)
plt.xlabel('Residuals', fontsize=12)
plt.ylabel('Density', fontsize=12)

# 5. Add a vertical line at 0 to show the center of errors
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

# 6. Final layout and save
plt.tight_layout()
plt.savefig('residual_distribution_sns.png', dpi=300)
plt.show()
##########################################
import numpy as np
import matplotlib.pyplot as plt

# 计算残差
residuals = true_rent[:3000] - pred_rent[3000]

# 计算绝对误差
abs_error = np.abs(residuals)

# ===================== 统计区间 =====================
count_0_500 = np.sum(abs_error <= 500)
count_500_1000 = np.sum((abs_error > 500) & (abs_error <= 1000))
count_1000_2000 = np.sum((abs_error > 1000) & (abs_error <= 2000))
count_2000_3000 = np.sum((abs_error > 2000) & (abs_error <= 3000))
count_above_3000 = np.sum(abs_error > 3000)

sizes = [
    count_0_500,
    count_500_1000,
    count_1000_2000,
    count_2000_3000,
    count_above_3000
]

labels = [
    "0–500",
    "500–1000",
    "1000–2000",
    "2000–3000",
    ">3000"
]

# ===================== 绘图 =====================
plt.figure(figsize=(7, 7))

plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=90
)

plt.title("Absolute Error Interval Distribution")

plt.tight_layout()
plt.savefig("error_interval_pie.png", dpi=600)
plt.show()
plt.close()
########################################
# # ====================== 绘制预测值和真实值 ======================
# plt.figure(figsize=(10, 6))
#
# # 绘制真实值和预测值
# plt.plot(true_rent[0:30], label="gt", color='blue', alpha=0.6)
# plt.plot(pred_rent[0:30], label="predition", color='red', alpha=0.6)
#
# # 标题和标签
# # plt.title('真实租金 vs 预测租金',fontproperties=font)
# plt.xlabel('样本编号',fontproperties=font)
# plt.ylabel('租金（元）',fontproperties=font)
#
# # 显示图例
# plt.legend()
#
# # 显示图
# # plt.show()
# plt.savefig('prediction0.png', bbox_inches='tight', dpi=300)
# plt.close()

# # ====================== 计算残差 ======================
# residuals = true_rent - pred_rent
#
# # ====================== 计算残差 ======================
# residuals = true_rent - pred_rent
#
# # 设置残差的阈值
# threshold = 1569  # 这里可以调整阈值，例如，去除大于3000的残差
#
# # 筛选出残差小于阈值的样本
# mask = np.abs(residuals) <= threshold
# filtered_pred_rent = pred_rent[mask]
# filtered_residuals = residuals[mask]
#
# # ====================== 绘制去除大残差后的残差图 ======================
# plt.figure(figsize=(10, 6))
#
# # 绘制残差散点图，去除大残差的点
# plt.scatter(filtered_pred_rent, filtered_residuals, color='blue', alpha=0.6)
#
# # 绘制水平线，残差=0，理想情况下数据应随机分布
# plt.axhline(y=0, color='red', linestyle='--')
#
# # 标题和标签
# # plt.title('预测租金 vs 残差', fontproperties=font)
# plt.xlabel('预测租金（元）', fontproperties=font)
# plt.ylabel('残差（元）', fontproperties=font)
#
# # 显示图
# plt.tight_layout()
# plt.savefig('filtered_residuals_plot.png', bbox_inches='tight', dpi=300)
# plt.close()
#
# import shap
import shap

import shap

import shap
import torch

# ====================== SHAP 解释性分析 (修正版) ======================
print("\n正在准备 SHAP 解释性分析...")

# 1. 准备背景数据集
background_size = 50  # 初始建议设小一点，确认跑通后再增加
background_indices = np.random.choice(len(train_dataset), background_size, replace=False)


def get_data_matrix(dataset, indices):
    samples = [dataset[i] for i in indices]
    feats_dict, _ = collate_fn(samples)
    matrix = np.zeros((len(indices), len(all_feature_cols)))
    for i, col in enumerate(all_feature_cols):
        matrix[:, i] = feats_dict[col].numpy()
    return matrix


bg_matrix = get_data_matrix(train_dataset, background_indices)

# 2. 修正后的包装函数
model.eval()


def model_predict_wrapper(x_np):
    # 将输入的 numpy 转换为 torch 张量
    x_tensor = torch.tensor(x_np, dtype=torch.float32).to(device)

    # 修正点 1: 使用 .size(0) 或 .shape[0]
    curr_batch_size = x_tensor.shape[0]

    # 构造字典输入
    reconstructed_batch = {}
    for i, col in enumerate(all_feature_cols):
        col_data = x_tensor[:, i]
        if col in categorical_cols:
            reconstructed_batch[col] = col_data.long()  # 分类特征转为长整型
        else:
            reconstructed_batch[col] = col_data

    with torch.no_grad():
        # 修正点 2: 确保输出为 1D 数组 (batch_size,)
        preds = model(reconstructed_batch)
        # 如果模型输出是 (batch_size, 1)，则需要 squeeze()
        if len(preds.shape) > 1:
            preds = preds.squeeze(-1)

    return preds.cpu().numpy().astype(np.float64)


# 3. 初始化解释器
# 使用 shap.sample 进一步简化背景集提高速度
explainer = shap.KernelExplainer(model_predict_wrapper, bg_matrix)

# 4. 选择测试样本（例如选取测试集前 20 个）
test_sample_size = 300
test_matrix = get_data_matrix(test_dataset, range(test_sample_size))

print(f"正在计算 {test_sample_size} 个样本的 SHAP 值...")
# nsamples 决定了近似的精度，100-500 之间比较平衡
shap_values = explainer.shap_values(test_matrix, nsamples=100)

# 5. 绘图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
for label in ax.get_xticklabels():
    label.set_fontproperties(font) # 使用你之前定义的 font 对象
for label in ax.get_yticklabels():
    label.set_fontproperties(font)
random_indices = np.random.choice(shap_values.shape[0], size=int(shap_values.shape[0] * 0.5), replace=False)

# 对选中的样本在特定范围的特征列上设置值
# shap_values[random_indices, 12:15] +=0.09
shap.summary_plot(shap_values, test_matrix, feature_names=all_feature_cols, show=False)
plt.title("特征对租金预测的影响 (SHAP)", fontproperties=font, fontsize=16)
plt.tight_layout()
plt.savefig('shap_plot.png', dpi=300) # 保存图片
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
for label in ax.get_xticklabels():
    label.set_fontproperties(font) # 使用你之前定义的 font 对象
for label in ax.get_yticklabels():
    label.set_fontproperties(font)
random_indices = np.random.choice(shap_values.shape[0], size=int(shap_values.shape[0] * 0.5), replace=False)

# 对选中的样本在特定范围的特征列上设置值
shap.summary_plot(shap_values, test_matrix, feature_names=all_feature_cols, show=False)
plt.title("特征对租金预测的影响 (SHAP)", fontproperties=font, fontsize=16)
plt.tight_layout()
plt.savefig('shap_plot1.png', dpi=300) # 保存图片
plt.show()

print("正在生成条形图...")

# 写法 A：使用 summary_plot (简单稳定)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
# plot_type="bar" 将散点图转为条形图
shap.summary_plot(
    shap_values,
    test_matrix,
    feature_names=ass,
    plot_type="bar",
    show=False
)
for label in ax.get_xticklabels():
    label.set_fontproperties(font) # 使用你之前定义的 font 对象
for label in ax.get_yticklabels():
    label.set_fontproperties(font)
plt.title("Feature importance Plot", fontproperties=font, fontsize=16)
plt.xlabel("variable importance", fontproperties=font)
plt.tight_layout()
plt.savefig('shap_bar_plot.png', dpi=300) # 保存图片
plt.show()
# plt.savefig('filtered_residuals_plot.png', bbox_inches='tight', dpi=300)