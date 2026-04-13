import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm

# ── 폰트 설정 ─────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False})

# =============================================================================
# 1. 데이터 로드 및 전처리
# =============================================================================
df = pd.read_csv('data/data_final_reduced_noised.csv')
df = df[df['거주지'] != '해당없음'].copy()

num_cols = [
    '월소득액', '공제비율', '세후_소득액', '주거·수도·광열', '정보통신', 
    '오락·문화', '음식·숙박', '월이자율', '월상환액', '남은현금', '대출금', '전세금'
]
df[num_cols] = (df[num_cols].replace({'%': '', ',': ''}, regex=True)
                            .apply(pd.to_numeric, errors='coerce')
                            .fillna(0))

df = df[df['전세금'] > 0].reset_index(drop=True)

# =============================================================================
# 2. Train/Test 분할 및 텐서 변환
# =============================================================================
drop_cols = [c for c in ['전세금', 'Unnamed: 0', 'm^2', '평수', '거주지'] if c in df.columns]
X = df.drop(columns=drop_cols)
y = np.log1p(df[['전세금']].values)

X_train_raw, X_test_raw, y_train_raw, y_test_raw, _, df_test = train_test_split(
    X, y, df, test_size=100, random_state=42
)
df_test.reset_index(drop=True, inplace=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

X_tr = torch.FloatTensor(X_train_scaled)
X_te = torch.FloatTensor(X_test_scaled)
y_tr = torch.FloatTensor(y_train_raw)
y_te = torch.FloatTensor(y_test_raw)

# =============================================================================
# 3. 모델 설계
# =============================================================================
class JeonseNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.in_block = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.res_block = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.out_block = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        h = self.in_block(x)
        res = self.res_block(h)
        h = F.gelu(res + h) 
        return self.out_block(h)

# =============================================================================
# 4. 모델 학습 설정 및 루프
# =============================================================================
model = JeonseNet(X_tr.shape[1])
criterion = nn.HuberLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50)

dataset = TensorDataset(X_tr, y_tr)
loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

train_losses, test_losses = [], []
test_r2_scores = [] # 정확도(R2) 추적 리스트 추가
best_loss = float('inf')
best_weights = None
best_epoch = 0 # 최고 성능 에포크 추적

for epoch in tqdm(range(600), desc="학습 진행도"):
    model.train()
    running_loss = 0.0
    
    # Train Phase
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(batch_X)
        
    avg_train_loss = running_loss / len(y_tr)
    train_losses.append(avg_train_loss)
    
    # Eval Phase
    model.eval()
    with torch.no_grad():
        pred_te = model(X_te)
        test_loss = criterion(pred_te, y_te).item()
        test_losses.append(test_loss)
        
        # 에포크별 R2 Score(정확도) 계산
        y_te_exp = np.expm1(y_te.numpy()).flatten()
        pred_te_exp = np.expm1(pred_te.numpy()).flatten()
        r2 = r2_score(y_te_exp, pred_te_exp)
        test_r2_scores.append(r2)
        
    scheduler.step(test_loss)
    
    # Best Model Save
    if test_loss < best_loss:
        best_loss = test_loss
        best_weights = model.state_dict().copy()
        best_epoch = epoch

# 최적 가중치 복원 및 결과 출력
model.load_state_dict(best_weights)
print("\n" + "="*60)
print(f"🎉 학습 완료! 최적 모델 로드 완료 (Epoch: {best_epoch + 1})")
print(f"🎯 최적 Test Loss (Huber) : {best_loss:.4f}")
print("="*60)

# =============================================================================
# 5. 미사용 유틸리티 함수
# =============================================================================
def load_cost_table(meters, prefix):
    regions = ['대도시', '중도시', '지방']
    levels = ['high', 'mid', 'low']
    result = {}
    
    for region, level in zip(regions, levels):
        try:
            df_cost = pd.read_csv(f'data/{prefix}_{level}_cost_average.csv')
            result[region] = dict(zip(meters, df_cost['평균전세(만원)']))
        except FileNotFoundError:
            pass 
    return result

def get_pyeong_string(target_cost, cost_dict, max_val=float('inf')):
    closest_meter = min(cost_dict.keys(), key=lambda m: abs(cost_dict[m] - target_cost))
    pyeong = round(closest_meter / 3.3058, 1)
    suffix = '+' if closest_meter >= max_val else ''
    return f"{pyeong}평{suffix}"

# =============================================================================
# 6. 예측 및 결과 데이터 프레임 구성
# =============================================================================
model.eval()
with torch.no_grad():
    preds = np.expm1(model(X_te).numpy()).flatten()
    actuals = np.expm1(y_te.numpy()).flatten()
    preds = np.maximum(preds, 1000.0)

def calculate_monthly_savings(row):
    if row['비율'] < 1:
        living_costs = row[['주거·수도·광열', '정보통신', '오락·문화', '음식·숙박']].sum()
        savings = row['세후_소득액'] - living_costs - row['월상환액']
        return max(savings, 0)
    return 0

res = df_test.copy()
res['예측전세금'] = preds.astype(int)
res['실제전세금'] = actuals.astype(int)
res['비율'] = res['실제전세금'] / res['예측전세금']

bins = [-np.inf, 0.9, 1.0, 1.1, np.inf]
labels = ['저지출', '적정', '약간과다', '과다지출']
res['판정'] = pd.cut(res['비율'], bins=bins, labels=labels)
res['월저축금'] = res.apply(calculate_monthly_savings, axis=1)

print(f"\n✅ 최종 모델 R² Score: {r2_score(actuals, preds):.4f}")
print("-" * 60)

format_dict = {
    '실제전세금': '{:>12,.0f}'.format,
    '예측전세금': '{:>12,.0f}'.format,
    '비율': '{:>8.2f}'.format,
    '월저축금': '{:>10,.0f}'.format
}
print(res[['실제전세금', '예측전세금', '비율', '판정', '월저축금']].head(20).to_string(formatters=format_dict))

# =============================================================================
# 7. 시각화 및 모델 저장
# =============================================================================
fig, axes = plt.subplots(4, 1, figsize=(5, 16))

# 공통 그리드 설정 (이중축이 있는 0번 제외)
for ax in axes[1:]:
    ax.grid(alpha=0.3)

# 1) 학습 손실 및 정확도(R²) 그래프 (이중 y축 적용)
ax1 = axes[0]
ax2 = ax1.twinx() # R² 표시를 위한 오른쪽 y축

line1 = ax1.plot(train_losses, label='Train Loss', color='tab:blue')
line2 = ax1.plot(test_losses, label='Test Loss', color='tab:orange')
line3 = ax2.plot(test_r2_scores, label='Test R² (정확도)', color='tab:green', alpha=0.7)

ax1.set_ylabel('Loss (Huber)')
ax2.set_ylabel('R² Score')
ax1.set_title('학습 손실 및 성능(R²) 추이')
ax1.grid(alpha=0.3)

# 범례 합치기
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

# 2) 실제 vs 예측 스캐터 플롯
axes[1].scatter(res['실제전세금'], res['예측전세금'], alpha=0.6, ec='navy', color='skyblue')
axes[1].axline((0, 0), slope=1, color='r', ls='--', label='기준선')
axes[1].set_title(f'실제 vs 예측 (최종 R²={r2_score(actuals, preds):.3f})')
axes[1].set_xlabel('실제 전세금')
axes[1].set_ylabel('예측 전세금')
axes[1].legend()

# 3) 전세금 상태 분포 (막대 그래프)
vc = res['판정'].value_counts()
color_map = {'저지출': 'cornflowerblue', '적정': 'mediumseagreen', '약간과다': 'orange', '과다지출': 'tomato'}
colors = [color_map.get(x, 'gray') for x in vc.index]

axes[2].bar(vc.index, vc.values, color=colors, ec='k', alpha=0.8)
for i, v in enumerate(vc.values):
    axes[2].text(i, v + 0.3, str(v), ha='center', fontweight='bold')
axes[2].set_title('전세금 상태 판정 분포')

# 4) 월 저축금 분포
savings = res[res['월저축금'] > 0]['월저축금']
if not savings.empty:
    axes[3].hist(savings, bins=20, color='mediumpurple', ec='k', alpha=0.8)
    axes[3].axvline(savings.mean(), color='r', ls='--', label=f"평균: {savings.mean():.0f}만")
    axes[3].axvline(savings.median(), color='b', ls='--', label=f"중위: {savings.median():.0f}만")
    axes[3].legend()
axes[3].set_title('월 저축금 분포 (저축 발생자 대상)')
axes[3].set_xlabel('만원')

plt.tight_layout()
plt.show()

# 8. 모델 저장
save_dict = {
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'numeric_cols': num_cols,
    'input_dim': X_tr.shape[1]
}
torch.save(save_dict, "jeonse_model_package.pth")