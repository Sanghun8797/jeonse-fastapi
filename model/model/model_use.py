import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import sys
import datetime

# [모델 클래스 및 로드 부분은 동일하므로 생략 가능하나 전체 흐름을 위해 유지]
class JeonseNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1))
        self.res_block1 = nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2))
        self.output_layer = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        out = self.input_layer(x)
        out = F.gelu(self.res_block1(out) + out) 
        return self.output_layer(out)

# 전역 변수 및 데이터 로드 (에러 방지용 전처리 포함)
try:
    # 모델 로드 함수 (이전과 동일)
    def load_jeonse_ai(path="jeonse_model_package.pth"):
        checkpoint = torch.load(path, weights_only=False)
        model = JeonseNet(checkpoint['input_dim'])
        mapping = {'in_block.': 'input_layer.', 'res_block.': 'res_block1.', 'out_block.': 'output_layer.'}
        new_state_dict = { (k.replace(old, new) if old in k else k): v 
                          for k, v in checkpoint['model_state_dict'].items() 
                          for old, new in mapping.items() if old in k or k == k }
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model, checkpoint['scaler'], checkpoint['numeric_cols']

    model, preprocessor, numeric_cols = load_jeonse_ai()
    global_df = pd.read_csv("data/data_final_reduced_noised.csv")
    global_estate_df = pd.read_csv("data/house.csv", encoding='utf-8-sig')
    
    global_df = global_df[(global_df['거주지'] != '해당없음') & (global_df['전세금'] > 0)].reset_index(drop=True)
    if global_df['공제비율'].dtype == 'object':
        global_df['공제비율'] = global_df['공제비율'].str.replace('%', '').astype(float)
    
    global_estate_df['건축년도'] = pd.to_numeric(global_estate_df['건축년도'], errors='coerce')
    global_estate_df = global_estate_df.dropna(subset=['건축년도'])
except Exception as e:
    print(f"❌ 초기화 오류: {e}")
    sys.exit()

# =============================================================================
# 3. [복구] 진단 및 상세 금융 분석 엔진
# =============================================================================
def diagnose_jeonse(user_input, target_jeonse, market_price=None):
    cleaned_input = {k: (float(v.replace('%', '')) if isinstance(v, str) and '%' in v else v) 
                     for k, v in user_input.items()}
    
    input_data = pd.DataFrame([cleaned_input], columns=numeric_cols[:-1])
    input_tensor = torch.tensor(preprocessor.transform(input_data), dtype=torch.float32)
    
    with torch.no_grad():
        predicted_price = np.expm1(model(input_tensor).item())

    ltv = (target_jeonse / market_price * 100) if market_price else 0
    if not market_price: safety_msg = "정보 없음"
    elif ltv >= 80: safety_msg = f"🚨 위험! (전세가율 {ltv:.1f}%) - 깡통전세 주의"
    elif ltv >= 70: safety_msg = f"⚠️ 주의! (전세가율 {ltv:.1f}%) - 반환 리스크 존재"
    else: safety_msg = f"✅ 안전! (전세가율 {ltv:.1f}%) - 보증금 보호 가능"

    price_diff = round(target_jeonse - predicted_price)
    diff_msg = f"+{price_diff:,}만원 ⚠️ (고평가)" if price_diff > 500 else f"-{abs(price_diff):,}만원 ✨ (저평가/안전)"

    report = {
        "내 계약 전세금": f"{target_jeonse:,} 만원",
        "AI 예측 적정가": f"{round(predicted_price):,} 만원",
        "보증금 안전성": safety_msg,
        "예측가 대비": diff_msg
    }
    return report, round(predicted_price)

def diagnose_peer_analysis_full(user_input, df):
    # 1. 피어 그룹 설정 (비슷한 소득 수준)
    inc = user_input['월소득액']
    peer_group = df[df['월소득액'].between(inc * 0.85, inc * 1.15)]
    if len(peer_group) < 5: 
        peer_group = df

    keys = ['주거·수도·광열', '정보통신', '오락·문화', '음식·숙박']
    
    # 2. 비중 계산 함수 (지출 / 소득 * 100)
    def get_ratio(income, spending):
        return (spending / income * 100) if income > 0 else 0

    print("\n📊 [소득 대비 지출 비중 분석]")
    print("-" * 65)
    print(f"{'지출 항목':<12} | {'내 비중(%)':>10} | {'평균(%)':>10} | {'차이(%p)':>8}")
    print("-" * 65)

    user_ratios = {}
    peer_ratios_med = {}

    for k in keys:
        # 내 소득 대비 비중
        u_ratio = get_ratio(inc, user_input[k])
        # 또래들의 소득 대비 비중 평균(중앙값)
        p_ratio = (peer_group[k] / peer_group['월소득액'] * 100).median()
        
        user_ratios[k] = u_ratio
        diff = u_ratio - p_ratio
        
        print(f"{k:<12} | {u_ratio:>10.1f}% | {p_ratio:>10.1f}% | {diff:>+8.1f}%p")
    
    print("-" * 65)

    # 3. 비중 차이가 가장 큰 항목 추출 (과소비 지표)
    # 단순히 금액이 큰 게 아니라, 평균보다 '비중'을 훨씬 많이 차지하는 항목 찾기
    top_item = max(keys, key=lambda k: user_ratios[k] - (peer_group[k] / peer_group['월소득액'] * 100).median())
    u_ratio_top = user_ratios[top_item]
    p_ratio_top = (peer_group[top_item] / peer_group['월소득액'] * 100).median()

    if u_ratio_top > p_ratio_top:
        # 평균 비중으로 줄였을 때 절약되는 '금액' 계산
        target_spending = inc * (p_ratio_top / 100)
        save_monthly = user_input[top_item] - target_spending
        
        print(f"⚠️ '{top_item}'에 쓰는 비중이 평균보다 {u_ratio_top - p_ratio_top:.1f}%p 높습니다.")
        print(f"💡 해당 비중을 평균({p_ratio_top:.1f}%) 수준으로 낮추면 월 {round(save_monthly):,}만원 저축 가능!")
        
        # 금융 솔루션 연동
        interest_rate = user_input.get('월이자율', 0)
        if isinstance(interest_rate, str):
            interest_rate = float(interest_rate.replace('%', '').strip())
        
        if interest_rate > 0:
            loan_offset = (save_monthly * 12) / (interest_rate / 100)
            print(f"💰 연간 절약액({round(save_monthly*12):,}만원)은 대출금 약 {round(loan_offset):,}만원의 이자와 맞먹습니다.")
    else:
        print("✅ 소득 수준에 비해 모든 지출 비중이 평균 이하입니다. 매우 건전한 소비 중입니다!")

# =============================================================================
# 4. [업데이트] 10년/20년 필터 매물 추천
# =============================================================================
def recommend_properties(predicted_price, df_estate, city_type, min_year=1900):
    print(f"\n🏢 [ 3. {city_type} 맞춤형 매물 추천 ]")
    print(f"   (조건: {min_year}년 이후 건축 / 예산 90% ~ 100% 내외)")
    
    min_p, max_p = predicted_price * 0.9, predicted_price * 1.0
    f_df = df_estate[(df_estate['도시분류'] == city_type) & 
                     (df_estate['추정전세'].between(min_p, max_p)) &
                     (df_estate['건축년도'] >= min_year)].copy()

    if f_df.empty:
        print("    ℹ️ 선택하신 조건의 매물이 없습니다. 조건을 완화해 보세요.")
        return

    # [수정 포인트] 면적을 일정 단위(예: 5㎡ 또는 10㎡)로 구간화하여 중복 방지
    # 여기서는 단순하게 정수형으로 변환하여 '거의 같은 평수'를 하나로 묶습니다.
    f_df['면적그룹'] = (f_df['전용면적'] / 5).round() * 5  # 5㎡ 단위로 그룹핑

    for cat in ['아파트', '오피스텔']:
        cat_df = f_df[f_df['분류'] == cat]
        print(f"\n   [{cat} 추천]")
        
        if not cat_df.empty:
            # 면적그룹별로 하나씩만 선택 (가장 신축이거나 가장 넓은 것 등 기준 설정 가능)
            # 여기서는 면적그룹별로 가장 건축년도가 최신인 것을 우선 추출
            display_df = cat_df.sort_values(['면적그룹', '건축년도'], ascending=[False, False]) \
                               .drop_duplicates(subset=['면적그룹']).head(5)
            
            # 다시 면적순으로 정렬해서 출력
            display_df = display_df.sort_values('전용면적', ascending=False)

            for _, row in display_df.iterrows():
                pyeong = round(row['전용면적'] / 3.3058, 1)
                print(f"    - {row['전용면적']:>6.2f}㎡ ({pyeong:>4}평) | {int(row['건축년도'])}년식 | {int(row['추정전세']):,}만원")
        else:
            print("    조건에 맞는 매물이 없습니다.")

# =============================================================================
# 5. 실행부 (연식 선택지 업데이트)
# =============================================================================
def run_console_app(user_idx=15):
    print("\n" + "="*60 + "\n   🏠 AI 주거 안심 & 소비 진단 서비스\n" + "="*60)

    sample_row = global_df.iloc[user_idx].to_dict()
    test_user = {k: (v if pd.notna(v) else 0) for k, v in sample_row.items()}
    target_jeonse = test_user.get('전세금', 0)
    market_price = test_user.get('매매가', int(target_jeonse / 0.8))

    # 입력 받기
    print(f"👤 진단 대상: {test_user['거주지']} 고객님")
    city_choice = input("📍 지역 선택 (1.대도시 2.중도시 3.지방): ").strip()
    city_map = {"1": "대도시", "2": "중형 도시", "3": "지방"}
    
    print("\n🏗️ 선호 연식 선택")
    print("  1. 신축급 (10년 이내) / 2. 준신축 (20년 이내) / 3. 상관없음")
    
    year_choice = input("선택: ").strip()
    
    curr_year = datetime.datetime.now().year
    year_map = {"1": curr_year - 10, "2": curr_year - 20, "3": 1900}

    # 결과 출력
    print("\n" + "★"*20 + " 종합 분석 결과 " + "★"*20)
    report, pred_p = diagnose_jeonse(test_user, target_jeonse, market_price)
    for k, v in report.items(): print(f"▶ {k:<12} : {v}")
    
    # 피어 분석 및 솔루션 출력
    diagnose_peer_analysis_full(test_user, global_df)
    
    # 추천 매물 출력
    recommend_properties(pred_p, global_estate_df, city_map.get(city_choice, "대도시"), year_map.get(year_choice, 1900))

if __name__ == "__main__":
    run_console_app(user_idx=25)