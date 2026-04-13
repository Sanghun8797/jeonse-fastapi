import pandas as pd
import numpy as np

def prepare_merged_dataset(apt_path, office_path, save_path="data/house.csv"):
    """
    아파트와 오피스텔 데이터를 합치고 '분류' 항목을 생성합니다.
    """
    # 1. 각 데이터 로드
    try:
        apt_df = pd.read_csv(apt_path)
        office_df = pd.read_csv(office_path)
        
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}")
        return None
    
    apt_df.columns = apt_df.columns.str.strip()
    office_df.columns = office_df.columns.str.strip()

    # 2. 가져올 핵심 항목 정의 (사용자 요청 기준)
    # 실제 데이터의 컬럼명과 일치하는지 꼭 확인하세요!
    target_cols = ["보증금", "월세", "추정전세", "전용면적", "도시분류", "건축년도"]
    
    # 3. 각 데이터프레임에서 필요한 항목만 복사 및 '분류' 태그 추가
    apt_sub = apt_df[target_cols].copy()
    apt_sub["분류"] = "아파트"
    
    office_sub = office_df[target_cols].copy()
    office_sub["분류"] = "오피스텔"

    # 4. 데이터 합치기 (위아래로 연결)
    merged_df = pd.concat([apt_sub, office_sub], axis=0, ignore_index=True)

    # 5. 모델 학습을 위한 숫자 변환 (Label Encoding)
    # PyTorch는 '아파트'라는 글자를 읽지 못하므로 숫자로 바꿔줘야 합니다.
    # 아파트: 0, 오피스텔: 1
    merged_df["분류_코드"] = merged_df["분류"].map({"아파트": 0, "오피스텔": 1})

    # 6. 결측치(NaN) 제거 및 정수 변환 (필요시)
    merged_df = merged_df.dropna()
    # 🔴 [추가] 건축년도를 정수형(int)으로 변환 (.0 제거)
    # 혹시 모를 에러를 방지하기 위해 정수 타입으로 변경합니다.
    merged_df["건축년도"] = merged_df["건축년도"].astype(int)
    
    # 7. 결과 저장 (나중에 엑셀로 보기 편하게 utf-8-sig 사용)
    merged_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print("--- 데이터 합치기 완료 ---")
    print(f"저장 위치: {save_path}")
    print(f"데이터 구성: 아파트 {len(apt_sub)}건, 오피스텔 {len(office_sub)}건 (총 {len(merged_df)}건)")
    
    return merged_df

# --- 사용 예시 ---
df = prepare_merged_dataset("data/apt_data_final.csv", "data/office_data_final.csv")
if df is not None:
    print(df.head())