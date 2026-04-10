# -*- coding: utf-8 -*-
import os
import pickle
from typing import Literal, Optional

import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# =============================================================================
# 0. 경로 설정
# =============================================================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR  = os.path.join(BASE_DIR, "saved_model")
TMPL_DIR  = os.path.join(BASE_DIR, "templates")

for path in [
    os.path.join(SAVE_DIR, "jeonse_model.pt"),
    os.path.join(SAVE_DIR, "preprocessor.pkl"),
    os.path.join(SAVE_DIR, "feature_columns.pkl"),
    os.path.join(SAVE_DIR, "tables.pkl"),
]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"저장된 파일 없음: {path}\n먼저 model_training.py를 실행하세요.")

# =============================================================================
# 1. 앱 및 템플릿
# =============================================================================
app       = FastAPI(title="전세 추천 API", version="1.0.0")
templates = Jinja2Templates(directory=TMPL_DIR)

# =============================================================================
# 2. 공통 데이터 / 함수
# =============================================================================
# 소득 구간별 평균 지출 기준 — 실제 데이터셋 기반 (만원)
EXPENSE_STD = [
    (0,      300, 32.6, 11.8, 12.3, 35.9),
    (300,    500, 34.8, 15.1, 16.4, 42.6),
    (500,    700, 38.5, 20.0, 20.7, 54.7),
    (700,    900, 40.6, 23.4, 25.1, 65.0),
    (900, 999999, 42.5, 24.8, 25.1, 68.2),
]

DEFAULT_ANNUAL_RATE = 0.04  # 전세대출 기본 연이자율 4%

def get_expense_std(salary: float) -> dict:
    for row in EXPENSE_STD:
        if row[0] <= salary < row[1]:
            return {"주거·수도·광열": row[2], "정보통신": row[3], "오락·문화": row[4], "음식·숙박": row[5]}
    return {"주거·수도·광열": 42.5, "정보통신": 24.8, "오락·문화": 25.1, "음식·숙박": 68.2}

def get_pyeong(price: float, table: dict, max_m2: Optional[float] = None) -> str:
    closest_m2 = min(table, key=lambda m: abs(table[m] - price))
    pyeong = round(closest_m2 / 3.305785, 1)
    if max_m2 and closest_m2 >= max_m2:
        return f"{pyeong}평 이상"
    return f"{pyeong}평"

def get_all_options(price: float) -> dict:
    return {
        grade: {
            "아파트":   get_pyeong(price, apt_tables[grade]),
            "오피스텔": get_pyeong(price, opi_tables[grade], max_m2=120.0),
        }
        for grade in ["대도시", "중도시", "지방"]
    }

def compare_item(value: float, avg: float) -> dict:
    diff    = value - avg
    percent = (value / avg * 100) if avg > 0 else 0.0
    if value > avg * 1.2:   tag = "과다"
    elif value > avg:        tag = "약간높음"
    elif value >= avg * 0.8: tag = "적정"
    else:                    tag = "절약"
    return {"평균": round(avg, 1), "차이": round(diff, 1), "비율": round(percent, 1), "태그": tag}

def build_custom_recommendation(city_grade: str, housing_type: str, options: dict) -> dict:
    valid_city    = city_grade    in ["대도시", "중도시", "지방"]
    valid_housing = housing_type  in ["아파트", "오피스텔"]

    if valid_city and valid_housing:
        return {"유형": "exact", "지역": city_grade, "주거형태": housing_type,
                "추천평수": options[city_grade][housing_type]}
    if valid_city:
        return {"유형": "city", "지역": city_grade, "추천": options[city_grade]}
    if valid_housing:
        return {"유형": "type", "주거형태": housing_type,
                "추천": {g: options[g][housing_type] for g in ["대도시", "중도시", "지방"]}}
    return {"유형": "none"}

def build_result(salary, housing, comm, culture, food, debt, city_grade, housing_type):
    # 대출금 역산 (월상환액 / 월이자율 = 대출원금)
    loan          = round(debt / (DEFAULT_ANNUAL_RATE / 12), 1) if debt > 0 else 0
    total_expense = housing + comm + culture + food
    remain        = salary - total_expense - debt
    expense_ratio = total_expense / salary
    debt_ratio    = debt / salary if salary > 0 else 0
    std           = get_expense_std(salary)

    # ML 예측
    user_data = {col: 0 for col in feature_columns}
    user_data.update({
        "월소득액": salary, "공제비율": 0, "세후_소득액": salary,
        "주거·수도·광열": housing, "정보통신": comm, "오락·문화": culture, "음식·숙박": food,
        "월이자율": 0, "월상환액": debt, "남은현금": max(remain, 0), "대출금": loan,
    })
    with torch.no_grad():
        predicted = model(torch.FloatTensor(preprocessor.transform(pd.DataFrame([user_data])))).item() * Y_SCALE_FACTOR
    predicted = max(1000, int(round(predicted)))

    # 전세금 범위 — 대출 부담 반영
    range_up  = 0.10 if debt_ratio < 0.2 else 0.07 if debt_ratio < 0.35 else 0.05
    min_j     = int(round(predicted * 0.90))
    max_j     = int(round(predicted * (1 + range_up)))

    # 상태 판정
    if remain < 0:              status, cls = "적자",    "danger"
    elif expense_ratio >= 0.85: status, cls = "과다지출", "danger"
    elif expense_ratio >= 0.70: status, cls = "빠듯함",  "warning"
    else:                        status, cls = "적정",    "success"

    options   = get_all_options(predicted)
    item_cmp  = {k: compare_item(v, std[k]) for k, v in
                 [("주거·수도·광열", housing), ("정보통신", comm), ("오락·문화", culture), ("음식·숙박", food)]}
    over_items = [{"항목": k, "현재": round(v, 1), "평균": round(std[k], 1),
                   "절감": round(max(0, v - std[k]), 1)}
                  for k, v in [("주거·수도·광열", housing), ("정보통신", comm),
                                ("오락·문화", culture), ("음식·숙박", food)] if v > std[k] * 1.2]

    return {
        "요약": {
            "세후_월소득": salary, "월_총지출": round(total_expense, 1),
            "월상환액": debt, "추정_대출금": loan,
            "남은현금": round(remain, 1), "상태": status, "상태클래스": cls,
            "대출부담경고": debt_ratio >= 0.35,
        },
        "전세추천": {"예측": predicted, "최소": min_j, "최대": max_j},
        "지출비교": item_cmp,
        "거주옵션": options,
        "희망조건": build_custom_recommendation(city_grade, housing_type, options),
        "과다지출": over_items,
        "저축": {"6개월": round(remain * 6, 1) if remain > 0 else 0,
                 "1년":   round(remain * 12, 1) if remain > 0 else 0},
    }

# =============================================================================
# 3. 모델 로드
# =============================================================================
class JeonseNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
    def forward(self, x): return self.net(x)

checkpoint = torch.load(os.path.join(SAVE_DIR, "jeonse_model.pt"), map_location="cpu")
with open(os.path.join(SAVE_DIR, "preprocessor.pkl"),    "rb") as f: preprocessor    = pickle.load(f)
with open(os.path.join(SAVE_DIR, "feature_columns.pkl"), "rb") as f: feature_columns = pickle.load(f)
with open(os.path.join(SAVE_DIR, "tables.pkl"),          "rb") as f: tables_data     = pickle.load(f)

apt_tables     = tables_data["apt_tables"]
opi_tables     = tables_data["opi_tables"]
model          = JeonseNet(checkpoint["input_dim"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
Y_SCALE_FACTOR = checkpoint["y_scale_factor"]

# =============================================================================
# 4. 요청 스키마
# =============================================================================
class JeonseRequest(BaseModel):
    salary:       float = Field(..., gt=0)
    housing:      float = Field(..., ge=0)
    comm:         float = Field(..., ge=0)
    culture:      float = Field(..., ge=0)
    food:         float = Field(..., ge=0)
    debt:         float = Field(..., ge=0)
    city_grade:   Optional[Literal["대도시", "중도시", "지방"]] = None
    housing_type: Optional[Literal["아파트", "오피스텔"]]       = None

# =============================================================================
# 5. 라우트
# =============================================================================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", 
                                      context={"result": None, "error": None})

@app.post("/predict")
def predict_api(payload: JeonseRequest):
    try:
        return build_result(payload.salary, payload.housing, payload.comm, payload.culture,
                            payload.food, payload.debt,
                            payload.city_grade or "", payload.housing_type or "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(
    request:      Request,
    salary:       float = Form(...),
    housing:      float = Form(...),
    comm:         float = Form(...),
    culture:      float = Form(...),
    food:         float = Form(...),
    debt:         float = Form(...),
    city_grade:   str   = Form(""),
    housing_type: str   = Form(""),
):
    try:
        if salary <= 0:
            return templates.TemplateResponse(request=request, name="index.html", 
                                              context={"result": None, "error": "세후 월소득은 0보다 커야 합니다."})
        if min(housing, comm, culture, food, debt) < 0:
            return templates.TemplateResponse(request=request, name="index.html", 
                                              context={"result": None, "error": "지출 항목은 음수일 수 없습니다."})

        result = build_result(salary, housing, comm, culture, food, debt, city_grade, housing_type)
        return templates.TemplateResponse(request=request, name="index.html", 
                                          context={"result": result, "error": None})

    except Exception as e:
        return templates.TemplateResponse(request=request, name="index.html", 
                                          context={"result": None, "error": f"오류 발생: {e}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="127.0.0.1", port=8000, reload=True)
