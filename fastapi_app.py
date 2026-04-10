import os
import pickle
from typing import Literal, Optional

import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# =============================================================================
# 0. 경로 설정
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_model")

MODEL_PATH = os.path.join(SAVE_DIR, "jeonse_model.pt")
PREPROCESSOR_PATH = os.path.join(SAVE_DIR, "preprocessor.pkl")
FEATURE_COLUMNS_PATH = os.path.join(SAVE_DIR, "feature_columns.pkl")
TABLES_PATH = os.path.join(SAVE_DIR, "tables.pkl")

required_files = [MODEL_PATH, PREPROCESSOR_PATH, FEATURE_COLUMNS_PATH, TABLES_PATH]
missing_files = [p for p in required_files if not os.path.exists(p)]
if missing_files:
    missing_text = "\n".join(missing_files)
    raise FileNotFoundError(
        "저장된 파일이 없습니다. 먼저 train_model.py를 실행하세요.\n"
        f"{missing_text}"
    )

# =============================================================================
# 1. FastAPI 앱
# =============================================================================
app = FastAPI(
    title="전세 추천 API",
    description="학습된 모델을 이용해 사용자 입력 기반 전세 추천 결과를 반환합니다.",
    version="1.0.0",
)

# =============================================================================
# 2. 공통 함수
# =============================================================================
expense_std = [
    (0,   300, 34.1, 12.7, 13.6, 38.2),
    (300, 500, 35.0, 14.9, 16.4, 42.6),
    (500, 700, 38.3, 19.7, 20.8, 54.9),
    (700, 900, 41.1, 23.4, 25.4, 65.1),
    (900, 999999, 42.3, 24.9, 25.2, 68.9),
]


def get_expense_std(salary: float) -> dict:
    for row in expense_std:
        if row[0] <= salary < row[1]:
            return {
                "주거·수도·광열": row[2],
                "정보통신": row[3],
                "오락·문화": row[4],
                "음식·숙박": row[5],
            }
    return {
        "주거·수도·광열": 42.3,
        "정보통신": 24.9,
        "오락·문화": 25.2,
        "음식·숙박": 68.9,
    }


def get_pyeong(price: float, table: dict, max_m2: Optional[float] = None) -> str:
    closest_m2 = min(table, key=lambda m: abs(table[m] - price))
    pyeong = round(closest_m2 / 3.305785, 1)
    if max_m2 and closest_m2 >= max_m2:
        return f"{pyeong}평 이상"
    return f"{pyeong}평"


def get_all_options(price: float, apt_tables: dict, opi_tables: dict) -> dict:
    return {
        grade: {
            "아파트": get_pyeong(price, apt_tables[grade]),
            "오피스텔": get_pyeong(price, opi_tables[grade], max_m2=120.0),
        }
        for grade in ["대도시", "중도시", "지방"]
    }


def compare_item(value: float, avg: float) -> dict:
    diff = value - avg
    percent = (value / avg * 100) if avg > 0 else 0.0

    if value > avg * 1.2:
        item_status = "🚨 과다"
    elif value > avg:
        item_status = "⚠️ 약간 높음"
    elif value >= avg * 0.8:
        item_status = "✅ 적정"
    else:
        item_status = "💡 절약"

    return {
        "평균": round(avg, 1),
        "차이": round(diff, 1),
        "비율(%)": round(percent, 1),
        "상태": item_status,
    }


def build_custom_recommendation(
    city_grade: str,
    housing_type: str,
    options: dict,
) -> dict:
    valid_city = city_grade in ["대도시", "중도시", "지방"]
    valid_housing = housing_type in ["아파트", "오피스텔"]

    if valid_city and valid_housing:
        return {
            "유형": "정확 추천",
            "내용": f"{city_grade} {housing_type} 추천 평수",
            "추천평수": options[city_grade][housing_type],
        }

    if valid_city:
        return {
            "유형": "지역 기준 추천",
            "지역": city_grade,
            "추천": {
                "아파트": options[city_grade]["아파트"],
                "오피스텔": options[city_grade]["오피스텔"],
            },
        }

    if valid_housing:
        return {
            "유형": "주거형태 기준 추천",
            "주거형태": housing_type,
            "추천": {
                grade: options[grade][housing_type]
                for grade in ["대도시", "중도시", "지방"]
            },
        }

    return {
        "유형": "안내",
        "내용": "희망 지역이나 주거형태를 입력하면 맞춤 추천을 추가로 볼 수 있습니다.",
    }


def build_result(
    salary: float,
    housing: float,
    comm: float,
    culture: float,
    food: float,
    debt: float,
    city_grade: str,
    housing_type: str,
):
    total_expense = housing + comm + culture + food
    remain = salary - total_expense - debt
    expense_ratio = total_expense / salary

    user_data = {col: 0 for col in feature_columns}
    user_data["월소득액"] = salary
    user_data["공제비율"] = 0
    user_data["세후_소득액"] = salary
    user_data["주거·수도·광열"] = housing
    user_data["정보통신"] = comm
    user_data["오락·문화"] = culture
    user_data["음식·숙박"] = food
    user_data["월이자율"] = 0
    user_data["월상환액"] = debt
    user_data["남은현금"] = remain
    user_data["대출금"] = 0

    input_df = pd.DataFrame([user_data])
    X_user = preprocessor.transform(input_df)
    X_user_t = torch.FloatTensor(X_user)

    with torch.no_grad():
        predicted_jeonse = model(X_user_t).item() * Y_SCALE_FACTOR

    predicted_jeonse = max(1000, int(round(predicted_jeonse)))
    min_jeonse = int(round(predicted_jeonse * 0.9))
    max_jeonse = int(round(predicted_jeonse * 1.1))

    std = get_expense_std(salary)

    if remain < 0:
        status = "🚨 적자"
    elif expense_ratio >= 0.85:
        status = "🚨 과다지출"
    elif expense_ratio >= 0.70:
        status = "⚠️ 빠듯함"
    else:
        status = "✅ 적정"

    options = get_all_options(predicted_jeonse, apt_tables, opi_tables)

    over_items = []
    for label, value in [
        ("주거·수도·광열", housing),
        ("정보통신", comm),
        ("오락·문화", culture),
        ("음식·숙박", food),
    ]:
        avg = std[label]
        if value > avg * 1.2:
            over_items.append({
                "항목": label,
                "현재": round(value, 1),
                "평균": round(avg, 1),
                "절감여지": round(max(0, value - avg), 1),
            })

    item_compare = {
        "주거·수도·광열": compare_item(housing, std["주거·수도·광열"]),
        "정보통신": compare_item(comm, std["정보통신"]),
        "오락·문화": compare_item(culture, std["오락·문화"]),
        "음식·숙박": compare_item(food, std["음식·숙박"]),
    }

    custom_recommendation = build_custom_recommendation(
        city_grade=city_grade,
        housing_type=housing_type,
        options=options,
    )

    return {
        "입력값": {
            "salary": salary,
            "housing": housing,
            "comm": comm,
            "culture": culture,
            "food": food,
            "debt": debt,
            "city_grade": city_grade,
            "housing_type": housing_type,
        },
        "요약": {
            "세후_월소득": round(salary, 1),
            "월_총지출": round(total_expense, 1),
            "월상환액": round(debt, 1),
            "남은현금": round(remain, 1),
            "소비상태": status,
        },
        "전세추천": {
            "모델_예측_전세금": predicted_jeonse,
            "적정_전세금_범위": {
                "최소": min_jeonse,
                "최대": max_jeonse,
            },
        },
        "평균지출비교": item_compare,
        "거주가능평수": options,
        "사용자희망조건기준": custom_recommendation,
        "소비조언": over_items if over_items else ["평균 대비 특별히 과다한 지출 항목이 없습니다."],
        "저축가능액": {
            "6개월": round(remain * 6, 1) if remain > 0 else 0,
            "1년": round(remain * 12, 1) if remain > 0 else 0,
        },
    }


def render_html(result=None, error_message=""):
    result_html = ""

    if result:
        summary = result["요약"]
        jeonse = result["전세추천"]
        compare = result["평균지출비교"]
        options = result["거주가능평수"]
        custom = result["사용자희망조건기준"]
        advice = result["소비조언"]
        saving = result["저축가능액"]

        compare_rows = ""
        for label, info in compare.items():
            compare_rows += f"""
            <tr>
                <td>{label}</td>
                <td>{info['평균']}</td>
                <td>{info['차이']}</td>
                <td>{info['비율(%)']}%</td>
                <td>{info['상태']}</td>
            </tr>
            """

        options_rows = ""
        for grade in ["대도시", "중도시", "지방"]:
            options_rows += f"""
            <tr>
                <td>{grade}</td>
                <td>{options[grade]['아파트']}</td>
                <td>{options[grade]['오피스텔']}</td>
            </tr>
            """

        advice_html = ""
        if isinstance(advice, list):
            for item in advice:
                if isinstance(item, dict):
                    advice_html += f"<li>{item['항목']}: 현재 {item['현재']}만원 / 평균 {item['평균']}만원 / 절감여지 {item['절감여지']}만원</li>"
                else:
                    advice_html += f"<li>{item}</li>"

        custom_html = ""
        if custom["유형"] == "정확 추천":
            custom_html = f"<p>{custom['내용']}: <strong>{custom['추천평수']}</strong></p>"
        elif custom["유형"] == "지역 기준 추천":
            custom_html = f"""
            <p><strong>{custom['지역']}</strong> 기준 추천</p>
            <ul>
                <li>아파트: {custom['추천']['아파트']}</li>
                <li>오피스텔: {custom['추천']['오피스텔']}</li>
            </ul>
            """
        elif custom["유형"] == "주거형태 기준 추천":
            rows = ""
            for grade, p in custom["추천"].items():
                rows += f"<li>{grade}: {p}</li>"
            custom_html = f"""
            <p><strong>{custom['주거형태']}</strong> 기준 추천</p>
            <ul>{rows}</ul>
            """
        else:
            custom_html = f"<p>{custom['내용']}</p>"

        result_html = f"""
        <div class="card">
            <h2>📋 결과 리포트</h2>
            <p><strong>세후 월소득:</strong> {summary['세후_월소득']}만원</p>
            <p><strong>월 총지출:</strong> {summary['월_총지출']}만원</p>
            <p><strong>월상환액:</strong> {summary['월상환액']}만원</p>
            <p><strong>남은현금:</strong> {summary['남은현금']}만원</p>
            <p><strong>소비 상태:</strong> {summary['소비상태']}</p>
            <p><strong>모델 예측 전세금:</strong> {jeonse['모델_예측_전세금']}만원</p>
            <p><strong>적정 전세금 범위:</strong> {jeonse['적정_전세금_범위']['최소']}만원 ~ {jeonse['적정_전세금_범위']['최대']}만원</p>
        </div>

        <div class="card">
            <h2>📊 평균 지출 기준 비교</h2>
            <table>
                <thead>
                    <tr>
                        <th>항목</th>
                        <th>평균(만원)</th>
                        <th>차이(만원)</th>
                        <th>비율</th>
                        <th>상태</th>
                    </tr>
                </thead>
                <tbody>
                    {compare_rows}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>🏠 예상 전세금으로 거주 가능한 평수</h2>
            <table>
                <thead>
                    <tr>
                        <th>지역</th>
                        <th>아파트</th>
                        <th>오피스텔</th>
                    </tr>
                </thead>
                <tbody>
                    {options_rows}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>🎯 사용자 희망 조건 기준</h2>
            {custom_html}
        </div>

        <div class="card">
            <h2>💡 소비 조언</h2>
            <ul>{advice_html}</ul>
        </div>

        <div class="card">
            <h2>💰 예상 저축 가능액</h2>
            <p><strong>6개월:</strong> {saving['6개월']}만원</p>
            <p><strong>1년:</strong> {saving['1년']}만원</p>
        </div>
        """

    error_html = f'<div class="error">{error_message}</div>' if error_message else ""

    return f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>전세 추천 프로그램</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #f5f7fb;
                margin: 0;
                padding: 0;
            }}
            .container {{
                max-width: 900px;
                margin: 40px auto;
                padding: 20px;
            }}
            .card {{
                background: white;
                border-radius: 14px;
                padding: 24px;
                margin-bottom: 20px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            }}
            h1, h2 {{
                margin-top: 0;
            }}
            label {{
                display: block;
                margin-top: 14px;
                font-weight: bold;
            }}
            input, select {{
                width: 100%;
                padding: 10px;
                margin-top: 6px;
                border: 1px solid #ccc;
                border-radius: 8px;
                box-sizing: border-box;
            }}
            button {{
                width: 100%;
                margin-top: 20px;
                padding: 14px;
                border: none;
                border-radius: 10px;
                background: #2563eb;
                color: white;
                font-size: 16px;
                cursor: pointer;
            }}
            button:hover {{
                background: #1d4ed8;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 10px;
                text-align: center;
            }}
            th {{
                background: #f0f4ff;
            }}
            .desc {{
                color: #555;
                font-size: 14px;
                margin-top: 6px;
            }}
            .error {{
                background: #fee2e2;
                color: #b91c1c;
                padding: 12px;
                border-radius: 10px;
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>🏠 사용자 직접 입력 기반 전세 추천</h1>
                <p class="desc">모든 금액 단위는 <strong>만원</strong>입니다. 예: 280 입력 = 280만원</p>
                {error_html}
                <form method="post" action="/predict-form">
                    <label>세후 월소득 (만원)</label>
                    <input type="number" step="0.1" min="0.1" name="salary" placeholder="예: 280" required>

                    <label>주거·수도·광열비 (만원)</label>
                    <input type="number" step="0.1" min="0" name="housing" placeholder="예: 40" required>

                    <label>정보통신비 (만원)</label>
                    <input type="number" step="0.1" min="0" name="comm" placeholder="예: 10" required>

                    <label>오락·문화비 (만원)</label>
                    <input type="number" step="0.1" min="0" name="culture" placeholder="예: 15" required>

                    <label>음식·숙박비 (만원)</label>
                    <input type="number" step="0.1" min="0" name="food" placeholder="예: 50" required>

                    <label>월상환액 (만원)</label>
                    <input type="number" step="0.1" min="0" name="debt" placeholder="예: 20" required>

                    <label>희망 지역 수준</label>
                    <select name="city_grade">
                        <option value="">선택 안 함</option>
                        <option value="대도시">대도시</option>
                        <option value="중도시">중도시</option>
                        <option value="지방">지방</option>
                    </select>

                    <label>희망 주거형태</label>
                    <select name="housing_type">
                        <option value="">선택 안 함</option>
                        <option value="아파트">아파트</option>
                        <option value="오피스텔">오피스텔</option>
                    </select>

                    <button type="submit">전세 추천 결과 보기</button>
                </form>
            </div>

            {result_html}
        </div>
    </body>
    </html>
    """


# =============================================================================
# 3. 모델 정의
# =============================================================================
class JeonseNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# 4. 저장된 객체 불러오기
# =============================================================================
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

with open(FEATURE_COLUMNS_PATH, "rb") as f:
    feature_columns = pickle.load(f)

with open(TABLES_PATH, "rb") as f:
    tables_data = pickle.load(f)

apt_tables = tables_data["apt_tables"]
opi_tables = tables_data["opi_tables"]

model = JeonseNet(checkpoint["input_dim"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

Y_SCALE_FACTOR = checkpoint["y_scale_factor"]


# =============================================================================
# 5. 요청 스키마
# =============================================================================
class JeonseRequest(BaseModel):
    salary: float = Field(..., gt=0, description="세후 월소득 (만원)", example=280)
    housing: float = Field(..., ge=0, description="주거·수도·광열 (만원)", example=40)
    comm: float = Field(..., ge=0, description="정보통신 (만원)", example=10)
    culture: float = Field(..., ge=0, description="오락·문화 (만원)", example=15)
    food: float = Field(..., ge=0, description="음식·숙박 (만원)", example=50)
    debt: float = Field(..., ge=0, description="월상환액 (만원)", example=20)
    city_grade: Optional[Literal["대도시", "중도시", "지방"]] = Field(default=None, description="희망 지역 수준")
    housing_type: Optional[Literal["아파트", "오피스텔"]] = Field(default=None, description="희망 주거형태")


# =============================================================================
# 6. API
# =============================================================================
@app.get("/", response_class=HTMLResponse)
def home():
    return render_html()


@app.post("/predict")
def predict_jeonse(payload: JeonseRequest):
    try:
        result = build_result(
            salary=payload.salary,
            housing=payload.housing,
            comm=payload.comm,
            culture=payload.culture,
            food=payload.food,
            debt=payload.debt,
            city_grade=payload.city_grade or "",
            housing_type=payload.housing_type or "",
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류가 발생했습니다: {e}")


@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(
    salary: float = Form(...),
    housing: float = Form(...),
    comm: float = Form(...),
    culture: float = Form(...),
    food: float = Form(...),
    debt: float = Form(...),
    city_grade: str = Form(""),
    housing_type: str = Form(""),
):
    try:
        if salary <= 0:
            return render_html(error_message="세후 월소득은 0보다 커야 합니다.")

        if min(housing, comm, culture, food, debt) < 0:
            return render_html(error_message="지출 항목과 월상환액은 음수일 수 없습니다.")

        result = build_result(
            salary=salary,
            housing=housing,
            comm=comm,
            culture=culture,
            food=food,
            debt=debt,
            city_grade=city_grade,
            housing_type=housing_type,
        )
        return render_html(result=result)

    except Exception as e:
        return render_html(error_message=f"오류가 발생했습니다: {e}")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="127.0.0.1", port=8000, reload=True)
