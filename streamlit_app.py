import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
try:
    import koreanize_matplotlib
except ImportError:
    st.warning("Matplotlib 한글 폰트 패키지(koreanize-matplotlib)가 설치되지 않았습니다. 차트 내 한글이 깨질 수 있습니다.")
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import platform

# ══════════════════════════════════════════════════════════════
#  LSTM 모델 구조 정의
# ══════════════════════════════════════════════════════════════
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ══════════════════════════════════════════════════════════════
#  페이지 설정
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="💰 Seoul Price Insight", layout="wide")
st.title("💰 Seoul Essential Goods Price Dashboard")
st.caption("서울시 생필품 농수축산물 가격 정보 및 AI 예측 (AI-Powered Consumer Price Index)")

# ══════════════════════════════════════════════════════════════
#  데이터 시뮬레이션 및 API 연동 준비
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_price_data(item_name, district):
    """
    서울시 물가 API를 시뮬레이션하거나 실제 데이터를 로드합니다.
    """
    # 실제로는 서울 열린데이터 광장 API를 연동할 수 있으나, 현재는 시뮬레이션 데이터를 생성합니다.
    np.random.seed(hash(item_name + district) % 123456789)
    n_days = 365
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days)
    
    # 기본 가격대 설정 (단위: 원)
    base_prices = {
        "배추": 5500, "무": 2800, "상추": 1800, "오이": 1500, "양파": 3500, "고추": 2500,
        "삼겹살": 29000, "쇠고기": 125000, "닭고기": 6500, "달걀": 7200, "돼지갈비": 18000,
        "고등어": 4800, "오징어": 8500, "명태": 5500, "갈치": 12000, "김": 9000
    }
    base = base_prices.get(item_name, 3000)
    
    # 추세 및 계절성 생성
    trend = np.linspace(0, base * 0.1, n_days)
    seasonality = (base * 0.2) * np.sin(np.arange(n_days) * 2 * np.pi / 365 + np.random.rand() * np.pi)
    noise = np.random.normal(0, base * 0.05, n_days)
    
    prices = base + trend + seasonality + noise
    df = pd.DataFrame({"date": dates, "price": prices})
    df.set_index("date", inplace=True)
    return df

# ══════════════════════════════════════════════════════════════
#  모델 학습 (인라인 학습)
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def train_selected_model(df, model_type="LSTM"):
    y = df["price"].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    if model_type == "Linear Regression":
        X = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        return model, None
    
    else: # LSTM
        window = 30
        X_seq, y_seq = [], []
        for i in range(len(y_scaled) - window):
            X_seq.append(y_scaled[i : i + window])
            y_seq.append(y_scaled[i + window])
        
        X_tensor = torch.FloatTensor(np.array(X_seq)).unsqueeze(-1)
        y_tensor = torch.FloatTensor(np.array(y_seq)).unsqueeze(-1)
        
        model = LSTMPredictor()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        for epoch in range(30):
            model.train()
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
        
        model.eval()
        return model, scaler

# ─── 사이드바 ───
st.sidebar.header("⚙️ Dashboard Settings")
category = st.sidebar.selectbox("Category", ["농산물", "축산물", "수산물"])
items = {
    "농산물": ["배추", "무", "상추", "오이", "양파", "고추"],
    "축산물": ["삼겹살", "쇠고기", "닭고기", "달걀", "돼지갈비"],
    "수산물": ["고등어", "오징어", "명태", "갈치", "김"]
}
item_name = st.sidebar.selectbox("Item", items[category])
district = st.sidebar.selectbox("District", ["종로구", "중구", "용산구", "송파구", "강남구"])
forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 7)
model_type = st.sidebar.radio("AI Model", ["Linear Regression", "LSTM"])

# 데이터 로드
df = load_price_data(item_name, district)

# ─── KPI ───
latest_price = df["price"].iloc[-1]
prev_price = df["price"].iloc[-7] # 1주일 전 비교
change = latest_price - prev_price

c1, c2, c3 = st.columns(3)
c1.metric(f"Latest {item_name} Price", f"{int(latest_price):,}원")
c2.metric("Weekly Change", f"{int(change):+,}원", delta_color="inverse")
c3.metric("District", district)

st.divider()

# ─── 탭 구성 ───
tab1, tab2, tab3 = st.tabs(["Price Trend", "AI Forecast", "Raw Data"])

with tab1:
    st.subheader(f"📈 {district} {item_name} 가격 변동 추이")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["price"], color="#e74c3c", linewidth=2)
    ax.set_ylabel("Price (KRW)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab2:
    st.subheader(f"🤖 AI {model_type} 기반 가격 예측")
    
    with st.spinner("모델 학습 및 예측 중..."):
        model, scaler = train_selected_model(df, model_type)
        
        # 예측 시작
        if model_type == "Linear Regression":
            future_indices = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
            preds = model.predict(future_indices)
        else:
            last_window = scaler.transform(df["price"].values[-30:].reshape(-1, 1)).flatten()
            preds_scaled = []
            curr_seq = last_window.tolist()
            for _ in range(forecast_days):
                with torch.no_grad():
                    input_t = torch.FloatTensor(curr_seq).unsqueeze(0).unsqueeze(-1)
                    p = model(input_t).item()
                    preds_scaled.append(p)
                    curr_seq.append(p)
                    curr_seq.pop(0)
            preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        
        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index[-60:], df["price"].values[-60:], label="Recent Actual", color="black", alpha=0.6)
        ax.plot(future_dates, preds, label="AI Forecast", color="#3498db", marker="o", linestyle="dashed")
        ax.set_ylabel("Price (KRW)")
        ax.set_title(f"{item_name} 미래 {forecast_days}일 가격 예측")
        ax.legend()
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)
        
        st.info(f"💡 AI 예측 결과: {forecast_days}일 후 예상 가격은 약 {int(preds[-1]):,}원입니다.")

        # ─── AI Insight Summary ───
        st.divider()
        st.subheader("📝 AI 분석 리포트")
        trend_val = preds[-1] - latest_price
        trend_percent = (trend_val / latest_price) * 100
        
        if trend_val > 0:
            st.warning(f"⚠️ **가격 상승 주의**: {item_name}의 가격이 향후 {forecast_days}일간 약 {trend_percent:.1f}% 상승할 것으로 예측됩니다. 구매 계획에 참고하세요.")
        else:
            st.success(f"✅ **가격 하락 전망**: {item_name}의 가격이 향후 {forecast_days}일간 약 {abs(trend_percent):.1f}% 하락할 것으로 보입니다. 조금 더 기다려보시는 것은 어떨까요?")

with tab3:
    st.subheader("📋 데이터 상세 정보")
    st.dataframe(df.sort_index(ascending=False), use_container_width=True)
    st.download_button("Download CSV", df.to_csv(), "price_data.csv", "text/csv")
