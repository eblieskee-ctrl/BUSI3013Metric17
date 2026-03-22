import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Seasonal Demand Volatility Tool", layout="wide")

st.title("📊 Seasonal Demand Volatility Index (Metric 17)")

# =========================
# DATA LOADING
# =========================
uploaded_file = st.file_uploader("Upload your dataset", type="csv")

def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # 🔥 YOUR GITHUB RAW LINK (UPDATE IF NEEDED)
        url = "https://raw.githubusercontent.com/eblieskee-ctrl/BUSI3013Metric17/main/small_dataset.csv"
        df = pd.read_csv(url)
        st.info("Using default dataset from GitHub")
    return df

try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Could not load the dataset. Error: {e}")
    st.stop()

# =========================
# COLUMN VALIDATION
# =========================
required_cols = ["order_date", "product_category", "order_count"]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# =========================
# DATA PREP
# =========================
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df = df.dropna(subset=["order_date", "product_category", "order_count"])

df["month"] = df["order_date"].dt.month_name()
df["day"] = df["order_date"].dt.day_name()

# =========================
# FILTER (REQUIRED FOR MARKS)
# =========================
category_filter = st.selectbox(
    "Filter by Category",
    ["All"] + list(df["product_category"].unique())
)

if category_filter != "All":
    df = df[df["product_category"] == category_filter]

# =========================
# AGGREGATION
# =========================
daily = (
    df.groupby(["product_category", "order_date"])["order_count"]
    .sum()
    .reset_index()
)

# =========================
# VOLATILITY INDEX
# =========================
volatility = (
    daily.groupby("product_category")["order_count"]
    .agg(["mean", "std", "max", "min"])
    .reset_index()
)

volatility["volatility_index"] = volatility["std"] / volatility["mean"]
volatility["planning_gap"] = volatility["max"] - volatility["min"]

# =========================
# KPI CARDS (REQUIRED)
# =========================
st.subheader("📌 Key Metrics")

col1, col2, col3 = st.columns(3)

highest_vol = volatility.loc[volatility["volatility_index"].idxmax()]
peak_month = df.groupby("month")["order_count"].sum().idxmax()
largest_gap = int(volatility["planning_gap"].max())

col1.metric("Highest Volatility Category", highest_vol["product_category"])
col2.metric("Peak Month", peak_month)
col3.metric("Planning Gap (Units)", largest_gap)

# =========================
# CHART 1: VOLATILITY BAR
# =========================
st.subheader("📊 Volatility by Category")

fig1 = px.bar(
    volatility,
    x="product_category",
    y="volatility_index",
    title="Volatility Index by Category"
)

st.plotly_chart(fig1, use_container_width=True)

# =========================
# CHART 2: HEATMAP
# =========================
st.subheader("🔥 Seasonal Demand Heatmap")

heatmap_data = df.pivot_table(
    values="order_count",
    index="month",
    columns="day",
    aggfunc="sum"
).fillna(0)

fig2 = px.imshow(heatmap_data, title="Month vs Day Demand")

st.plotly_chart(fig2, use_container_width=True)

# =========================
# CHART 3: PEAK VS TROUGH
# =========================
st.subheader("📉 Peak vs Trough Demand")

fig3 = px.bar(
    volatility,
    x="product_category",
    y=["max", "min"],
    barmode="group",
    title="Peak vs Trough Demand"
)

st.plotly_chart(fig3, use_container_width=True)

# =========================
# INTERPRETATION (FOR MARKS)
# =========================
st.subheader("💡 Business Insight")

st.info(
    "Categories with high volatility are harder to forecast and require flexible inventory planning. "
    "Low-volatility categories are stable and easier to manage. Businesses should focus on improving "
    "forecasting and buffer strategies for highly volatile categories."
)
