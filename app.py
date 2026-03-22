import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Metric 17: Seasonal Demand Volatility Index", layout="wide")

st.title("Seasonal Demand Volatility Index")
st.write(
    "This tool measures how much demand fluctuates over time by product category or service type "
    "so a business can identify unstable demand patterns, seasonal peaks, and planning gaps."
)

DEFAULT_DATA_URL = "https://raw.githubusercontent.com/eblieskee-ctrl/BUSI3013Metric17/main/processed_orders_data.csv"

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# STUDENT NOTE: This helper loads either a user-uploaded CSV or the default GitHub dataset.
# Using both options makes the app reusable for graders while still allowing a fallback demo file.
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), "uploaded"
    return pd.read_csv(DEFAULT_DATA_URL), "github"


# STUDENT NOTE: This helper safely converts a pandas Series into numeric values.
# Invalid values become missing so they can be removed before metric calculations.
def to_numeric_safe(series):
    return pd.to_numeric(series, errors="coerce")


uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

try:
    raw_df, source_used = load_data(uploaded_file)
except Exception as e:
    st.error(f"Could not load the dataset. Error: {e}")
    st.stop()

if source_used == "github":
    st.info("Using default dataset from your GitHub repository.")

st.subheader("Data Preview")
st.dataframe(raw_df.head(10), use_container_width=True)

columns = raw_df.columns.tolist()

st.subheader("Column Mapping")

col1, col2 = st.columns(2)
with col1:
    date_col = st.selectbox("Select the date column", columns, index=0 if columns else None)
    category_col = st.selectbox("Select the product category / service type column", columns, index=1 if len(columns) > 1 else 0)
with col2:
    demand_col = st.selectbox("Select the quantity / units sold / order count column", columns, index=2 if len(columns) > 2 else 0)
    revenue_col = st.selectbox("Select the revenue / order value column", columns, index=3 if len(columns) > 3 else 0)

# STUDENT NOTE: Standardize mapped column names so the rest of the code uses one reusable logic path.
df = raw_df[[date_col, category_col, demand_col, revenue_col]].copy()
df.columns = ["order_date", "product_category", "demand_units", "order_value"]

# STUDENT NOTE: Convert types before analysis so time-based grouping and numeric calculations work correctly.
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df["demand_units"] = to_numeric_safe(df["demand_units"])
df["order_value"] = to_numeric_safe(df["order_value"])
df["product_category"] = df["product_category"].astype(str).str.strip()

# STUDENT NOTE: Drop invalid rows because the metric cannot be computed without valid date, category, demand, and revenue values.
df = df.dropna(subset=["order_date", "product_category", "demand_units", "order_value"])
df = df[df["product_category"] != ""]

if df.empty:
    st.error("No valid rows remain after cleaning. Check your column mapping and dataset values.")
    st.stop()

# STUDENT NOTE: Create calendar features for the seasonal heatmap and KPI calculations.
df["month_name"] = df["order_date"].dt.month_name()
df["day_of_week"] = df["order_date"].dt.day_name()
df["year_month"] = df["order_date"].dt.to_period("M").astype(str)

st.subheader("Interactive Filter")

available_categories = sorted(df["product_category"].unique())
selected_categories = st.multiselect(
    "Select categories to include",
    available_categories,
    default=available_categories
)

filtered_df = df[df["product_category"].isin(selected_categories)].copy()

if filtered_df.empty:
    st.warning("No data matches the selected categories.")
    st.stop()

# STUDENT NOTE: Aggregate demand to category-date level so volatility is measured over time rather than per transaction row.
daily_demand = (
    filtered_df.groupby(["product_category", "order_date"], as_index=False)
    .agg(
        demand_units=("demand_units", "sum"),
        order_value=("order_value", "sum")
    )
)

# STUDENT NOTE: Calculate mean, standard deviation, peak, and trough demand for each category.
volatility_df = (
    daily_demand.groupby("product_category")["demand_units"]
    .agg(["mean", "std", "max", "min"])
    .reset_index()
)

# STUDENT NOTE: Fill missing standard deviation values with zero for categories that appear only once.
volatility_df["std"] = volatility_df["std"].fillna(0)

# STUDENT NOTE: Compute the coefficient of variation as the Seasonal Demand Volatility Index.
# This compares variability relative to average demand, which makes categories comparable.
volatility_df["volatility_index"] = volatility_df["std"] / volatility_df["mean"]

# STUDENT NOTE: Compute the planning gap in units as the difference between the highest and lowest observed demand.
volatility_df["planning_gap_units"] = volatility_df["max"] - volatility_df["min"]

# STUDENT NOTE: Sort categories from highest to lowest volatility so the hardest-to-plan categories appear first.
volatility_df = volatility_df.sort_values("volatility_index", ascending=False).reset_index(drop=True)

# STUDENT NOTE: Build the month-by-day demand matrix required for the seasonal heatmap.
heatmap_source = (
    filtered_df.groupby(["month_name", "day_of_week"], as_index=False)["demand_units"]
    .sum()
)

heatmap_data = heatmap_source.pivot(
    index="month_name",
    columns="day_of_week",
    values="demand_units"
).fillna(0)

heatmap_data = heatmap_data.reindex(MONTH_ORDER)
existing_days = [d for d in DAY_ORDER if d in heatmap_data.columns]
heatmap_data = heatmap_data.reindex(columns=existing_days)

# STUDENT NOTE: Find the peak month across the currently filtered data.
monthly_demand = (
    filtered_df.groupby("month_name")["demand_units"]
    .sum()
    .reindex(MONTH_ORDER)
)

peak_month = monthly_demand.idxmax() if monthly_demand.notna().any() else "N/A"

# STUDENT NOTE: Extract headline KPI values for the dashboard cards.
highest_vol_category = volatility_df.iloc[0]["product_category"]
highest_vol_value = volatility_df.iloc[0]["volatility_index"]
planning_gap_units = int(volatility_df["planning_gap_units"].max())

st.subheader("Headline KPIs")
k1, k2, k3 = st.columns(3)
k1.metric("Highest-Volatility Category", highest_vol_category)
k2.metric("Peak Month", peak_month)
k3.metric("Demand Planning Gap (Units)", f"{planning_gap_units:,}")

st.subheader("Metric Results")

# STUDENT NOTE: Bar chart comparing volatility index across categories.
fig_bar = px.bar(
    volatility_df,
    x="product_category",
    y="volatility_index",
    title="Volatility Index by Category",
    labels={
        "product_category": "Category",
        "volatility_index": "Volatility Index (Std Dev / Mean)"
    }
)
st.plotly_chart(fig_bar, use_container_width=True)

# STUDENT NOTE: Heatmap of demand by month and day of week to reveal seasonal patterns.
fig_heatmap = px.imshow(
    heatmap_data,
    title="Seasonal Demand Heatmap (Month × Day of Week)",
    labels={
        "x": "Day of Week",
        "y": "Month",
        "color": "Demand Units"
    },
    aspect="auto"
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# STUDENT NOTE: Grouped bar chart comparing the peak and trough demand levels by category.
peak_trough_df = volatility_df[["product_category", "max", "min"]].copy()
peak_trough_long = peak_trough_df.melt(
    id_vars="product_category",
    value_vars=["max", "min"],
    var_name="demand_level",
    value_name="units"
)

fig_peak_trough = px.bar(
    peak_trough_long,
    x="product_category",
    y="units",
    color="demand_level",
    barmode="group",
    title="Peak-to-Trough Demand Comparison by Category",
    labels={
        "product_category": "Category",
        "units": "Demand Units",
        "demand_level": "Demand Level"
    }
)
st.plotly_chart(fig_peak_trough, use_container_width=True)

st.subheader("Volatility Table")
display_df = volatility_df.copy()
display_df["volatility_index"] = display_df["volatility_index"].round(3)
display_df["mean"] = display_df["mean"].round(2)
display_df["std"] = display_df["std"].round(2)
st.dataframe(display_df, use_container_width=True)

st.subheader("Interpretation")
st.info(
    f"The highest-volatility category in the current view is **{highest_vol_category}**, which means its demand "
    f"changes more sharply over time than the other selected categories. The peak month is **{peak_month}**, so that "
    f"period deserves closer attention for inventory, staffing, or capacity planning. The largest observed planning gap "
    f"is **{planning_gap_units:,} units**, showing the size of the swing between peak and trough demand. Categories "
    f"with high volatility should not be managed with static replenishment assumptions because they carry a higher risk "
    f"of stockouts during peak periods and overstock during slower periods."
)
