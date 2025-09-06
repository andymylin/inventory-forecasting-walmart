import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Walmart Inventory Forecasting", layout="wide")

st.title("🛒 Walmart Inventory Demand Forecasting (M5 Subset)")

# Load dataset from GitHub raw link
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/kaushiksuresh147/m5-forecasting-small-dataset/main/sales_train_validation_small.csv"
    df = pd.read_csv(url)
    
    # Reshape: Convert from wide to long format
    df = df.melt(id_vars=["item_id", "store_id"], var_name="day", value_name="sales")
    
    # Convert day labels (d_1, d_2, …) into sequential dates
    df["day_num"] = df["day"].str.replace("d_", "").astype(int)
    df["ds"] = pd.to_datetime("2011-01-29") + pd.to_timedelta(df["day_num"] - 1, unit="D")
    
    # Prophet needs columns: ds, y
    df = df.rename(columns={"sales": "y"})
    return df

df = load_data()

st.subheader("📊 Sample Data")
st.dataframe(df.head())

# Sidebar for product selection
product = st.sidebar.selectbox("Choose a Product", df["item_id"].unique())
store = st.sidebar.selectbox("Choose a Store", df["store_id"].unique())

# Filter data
product_df = df[(df["item_id"] == product) & (df["store_id"] == store)][["ds", "y"]]

st.subheader(f"📦 Sales History for {product} in {store}")
st.line_chart(product_df.set_index("ds"))

# Prophet Forecast
model = Prophet()
model.fit(product_df)

future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)

# Plot Forecast
st.subheader("📈 60-Day Forecast")
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Show Forecast Table
st.subheader("🔎 Forecast Table (Last 10 Days)")
st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10))
