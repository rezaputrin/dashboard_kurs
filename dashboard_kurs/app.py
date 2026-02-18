import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dashboard Prediksi USD/IDR", layout="wide")

st.title("Dashboard Prediksi Kurs USD/IDR")
st.caption("Perbandingan ARIMA vs Random Forest (Tuned) | Data bulanan 2015–2025")

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def to_datetime(df, col="date"):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])
    return df

# === Load data dari folder data/ ===
df = load_csv("data/dataset_final_kurs_midrate_m2_policyrate_monthly_2015_2025.csv")
pred = load_csv("data/predictions_test.csv")
met = load_csv("data/metrics.csv")
try:
    fi = load_csv("data/rf_feature_importance.csv")
except:
    fi = None

df = to_datetime(df, "date")
pred = to_datetime(pred, "date")

# === Sidebar ===
st.sidebar.header("Pengaturan")

available_pred_cols = [c for c in pred.columns if c.lower().startswith("pred")]
default_models = []
if "pred_arima" in pred.columns: default_models.append("pred_arima")
if "pred_rf_tuned" in pred.columns: default_models.append("pred_rf_tuned")
if not default_models and available_pred_cols:
    default_models = available_pred_cols[:1]

models_to_show = st.sidebar.multiselect(
    "Model prediksi yang ditampilkan",
    options=available_pred_cols,
    default=default_models
)

# filter tanggal historis
min_date = df["date"].min()
max_date = df["date"].max()
date_range = st.sidebar.slider(
    "Rentang tanggal historis",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime())
)
df_view = df[(df["date"] >= pd.to_datetime(date_range[0])) & (df["date"] <= pd.to_datetime(date_range[1]))].copy()

# === Section 1: Metrik ===
st.subheader("Ringkasan Evaluasi Model")

colA, colB = st.columns([2, 3])
with colA:
    st.write("**Tabel Metrik**")
    st.dataframe(met, use_container_width=True)

with colB:
    if "MAPE" in met.columns:
        fig_mape = px.bar(met, x="Model", y="MAPE",
                          title="Perbandingan MAPE (lebih kecil lebih baik)",
                          text="MAPE")
        fig_mape.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig_mape, use_container_width=True)
    else:
        st.info("Kolom 'MAPE' tidak ditemukan di metrics.csv")

st.divider()

# === Section 2: Tren historis ===
st.subheader("Tren Historis USD/IDR")

if "mid_rate_monthly_mean" not in df_view.columns:
    st.error("Kolom 'mid_rate_monthly_mean' tidak ditemukan di dataset historis.")
    st.stop()

fig_hist = px.line(df_view, x="date", y="mid_rate_monthly_mean", title="USD/IDR Bulanan (Mid Rate)")
st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# === Section 3: Aktual vs Prediksi ===
st.subheader("Aktual vs Prediksi (Periode Uji)")

need_cols = ["date", "actual"]
for c in need_cols:
    if c not in pred.columns:
        st.error(f"Kolom '{c}' tidak ditemukan di predictions_test.csv")
        st.stop()

plot_cols = ["actual"] + models_to_show
if len(plot_cols) == 1:
    st.warning("Pilih minimal 1 kolom prediksi (pred_...) di sidebar.")
else:
    fig_pred = px.line(pred, x="date", y=plot_cols, title="Aktual vs Prediksi")
    st.plotly_chart(fig_pred, use_container_width=True)

with st.expander("Lihat tabel prediksi (data uji)"):
    st.dataframe(pred.sort_values("date"), use_container_width=True)

st.divider()

# === Section 4: Feature Importance (opsional) ===
st.subheader("Feature Importance Random Forest (Opsional)")

if fi is None:
    st.info("File rf_feature_importance.csv belum ada / tidak dimuat.")
else:
    if not {"feature", "importance"}.issubset(set(fi.columns)):
        st.warning("Format rf_feature_importance.csv harus punya kolom: feature, importance")
    else:
        top_n = st.slider("Tampilkan Top-N fitur", min_value=5, max_value=25, value=12)
        fi_sorted = fi.sort_values("importance", ascending=False).head(top_n)
        fig_fi = px.bar(fi_sorted, x="importance", y="feature", orientation="h",
                        title=f"Top-{top_n} Feature Importance")
        st.plotly_chart(fig_fi, use_container_width=True)

st.divider()

# === Section 5: Download ===
st.subheader("Unduh Output")

pred_csv = pred.to_csv(index=False).encode("utf-8")
met_csv = met.to_csv(index=False).encode("utf-8")

st.download_button("Download predictions_test.csv", pred_csv, file_name="predictions_test.csv", mime="text/csv")
st.download_button("Download metrics.csv", met_csv, file_name="metrics.csv", mime="text/csv")

if fi is not None:
    fi_csv = fi.to_csv(index=False).encode("utf-8")
    st.download_button("Download rf_feature_importance.csv", fi_csv, file_name="rf_feature_importance.csv", mime="text/csv")