"""Electricity Consumption Forecasting Using Machine Learning — Streamlit App.

A modern, dark-navy and electric-blue interactive Machine Learning dashboard
using the UCI Appliances Energy Prediction dataset (Dataset ID: 374).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.model import (
    CORE_FEATURES,
    DATASET_ID,
    DAY_NAMES,
    FEATURE_DESCRIPTIONS,
    FEATURE_SETS,
    MODEL_NAMES,
    TARGET_COL,
    UCI_DATASET_PAGE,
    DatasetMetadata,
    SplitData,
    TrainingBundle,
    compute_training_feature_stats,
    load_appliances_dataset,
    predict_consumption,
    prepare_chronological_split,
    train_and_evaluate_models,
)

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Electricity Consumption Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY_BG = "#060D1A"
CARD_BG = "#0D1B2E"
CARD_BG_ALT = "#112240"
ELECTRIC_BLUE = "#00B4FF"
CYAN_ACCENT = "#22D3EE"
EMERALD_ACCENT = "#10B981"
AMBER_ACCENT = "#F59E0B"
CORAL_ACCENT = "#F43F5E"
TEXT_PRIMARY = "#F1F5F9"
TEXT_MUTED = "#94A3B8"
GRID_COLOR = "#1E3A5F"

GRAPHS_DIR = Path(__file__).resolve().parent / "graphs"


def inject_custom_css() -> None:
    """Inject a polished dark navy and electric blue dashboard stylesheet."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: #F1F5F9;
        }

        .stApp {
            background: radial-gradient(circle at 12% 8%, rgba(0, 180, 255, 0.10), transparent 35%),
                        radial-gradient(circle at 88% 15%, rgba(34, 211, 238, 0.08), transparent 30%),
                        #060D1A;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #091326 0%, #070F1F 100%);
            border-right: 1px solid rgba(0, 180, 255, 0.18);
        }

        section[data-testid="stSidebar"] .stRadio > label {
            font-weight: 600;
            color: #94A3B8;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            font-size: 0.75rem;
        }

        .hero-card {
            background: linear-gradient(135deg, rgba(13, 27, 46, 0.96) 0%, rgba(17, 34, 64, 0.92) 100%);
            border: 1px solid rgba(0, 180, 255, 0.30);
            border-radius: 16px;
            padding: 1.75rem 2rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(56, 189, 248, 0.15);
        }

        .hero-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 0.85rem;
        }

        .tech-pill {
            display: inline-block;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            background: rgba(0, 180, 255, 0.14);
            color: #38BDF8;
            border: 1px solid rgba(0, 180, 255, 0.35);
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.2;
            margin: 0 0 0.5rem 0;
            background: linear-gradient(90deg, #F8FAFC 0%, #38BDF8 65%, #22D3EE 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-subtitle {
            color: #CBD5E1;
            font-size: 1rem;
            line-height: 1.55;
            margin: 0;
            max-width: 880px;
        }

        .kpi-card {
            background: linear-gradient(160deg, #0D1B2E 0%, #0A1526 100%);
            border: 1px solid rgba(0, 180, 255, 0.22);
            border-radius: 14px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
            height: 100%;
        }

        .kpi-label {
            color: #94A3B8;
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.35rem;
        }

        .kpi-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.55rem;
            font-weight: 700;
            color: #F8FAFC;
            line-height: 1.25;
        }

        .kpi-sub {
            color: #38BDF8;
            font-size: 0.80rem;
            margin-top: 0.35rem;
            font-weight: 500;
        }

        .section-card {
            background: rgba(13, 27, 46, 0.88);
            border: 1px solid rgba(56, 189, 248, 0.18);
            border-radius: 14px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 1rem;
        }

        .section-heading {
            font-size: 1.12rem;
            font-weight: 700;
            color: #E2E8F0;
            margin: 0 0 0.45rem 0;
        }

        .winner-banner {
            background: linear-gradient(135deg, rgba(0, 180, 255, 0.16) 0%, rgba(16, 185, 129, 0.14) 100%);
            border: 1px solid rgba(34, 211, 238, 0.45);
            border-radius: 14px;
            padding: 1.2rem 1.4rem;
            margin: 1rem 0;
        }

        .prediction-result-card {
            background: linear-gradient(135deg, rgba(0, 180, 255, 0.20) 0%, rgba(17, 34, 64, 0.95) 100%);
            border: 1px solid rgba(0, 180, 255, 0.55);
            border-radius: 16px;
            padding: 1.6rem 1.8rem;
            box-shadow: 0 0 28px rgba(0, 180, 255, 0.16);
            margin-top: 0.75rem;
        }

        .prediction-number {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.6rem;
            font-weight: 800;
            color: #38BDF8;
            line-height: 1.15;
            margin: 0.25rem 0;
        }

        .stButton > button, .stFormSubmitButton > button {
            background: linear-gradient(90deg, #0284C7 0%, #00B4FF 100%);
            color: #FFFFFF;
            border: 1px solid rgba(56, 189, 248, 0.6);
            border-radius: 10px;
            font-weight: 600;
            padding: 0.55rem 1.2rem;
            transition: all 0.18s ease;
        }

        .stButton > button:hover, .stFormSubmitButton > button:hover {
            background: linear-gradient(90deg, #0369A1 0%, #38BDF8 100%);
            border-color: #7DD3FC;
            transform: translateY(-1px);
        }

        div[data-testid="stMetric"] {
            background: rgba(13, 27, 46, 0.85);
            border: 1px solid rgba(0, 180, 255, 0.20);
            border-radius: 12px;
            padding: 0.85rem 1rem;
        }

        .sidebar-brand {
            padding: 0.75rem 0.9rem;
            background: linear-gradient(135deg, rgba(0, 180, 255, 0.16), rgba(17, 34, 64, 0.9));
            border: 1px solid rgba(0, 180, 255, 0.30);
            border-radius: 12px;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_matplotlib_axes(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """Apply consistent dark-navy and electric-blue chart styling."""
    ax.set_facecolor("#091426")
    ax.figure.set_facecolor(CARD_BG)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=12.5, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color="#CBD5E1", fontsize=10, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color="#CBD5E1", fontsize=10, labelpad=8)
    ax.tick_params(colors="#CBD5E1", labelsize=9)
    ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color("#1E3A5F")


def render_kpi_card(label: str, value: str, subtitle: str) -> None:
    """Render a themed metric KPI card."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached Data & Model Resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading UCI Appliances Energy dataset (ID: 374)...")
def get_dataset_cached() -> tuple[pd.DataFrame, DatasetMetadata]:
    """Load and cache the chronologically sorted dataset and metadata."""
    return load_appliances_dataset()


@st.cache_resource(show_spinner="Training ML models on chronological split...")
def get_trained_bundle_cached(
    feature_set_name: str,
    train_ratio: float,
    dt_max_depth: int,
    dt_min_samples_leaf: int,
    gb_n_estimators: int,
    gb_learning_rate: float,
    gb_max_depth: int,
    gb_subsample: float,
) -> tuple[SplitData, TrainingBundle]:
    """Split chronologically and train/evaluate all three regression models."""
    df, _ = get_dataset_cached()
    feature_cols = FEATURE_SETS[feature_set_name]
    split_data = prepare_chronological_split(df, feature_cols=feature_cols, train_ratio=train_ratio)
    bundle = train_and_evaluate_models(
        split_data=split_data,
        dt_max_depth=dt_max_depth,
        dt_min_samples_leaf=dt_min_samples_leaf,
        gb_n_estimators=gb_n_estimators,
        gb_learning_rate=gb_learning_rate,
        gb_max_depth=gb_max_depth,
        gb_subsample=gb_subsample,
    )
    return split_data, bundle


# ---------------------------------------------------------------------------
# Page Renderers
# ---------------------------------------------------------------------------
def render_home_page(
    df: pd.DataFrame,
    meta: DatasetMetadata,
    split_data: SplitData,
    bundle: TrainingBundle,
) -> None:
    """Page 1: Home Dashboard."""
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge-row">
                <span class="tech-pill">UCI Dataset ID: 374</span>
                <span class="tech-pill">Chronological Split · Zero Data Leakage</span>
                <span class="tech-pill">Linear · Decision Tree · Gradient Boosting</span>
            </div>
            <h1 class="hero-title">⚡ Electricity Consumption Forecasting Using Machine Learning</h1>
            <p class="hero-subtitle">
                Interactive analytics and forecasting platform built on real 10-minute household energy measurements.
                Explore sensor patterns, evaluate three regression models under strict time-ordered validation,
                and generate real-time appliance consumption forecasts in Watt-hours (Wh).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    best_row = bundle.metrics_df.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card(
            "Total Records",
            f"{meta.total_records:,}",
            f"10-min intervals · {meta.duplicates_removed} duplicates",
        )
    with col2:
        render_kpi_card(
            "Dataset Features",
            f"{meta.raw_feature_count}",
            f"{len(bundle.feature_cols)} active features in model",
        )
    with col3:
        render_kpi_card(
            "Prediction Target",
            f"{meta.target_col} (Wh)",
            f"Median {df[TARGET_COL].median():.0f} Wh · Mean {df[TARGET_COL].mean():.1f} Wh",
        )
    with col4:
        render_kpi_card(
            "Best Model (RMSE / R²)",
            f"{best_row['Model'].replace(' Regression', '')}",
            f"MAE {best_row['MAE (Wh)']:.2f} Wh · R² {best_row['R² Score']:.4f}",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.1, 1.3], gap="large")
    with left_col:
        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-heading">📋 Dataset & Split Overview</div>
                <p style="color:#CBD5E1; font-size:0.93rem; line-height:1.6;">
                    <strong>Source:</strong> {meta.source_used}<br>
                    <strong>Observation Span:</strong> {meta.date_start:%Y-%m-%d %H:%M} to {meta.date_end:%Y-%m-%d %H:%M}<br>
                    <strong>Missing Values (Raw):</strong> {meta.total_missing} across {meta.raw_column_count} columns<br>
                    <strong>Chronological Training Window ({split_data.train_ratio:.0%}):</strong>
                    {len(split_data.X_train):,} rows ({split_data.train_start:%Y-%m-%d} → {split_data.train_end:%Y-%m-%d})<br>
                    <strong>Chronological Testing Window ({1 - split_data.train_ratio:.0%}):</strong>
                    {len(split_data.X_test):,} rows ({split_data.test_start:%Y-%m-%d} → {split_data.test_end:%Y-%m-%d})
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-card">
                <div class="section-heading">🛡️ Anti-Leakage Architecture</div>
                <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.6; margin-bottom:0;">
                    • Records are sorted strictly by timestamp before splitting with <code>shuffle=False</code>.<br>
                    • Median imputation and feature scaling are encapsulated inside <code>scikit-learn</code> Pipelines fit exclusively on <code>X_train</code>.<br>
                    • Models are evaluated strictly on unseen future test observations using <strong>MAE</strong>, <strong>RMSE</strong>, and <strong>R²</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown("#### 🏆 Model Performance Summary")
        display_df = bundle.metrics_df.copy()
        display_df["MAE (Wh)"] = display_df["MAE (Wh)"].map(lambda v: f"{v:.2f}")
        display_df["RMSE (Wh)"] = display_df["RMSE (Wh)"].map(lambda v: f"{v:.2f}")
        display_df["R² Score"] = display_df["R² Score"].map(lambda v: f"{v:.4f}")
        display_df["Train Time (s)"] = display_df["Train Time (s)"].map(lambda v: f"{v:.3f}s")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        models_short = [m.replace(" Regression", "") for m in bundle.metrics_df["Model"]]
        x = np.arange(len(models_short))
        width = 0.35
        ax.bar(x - width / 2, bundle.metrics_df["MAE (Wh)"], width, label="MAE (Wh)", color=ELECTRIC_BLUE)
        ax.bar(x + width / 2, bundle.metrics_df["RMSE (Wh)"], width, label="RMSE (Wh)", color=CYAN_ACCENT)
        ax.set_xticks(x)
        ax.set_xticklabels(models_short)
        style_matplotlib_axes(ax, "Error Comparison on Chronological Test Set (Lower is Better)", ylabel="Error (Wh)")
        ax.legend(facecolor=CARD_BG_ALT, edgecolor=GRID_COLOR, labelcolor=TEXT_PRIMARY)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def render_dataset_explorer_page(df: pd.DataFrame, meta: DatasetMetadata) -> None:
    """Page 2: Dataset Explorer."""
    st.markdown("## 🗂️ Dataset Explorer")
    st.caption(
        f"Inspecting the UCI Appliances Energy Prediction dataset (ID: {DATASET_ID}) loaded from {meta.source_used}."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Rows", f"{df.shape[0]:,}", "After deduplication & sorting")
    with c2:
        render_kpi_card("Columns", f"{df.shape[1]}", f"{meta.raw_column_count} raw + 5 derived time cols")
    with c3:
        render_kpi_card("Missing Values", f"{meta.total_missing}", "Across all raw columns")
    with c4:
        mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        render_kpi_card("Memory Footprint", f"{mem_mb:.2f} MB", "In-memory DataFrame")

    st.markdown("<br>", unsafe_allow_html=True)

    tab_preview, tab_missing, tab_stats = st.tabs(
        ["🔍 Interactive Data Preview", "🩺 Missing Values & Schema", "📊 Statistical Summary"]
    )

    with tab_preview:
        ctrl1, ctrl2, ctrl3 = st.columns([1.1, 1.1, 2.0])
        with ctrl1:
            preview_mode = st.selectbox(
                "View Mode",
                ["First N Rows", "Last N Rows", "Random Sample"],
                index=0,
            )
        with ctrl2:
            n_rows = st.slider("Number of Rows", min_value=5, max_value=200, value=25, step=5)
        with ctrl3:
            default_cols = ["date", "Appliances", "lights", "T1", "RH_1", "T2", "RH_2", "T_out", "RH_out", "Windspeed", "hour"]
            selected_cols = st.multiselect(
                "Columns to Display",
                options=list(df.columns),
                default=[c for c in default_cols if c in df.columns],
            )

        if not selected_cols:
            st.warning("Select at least one column to display the preview table.")
        else:
            if preview_mode == "First N Rows":
                preview_df = df[selected_cols].head(n_rows)
            elif preview_mode == "Last N Rows":
                preview_df = df[selected_cols].tail(n_rows)
            else:
                preview_df = df[selected_cols].sample(n=min(n_rows, len(df)), random_state=42)

            st.dataframe(preview_df, use_container_width=True, height=400)
            st.download_button(
                label="⬇️ Download Preview CSV",
                data=preview_df.to_csv(index=False).encode("utf-8"),
                file_name="appliances_preview.csv",
                mime="text/csv",
            )

    with tab_missing:
        schema_rows = []
        for col in df.columns:
            raw_missing = meta.missing_by_col.get(col, int(df[col].isna().sum()))
            schema_rows.append(
                {
                    "Column": col,
                    "Description": FEATURE_DESCRIPTIONS.get(col, "Derived temporal feature"),
                    "Data Type": str(df[col].dtype),
                    "Non-Null Count": int(df[col].notna().sum()),
                    "Missing Count": raw_missing,
                    "Missing (%)": round(100.0 * raw_missing / max(1, meta.total_records), 4),
                }
            )
        schema_df = pd.DataFrame(schema_rows)
        st.dataframe(schema_df, use_container_width=True, hide_index=True, height=440)

    with tab_stats:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        stat_cols = st.multiselect(
            "Select Numeric Columns for Summary",
            options=numeric_cols,
            default=[c for c in [TARGET_COL] + CORE_FEATURES if c in numeric_cols],
        )
        if stat_cols:
            summary_df = df[stat_cols].describe().T
            summary_df["median"] = df[stat_cols].median()
            summary_df = summary_df[["count", "mean", "std", "min", "25%", "median", "75%", "max"]]
            st.dataframe(summary_df.map(lambda v: f"{v:.3f}"), use_container_width=True, height=420)
            st.download_button(
                label="⬇️ Download Statistical Summary CSV",
                data=summary_df.round(4).to_csv().encode("utf-8"),
                file_name="appliances_statistical_summary.csv",
                mime="text/csv",
            )
        else:
            st.info("Select one or more numeric columns to view summary statistics.")


def render_data_visualization_page(
    df: pd.DataFrame,
    split_data: SplitData,
    bundle: TrainingBundle,
) -> None:
    """Page 3: Data Visualization."""
    st.markdown("## 📊 Data Visualization")
    st.caption(
        "Interactive exploratory and diagnostic charts generated dynamically from the real UCI Appliances Energy dataset."
    )

    # 1. Electricity Consumption Distribution
    st.markdown("### 1. Electricity Consumption Distribution")
    d_col1, d_col2 = st.columns([1.2, 1.2])
    with d_col1:
        bins = st.slider("Histogram Bins", min_value=15, max_value=80, value=35, step=5)
    with d_col2:
        cap_option = st.selectbox(
            "Display Range",
            ["Full Range (10 – 1080 Wh)", "Up to 99th Percentile", "Up to 95th Percentile"],
            index=0,
        )

    appliances_series = df[TARGET_COL].dropna()
    if cap_option == "Up to 99th Percentile":
        appliances_plot = appliances_series[appliances_series <= appliances_series.quantile(0.99)]
    elif cap_option == "Up to 95th Percentile":
        appliances_plot = appliances_series[appliances_series <= appliances_series.quantile(0.95)]
    else:
        appliances_plot = appliances_series

    fig1, ax1 = plt.subplots(figsize=(10, 4.0))
    sns.histplot(appliances_plot, bins=bins, kde=True, color=ELECTRIC_BLUE, edgecolor="#060D1A", alpha=0.82, ax=ax1)
    mean_val = float(appliances_series.mean())
    median_val = float(appliances_series.median())
    ax1.axvline(mean_val, color=AMBER_ACCENT, linestyle="--", linewidth=1.8, label=f"Mean: {mean_val:.1f} Wh")
    ax1.axvline(median_val, color=EMERALD_ACCENT, linestyle="-.", linewidth=1.8, label=f"Median: {median_val:.1f} Wh")
    style_matplotlib_axes(
        ax1,
        "Appliance Electricity Consumption Distribution",
        xlabel="Appliances Energy Consumption (Wh)",
        ylabel="Frequency (10-min intervals)",
    )
    ax1.legend(facecolor=CARD_BG_ALT, edgecolor=GRID_COLOR, labelcolor=TEXT_PRIMARY)
    fig1.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    st.divider()

    # 2. Hourly Electricity Consumption
    st.markdown("### 2. Hourly Electricity Consumption Pattern")
    split_weekend = st.checkbox("Compare Weekday vs Weekend Hourly Profiles", value=True)

    fig2, ax2 = plt.subplots(figsize=(10, 4.2))
    if split_weekend:
        weekday_hourly = df[df["is_weekend"] == 0].groupby("hour")[TARGET_COL].mean()
        weekend_hourly = df[df["is_weekend"] == 1].groupby("hour")[TARGET_COL].mean()
        overall_hourly = df.groupby("hour")[TARGET_COL].mean()
        ax2.plot(overall_hourly.index, overall_hourly.values, marker="o", linewidth=2.4, color=ELECTRIC_BLUE, label="All Days (Mean)")
        ax2.plot(weekday_hourly.index, weekday_hourly.values, marker="s", linewidth=1.8, linestyle="--", color=CYAN_ACCENT, label="Weekdays (Mean)")
        ax2.plot(weekend_hourly.index, weekend_hourly.values, marker="^", linewidth=1.8, linestyle=":", color=AMBER_ACCENT, label="Weekends (Mean)")
    else:
        hourly_stats = df.groupby("hour")[TARGET_COL].agg(
            mean="mean",
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
        )
        ax2.fill_between(
            hourly_stats.index,
            hourly_stats["q25"],
            hourly_stats["q75"],
            color=ELECTRIC_BLUE,
            alpha=0.18,
            label="IQR (25th–75th pct)",
        )
        ax2.plot(hourly_stats.index, hourly_stats["mean"], marker="o", linewidth=2.4, color=ELECTRIC_BLUE, label="Mean (Wh)")
        ax2.plot(hourly_stats.index, hourly_stats["median"], marker="s", linewidth=2.0, color=EMERALD_ACCENT, label="Median (Wh)")

    ax2.set_xticks(range(24))
    style_matplotlib_axes(
        ax2,
        "Average Electricity Consumption by Hour of Day",
        xlabel="Hour of Day (00:00 – 23:00)",
        ylabel="Energy Consumption (Wh)",
    )
    ax2.legend(facecolor=CARD_BG_ALT, edgecolor=GRID_COLOR, labelcolor=TEXT_PRIMARY)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.divider()

    # 3. Correlation Heatmap
    st.markdown("### 3. Feature Correlation Heatmap")
    corr_groups = {
        "Core Prediction Features": [TARGET_COL] + CORE_FEATURES,
        "Indoor Room Temperatures (T1–T9)": [TARGET_COL, "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T_out"],
        "Humidity Sensors (RH_1–RH_9)": [TARGET_COL, "RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6", "RH_7", "RH_8", "RH_9", "RH_out"],
        "Weather & Temporal Drivers": [TARGET_COL, "lights", "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint", "hour", "dayofweek"],
    }
    selected_group = st.selectbox("Feature Group for Correlation Matrix", list(corr_groups.keys()), index=0)
    corr_cols = [c for c in corr_groups[selected_group] if c in df.columns]
    corr_matrix = df[corr_cols].corr()

    fig3, ax3 = plt.subplots(figsize=(10.5, 6.2))
    fig3.set_facecolor(CARD_BG)
    ax3.set_facecolor(CARD_BG)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0.0,
        linewidths=0.6,
        linecolor="#060D1A",
        annot_kws={"size": 8.5},
        cbar_kws={"shrink": 0.85},
        ax=ax3,
    )
    ax3.set_title(f"Correlation Heatmap — {selected_group}", color=TEXT_PRIMARY, fontsize=12.5, fontweight="bold", pad=12)
    ax3.tick_params(colors="#CBD5E1", labelsize=9)
    cbar = ax3.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(colors="#CBD5E1", labelsize=8.5)
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    st.divider()

    # 4. Actual vs Predicted Graph
    st.markdown("### 4. Actual vs Predicted Electricity Consumption (Chronological Test Set)")
    avp1, avp2, avp3 = st.columns([1.5, 1.0, 1.0])
    with avp1:
        models_to_plot = st.multiselect(
            "Models to Overlay",
            options=MODEL_NAMES,
            default=[bundle.best_model_name],
        )
    with avp2:
        max_win = min(500, len(bundle.y_test))
        min_win = min(40, max_win)
        window_size = st.slider(
            "Window Length (records)",
            min_value=min_win,
            max_value=max_win,
            value=min(144, max_win),
            step=12,
        )
    with avp3:
        max_start = max(0, len(bundle.y_test) - window_size)
        start_idx = st.slider("Test Window Start Index", min_value=0, max_value=max_start, value=0, step=10)

    end_idx = start_idx + window_size
    test_dates_slice = bundle.test_dates.iloc[start_idx:end_idx]
    y_actual_slice = bundle.y_test.iloc[start_idx:end_idx].values

    fig4, ax4 = plt.subplots(figsize=(10.5, 4.4))
    ax4.plot(
        test_dates_slice,
        y_actual_slice,
        color="#F8FAFC",
        linewidth=2.0,
        alpha=0.9,
        label="Actual Appliances (Wh)",
    )
    model_palette = {
        "Linear Regression": ELECTRIC_BLUE,
        "Decision Tree Regression": AMBER_ACCENT,
        "Gradient Boosting Regression": EMERALD_ACCENT,
    }
    for m_name in models_to_plot:
        y_pred_slice = bundle.predictions[m_name][start_idx:end_idx]
        ax4.plot(
            test_dates_slice,
            y_pred_slice,
            color=model_palette.get(m_name, CYAN_ACCENT),
            linewidth=2.0,
            linestyle="-",
            alpha=0.92,
            label=f"Predicted ({m_name})",
        )

    style_matplotlib_axes(
        ax4,
        f"Actual vs Predicted Consumption ({test_dates_slice.iloc[0]:%Y-%m-%d %H:%M} to {test_dates_slice.iloc[-1]:%Y-%m-%d %H:%M})",
        xlabel="Chronological Test Timestamp",
        ylabel="Energy Consumption (Wh)",
    )
    fig4.autofmt_xdate(rotation=25)
    ax4.legend(facecolor=CARD_BG_ALT, edgecolor=GRID_COLOR, labelcolor=TEXT_PRIMARY)
    fig4.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)

    with st.expander("📁 View Static Reference Graphs from Repository (`graphs/`)"):
        graph_files = [
            ("consumption_distribution.png", "Consumption Distribution"),
            ("hourly_consumption.png", "Hourly Consumption"),
            ("correlation_heatmap.png", "Correlation Heatmap"),
            ("actual_vs_predicted.png", "Actual vs Predicted"),
        ]
        g_cols = st.columns(2)
        for idx, (fname, caption) in enumerate(graph_files):
            fpath = GRAPHS_DIR / fname
            with g_cols[idx % 2]:
                if fpath.exists():
                    st.image(str(fpath), caption=caption, use_container_width=True)
                else:
                    st.info(f"Reference image `{fname}` not found in `graphs/`.")


def render_ml_models_page(df: pd.DataFrame) -> tuple[SplitData, TrainingBundle]:
    """Page 4: Machine Learning Models Training & Evaluation."""
    st.markdown("## 🤖 Machine Learning Models")
    st.caption(
        "Train, tune, and compare Linear Regression, Decision Tree Regression, and Gradient Boosting Regression "
        "on a strict chronological split with zero data leakage."
    )

    if "ml_config" not in st.session_state:
        st.session_state.ml_config = {
            "feature_set_name": "Core Prediction Features (10)",
            "train_ratio": 0.80,
            "dt_max_depth": 6,
            "dt_min_samples_leaf": 25,
            "gb_n_estimators": 120,
            "gb_learning_rate": 0.05,
            "gb_max_depth": 3,
            "gb_subsample": 0.80,
        }

    cfg = st.session_state.ml_config

    with st.form("ml_training_form"):
        c1, c2 = st.columns([1.3, 1.0])
        with c1:
            feature_set_name = st.selectbox(
                "Feature Set",
                options=list(FEATURE_SETS.keys()),
                index=list(FEATURE_SETS.keys()).index(cfg["feature_set_name"]),
                help="Core 10 features match the interactive Prediction System inputs 1-to-1; 27 clean sensor features exclude random rv1/rv2 noise and non-repeating calendar columns.",
            )
        with c2:
            train_ratio = st.slider(
                "Chronological Train Split Ratio",
                min_value=0.70,
                max_value=0.90,
                value=float(cfg["train_ratio"]),
                step=0.05,
            )

        with st.expander("⚙️ Model Hyperparameters (Decision Tree & Gradient Boosting)", expanded=False):
            hp1, hp2 = st.columns(2)
            with hp1:
                st.markdown("**Decision Tree Regression**")
                dt_max_depth = st.slider("Max Depth (DT)", 2, 20, int(cfg["dt_max_depth"]))
                dt_min_samples_leaf = st.slider("Min Samples per Leaf (DT)", 5, 100, int(cfg["dt_min_samples_leaf"]), step=5)
            with hp2:
                st.markdown("**Gradient Boosting Regression**")
                gb_n_estimators = st.slider("Number of Estimators (GB)", 50, 300, int(cfg["gb_n_estimators"]), step=10)
                gb_learning_rate = st.select_slider(
                    "Learning Rate (GB)",
                    options=[0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
                    value=float(cfg["gb_learning_rate"]),
                )
                gb_max_depth = st.slider("Tree Max Depth (GB)", 2, 6, int(cfg["gb_max_depth"]))
                gb_subsample = st.slider("Subsample Fraction (GB)", 0.6, 1.0, float(cfg["gb_subsample"]), step=0.1)

        train_clicked = st.form_submit_button("🚀 Train & Evaluate All 3 Models")

    if train_clicked:
        st.session_state.ml_config = {
            "feature_set_name": feature_set_name,
            "train_ratio": float(train_ratio),
            "dt_max_depth": int(dt_max_depth),
            "dt_min_samples_leaf": int(dt_min_samples_leaf),
            "gb_n_estimators": int(gb_n_estimators),
            "gb_learning_rate": float(gb_learning_rate),
            "gb_max_depth": int(gb_max_depth),
            "gb_subsample": float(gb_subsample),
        }
        st.rerun()

    split_data, bundle = get_trained_bundle_cached(
        feature_set_name=cfg["feature_set_name"],
        train_ratio=cfg["train_ratio"],
        dt_max_depth=cfg["dt_max_depth"],
        dt_min_samples_leaf=cfg["dt_min_samples_leaf"],
        gb_n_estimators=cfg["gb_n_estimators"],
        gb_learning_rate=cfg["gb_learning_rate"],
        gb_max_depth=cfg["gb_max_depth"],
        gb_subsample=cfg["gb_subsample"],
    )

    st.info(
        f"**Chronological Split Summary:** Train = `{len(split_data.X_train):,}` samples "
        f"({split_data.train_start:%Y-%m-%d %H:%M} → {split_data.train_end:%Y-%m-%d %H:%M}) | "
        f"Test = `{len(split_data.X_test):,}` future samples "
        f"({split_data.test_start:%Y-%m-%d %H:%M} → {split_data.test_end:%Y-%m-%d %H:%M}) | "
        f"Features = `{len(bundle.feature_cols)}`."
    )

    best_row = bundle.metrics_df.iloc[0]
    mae_note = ""
    if bundle.best_mae_model_name != bundle.best_model_name:
        mae_val = bundle.metrics_df.loc[
            bundle.metrics_df["Model"] == bundle.best_mae_model_name, "MAE (Wh)"
        ].iloc[0]
        mae_note = (
            f" Note: <strong>{bundle.best_mae_model_name}</strong> achieved the lowest MAE "
            f"(<code>{mae_val:.2f} Wh</code>), while <strong>{bundle.best_model_name}</strong> "
            f"achieved the best overall RMSE and R²."
        )

    st.markdown(
        f"""
        <div class="winner-banner">
            <div style="font-size:0.82rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#22D3EE;">
                🏆 Best-Performing Model (Ranked by Lowest RMSE & Highest R²)
            </div>
            <div style="font-size:1.45rem; font-weight:800; color:#F8FAFC; margin:0.25rem 0;">
                {best_row['Model']}
            </div>
            <div style="color:#E2E8F0; font-size:0.95rem;">
                MAE: <strong>{best_row['MAE (Wh)']:.2f} Wh</strong> &nbsp;|&nbsp;
                RMSE: <strong>{best_row['RMSE (Wh)']:.2f} Wh</strong> &nbsp;|&nbsp;
                R² Score: <strong>{best_row['R² Score']:.4f}</strong> &nbsp;|&nbsp;
                Train Time: <strong>{best_row['Train Time (s)']:.3f}s</strong>
                <br><span style="color:#CBD5E1; font-size:0.88rem;">{mae_note}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📊 Model Evaluation Metrics Comparison")
    formatted_metrics = bundle.metrics_df.copy()
    formatted_metrics["Rank"] = [f"#{i + 1}" for i in range(len(formatted_metrics))]
    formatted_metrics["MAE (Wh)"] = formatted_metrics["MAE (Wh)"].map(lambda v: f"{v:.3f}")
    formatted_metrics["RMSE (Wh)"] = formatted_metrics["RMSE (Wh)"].map(lambda v: f"{v:.3f}")
    formatted_metrics["R² Score"] = formatted_metrics["R² Score"].map(lambda v: f"{v:.4f}")
    formatted_metrics["Train Time (s)"] = formatted_metrics["Train Time (s)"].map(lambda v: f"{v:.3f}s")
    formatted_metrics = formatted_metrics[["Rank", "Model", "MAE (Wh)", "RMSE (Wh)", "R² Score", "Train Time (s)"]]
    st.dataframe(formatted_metrics, use_container_width=True, hide_index=True)

    # Bar charts for MAE, RMSE, and R²
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    fig.set_facecolor(CARD_BG)
    short_names = [m.replace(" Regression", "") for m in bundle.metrics_df["Model"]]
    colors = [
        EMERALD_ACCENT if m == bundle.best_model_name else ELECTRIC_BLUE
        for m in bundle.metrics_df["Model"]
    ]

    metrics_specs = [
        ("MAE (Wh)", "Mean Absolute Error (Lower = Better)"),
        ("RMSE (Wh)", "Root Mean Squared Error (Lower = Better)"),
        ("R² Score", "Coefficient of Determination (Higher = Better)"),
    ]
    for ax, (col_name, chart_title) in zip(axes, metrics_specs):
        vals = bundle.metrics_df[col_name].values
        bars = ax.bar(short_names, vals, color=colors, width=0.52, edgecolor="#060D1A")
        style_matplotlib_axes(ax, chart_title, ylabel=col_name)
        for bar, val in zip(bars, vals):
            fmt = f"{val:.3f}" if col_name == "R² Score" else f"{val:.1f}"
            y_pos = bar.get_height() + (0.02 * (max(abs(vals)) or 1.0)) if val >= 0 else bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                fmt,
                ha="center",
                va="bottom" if val >= 0 else "top",
                color=TEXT_PRIMARY,
                fontsize=8.5,
                fontweight="bold",
            )
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### 🔎 Feature Importance / Standardized Drivers")
    imp_model = st.selectbox("Inspect Feature Importance for Model", options=MODEL_NAMES, index=MODEL_NAMES.index(bundle.best_model_name))
    imp_df = bundle.feature_importances[imp_model].head(12).iloc[::-1]

    fig_imp, ax_imp = plt.subplots(figsize=(10, 4.2))
    ax_imp.barh(imp_df["Feature"], imp_df["Importance"], color=CYAN_ACCENT, edgecolor="#060D1A")
    style_matplotlib_axes(
        ax_imp,
        f"Top Feature Drivers — {imp_model}",
        xlabel="Normalized Importance" + (" (|Standardized Coef|)" if imp_model == "Linear Regression" else " (Gini / Gain)"),
    )
    fig_imp.tight_layout()
    st.pyplot(fig_imp, use_container_width=True)
    plt.close(fig_imp)

    return split_data, bundle


def render_prediction_system_page(
    split_data_active: SplitData,
    bundle_active: TrainingBundle,
) -> None:
    """Page 5: Real-Time Prediction System."""
    st.markdown("## ⚡ Prediction System")
    st.caption(
        "Enter indoor/outdoor temperature, humidity, lighting, wind speed, and schedule parameters "
        "to predict household appliance electricity consumption in Watt-hours (Wh)."
    )

    # Ensure Prediction System always uses a pipeline matching the 10 interactive inputs cleanly.
    if set(bundle_active.feature_cols) == set(CORE_FEATURES):
        pred_split, pred_bundle = split_data_active, bundle_active
    else:
        cfg = st.session_state.get(
            "ml_config",
            {
                "train_ratio": 0.80,
                "dt_max_depth": 6,
                "dt_min_samples_leaf": 25,
                "gb_n_estimators": 120,
                "gb_learning_rate": 0.05,
                "gb_max_depth": 3,
                "gb_subsample": 0.80,
            },
        )
        pred_split, pred_bundle = get_trained_bundle_cached(
            feature_set_name="Core Prediction Features (10)",
            train_ratio=float(cfg["train_ratio"]),
            dt_max_depth=int(cfg["dt_max_depth"]),
            dt_min_samples_leaf=int(cfg["dt_min_samples_leaf"]),
            gb_n_estimators=int(cfg["gb_n_estimators"]),
            gb_learning_rate=float(cfg["gb_learning_rate"]),
            gb_max_depth=int(cfg["gb_max_depth"]),
            gb_subsample=float(cfg["gb_subsample"]),
        )
        st.info(
            "Note: The Machine Learning Models page is currently set to the 27-feature sensor set. "
            "For live predictions from the 10 user inputs below, this page uses the models trained on "
            "the Core 10 Prediction Features."
        )

    stats = compute_training_feature_stats(pred_split.X_train)

    # Initialize widget-bound session state keys from training medians (no test leakage).
    median_defaults = {
        "inp_lights": round(stats["lights"]["median"], 1),
        "inp_T1": round(stats["T1"]["median"], 2),
        "inp_RH_1": round(stats["RH_1"]["median"], 2),
        "inp_T2": round(stats["T2"]["median"], 2),
        "inp_RH_2": round(stats["RH_2"]["median"], 2),
        "inp_T_out": round(stats["T_out"]["median"], 2),
        "inp_RH_out": round(stats["RH_out"]["median"], 2),
        "inp_Windspeed": round(stats["Windspeed"]["median"], 2),
        "inp_hour": 18,
        "inp_dayofweek": 2,
    }
    for key, val in median_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Quick scenario preset buttons
    st.markdown("**Quick Scenario Presets:**")
    p1, p2, p3, p4 = st.columns(4)
    if p1.button("⚖️ Training Median Baseline", use_container_width=True):
        st.session_state.update({**median_defaults, "inp_hour": 12, "inp_dayofweek": 2})
    if p2.button("🍳 Evening Cooking Peak", use_container_width=True):
        st.session_state.update(
            {
                "inp_lights": 30.0,
                "inp_T1": 23.0,
                "inp_RH_1": 45.0,
                "inp_T2": 21.5,
                "inp_RH_2": 42.0,
                "inp_T_out": 8.0,
                "inp_RH_out": 80.0,
                "inp_Windspeed": 4.5,
                "inp_hour": 18,
                "inp_dayofweek": 4,
            }
        )
    if p3.button("🌅 Morning Routine", use_container_width=True):
        st.session_state.update(
            {
                "inp_lights": 10.0,
                "inp_T1": 20.8,
                "inp_RH_1": 41.0,
                "inp_T2": 19.5,
                "inp_RH_2": 41.5,
                "inp_T_out": 5.5,
                "inp_RH_out": 85.0,
                "inp_Windspeed": 3.5,
                "inp_hour": 8,
                "inp_dayofweek": 0,
            }
        )
    if p4.button("🌙 Quiet Night", use_container_width=True):
        st.session_state.update(
            {
                "inp_lights": 0.0,
                "inp_T1": 20.0,
                "inp_RH_1": 38.5,
                "inp_T2": 18.8,
                "inp_RH_2": 39.5,
                "inp_T_out": 4.0,
                "inp_RH_out": 88.0,
                "inp_Windspeed": 2.0,
                "inp_hour": 2,
                "inp_dayofweek": 6,
            }
        )

    with st.form("prediction_form"):
        selected_model_name = st.selectbox(
            "Select Forecasting Model",
            options=MODEL_NAMES,
            index=MODEL_NAMES.index(pred_bundle.best_model_name),
            help=f"Defaults to best evaluated model ({pred_bundle.best_model_name}).",
        )

        col_a, col_b, col_c = st.columns(3, gap="medium")

        with col_a:
            st.markdown("##### 🌡️ Indoor Climate")
            t1 = st.number_input(
                "Kitchen Temperature — T1 (°C)",
                min_value=0.0,
                max_value=45.0,
                step=0.1,
                key="inp_T1",
            )
            rh1 = st.number_input(
                "Kitchen Humidity — RH_1 (%)",
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                key="inp_RH_1",
            )
            t2 = st.number_input(
                "Living Room Temperature — T2 (°C)",
                min_value=0.0,
                max_value=45.0,
                step=0.1,
                key="inp_T2",
            )
            rh2 = st.number_input(
                "Living Room Humidity — RH_2 (%)",
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                key="inp_RH_2",
            )

        with col_b:
            st.markdown("##### 🌤️ Outdoor Weather")
            t_out = st.number_input(
                "Outside Temperature — T_out (°C)",
                min_value=-25.0,
                max_value=45.0,
                step=0.5,
                key="inp_T_out",
            )
            rh_out = st.number_input(
                "Outside Humidity — RH_out (%)",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                key="inp_RH_out",
            )
            windspeed = st.number_input(
                "Wind Speed (m/s)",
                min_value=0.0,
                max_value=40.0,
                step=0.5,
                key="inp_Windspeed",
            )

        with col_c:
            st.markdown("##### 💡 Lighting & Schedule")
            lights = st.number_input(
                "Lighting Consumption — lights (Wh)",
                min_value=0.0,
                max_value=150.0,
                step=5.0,
                key="inp_lights",
            )
            hour = st.slider(
                "Time of Day (Hour)",
                min_value=0,
                max_value=23,
                format="%02d:00",
                key="inp_hour",
            )
            dayofweek = st.selectbox(
                "Day of Week",
                options=list(range(7)),
                format_func=lambda idx: DAY_NAMES[idx],
                key="inp_dayofweek",
            )

        submitted = st.form_submit_button("⚡ Predict Electricity Consumption", use_container_width=True)

    if submitted:
        user_inputs: dict[str, float | int] = {
            "lights": lights,
            "T1": t1,
            "RH_1": rh1,
            "T2": t2,
            "RH_2": rh2,
            "T_out": t_out,
            "RH_out": rh_out,
            "Windspeed": windspeed,
            "hour": hour,
            "dayofweek": dayofweek,
        }

        try:
            chosen_pipeline = pred_bundle.models[selected_model_name]
            raw_pred, pred_wh = predict_consumption(chosen_pipeline, pred_bundle.feature_cols, user_inputs)

            train_q33 = float(pred_split.y_train.quantile(0.33))
            train_q75 = float(pred_split.y_train.quantile(0.75))
            if pred_wh <= train_q33:
                load_tier = "Low / Baseline Load"
                tier_color = EMERALD_ACCENT
            elif pred_wh <= train_q75:
                load_tier = "Moderate Load"
                tier_color = ELECTRIC_BLUE
            else:
                load_tier = "High Demand"
                tier_color = AMBER_ACCENT

            st.markdown(
                f"""
                <div class="prediction-result-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                        <span style="font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#CBD5E1;">
                            Forecasted Appliance Energy Consumption ({selected_model_name})
                        </span>
                        <span style="padding:0.25rem 0.75rem; border-radius:999px; font-size:0.78rem; font-weight:700; background:rgba(15,23,42,0.7); color:{tier_color}; border:1px solid {tier_color};">
                            {load_tier}
                        </span>
                    </div>
                    <div class="prediction-number">{pred_wh:.2f} Wh</div>
                    <div style="color:#CBD5E1; font-size:0.92rem;">
                        Equivalent to <strong>{pred_wh / 1000:.4f} kWh</strong> per 10-minute interval
                        (≈ <strong>{(pred_wh * 6) / 1000:.3f} kW</strong> average instantaneous draw) ·
                        Training Median: <strong>{pred_split.y_train.median():.0f} Wh</strong> |
                        Training Mean: <strong>{pred_split.y_train.mean():.1f} Wh</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if raw_pred < 0:
                st.warning(
                    f"Raw linear extrapolation produced `{raw_pred:.2f} Wh`, which was clamped to `0.00 Wh` "
                    "because physical appliance energy consumption cannot be negative."
                )

            st.markdown("#### 🔄 Cross-Model Prediction Comparison for This Input")
            comp_rows = []
            for m_name in MODEL_NAMES:
                m_raw, m_clamped = predict_consumption(
                    pred_bundle.models[m_name], pred_bundle.feature_cols, user_inputs
                )
                m_metrics = pred_bundle.metrics_df.loc[pred_bundle.metrics_df["Model"] == m_name].iloc[0]
                comp_rows.append(
                    {
                        "Model": m_name + (" ⭐ (Best)" if m_name == pred_bundle.best_model_name else ""),
                        "Predicted Consumption (Wh)": round(m_clamped, 2),
                        "Test MAE (Wh)": round(float(m_metrics["MAE (Wh)"]), 2),
                        "Test RMSE (Wh)": round(float(m_metrics["RMSE (Wh)"]), 2),
                        "Test R²": round(float(m_metrics["R² Score"]), 4),
                    }
                )
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def render_about_project_page(meta: DatasetMetadata) -> None:
    """Page 6: About Project."""
    st.markdown("## 📘 About Project")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-heading">🎯 Problem Statement</div>
                <p style="color:#CBD5E1; font-size:0.94rem; line-height:1.6;">
                    Residential buildings account for a substantial share of global electricity demand, with
                    appliance loads fluctuating sharply based on occupancy habits, indoor thermal comfort,
                    lighting usage, and external weather conditions. Accurately forecasting appliance energy
                    consumption enables smarter demand management, anomaly detection, and household energy efficiency.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-card">
                <div class="section-heading">🚀 Project Objectives</div>
                <ul style="color:#CBD5E1; font-size:0.94rem; line-height:1.65; margin-bottom:0;">
                    <li>Ingest and audit real 10-minute residential energy & environmental sensor data.</li>
                    <li>Perform chronological preprocessing without future data leakage.</li>
                    <li>Visualize consumption distributions, hourly demand cycles, and sensor correlations.</li>
                    <li>Train and objectively compare Linear Regression, Decision Tree Regression, and Gradient Boosting Regression using MAE, RMSE, and R².</li>
                    <li>Deliver a real-time interactive forecasting interface in Wh.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-heading">🗃️ Dataset Information</div>
                <p style="color:#CBD5E1; font-size:0.94rem; line-height:1.6;">
                    <strong>Name:</strong> Appliances Energy Prediction (UCI Dataset ID: {DATASET_ID})<br>
                    <strong>Source:</strong> <a href="{UCI_DATASET_PAGE}" target="_blank" style="color:#38BDF8;">UCI Machine Learning Repository</a><br>
                    <strong>Records:</strong> {meta.total_records:,} observations at 10-minute intervals<br>
                    <strong>Time Span:</strong> {meta.date_start:%Y-%m-%d} to {meta.date_end:%Y-%m-%d} (~4.5 months)<br>
                    <strong>Target:</strong> <code>Appliances</code> — energy use in Watt-hours (Wh)<br>
                    <strong>Signals:</strong> Lighting energy (<code>lights</code>), indoor room temperatures & humidities
                    (<code>T1–T9</code>, <code>RH_1–RH_9</code>), outdoor weather station readings (<code>T_out</code>,
                    <code>RH_out</code>, <code>Windspeed</code>, <code>Press_mm_hg</code>, <code>Visibility</code>, <code>Tdewpoint</code>),
                    plus two random control variables (<code>rv1</code>, <code>rv2</code>).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-card">
                <div class="section-heading">🛠️ Technologies Used</div>
                <div style="display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.5rem;">
                    <span class="tech-pill">Python 3</span>
                    <span class="tech-pill">Streamlit</span>
                    <span class="tech-pill">Pandas</span>
                    <span class="tech-pill">NumPy</span>
                    <span class="tech-pill">Matplotlib</span>
                    <span class="tech-pill">Seaborn</span>
                    <span class="tech-pill">Scikit-learn</span>
                    <span class="tech-pill">ucimlrepo</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="section-card">
            <div class="section-heading">🔄 Machine Learning Workflow</div>
            <p style="color:#CBD5E1; font-size:0.94rem; line-height:1.7; margin-bottom:0;">
                <strong>1. Data Ingestion & Audit:</strong> Load UCI Dataset #374, normalize timestamp formatting, check missing values and duplicates.<br>
                <strong>2. Chronological Ordering & Feature Engineering:</strong> Sort records strictly by <code>date</code> and extract deterministic time features (<code>hour</code>, <code>dayofweek</code>).<br>
                <strong>3. Time-Ordered Split:</strong> Partition the first 80% (or user-configured ratio) for training and the remaining future 20% for testing with <code>shuffle=False</code>.<br>
                <strong>4. Leakage-Free Pipelines:</strong> Wrap <code>SimpleImputer(strategy="median")</code> (and <code>StandardScaler</code> for Linear Regression) inside scikit-learn <code>Pipeline</code> objects fit strictly on <code>X_train</code>.<br>
                <strong>5. Model Training & Comparison:</strong> Train Linear Regression, Decision Tree Regression, and Gradient Boosting Regression; evaluate with MAE, RMSE, and R².<br>
                <strong>6. Live Inference:</strong> Serve interactive predictions in Wh alongside cross-model comparisons.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
            <div class="section-heading">📝 Conclusion & Limitations</div>
            <p style="color:#CBD5E1; font-size:0.94rem; line-height:1.65; margin-bottom:0;">
                <strong>Conclusion:</strong> Combining environmental sensors, lighting power, and time-of-day signals captures
                meaningful diurnal appliance demand trends (such as evening cooking peaks around 18:00 and low overnight baselines).
                Strict chronological validation reveals how models generalize into future weeks rather than memorizing adjacent 10-minute samples.<br><br>
                <strong>Limitations:</strong><br>
                • <strong>Spiky Unobserved Events:</strong> Sudden high-wattage appliance bursts (ovens, washing machines, dryers) are driven by unobserved occupant actions, limiting R² when predicting 10-minute instantaneous spikes without lag features.<br>
                • <strong>Seasonal Drift:</strong> Because the dataset spans January through May, late-spring outdoor temperatures in the chronological test split exceed winter training temperatures, where tree-based models cannot extrapolate linearly beyond seen splits.<br>
                • <strong>Future Scope:</strong> Adding lagged appliance consumption (<code>t-1</code>, rolling means), smart-plug sub-metering, or full-year seasonal coverage would substantially boost predictive accuracy.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Application Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    inject_custom_css()

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div style="font-size:1.15rem; font-weight:800; color:#F8FAFC;">⚡ EnergyForecast AI</div>
                <div style="font-size:0.78rem; color:#38BDF8; margin-top:0.15rem;">
                    UCI Appliances Dataset · ML Suite
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            options=[
                "🏠 Home Dashboard",
                "🗂️ Dataset Explorer",
                "📊 Data Visualization",
                "🤖 Machine Learning Models",
                "⚡ Prediction System",
                "📘 About Project",
            ],
            index=0,
        )

    try:
        df, meta = get_dataset_cached()
    except Exception as exc:
        st.error(f"Unable to load the UCI Appliances Energy Prediction dataset: {exc}")
        st.info("Please verify your internet connection or try refreshing the application.")
        st.stop()

    cfg = st.session_state.get(
        "ml_config",
        {
            "feature_set_name": "Core Prediction Features (10)",
            "train_ratio": 0.80,
            "dt_max_depth": 6,
            "dt_min_samples_leaf": 25,
            "gb_n_estimators": 120,
            "gb_learning_rate": 0.05,
            "gb_max_depth": 3,
            "gb_subsample": 0.80,
        },
    )

    split_data, bundle = get_trained_bundle_cached(
        feature_set_name=cfg["feature_set_name"],
        train_ratio=float(cfg["train_ratio"]),
        dt_max_depth=int(cfg["dt_max_depth"]),
        dt_min_samples_leaf=int(cfg["dt_min_samples_leaf"]),
        gb_n_estimators=int(cfg["gb_n_estimators"]),
        gb_learning_rate=float(cfg["gb_learning_rate"]),
        gb_max_depth=int(cfg["gb_max_depth"]),
        gb_subsample=float(cfg["gb_subsample"]),
    )

    with st.sidebar:
        st.divider()
        best_row = bundle.metrics_df.iloc[0]
        st.markdown(
            f"""
            <div style="font-size:0.80rem; color:#94A3B8; line-height:1.55;">
                <strong style="color:#E2E8F0;">Pipeline Status:</strong> Ready ✅<br>
                <strong style="color:#E2E8F0;">Records:</strong> {meta.total_records:,}<br>
                <strong style="color:#E2E8F0;">Best Model:</strong> {best_row['Model'].replace(' Regression', '')}<br>
                <strong style="color:#E2E8F0;">Test R²:</strong> {best_row['R² Score']:.4f}<br>
                <strong style="color:#E2E8F0;">Test MAE:</strong> {best_row['MAE (Wh)']:.2f} Wh
            </div>
            """,
            unsafe_allow_html=True,
        )

    if page == "🏠 Home Dashboard":
        render_home_page(df, meta, split_data, bundle)
    elif page == "🗂️ Dataset Explorer":
        render_dataset_explorer_page(df, meta)
    elif page == "📊 Data Visualization":
        render_data_visualization_page(df, split_data, bundle)
    elif page == "🤖 Machine Learning Models":
        render_ml_models_page(df)
    elif page == "⚡ Prediction System":
        render_prediction_system_page(split_data, bundle)
    elif page == "📘 About Project":
        render_about_project_page(meta)

    st.divider()
    st.caption(
        "Electricity Consumption Forecasting Using Machine Learning · "
        "UCI Appliances Energy Prediction Dataset (ID: 374) · Built with Python, Scikit-learn & Streamlit."
    )


if __name__ == "__main__":
    main()
