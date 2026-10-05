import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

from core.data_loader import DataLoader
from core.cleaner import DataCleaner
from core.slicer import DataSlicer
from core.eda import EDAEngine, CHART_THEME
from core.ai_engine import AIEngine
from core.reporter import ReportGenerator

# ---------------------------------------------------------
# Page Configuration & Executive Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Nexus Analytics | Auto-EDA & Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Gradient Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.7) 0%, rgba(49, 46, 129, 0.5) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FFFFFF 0%, #C7D2FE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
    }

    /* Stat Cards */
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #6366F1;
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .metric-val {
        font-size: 1.7rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 4px;
    }
    .metric-meta {
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* Badge & Tag */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Clean Card Container */
    .content-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        background-color: #1E293B;
        border: 1px solid #334155;
        color: #94A3B8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4F46E5 !important;
        color: #FFFFFF !important;
        border-color: #6366F1 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = None
if "filter_summary" not in st.session_state:
    st.session_state.filter_summary = "All Records (Unfiltered)"
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "None"
if "ai_insights" not in st.session_state:
    st.session_state.ai_insights = None

# ---------------------------------------------------------
# Sidebar: Ingestion & Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Nexus Engine")
    st.caption("Automated Data Analytics & BI Studio")
    st.markdown("---")

    # 1. Dataset Source
    st.markdown("#### 📂 1. Ingest Dataset")
    data_source = st.radio(
        "Source Type",
        ["Demo E-Commerce Data (15k rows)", "Upload Your File (CSV, Excel, Parquet)"],
        label_visibility="collapsed"
    )

    if data_source == "Demo E-Commerce Data (15k rows)":
        if st.button("🚀 Load Demo Dataset", use_container_width=True, type="primary"):
            demo_path = "sample_data/ecommerce_analytics_data.csv"
            if os.path.exists(demo_path):
                df = DataLoader.load_file(demo_path)
                st.session_state.raw_df = df
                st.session_state.cleaned_df = df.copy()
                st.session_state.filtered_df = df.copy()
                st.session_state.dataset_name = "ecommerce_analytics_data.csv"
                st.session_state.filter_summary = "All Records (Demo E-Commerce)"
                st.session_state.ai_insights = None
                st.success("Loaded 15,000+ demo transactions!")
            else:
                st.error("Demo file not found. Generating...")
    else:
        uploaded_file = st.file_uploader(
            "Upload any dataset",
            type=["csv", "xlsx", "xls", "parquet", "json"],
            help="Handles large files seamlessly with DuckDB memory optimization."
        )
        if uploaded_file is not None:
            if st.session_state.dataset_name != uploaded_file.name:
                with st.spinner("Parsing & optimizing schema..."):
                    df = DataLoader.load_file(uploaded_file)
                    st.session_state.raw_df = df
                    st.session_state.cleaned_df = df.copy()
                    st.session_state.filtered_df = df.copy()
                    st.session_state.dataset_name = uploaded_file.name
                    st.session_state.filter_summary = "All Records (Raw Upload)"
                    st.session_state.ai_insights = None
                st.success(f"Loaded `{uploaded_file.name}` ({len(df):,} rows)")

    st.markdown("---")
    
    # 2. AI Intelligence Provider
    st.markdown("#### 🤖 2. AI Intelligence Config")
    ai_provider = st.selectbox("LLM Engine", ["Google Gemini", "OpenAI"])
    
    if ai_provider == "Google Gemini":
        api_key = st.text_input("Gemini API Key", type="password", placeholder="AIzaSy...", help="Leave blank to use intelligent offline statistical engine.")
    else:
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", help="Leave blank to use intelligent offline statistical engine.")

    if not api_key:
        st.info("💡 Running in **Smart Statistical Engine** mode (100% offline & free). Paste API key to enable generative LLM narratives.")

    st.markdown("---")
    # Quick Health score indicator if data is present
    if st.session_state.raw_df is not None:
        health = DataCleaner.calculate_health_score(st.session_state.cleaned_df)
        st.markdown("#### 🩺 Data Health Score")
        st.markdown(f"""
        <div style="background-color: #1E293B; border: 1px solid #334155; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 1.6rem; font-weight: 800; color: {health['color']};">{health['score']}%</div>
            <div style="font-size: 0.8rem; color: #94A3B8;">Grade: {health['grade']}</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px;">Nulls: {health['missing_pct']}% | Dups: {health['duplicate_pct']}%</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Top Banner / Hero
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Nexus Analytics Studio</div>
    <div class="hero-subtitle">
        Universal Auto-Cleaning, Dynamic Multi-Condition Slicing, Deep EDA & Executive AI Intelligence
    </div>
</div>
""", unsafe_allow_html=True)

# Guard: No dataset loaded yet
if st.session_state.raw_df is None:
    st.info("👋 Welcome! Click **'Load Demo Dataset'** in the sidebar to test immediately, or upload your own CSV/Excel/Parquet file.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">1. Universal Ingestion</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #FFFFFF; margin: 8px 0;">Any Format & Size</div>
            <div style="font-size: 0.85rem; color: #94A3B8;">Stream CSV, Excel, Parquet, or JSON with DuckDB & Polars speed.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">2. Smart Condition Slicing</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #FFFFFF; margin: 8px 0;">Last 2 Months, Cohorts & Rules</div>
            <div style="font-size: 0.85rem; color: #94A3B8;">Slice timeframes, repeat customers, age brackets, or ask in plain English.</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">3. 1-Click Executive Reports</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #FFFFFF; margin: 8px 0;">Interactive HTML & PDF</div>
            <div style="font-size: 0.85rem; color: #94A3B8;">Generate publication-ready dashboards and PDF summaries for executives.</div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------
tab_overview, tab_clean, tab_slice, tab_eda, tab_report = st.tabs([
    "📁 1. Dataset Overview",
    "🧼 2. Auto-Cleaning & Health",
    "🎯 3. Slicer & Business Queries",
    "📊 4. Deep EDA Intelligence",
    "🤖 5. Executive AI Report"
])

current_df = st.session_state.filtered_df if st.session_state.filtered_df is not None else st.session_state.cleaned_df

# =========================================================
# TAB 1: DATASET OVERVIEW
# =========================================================
with tab_overview:
    st.markdown("### 📋 Dataset Profile & Raw Inspector")
    
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Records</div>
            <div class="metric-val">{len(st.session_state.raw_df):,}</div>
            <div class="metric-meta" style="color: #6366F1;">Source: {st.session_state.dataset_name}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Columns</div>
            <div class="metric-val">{len(st.session_state.raw_df.columns)}</div>
            <div class="metric-meta" style="color: #10B981;">Features Tracked</div>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        cols_meta = DataLoader.inspect_columns(st.session_state.raw_df)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Numeric / Categorical</div>
            <div class="metric-val">{len(cols_meta['numeric_cols'])} / {len(cols_meta['categorical_cols'])}</div>
            <div class="metric-meta" style="color: #06B6D4;">Date Columns: {len(cols_meta['date_cols'])}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_d:
        mem_mb = round(st.session_state.raw_df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">In-Memory Size</div>
            <div class="metric-val">{mem_mb} MB</div>
            <div class="metric-meta" style="color: #F59E0B;">Optimized Types</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Data View & Column Summary
    c_left, c_right = st.columns([3, 1])
    with c_left:
        st.markdown("#### Sample Records (Top 25)")
        st.dataframe(st.session_state.raw_df.head(25), use_container_width=True, height=350)
    with c_right:
        st.markdown("#### Schema Breakdown")
        schema_df = pd.DataFrame({
            "Column": st.session_state.raw_df.columns,
            "Type": [str(t) for t in st.session_state.raw_df.dtypes],
            "Nulls": st.session_state.raw_df.isna().sum().values
        })
        st.dataframe(schema_df, use_container_width=True, height=350)

# =========================================================
# TAB 2: AUTO-CLEANING & HEALTH AUDIT
# =========================================================
with tab_clean:
    st.markdown("### 🧼 Automated Data Hygiene & Quality Engineering")
    st.caption("Detect anomalies, impute missing values, eradicate duplicates, and trace transformations.")

    health = DataCleaner.calculate_health_score(st.session_state.cleaned_df)
    
    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    with col_h1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Health Score</div>
            <div class="metric-val" style="color: {health['color']};">{health['score']}%</div>
            <div class="metric-meta" style="color: {health['color']};">Grade: {health['grade']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_h2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Missing Cells</div>
            <div class="metric-val">{health['missing_cells']:,}</div>
            <div class="metric-meta" style="color: #EF4444;">{health['missing_pct']}% of dataset</div>
        </div>
        """, unsafe_allow_html=True)
    with col_h3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Duplicate Rows</div>
            <div class="metric-val">{health['duplicate_rows']:,}</div>
            <div class="metric-meta" style="color: #F59E0B;">{health['duplicate_pct']}% redundancy</div>
        </div>
        """, unsafe_allow_html=True)
    with col_h4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Cleaned Records</div>
            <div class="metric-val">{len(st.session_state.cleaned_df):,}</div>
            <div class="metric-meta" style="color: #10B981;">Ready for Analytics</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Cleaning Control Panel
    st.markdown("#### ⚙️ Auto-Cleaning Controls")
    with st.expander("Configure Cleaning Rules", expanded=True):
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            clean_dups = st.checkbox("Purge Exact Duplicate Rows", value=True)
            clean_strings = st.checkbox("Trim String Whitespace", value=True)
        with col_c2:
            clean_nulls = st.checkbox("Impute Missing Values (Median / Mode)", value=True)
        with col_c3:
            clean_outliers = st.checkbox("Clip Extreme Outliers (IQR Method)", value=False)
        
        if st.button("✨ Run Automated Cleaning Pipeline", type="primary"):
            with st.spinner("Executing cleaning pipeline..."):
                cleaned, logs = DataCleaner.auto_clean(
                    st.session_state.raw_df,
                    remove_duplicates=clean_dups,
                    impute_missing=clean_nulls,
                    trim_strings=clean_strings,
                    handle_outliers=clean_outliers
                )
                st.session_state.cleaned_df = cleaned
                st.session_state.filtered_df = cleaned.copy()
                st.session_state.audit_log = logs
            st.success("Automated cleaning pipeline completed successfully!")

    # Audit Trail
    st.markdown("#### 📜 Cleaning Audit Trail")
    if st.session_state.audit_log:
        for item in st.session_state.audit_log:
            st.markdown(f"""
            <div style="background-color: #1E293B; border-left: 4px solid #10B981; padding: 12px 16px; margin-bottom: 8px; border-radius: 6px;">
                <span style="font-weight: 700; color: #FFFFFF;">{item['action']}</span>: 
                <span style="color: #94A3B8;">{item['details']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Run the cleaning pipeline above to generate an audit log.")

# =========================================================
# TAB 3: SLICER & BUSINESS QUERIES (THE FILTER HUB)
# =========================================================
with tab_slice:
    st.markdown("### 🎯 Data Slicing & Condition Engine")
    st.caption("Filter data by specific timeframes (e.g. Last 2 Months), demographic criteria, customer loyalty, or natural language.")

    # 1. Quick Presets (One-click filters for e-commerce / analytics)
    st.markdown("#### ⚡ 1-Click Business Presets")
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
    
    with preset_col1:
        if st.button("📅 Last 2 Months (60 Days)", use_container_width=True):
            cols_info = DataLoader.inspect_columns(st.session_state.cleaned_df)
            date_col = cols_info["date_cols"][0] if cols_info["date_cols"] else None
            if date_col:
                sliced, desc = DataSlicer.slice_by_timeframe(st.session_state.cleaned_df, date_col, "Last 2 Months (60 Days)")
                st.session_state.filtered_df = sliced
                st.session_state.filter_summary = desc
                st.session_state.ai_insights = None
                st.rerun()
            else:
                st.warning("No date column detected for timeframe slicing.")

    with preset_col2:
        if st.button("🔁 Repeat Customers Only", use_container_width=True):
            # Check if repeat column or total orders exists
            target_col = None
            if "is_repeat_customer" in st.session_state.cleaned_df.columns:
                rules = [{"column": "is_repeat_customer", "operator": "==", "value": "Yes"}]
            elif "customer_total_orders" in st.session_state.cleaned_df.columns:
                rules = [{"column": "customer_total_orders", "operator": ">", "value": 1}]
            else:
                rules = []

            if rules:
                sliced, desc = DataSlicer.apply_rules(st.session_state.cleaned_df, rules)
                st.session_state.filtered_df = sliced
                st.session_state.filter_summary = desc
                st.session_state.ai_insights = None
                st.rerun()
            else:
                st.warning("Customer order frequency column not found in this dataset.")

    with preset_col3:
        if st.button("🎂 Age >= 25 & Repeat Buyers", use_container_width=True):
            rules = []
            if "customer_age" in st.session_state.cleaned_df.columns:
                rules.append({"column": "customer_age", "operator": ">=", "value": 25})
            if "is_repeat_customer" in st.session_state.cleaned_df.columns:
                rules.append({"column": "is_repeat_customer", "operator": "==", "value": "Yes"})
            elif "customer_total_orders" in st.session_state.cleaned_df.columns:
                rules.append({"column": "customer_total_orders", "operator": ">", "value": 1})

            if rules:
                sliced, desc = DataSlicer.apply_rules(st.session_state.cleaned_df, rules, logic="AND")
                st.session_state.filtered_df = sliced
                st.session_state.filter_summary = desc
                st.session_state.ai_insights = None
                st.rerun()
            else:
                st.warning("Required demographic columns not found.")

    with preset_col4:
        if st.button("🔄 Reset to Full Cleaned Data", use_container_width=True):
            st.session_state.filtered_df = st.session_state.cleaned_df.copy()
            st.session_state.filter_summary = "All Records (Unfiltered)"
            st.session_state.ai_insights = None
            st.rerun()

    st.markdown("---")

    # 2. Detailed Filter Builder
    col_filter_left, col_filter_right = st.columns(2)

    with col_filter_left:
        st.markdown("#### ⏳ Timeframe Filter")
        cols_info = DataLoader.inspect_columns(st.session_state.cleaned_df)
        date_cols = cols_info["date_cols"]
        
        if date_cols:
            selected_date_col = st.selectbox("Date Column Anchor", date_cols)
            timeframe_choice = st.selectbox(
                "Timeframe Window",
                [
                    "All Time",
                    "Last 1 Month (30 Days)",
                    "Last 2 Months (60 Days)",
                    "Last 3 Months (90 Days)",
                    "Last 6 Months (180 Days)",
                    "Year to Date (YTD)",
                    "Custom Range"
                ],
                index=0
            )

            c_start, c_end = None, None
            if timeframe_choice == "Custom Range":
                col_d1, col_d2 = st.columns(2)
                min_d = st.session_state.cleaned_df[selected_date_col].min().date()
                max_d = st.session_state.cleaned_df[selected_date_col].max().date()
                with col_d1:
                    c_start = st.date_input("Start Date", value=min_d)
                with col_d2:
                    c_end = st.date_input("End Date", value=max_d)

            if st.button("Apply Time Filter", key="apply_time_filter"):
                sliced, desc = DataSlicer.slice_by_timeframe(
                    st.session_state.cleaned_df,
                    selected_date_col,
                    timeframe_choice,
                    c_start,
                    c_end
                )
                st.session_state.filtered_df = sliced
                st.session_state.filter_summary = desc
                st.session_state.ai_insights = None
                st.success(desc)
                st.rerun()
        else:
            st.info("No datetime columns detected in dataset.")

    with col_filter_right:
        st.markdown("#### 🔍 Multi-Condition Rule Builder")
        all_cols = list(st.session_state.cleaned_df.columns)
        
        rule_col = st.selectbox("Column", all_cols)
        rule_op = st.selectbox("Operator", [">=", "<=", ">", "<", "==", "!=", "contains", "in"])
        
        # Smart value suggestions based on column type
        if pd.api.types.is_numeric_dtype(st.session_state.cleaned_df[rule_col]):
            median_val = float(st.session_state.cleaned_df[rule_col].median())
            rule_val = st.text_input("Comparison Value (Numeric)", value=str(round(median_val, 2)))
        elif rule_op == "in":
            sample_unique = list(st.session_state.cleaned_df[rule_col].dropna().unique()[:4])
            rule_val = st.text_input("Values (comma-separated)", value=", ".join([str(x) for x in sample_unique]))
        else:
            rule_val = st.text_input("Comparison Value (Text)", value="")

        if st.button("Apply Condition Rule", key="apply_rule_btn"):
            if rule_val:
                rule = [{"column": rule_col, "operator": rule_op, "value": rule_val}]
                # Apply on top of existing filtered df or cleaned df
                base_df = st.session_state.filtered_df if st.session_state.filtered_df is not None else st.session_state.cleaned_df
                sliced, desc = DataSlicer.apply_rules(base_df, rule)
                st.session_state.filtered_df = sliced
                st.session_state.filter_summary = f"{st.session_state.filter_summary} & [{desc}]"
                st.session_state.ai_insights = None
                st.success(desc)
                st.rerun()

    # Active Slice Status Card
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid #6366F1; border-radius: 12px; padding: 18px;">
        <span style="font-weight: 700; color: #818CF8; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.05em;">Currently Active Segment</span>
        <div style="font-size: 1.15rem; font-weight: 600; color: #FFFFFF; margin-top: 4px;">{st.session_state.filter_summary}</div>
        <div style="font-size: 0.9rem; color: #94A3B8; margin-top: 4px;">
            Matched Records: <strong style="color: #10B981;">{len(st.session_state.filtered_df):,}</strong> / {len(st.session_state.cleaned_df):,} total 
            ({round(len(st.session_state.filtered_df)/max(1, len(st.session_state.cleaned_df))*100, 1)}%)
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# TAB 4: DEEP EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================
with tab_eda:
    active_df = st.session_state.filtered_df if st.session_state.filtered_df is not None else st.session_state.cleaned_df
    
    st.markdown("### 📊 Automated Exploratory Data Analysis (EDA)")
    st.caption(f"Analyzing Active Segment: {st.session_state.filter_summary} ({len(active_df):,} rows)")

    if len(active_df) == 0:
        st.warning("Current filter returned 0 records. Adjust your conditions in the Slicer tab.")
        st.stop()

    # 1. KPI Metric Ribbon
    kpis = EDAEngine.get_kpis(active_df)
    num_summaries = kpis.get("numeric_summaries", {})
    
    kpi_cols = st.columns(min(4, max(1, len(num_summaries) + 1)))
    with kpi_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Analyzed Cohort</div>
            <div class="metric-val">{len(active_df):,}</div>
            <div class="metric-meta" style="color: #6366F1;">Transactions / Rows</div>
        </div>
        """, unsafe_allow_html=True)
    
    for idx, (col_name, stats) in enumerate(list(num_summaries.items())[:3]):
        with kpi_cols[idx + 1]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total {col_name.replace('_', ' ').title()}</div>
                <div class="metric-val">{stats['sum']:,.1f}</div>
                <div class="metric-meta" style="color: #10B981;">Avg: {stats['mean']:,.1f} | Med: {stats['median']:,.1f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Charts Section
    cols_meta = DataLoader.inspect_columns(active_df)
    
    # Time-Series Trajectory
    if cols_meta["date_cols"] and cols_meta["numeric_cols"]:
        st.markdown("#### 📈 Longitudinal Time-Series Trend")
        c_ts1, c_ts2, c_ts3 = st.columns([2, 1, 1])
        with c_ts1:
            ts_date_col = st.selectbox("Date Variable", cols_meta["date_cols"], key="eda_date_col")
        with c_ts2:
            ts_val_col = st.selectbox("Value Variable", cols_meta["numeric_cols"], key="eda_val_col")
        with c_ts3:
            ts_agg = st.selectbox("Aggregation", ["sum", "mean", "count"], key="eda_agg")

        fig_ts = EDAEngine.create_timeseries_chart(active_df, ts_date_col, ts_val_col, agg_func=ts_agg)
        st.plotly_chart(fig_ts, use_container_width=True)

    # Category Frequency & Distribution
    col_chart_left, col_chart_right = st.columns(2)
    
    with col_chart_left:
        st.markdown("#### 🏷️ Categorical Concentration")
        if cols_meta["categorical_cols"]:
            cat_choice = st.selectbox("Category Dimension", cols_meta["categorical_cols"], key="eda_cat_col")
            fig_cat = EDAEngine.create_categorical_chart(active_df, cat_choice)
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("No categorical dimensions found.")

    with col_chart_right:
        st.markdown("#### 📊 Numeric Spread & Distribution")
        if cols_meta["numeric_cols"]:
            num_choice = st.selectbox("Numeric Metric", cols_meta["numeric_cols"], key="eda_num_col")
            fig_dist = EDAEngine.create_distribution_chart(active_df, num_choice)
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("No numeric dimensions found.")

    # Correlation Matrix
    if len(cols_meta["numeric_cols"]) >= 2:
        st.markdown("#### 🧬 Correlation Heatmap")
        fig_corr = EDAEngine.create_correlation_heatmap(active_df)
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)

# =========================================================
# TAB 5: EXECUTIVE AI REPORT & EXPORT
# =========================================================
with tab_report:
    active_df = st.session_state.filtered_df if st.session_state.filtered_df is not None else st.session_state.cleaned_df
    
    st.markdown("### 🤖 Executive AI Report & Strategic Insights")
    st.caption("AI-driven synthesis with quantitative findings, vulnerability analysis, and exportable reports.")

    kpis = EDAEngine.get_kpis(active_df)

    # Generate insights button or auto-generate
    if st.session_state.ai_insights is None:
        with st.spinner("Synthesizing executive intelligence..."):
            provider_choice = "Gemini" if ai_provider == "Google Gemini" else "OpenAI"
            insights = AIEngine.generate_executive_insights(
                active_df,
                st.session_state.filter_summary,
                kpis,
                provider=provider_choice,
                api_key=api_key
            )
            st.session_state.ai_insights = insights

    insights = st.session_state.ai_insights

    if "api_error" in insights:
        st.warning(insights["api_error"])

    # Executive Summary Card
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(30, 27, 75, 0.6) 0%, rgba(49, 46, 129, 0.4) 100%); border: 1px solid #6366F1; border-radius: 14px; padding: 22px; margin-bottom: 24px;">
        <span style="font-weight: 700; color: #818CF8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;">Executive Brief</span>
        <div style="font-size: 1.15rem; color: #FFFFFF; font-weight: 500; margin-top: 8px; line-height: 1.6;">
            {insights.get('executive_summary', '')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Findings & Risks Grid
    c_find, c_risk = st.columns(2)
    with c_find:
        st.markdown("#### 🎯 Key Statistical Findings")
        for finding in insights.get("key_findings", []):
            st.markdown(f"""
            <div style="background-color: #1E293B; border-left: 4px solid #6366F1; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;">
                <span style="color: #E2E8F0; font-size: 0.95rem;">{finding}</span>
            </div>
            """, unsafe_allow_html=True)

    with c_risk:
        st.markdown("#### ⚠️ Vulnerabilities & Risk Flags")
        for risk in insights.get("anomalies_or_risks", []):
            st.markdown(f"""
            <div style="background-color: #1E293B; border-left: 4px solid #F43F5E; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;">
                <span style="color: #E2E8F0; font-size: 0.95rem;">{risk}</span>
            </div>
            """, unsafe_allow_html=True)

    # Strategic Action Plan
    st.markdown("#### 🚀 Actionable Recommendations")
    for rec in insights.get("strategic_recommendations", []):
        st.markdown(f"""
        <div style="background-color: #1E293B; border-left: 4px solid #10B981; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;">
            <span style="color: #E2E8F0; font-size: 0.95rem;">{rec}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Q&A Assistant
    st.markdown("#### 💬 Ask Ad-Hoc Questions About This Dataset")
    user_q = st.text_input("Ask any question (e.g. 'What is the top contributing category for repeat buyers?')", key="user_adhoc_q")
    if st.button("Ask Assistant", type="secondary"):
        if user_q:
            with st.spinner("Analyzing question..."):
                answer = AIEngine.answer_question(
                    user_q,
                    active_df,
                    provider="Gemini" if ai_provider == "Google Gemini" else "OpenAI",
                    api_key=api_key
                )
            st.markdown(f"""
            <div style="background-color: #1E293B; border: 1px solid #4F46E5; border-radius: 10px; padding: 16px; margin-top: 10px;">
                <strong style="color: #818CF8;">Assistant:</strong><br>
                <span style="color: #F8FAFC;">{answer}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Download & Export Hub
    st.markdown("#### 📥 Export Publication Reports & Filtered Data")
    exp_col1, exp_col2, exp_col3 = st.columns(3)

    cols_meta = DataLoader.inspect_columns(active_df)
    report_charts = []
    if cols_meta["date_cols"] and cols_meta["numeric_cols"]:
        report_charts.append(EDAEngine.create_timeseries_chart(active_df, cols_meta["date_cols"][0], cols_meta["numeric_cols"][0]))
    if cols_meta["categorical_cols"]:
        report_charts.append(EDAEngine.create_categorical_chart(active_df, cols_meta["categorical_cols"][0]))
    if cols_meta["numeric_cols"]:
        report_charts.append(EDAEngine.create_distribution_chart(active_df, cols_meta["numeric_cols"][0]))

    with exp_col1:
        html_report = ReportGenerator.generate_html_report(
            active_df,
            st.session_state.filter_summary,
            kpis,
            insights,
            report_charts
        )
        st.download_button(
            label="🌐 Download Standalone HTML Dashboard",
            data=html_report,
            file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            use_container_width=True
        )

    with exp_col2:
        try:
            pdf_bytes = ReportGenerator.generate_pdf_report(
                active_df,
                st.session_state.filter_summary,
                kpis,
                insights
            )
            st.download_button(
                label="📄 Download Executive PDF Report",
                data=pdf_bytes,
                file_name=f"analytics_executive_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"PDF generation error: {e}")

    with exp_col3:
        csv_data = active_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Download Cleaned & Filtered CSV",
            data=csv_data,
            file_name=f"filtered_dataset_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
