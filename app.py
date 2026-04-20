"""
╔══════════════════════════════════════════════════════════════════╗
║   Social Media Sentiment Analysis Dashboard                      ║
║   A production-quality Streamlit app for college final project   ║
║                                                                  ║
║   Run:                                                           ║
║     pip install streamlit pandas plotly openpyxl wordcloud       ║
║     streamlit run app.py                                         ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base & fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Main background ── */
.stApp {
    background: #f8f9fc;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e8ecf1;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* ── Hide default header ── */
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* ── KPI card styling ── */
div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e8ecf1;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s;
}
div[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.09);
}
div[data-testid="metric-container"] label {
    color: #6b7280 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #111827 !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* ── Section card wrapper ── */
.card {
    background: #ffffff;
    border: 1px solid #e8ecf1;
    border-radius: 16px;
    padding: 28px 28px 20px;
    margin-bottom: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

/* ── Page title ── */
.page-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 4px;
}
.page-subtitle {
    font-size: 0.95rem;
    color: #6b7280;
    margin-bottom: 28px;
}

/* ── Insight card ── */
.insight-card {
    background: linear-gradient(135deg, #f0f7ff 0%, #e8f4fd 100%);
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
    font-size: 0.92rem;
    color: #1e3a5f;
}

/* ── Home hero ── */
.hero-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 48px 40px;
    color: white;
    margin-bottom: 28px;
}
.hero-box h1 { font-size: 2.1rem; font-weight: 700; margin: 0 0 10px; }
.hero-box p  { font-size: 1.05rem; opacity: 0.9; margin: 0; line-height: 1.6; }

/* ── Tag badge ── */
.badge {
    display: inline-block;
    background: #eff6ff;
    color: #1d4ed8;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 4px 3px;
}

/* ── Divider ── */
hr.soft { border: none; border-top: 1px solid #e8ecf1; margin: 20px 0; }

/* ── DataFrame styling ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Radio nav buttons ── */
div[data-baseweb="radio"] > div { gap: 6px; }
div[data-baseweb="radio"] label {
    border-radius: 8px;
    padding: 8px 14px;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# COLOUR PALETTE  (shared across all charts)
# ─────────────────────────────────────────────
PALETTE = {
    "Positive":  "#22c55e",
    "Negative":  "#ef4444",
    "Neutral":   "#f59e0b",
}
COUNTRY_COLORS = px.colors.qualitative.Pastel


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    """Load CSV or Excel, clean, and return a tidy DataFrame."""
    # ── Read ──────────────────────────────────────────────────────
    if hasattr(file, "name"):
        name = file.name.lower()
    else:
        name = str(file).lower()

    if name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        try:
            df = pd.read_excel(file, sheet_name="Cleaned Data")
        except Exception:
            df = pd.read_excel(file)

    # ── Strip whitespace from string columns ──────────────────────
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # ── Remove duplicates ─────────────────────────────────────────
    df.drop_duplicates(inplace=True)

    # ── Standardise sentiment column ─────────────────────────────
    sent_aliases = {
        "Sentiment_Group": "Sentiment",
        "sentiment":       "Sentiment",
        "sentiment_group": "Sentiment",
    }
    df.rename(columns={k: v for k, v in sent_aliases.items() if k in df.columns}, inplace=True)

    if "Sentiment" not in df.columns:
        for col in df.columns:
            if "sent" in col.lower():
                df.rename(columns={col: "Sentiment"}, inplace=True)
                break

    # ── Standardise country column ────────────────────────────────
    if "Country" not in df.columns:
        for col in df.columns:
            if "country" in col.lower():
                df.rename(columns={col: "Country"}, inplace=True)
                break

    # ── Handle / synthesise Date column ──────────────────────────
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif "Hour" in df.columns:
        # Build a synthetic date from Hour so time charts work
        base = pd.Timestamp("2024-01-01")
        df["Date"] = df["Hour"].apply(
            lambda h: base + pd.Timedelta(hours=int(h)) if str(h).isdigit() else pd.NaT
        )
    else:
        df["Date"] = pd.NaT

    # ── Drop rows missing critical fields ─────────────────────────
    critical = [c for c in ["Sentiment", "Country"] if c in df.columns]
    df.dropna(subset=critical, inplace=True)

    # ── Numeric engagement columns ────────────────────────────────
    for col in ["Retweets", "Likes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ── Capitalise sentiment values ───────────────────────────────
    if "Sentiment" in df.columns:
        df["Sentiment"] = df["Sentiment"].str.capitalize()

    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

def sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 10px 0 20px;'>
            <div style='font-size:2rem;'>💬</div>
            <div style='font-weight:700; font-size:1.05rem; color:#111827;'>Sentiment Analyzer</div>
            <div style='font-size:0.75rem; color:#9ca3af; margin-top:2px;'>Social Media Dashboard</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Navigation ────────────────────────────────────────────
        st.markdown("**Navigation**")
        pages = ["🏠 Home", "📋 Data Overview", "📊 Dashboard", "🔬 Advanced Analysis", "💡 Insights"]
        page = st.radio("", pages, label_visibility="collapsed")

        st.markdown("<hr class='soft'>", unsafe_allow_html=True)

        # ── Filters ───────────────────────────────────────────────
        st.markdown("**Filters**")

        countries = sorted(df["Country"].dropna().unique()) if "Country" in df.columns else []
        selected_countries = st.multiselect("🌍 Country", countries, default=countries)

        sentiments = sorted(df["Sentiment"].dropna().unique()) if "Sentiment" in df.columns else []
        selected_sentiments = st.multiselect("💭 Sentiment", sentiments, default=sentiments)

        # Date filter only if we have real dates
        has_date = "Date" in df.columns and df["Date"].notna().any()
        if has_date:
            min_d = df["Date"].min().date()
            max_d = df["Date"].max().date()
            date_range = st.date_input("📅 Date Range", value=(min_d, max_d),
                                       min_value=min_d, max_value=max_d)
        else:
            date_range = None

        st.markdown("<hr class='soft'>", unsafe_allow_html=True)

        # ── File uploader ─────────────────────────────────────────
        st.markdown("**Upload your own file**")
        uploaded = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"],
                                    label_visibility="collapsed")

        st.markdown("<br><div style='font-size:0.72rem;color:#9ca3af;text-align:center;'>Built with Streamlit & Plotly</div>", unsafe_allow_html=True)

    return page, selected_countries, selected_sentiments, date_range, uploaded


# ═══════════════════════════════════════════════════════════════════
# FILTER DATAFRAME
# ═══════════════════════════════════════════════════════════════════

def apply_filters(df, selected_countries, selected_sentiments, date_range):
    fdf = df.copy()
    if selected_countries and "Country" in fdf.columns:
        fdf = fdf[fdf["Country"].isin(selected_countries)]
    if selected_sentiments and "Sentiment" in fdf.columns:
        fdf = fdf[fdf["Sentiment"].isin(selected_sentiments)]
    if date_range and len(date_range) == 2 and "Date" in fdf.columns:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        mask = fdf["Date"].notna()
        fdf = fdf[mask & (fdf["Date"] >= start) & (fdf["Date"] <= end)]
    return fdf


# ═══════════════════════════════════════════════════════════════════
# HELPER – plotly base layout
# ═══════════════════════════════════════════════════════════════════

def chart_layout(fig, title="", height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, family="Inter", color="#111827"), x=0),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="Inter", color="#374151"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, linecolor="#e5e7eb"),
        yaxis=dict(gridcolor="#f3f4f6", linecolor="#e5e7eb"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# PAGE 1 – HOME
# ═══════════════════════════════════════════════════════════════════

def page_home(df):
    st.markdown("""
    <div class='hero-box'>
        <h1>💬 Social Media Sentiment Analysis</h1>
        <p>An interactive intelligence dashboard that transforms raw social media data
        into actionable sentiment insights across platforms, countries, and time.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='card'>
            <div style='font-size:1.6rem;'>📊</div>
            <div style='font-weight:600;font-size:1rem;margin:8px 0 4px;color:#111827;'>Interactive Dashboard</div>
            <div style='font-size:0.85rem;color:#6b7280;'>Explore KPIs, charts, and sentiment breakdowns with real-time filters.</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='card'>
            <div style='font-size:1.6rem;'>🌍</div>
            <div style='font-weight:600;font-size:1rem;margin:8px 0 4px;color:#111827;'>Country Analysis</div>
            <div style='font-size:0.85rem;color:#6b7280;'>Compare sentiment distribution across countries and regions.</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='card'>
            <div style='font-size:1.6rem;'>💡</div>
            <div style='font-weight:600;font-size:1rem;margin:8px 0 4px;color:#111827;'>Auto Insights</div>
            <div style='font-size:0.85rem;color:#6b7280;'>Automatically generated insights and observations from your data.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 📌 Dataset at a Glance")
    g1, g2, g3, g4 = st.columns(4)
    with g1: st.metric("Total Records", f"{len(df):,}")
    with g2: st.metric("Countries", df["Country"].nunique() if "Country" in df.columns else "—")
    with g3: st.metric("Sentiments", df["Sentiment"].nunique() if "Sentiment" in df.columns else "—")
    with g4:
        platforms = df["Platform"].nunique() if "Platform" in df.columns else "—"
        st.metric("Platforms", platforms)

    st.markdown("<br>**Columns in dataset:** " +
                " ".join(f"<span class='badge'>{c}</span>" for c in df.columns),
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <b>ℹ️ How to use this dashboard</b><br><br>
        <span style='color:#374151;font-size:0.9rem;'>
        Use the <b>sidebar</b> to navigate between pages and apply filters by Country, Sentiment, or Date.
        All charts are <b>interactive</b> — hover to see details, click legends to toggle series, and drag to zoom.
        Upload your own dataset via the sidebar file uploader to analyse custom data.
        </span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 2 – DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════════

def page_data_overview(df):
    st.markdown("<div class='page-title'>📋 Data Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>A bird's-eye view of the raw dataset.</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", int(df.isnull().sum().sum()))
    c4.metric("Duplicate rows", int(df.duplicated().sum()))

    st.markdown("---")

    with st.container():
        st.markdown("#### 👀 Dataset Preview (first 20 rows)")
        st.dataframe(df.head(20), use_container_width=True, height=300)

    with st.container():
        st.markdown("#### 📐 Column Information")
        info = pd.DataFrame({
            "Column": df.columns,
            "Dtype": [str(df[c].dtype) for c in df.columns],
            "Non-Null": [df[c].notna().sum() for c in df.columns],
            "Unique": [df[c].nunique() for c in df.columns],
            "Sample": [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "—" for c in df.columns],
        })
        st.dataframe(info, use_container_width=True, hide_index=True)

    with st.container():
        st.markdown("#### 📊 Summary Statistics")
        num_cols = df.select_dtypes(include="number")
        if not num_cols.empty:
            st.dataframe(num_cols.describe().round(2), use_container_width=True)
        else:
            st.info("No numeric columns found for summary statistics.")


# ═══════════════════════════════════════════════════════════════════
# PAGE 3 – DASHBOARD
# ═══════════════════════════════════════════════════════════════════

def page_dashboard(df):
    st.markdown("<div class='page-title'>📊 Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Key metrics and interactive visualisations from filtered data.</div>", unsafe_allow_html=True)

    if df.empty:
        st.warning("⚠️ No data matches the selected filters. Please adjust the sidebar filters.")
        return

    # ── KPI Cards ────────────────────────────────────────────────
    total = len(df)
    sent_col = "Sentiment" if "Sentiment" in df.columns else None
    country_col = "Country" if "Country" in df.columns else None

    pos_pct = neg_pct = neu_pct = 0.0
    if sent_col:
        vc = df[sent_col].value_counts()
        pos_pct = round(vc.get("Positive", 0) / total * 100, 1)
        neg_pct = round(vc.get("Negative", 0) / total * 100, 1)
        neu_pct = round(vc.get("Neutral",  0) / total * 100, 1)

    top_country = df[country_col].value_counts().idxmax() if country_col else "—"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📝 Total Records",   f"{total:,}")
    k2.metric("✅ Positive",        f"{pos_pct}%",
              delta=f"+{pos_pct}%" if pos_pct > 33 else None)
    k3.metric("❌ Negative",        f"{neg_pct}%",
              delta=f"-{neg_pct}%" if neg_pct > 33 else None, delta_color="inverse")
    k4.metric("⚖️ Neutral",         f"{neu_pct}%")
    k5.metric("🏆 Most Active",     top_country)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Bar + Pie ─────────────────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if sent_col:
            sc = df[sent_col].value_counts().reset_index()
            sc.columns = ["Sentiment", "Count"]
            sc["Color"] = sc["Sentiment"].map(PALETTE)
            fig = px.bar(sc, x="Sentiment", y="Count", color="Sentiment",
                         color_discrete_map=PALETTE, text="Count")
            fig.update_traces(textposition="outside", marker_line_width=0,
                              textfont=dict(size=13, color="#111827"))
            fig = chart_layout(fig, "Sentiment Count")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if sent_col:
            pie_data = df[sent_col].value_counts().reset_index()
            pie_data.columns = ["Sentiment", "Count"]
            fig2 = px.pie(pie_data, names="Sentiment", values="Count",
                          color="Sentiment", color_discrete_map=PALETTE,
                          hole=0.45)
            fig2.update_traces(textinfo="percent+label", textfont_size=12,
                               marker=dict(line=dict(color="#ffffff", width=2)))
            fig2 = chart_layout(fig2, "Sentiment Distribution")
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 2: Line chart – Sentiment over time ──────────────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    has_date = "Date" in df.columns and df["Date"].notna().sum() > 2
    if has_date and sent_col:
        tdf = df.dropna(subset=["Date"]).copy()
        tdf["Month"] = tdf["Date"].dt.to_period("M").astype(str)
        trend = tdf.groupby(["Month", sent_col]).size().reset_index(name="Count")
        fig3 = px.line(trend, x="Month", y="Count", color=sent_col,
                       color_discrete_map=PALETTE, markers=True,
                       line_shape="spline")
        fig3.update_traces(line_width=2.5)
        fig3 = chart_layout(fig3, "Sentiment Trend Over Time", height=340)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        # Fallback: Hour-based trend if Date was synthetic
        if "Hour" in df.columns and sent_col:
            hourly = df.groupby(["Hour", sent_col]).size().reset_index(name="Count")
            fig3 = px.line(hourly, x="Hour", y="Count", color=sent_col,
                           color_discrete_map=PALETTE, markers=True,
                           labels={"Hour": "Hour of Day"}, line_shape="spline")
            fig3.update_traces(line_width=2.5)
            fig3 = chart_layout(fig3, "Sentiment Trend by Hour of Day", height=340)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("📅 No date/time column detected – time trend chart not available.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 3: Country-wise stacked bar ──────────────────────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if country_col and sent_col:
        top_n = df[country_col].value_counts().head(10).index
        cdf = df[df[country_col].isin(top_n)]
        country_sent = cdf.groupby([country_col, sent_col]).size().reset_index(name="Count")
        fig4 = px.bar(country_sent, x=country_col, y="Count", color=sent_col,
                      color_discrete_map=PALETTE, barmode="stack",
                      labels={country_col: "Country"})
        fig4.update_layout(xaxis_tickangle=-30)
        fig4 = chart_layout(fig4, "Country-wise Sentiment Comparison (Top 10)", height=400)
        st.plotly_chart(fig4, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 4 – ADVANCED ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def page_advanced(df):
    st.markdown("<div class='page-title'>🔬 Advanced Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Deeper dives into patterns, engagement, and text.</div>", unsafe_allow_html=True)

    if df.empty:
        st.warning("No data available. Adjust your filters.")
        return

    sent_col    = "Sentiment" if "Sentiment" in df.columns else None
    country_col = "Country"   if "Country"   in df.columns else None

    # ── Top countries by activity ────────────────────────────────
    if country_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🌍 Top 10 Countries by Activity")
        top_c = df[country_col].value_counts().head(10).reset_index()
        top_c.columns = ["Country", "Posts"]
        fig = px.bar(top_c, x="Posts", y="Country", orientation="h",
                     color="Posts", color_continuous_scale="Blues",
                     text="Posts")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis=dict(autorange="reversed"),
                          coloraxis_showscale=False)
        fig = chart_layout(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Platform analysis ────────────────────────────────────────
    if "Platform" in df.columns and sent_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 📱 Sentiment by Platform")
        plat = df.groupby(["Platform", sent_col]).size().reset_index(name="Count")
        fig2 = px.bar(plat, x="Platform", y="Count", color=sent_col,
                      barmode="group", color_discrete_map=PALETTE)
        fig2 = chart_layout(fig2, height=350)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Engagement analysis (Likes / Retweets) ───────────────────
    eng_cols = [c for c in ["Likes", "Retweets"] if c in df.columns]
    if eng_cols and sent_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 💥 Engagement by Sentiment")
        eng_agg = df.groupby(sent_col)[eng_cols].mean().reset_index().round(1)
        fig3 = go.Figure()
        colors = {"Likes": "#6366f1", "Retweets": "#06b6d4"}
        for col in eng_cols:
            fig3.add_trace(go.Bar(
                name=col,
                x=eng_agg[sent_col],
                y=eng_agg[col],
                marker_color=colors.get(col, "#94a3b8"),
                text=eng_agg[col],
                textposition="outside",
            ))
        fig3.update_layout(barmode="group")
        fig3 = chart_layout(fig3, "Avg Engagement per Sentiment", height=340)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Part of Day ──────────────────────────────────────────────
    if "Part_of_Day" in df.columns and sent_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🕐 Sentiment by Part of Day")
        pod_order = ["Morning", "Afternoon", "Evening", "Night"]
        pod = df.groupby(["Part_of_Day", sent_col]).size().reset_index(name="Count")
        fig4 = px.bar(pod, x="Part_of_Day", y="Count", color=sent_col,
                      color_discrete_map=PALETTE, barmode="stack",
                      category_orders={"Part_of_Day": pod_order})
        fig4 = chart_layout(fig4, height=340)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Word cloud ───────────────────────────────────────────────
    text_col = next((c for c in df.columns if "text" in c.lower() or "hashtag" in c.lower()), None)
    if text_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"#### ☁️ Word Frequency from `{text_col}`")
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            text_blob = " ".join(df[text_col].dropna().astype(str).tolist())
            wc = WordCloud(width=900, height=350, background_color="white",
                           colormap="Blues", max_words=120,
                           collocations=False).generate(text_blob)
            fig_wc, ax = plt.subplots(figsize=(11, 3.8))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            fig_wc.patch.set_alpha(0)
            st.pyplot(fig_wc, use_container_width=True)
        except ImportError:
            # Fallback: top words bar chart
            from collections import Counter
            words = " ".join(df[text_col].dropna().astype(str)).split()
            stop = {"the","a","an","and","is","in","to","of","for","on","are","was",
                    "it","this","that","with","at","be","as","we","not","nan"}
            freq = Counter(w.lower() for w in words if w.lower() not in stop and len(w) > 2)
            top_words = pd.DataFrame(freq.most_common(20), columns=["Word","Count"])
            fig_w = px.bar(top_words, x="Count", y="Word", orientation="h",
                           color="Count", color_continuous_scale="Blues",
                           text="Count")
            fig_w.update_traces(textposition="outside")
            fig_w.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            fig_w = chart_layout(fig_w, "Top 20 Most Frequent Words", height=500)
            st.plotly_chart(fig_w, use_container_width=True)
            st.caption("Install `wordcloud` (`pip install wordcloud`) for an actual word cloud.")
        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 5 – INSIGHTS
# ═══════════════════════════════════════════════════════════════════

def page_insights(df):
    st.markdown("<div class='page-title'>💡 Insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Automatically generated observations from the filtered dataset.</div>", unsafe_allow_html=True)

    if df.empty:
        st.warning("No data available. Adjust your filters.")
        return

    insights = []
    sent_col    = "Sentiment" if "Sentiment" in df.columns else None
    country_col = "Country"   if "Country"   in df.columns else None

    if sent_col:
        vc  = df[sent_col].value_counts()
        tot = len(df)
        dom = vc.idxmax()
        dom_pct = round(vc.max() / tot * 100, 1)
        insights.append(f"🏆 <b>Dominant Sentiment:</b> The most common sentiment across all records is <b>{dom}</b>, "
                        f"accounting for <b>{dom_pct}%</b> of total posts.")

        pos = round(vc.get("Positive", 0) / tot * 100, 1)
        neg = round(vc.get("Negative", 0) / tot * 100, 1)
        if pos > neg:
            diff = round(pos - neg, 1)
            insights.append(f"😊 <b>Overall Positivity:</b> Positive sentiment leads negative by <b>{diff} percentage points</b> "
                            f"({pos}% vs {neg}%), suggesting a generally favourable social media climate in this dataset.")
        elif neg > pos:
            diff = round(neg - pos, 1)
            insights.append(f"😟 <b>Negativity Spike:</b> Negative sentiment outweighs positive by <b>{diff} percentage points</b> "
                            f"({neg}% vs {pos}%). This may indicate underlying social tensions or crisis events during the period.")
        else:
            insights.append("⚖️ <b>Balanced Opinions:</b> Positive and negative sentiments are roughly equal, "
                            "reflecting a polarised but balanced discussion.")

    if country_col:
        top_country = df[country_col].value_counts().idxmax()
        top_count   = df[country_col].value_counts().max()
        insights.append(f"🌍 <b>Most Active Country:</b> <b>{top_country}</b> leads in social media activity "
                        f"with <b>{top_count:,}</b> posts — the highest contributor in the dataset.")

        if sent_col:
            pos_by_country = (
                df[df[sent_col] == "Positive"][country_col].value_counts() /
                df[country_col].value_counts()
            ).dropna().sort_values(ascending=False)
            if not pos_by_country.empty:
                best = pos_by_country.idxmax()
                best_pct = round(pos_by_country.max() * 100, 1)
                insights.append(f"😍 <b>Happiest Country:</b> <b>{best}</b> has the highest proportion of "
                                f"Positive sentiment at <b>{best_pct}%</b> of its total posts.")

    if "Platform" in df.columns:
        top_plat = df["Platform"].value_counts().idxmax()
        plat_pct = round(df["Platform"].value_counts().max() / len(df) * 100, 1)
        insights.append(f"📱 <b>Leading Platform:</b> <b>{top_plat}</b> dominates with <b>{plat_pct}%</b> of all posts, "
                        f"making it the primary platform for the discussions captured in this dataset.")

    if "Part_of_Day" in df.columns and sent_col:
        pod_neg = df[df[sent_col] == "Negative"]["Part_of_Day"].value_counts()
        if not pod_neg.empty:
            worst_pod = pod_neg.idxmax()
            insights.append(f"🌙 <b>Peak Negativity Window:</b> Negative posts peak during the <b>{worst_pod}</b>, "
                            f"which may correlate with news cycles, work stress, or online traffic patterns during that time of day.")

    if "Likes" in df.columns and sent_col:
        avg_likes = df.groupby(sent_col)["Likes"].mean().round(1)
        if not avg_likes.empty:
            top_sent = avg_likes.idxmax()
            top_avg  = avg_likes.max()
            insights.append(f"❤️ <b>Most Engaging Sentiment:</b> <b>{top_sent}</b> posts receive the highest average "
                            f"likes (<b>{top_avg}</b>), suggesting audiences engage more with {top_sent.lower()} content.")

    # ── Render insight cards ──────────────────────────────────────
    if not insights:
        st.info("Not enough data to generate insights. Try broadening your filters.")
        return

    for i, ins in enumerate(insights, 1):
        st.markdown(f"<div class='insight-card'><b>#{i}</b> &nbsp; {ins}</div>",
                    unsafe_allow_html=True)

    # ── Quick summary table ───────────────────────────────────────
    if sent_col and country_col:
        st.markdown("---")
        st.markdown("#### 📋 Sentiment Summary by Country")
        pivot = df.groupby([country_col, sent_col]).size().unstack(fill_value=0)
        pivot["Total"]   = pivot.sum(axis=1)
        pivot = pivot.sort_values("Total", ascending=False).head(12)
        st.dataframe(pivot.style.background_gradient(cmap="Blues", subset=["Total"]),
                     use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    # ── Default dataset path ─────────────────────────────────────
    DEFAULT_FILE = "final_data_set.csv"   # place in same folder as app.py

    # ── Sidebar (returns uploaded file if any) ────────────────────
    # We need raw df first for sidebar filters; reload after upload
    @st.cache_data(show_spinner=False)
    def load_default():
        import os, pathlib
        # Try CSV first (uploaded as CSV), then XLSX
        for path in [DEFAULT_FILE,
                     pathlib.Path(__file__).parent / DEFAULT_FILE,
                     "final_data_set.xlsx"]:
            try:
                return load_data(str(path))
            except Exception:
                pass
        return pd.DataFrame()   # empty fallback

    # Initial load
    df_raw = load_default()

    # Sidebar (may return a newly uploaded file)
    page, sel_countries, sel_sents, date_range, uploaded = sidebar(df_raw)

    # If user uploaded a file, reload from it
    if uploaded is not None:
        with st.spinner("Loading your file…"):
            try:
                df_raw = load_data(uploaded)
                st.sidebar.success(f"✅ Loaded {uploaded.name}")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")

    if df_raw.empty:
        st.warning("⚠️ Could not load the default dataset. Please upload a file via the sidebar.")
        return

    # Apply sidebar filters
    df = apply_filters(df_raw, sel_countries, sel_sents, date_range)

    # ── Route to selected page ────────────────────────────────────
    if   "Home"     in page: page_home(df_raw)          # home always uses full df for context
    elif "Overview" in page: page_data_overview(df_raw) # overview on raw (pre-filter) data
    elif "Dashboard" in page: page_dashboard(df)
    elif "Advanced" in page: page_advanced(df)
    elif "Insights" in page: page_insights(df)


if __name__ == "__main__":
    main()
