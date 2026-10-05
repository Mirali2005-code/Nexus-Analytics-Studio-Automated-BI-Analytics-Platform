import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional

# Consistent Modern Theme Palette
CHART_THEME = {
    "template": "plotly_dark",
    "paper_bgcolor": "#1E293B",
    "plot_bgcolor": "#0F172A",
    "font_color": "#F8FAFC",
    "accent_primary": "#6366F1",    # Indigo
    "accent_secondary": "#10B981",  # Emerald
    "accent_tertiary": "#06B6D4",   # Cyan
    "accent_coral": "#F43F5E",      # Rose
    "accent_amber": "#F59E0B",      # Amber
    "color_sequence": ["#6366F1", "#10B981", "#06B6D4", "#F43F5E", "#F59E0B", "#8B5CF6", "#EC4899"]
}

def style_fig(fig: go.Figure) -> go.Figure:
    """Applies modern executive dark styling to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=dict(color=CHART_THEME["font_color"], family="Inter, system-ui, sans-serif"),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(30, 41, 59, 0.6)"
        )
    )
    fig.update_xaxes(gridcolor="#334155", zerolinecolor="#334155")
    fig.update_yaxes(gridcolor="#334155", zerolinecolor="#334155")
    return fig

class EDAEngine:
    """
    Automated exploratory data analysis engine computing distributions,
    trends, correlations, category insights, and generating Plotly figures.
    """

    @staticmethod
    def get_kpis(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates headline metrics and dataset dimensions."""
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()

        kpis = {
            "total_records": len(df),
            "total_features": len(df.columns),
            "numeric_count": len(num_cols),
            "categorical_count": len(cat_cols),
            "date_count": len(date_cols),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "numeric_summaries": {}
        }

        # Look for target/money/count columns
        for col in num_cols[:6]: # Top 6 numeric columns
            s = df[col].dropna()
            if len(s) > 0:
                kpis["numeric_summaries"][col] = {
                    "sum": round(float(s.sum()), 2),
                    "mean": round(float(s.mean()), 2),
                    "median": round(float(s.median()), 2),
                    "std": round(float(s.std()), 2) if len(s) > 1 else 0.0,
                    "min": round(float(s.min()), 2),
                    "max": round(float(s.max()), 2)
                }

        return kpis

    @staticmethod
    def create_distribution_chart(df: pd.DataFrame, num_col: str) -> go.Figure:
        """Creates an enhanced histogram with boxplot overlay for distribution analysis."""
        fig = px.histogram(
            df,
            x=num_col,
            marginal="box",
            nbins=40,
            title=f"Distribution & Spread: {num_col}",
            color_discrete_sequence=[CHART_THEME["accent_primary"]]
        )
        return style_fig(fig)

    @staticmethod
    def create_categorical_chart(df: pd.DataFrame, cat_col: str, top_n: int = 10) -> go.Figure:
        """Creates a modern horizontal bar chart of category frequencies."""
        val_counts = df[cat_col].astype(str).value_counts().head(top_n).reset_index()
        val_counts.columns = [cat_col, "Count"]
        val_counts["Percentage"] = (val_counts["Count"] / len(df) * 100).round(1)

        fig = px.bar(
            val_counts,
            x="Count",
            y=cat_col,
            orientation='h',
            text=val_counts.apply(lambda r: f"{r['Count']:,} ({r['Percentage']}%)", axis=1),
            title=f"Top {top_n} Breakdown: {cat_col}",
            color="Count",
            color_continuous_scale=["#6366F1", "#10B981"]
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        return style_fig(fig)

    @staticmethod
    def create_timeseries_chart(
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        agg_func: str = "sum",
        freq: str = "D" # 'D', 'W', 'M'
    ) -> go.Figure:
        """Aggregates time series data and plots trend with moving average."""
        temp = df[[date_col, value_col]].copy().dropna()
        temp[date_col] = pd.to_datetime(temp[date_col])
        temp = temp.set_index(date_col)

        if agg_func == "sum":
            resampled = temp[value_col].resample(freq).sum().reset_index()
        elif agg_func == "mean":
            resampled = temp[value_col].resample(freq).mean().reset_index()
        else:
            resampled = temp[value_col].resample(freq).count().reset_index()

        resampled["7_period_ma"] = resampled[value_col].rolling(window=7, min_periods=1).mean()

        freq_labels = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
        label = freq_labels.get(freq, "Periodic")

        fig = go.Figure()
        # Bars or line for raw values
        fig.add_trace(go.Bar(
            x=resampled[date_col],
            y=resampled[value_col],
            name=f"{label} {agg_func.capitalize()}",
            marker_color="rgba(99, 102, 241, 0.4)",
            marker_line_color="#6366F1",
            marker_line_width=1
        ))
        # Smooth Trend Line
        fig.add_trace(go.Scatter(
            x=resampled[date_col],
            y=resampled["7_period_ma"],
            mode="lines",
            name="7-Period Moving Avg",
            line=dict(color="#10B981", width=3)
        ))

        fig.update_layout(
            title=f"{label} Trend: {value_col} ({agg_func.capitalize()})",
            xaxis_title="Date",
            yaxis_title=value_col
        )
        return style_fig(fig)

    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
        """Generates an interactive correlation heatmap of all numeric variables."""
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            return None

        corr = num_df.corr().round(2)
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Matrix Heatmap"
        )
        fig.update_layout(coloraxis_colorbar=dict(title="Correlation"))
        return style_fig(fig)

    @staticmethod
    def create_bivariate_scatter(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: Optional[str] = None
    ) -> go.Figure:
        """Scatter plot with trendline to explore relationships between variables."""
        sample_size = min(3000, len(df))
        sampled_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df

        fig = px.scatter(
            sampled_df,
            x=x_col,
            y=y_col,
            color=color_col if color_col and color_col in sampled_df.columns else None,
            title=f"Relationship: {x_col} vs {y_col}",
            opacity=0.7,
            color_discrete_sequence=CHART_THEME["color_sequence"]
        )
        return style_fig(fig)
