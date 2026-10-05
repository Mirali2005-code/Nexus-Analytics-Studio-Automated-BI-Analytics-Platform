import io
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

class ReportGenerator:
    """
    Generates presentation-grade executive reports in standalone interactive HTML
    and formatted PDF formats.
    """

    @staticmethod
    def generate_html_report(
        df: pd.DataFrame,
        filter_summary: str,
        kpi_data: Dict[str, Any],
        ai_insights: Dict[str, Any],
        charts: List[go.Figure]
    ) -> str:
        """
        Builds a single-file, highly polished HTML report with embedded styles,
        interactive Plotly charts, KPI cards, and strategic insights.
        """
        timestamp = datetime.now().strftime("%B %d, %Y - %H:%M")
        
        # Build KPI HTML cards
        kpi_cards_html = ""
        num_summaries = kpi_data.get("numeric_summaries", {})
        
        # Add basic dimensions card
        kpi_cards_html += f"""
        <div class="kpi-card">
            <div class="kpi-label">Analyzed Records</div>
            <div class="kpi-value">{len(df):,}</div>
            <div class="kpi-sub">Total Features: {len(df.columns)}</div>
        </div>
        """

        for col, stats in list(num_summaries.items())[:3]:
            kpi_cards_html += f"""
            <div class="kpi-card">
                <div class="kpi-label">Total {col.replace('_', ' ').title()}</div>
                <div class="kpi-value">{stats['sum']:,.2f}</div>
                <div class="kpi-sub">Avg: {stats['mean']:,.2f} | Med: {stats['median']:,.2f}</div>
            </div>
            """

        # Build Insights HTML
        findings_html = "".join([f"<li>{item}</li>" for item in ai_insights.get("key_findings", [])])
        risks_html = "".join([f"<li>{item}</li>" for item in ai_insights.get("anomalies_or_risks", [])])
        recs_html = "".join([f"<li>{item}</li>" for item in ai_insights.get("strategic_recommendations", [])])

        # Embed charts HTML
        charts_html = ""
        for i, fig in enumerate(charts):
            chart_div = fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
            charts_html += f"""
            <div class="chart-wrapper">
                {chart_div}
            </div>
            """

        # Sample preview table
        table_html = df.head(10).to_html(classes="report-table", index=False, border=0)

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executive Analytics Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0F172A;
            --surface-color: #1E293B;
            --surface-elevated: #334155;
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
            --accent-primary: #6366F1;
            --accent-emerald: #10B981;
            --accent-rose: #F43F5E;
            --border-color: #334155;
        }}
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        body {{
            background-color: var(--bg-color);
            color: var(--text-primary);
            padding: 40px 24px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 32px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }}
        .header-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 16px;
            margin-bottom: 16px;
        }}
        .badge {{
            background: rgba(99, 102, 241, 0.2);
            color: #818CF8;
            border: 1px solid #6366F1;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        h1 {{
            font-size: 2.2rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 8px;
        }}
        .context-tag {{
            color: #A5B4FC;
            font-size: 0.95rem;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
            margin-bottom: 32px;
        }}
        .kpi-card {{
            background-color: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease;
        }}
        .kpi-label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 4px;
        }}
        .kpi-sub {{
            font-size: 0.85rem;
            color: var(--accent-emerald);
        }}
        .section-card {{
            background-color: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 28px;
            margin-bottom: 32px;
        }}
        .section-title {{
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 10px;
            color: #FFFFFF;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
        }}
        .insights-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }}
        @media (max-width: 768px) {{
            .insights-grid {{ grid-template-columns: 1fr; }}
        }}
        .insight-box {{
            background: rgba(15, 23, 42, 0.6);
            border-radius: 10px;
            padding: 20px;
            border-left: 4px solid var(--accent-primary);
        }}
        .insight-box.risk {{
            border-left-color: var(--accent-rose);
        }}
        .insight-box.action {{
            border-left-color: var(--accent-emerald);
        }}
        .insight-box h3 {{
            font-size: 1rem;
            margin-bottom: 12px;
            color: #FFFFFF;
        }}
        ul {{
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 8px;
            color: var(--text-secondary);
            font-size: 0.95rem;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 28px;
            margin-bottom: 32px;
        }}
        .chart-wrapper {{
            background-color: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 20px;
            overflow: hidden;
        }}
        .report-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            text-align: left;
        }}
        .report-table th {{
            background-color: var(--surface-elevated);
            color: var(--text-primary);
            padding: 12px 16px;
            font-weight: 600;
        }}
        .report-table td {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-secondary);
        }}
        .report-table tr:hover td {{
            background-color: rgba(99, 102, 241, 0.05);
            color: var(--text-primary);
        }}
        footer {{
            text-align: center;
            padding: 24px;
            color: var(--text-secondary);
            font-size: 0.85rem;
            border-top: 1px solid var(--border-color);
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-top">
                <span class="badge">Executive Intelligence</span>
                <span style="color: #94A3B8; font-size: 0.9rem;">Generated: {timestamp}</span>
            </div>
            <h1>Automated Analytics & EDA Report</h1>
            <div class="context-tag">Active Slice: {filter_summary}</div>
        </header>

        <div class="kpi-grid">
            {kpi_cards_html}
        </div>

        <section class="section-card">
            <h2 class="section-title">Strategic Insights & Executive Summary</h2>
            <p style="font-size: 1.05rem; margin-bottom: 24px; color: #E2E8F0;">
                {ai_insights.get("executive_summary", "")}
            </p>
            
            <div class="insights-grid">
                <div class="insight-box">
                    <h3>Key Statistical Findings</h3>
                    <ul>{findings_html}</ul>
                </div>
                <div class="insight-box risk">
                    <h3>Identified Vulnerabilities & Risks</h3>
                    <ul>{risks_html}</ul>
                </div>
            </div>

            <div style="margin-top: 24px;" class="insight-box action">
                <h3>Recommended Action Plan</h3>
                <ul>{recs_html}</ul>
            </div>
        </section>

        <section class="chart-grid">
            {charts_html}
        </section>

        <section class="section-card">
            <h2 class="section-title">Data Sample Inspection (Top 10 Rows)</h2>
            <div style="overflow-x: auto;">
                {table_html}
            </div>
        </section>

        <footer>
            Prepared by Antigravity Smart Analytics Engine &bull; Confidential Business Intelligence
        </footer>
    </div>
</body>
</html>
"""
        return html_template

    @staticmethod
    def generate_pdf_report(
        df: pd.DataFrame,
        filter_summary: str,
        kpi_data: Dict[str, Any],
        ai_insights: Dict[str, Any]
    ) -> bytes:
        """
        Creates a clean, downloadable PDF report using ReportLab.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor('#1E1B4B'),
            fontName='Helvetica-Bold',
            spaceAfter=6
        )
        subtitle_style = ParagraphStyle(
            'ReportSubtitle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#64748B'),
            spaceAfter=14
        )
        h2_style = ParagraphStyle(
            'SectionH2',
            parent=styles['Heading2'],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor('#312E81'),
            fontName='Helvetica-Bold',
            spaceBefore=12,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            'ReportBody',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#334155'),
            spaceAfter=8
        )
        bullet_style = ParagraphStyle(
            'ReportBullet',
            parent=styles['Normal'],
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor('#1E293B'),
            leftIndent=15,
            spaceAfter=4
        )

        story = []

        # Header
        story.append(Paragraph("EXECUTIVE ANALYTICS & EDA REPORT", title_style))
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} | Context: {filter_summary}", subtitle_style))
        story.append(Spacer(1, 10))

        # KPI Table
        num_summaries = kpi_data.get("numeric_summaries", {})
        kpi_rows = [["Metric", "Value", "Key Statistics"]]
        kpi_rows.append(["Total Filtered Rows", f"{len(df):,}", f"Columns: {len(df.columns)}"])
        
        for col, stats in list(num_summaries.items())[:3]:
            kpi_rows.append([
                col.replace("_", " ").title(),
                f"{stats['sum']:,.2f}",
                f"Mean: {stats['mean']:,.2f} | Med: {stats['median']:,.2f}"
            ])

        kpi_table = Table(kpi_rows, colWidths=[150, 120, 260])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F46E5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8FAFC')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 14))

        # Executive Summary
        story.append(Paragraph("1. Executive Summary", h2_style))
        story.append(Paragraph(ai_insights.get("executive_summary", "No summary available."), body_style))
        story.append(Spacer(1, 10))

        # Key Findings
        story.append(Paragraph("2. Key Empirical Findings", h2_style))
        for finding in ai_insights.get("key_findings", []):
            story.append(Paragraph(f"&bull; {finding}", bullet_style))
        story.append(Spacer(1, 10))

        # Risks & Anomalies
        story.append(Paragraph("3. Vulnerabilities & Concentration Risks", h2_style))
        for risk in ai_insights.get("anomalies_or_risks", []):
            story.append(Paragraph(f"&bull; {risk}", bullet_style))
        story.append(Spacer(1, 10))

        # Action Recommendations
        story.append(Paragraph("4. Strategic Recommendations", h2_style))
        for rec in ai_insights.get("strategic_recommendations", []):
            story.append(Paragraph(f"&bull; {rec}", bullet_style))
        story.append(Spacer(1, 14))

        # Build document
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes
