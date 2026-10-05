import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

class DataSlicer:
    """
    Handles dynamic data slicing, date-range filtering (e.g. 'Last 2 Months'),
    multi-attribute business condition filtering, and custom query execution.
    """

    @staticmethod
    def slice_by_timeframe(
        df: pd.DataFrame,
        date_col: str,
        timeframe_option: str,
        custom_start: Optional[datetime] = None,
        custom_end: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, str]:
        """
        Filters dataframe based on timeframe selection (e.g. 'Last 2 Months', 'Last 30 Days', etc.)
        """
        if date_col not in df.columns or timeframe_option == "All Time":
            return df, "All Time (Full Dataset)"

        # Ensure datetime format
        series = pd.to_datetime(df[date_col], errors='coerce')
        valid_dates = series.dropna()
        if len(valid_dates) == 0:
            return df, "No valid dates found in column"

        max_date = valid_dates.max()

        if timeframe_option == "Last 1 Month (30 Days)":
            start_date = max_date - timedelta(days=30)
            end_date = max_date
        elif timeframe_option == "Last 2 Months (60 Days)":
            start_date = max_date - timedelta(days=60)
            end_date = max_date
        elif timeframe_option == "Last 3 Months (90 Days)":
            start_date = max_date - timedelta(days=90)
            end_date = max_date
        elif timeframe_option == "Last 6 Months (180 Days)":
            start_date = max_date - timedelta(days=180)
            end_date = max_date
        elif timeframe_option == "Year to Date (YTD)":
            start_date = datetime(max_date.year, 1, 1)
            end_date = max_date
        elif timeframe_option == "Custom Range":
            if custom_start is None or custom_end is None:
                return df, "Custom Range (Incomplete range provided)"
            start_date = pd.to_datetime(custom_start)
            end_date = pd.to_datetime(custom_end) + timedelta(days=1) - timedelta(seconds=1)
        else:
            return df, "All Time"

        mask = (series >= start_date) & (series <= end_date)
        sliced_df = df[mask].copy()
        
        summary = (
            f"Filtered by [{date_col}]: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} "
            f"({len(sliced_df):,} of {len(df):,} rows, {round(len(sliced_df)/max(1, len(df))*100, 1)}%)"
        )
        return sliced_df, summary

    @staticmethod
    def apply_rules(
        df: pd.DataFrame,
        rules: List[Dict[str, Any]],
        logic: str = "AND"
    ) -> Tuple[pd.DataFrame, str]:
        """
        Applies a list of filter conditions:
        Each rule has:
          - 'column': column name
          - 'operator': '>', '>=', '<', '<=', '==', '!=', 'in', 'contains'
          - 'value': comparison value
        """
        if not rules or len(df) == 0:
            return df, "No custom rules applied"

        masks = []
        rule_descriptions = []

        for r in rules:
            col = r.get("column")
            op = r.get("operator")
            val = r.get("value")

            if col not in df.columns:
                continue

            series = df[col]
            mask = None

            try:
                if op == ">":
                    mask = series > float(val)
                elif op == ">=":
                    mask = series >= float(val)
                elif op == "<":
                    mask = series < float(val)
                elif op == "<=":
                    mask = series <= float(val)
                elif op == "==":
                    if pd.api.types.is_numeric_dtype(series):
                        mask = series == float(val)
                    else:
                        mask = series.astype(str) == str(val)
                elif op == "!=":
                    if pd.api.types.is_numeric_dtype(series):
                        mask = series != float(val)
                    else:
                        mask = series.astype(str) != str(val)
                elif op == "in":
                    # Expecting a list or comma-separated string
                    if isinstance(val, str):
                        items = [x.strip() for x in val.split(",") if x.strip()]
                    else:
                        items = list(val)
                    mask = series.isin(items)
                elif op == "contains":
                    mask = series.astype(str).str.contains(str(val), case=False, na=False)

                if mask is not None:
                    masks.append(mask)
                    rule_descriptions.append(f"`{col}` {op} `{val}`")
            except Exception as e:
                rule_descriptions.append(f"[Failed rule on `{col}`: {e}]")

        if not masks:
            return df, "No valid rules evaluated"

        if logic == "AND":
            final_mask = masks[0]
            for m in masks[1:]:
                final_mask = final_mask & m
        else: # OR
            final_mask = masks[0]
            for m in masks[1:]:
                final_mask = final_mask | m

        filtered_df = df[final_mask].copy()
        desc = f"Applied Rules ({logic}): " + f" {logic} ".join(rule_descriptions)
        desc += f" -> Matched {len(filtered_df):,} / {len(df):,} records ({round(len(filtered_df)/max(1, len(df))*100, 1)}%)"

        return filtered_df, desc
