import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List

class DataCleaner:
    """
    Automated data hygiene engine that scores data quality,
    imputes missing values, eliminates duplicates, handles outliers,
    and produces an audit report.
    """

    @staticmethod
    def calculate_health_score(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates a comprehensive Data Health Score (0-100) and component metrics."""
        total_cells = df.size
        if total_cells == 0:
            return {"score": 0, "missing_pct": 100, "duplicate_pct": 100, "grade": "F"}

        missing_cells = df.isna().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100

        duplicate_rows = df.duplicated().sum()
        total_rows = len(df)
        duplicate_pct = (duplicate_rows / max(1, total_rows)) * 100

        # Score calculation: 100 minus weighted penalties
        score = 100.0 - (missing_pct * 1.5) - (duplicate_pct * 2.0)
        score = max(5.0, min(100.0, round(score, 1)))

        if score >= 90:
            grade = "A (Excellent)"
            color = "#10B981" # Emerald
        elif score >= 75:
            grade = "B (Good)"
            color = "#3B82F6" # Blue
        elif score >= 60:
            grade = "C (Fair)"
            color = "#F59E0B" # Amber
        else:
            grade = "D (Poor / Dirty)"
            color = "#EF4444" # Red

        return {
            "score": score,
            "grade": grade,
            "color": color,
            "total_rows": total_rows,
            "total_cols": len(df.columns),
            "missing_cells": int(missing_cells),
            "missing_pct": round(missing_pct, 2),
            "duplicate_rows": int(duplicate_rows),
            "duplicate_pct": round(duplicate_pct, 2),
        }

    @staticmethod
    def auto_clean(
        df: pd.DataFrame,
        remove_duplicates: bool = True,
        impute_missing: bool = True,
        trim_strings: bool = True,
        handle_outliers: bool = False,
        outlier_strategy: str = "clip" # "clip" or "none"
    ) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
        """
        Executes automated cleaning pipeline and generates an audit log.
        """
        cleaned_df = df.copy()
        audit_log = []

        # 1. Whitespace trimming
        if trim_strings:
            trimmed_cols = 0
            for col in cleaned_df.select_dtypes(include=['object', 'string']).columns:
                cleaned_df[col] = cleaned_df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                trimmed_cols += 1
            if trimmed_cols > 0:
                audit_log.append({
                    "action": "String Standardization",
                    "details": f"Trimmed leading/trailing whitespace across {trimmed_cols} text columns.",
                    "status": "Success"
                })

        # 2. Duplicate Removal
        if remove_duplicates:
            dup_count = cleaned_df.duplicated().sum()
            if dup_count > 0:
                cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
                audit_log.append({
                    "action": "Deduplication",
                    "details": f"Identified and purged {dup_count} exact duplicate rows ({round(dup_count / (len(df)) * 100, 2)}%).",
                    "status": "Success"
                })
            else:
                audit_log.append({
                    "action": "Deduplication",
                    "details": "Zero duplicate rows found. Dataset is unique.",
                    "status": "Clean"
                })

        # 3. Missing Value Imputation
        if impute_missing:
            imputed_details = []
            for col in cleaned_df.columns:
                null_count = cleaned_df[col].isna().sum()
                if null_count > 0:
                    pct = round(null_count / len(cleaned_df) * 100, 1)
                    if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        median_val = cleaned_df[col].median()
                        cleaned_df[col] = cleaned_df[col].fillna(median_val)
                        imputed_details.append(f"`{col}`: {null_count} nulls ({pct}%) filled with median ({round(median_val, 2) if isinstance(median_val, (int, float)) else median_val})")
                    elif pd.api.types.is_datetime64_any_dtype(cleaned_df[col]):
                        cleaned_df[col] = cleaned_df[col].ffill().bfill()
                        imputed_details.append(f"`{col}`: {null_count} nulls ({pct}%) forward/backward filled")
                    else:
                        # Categorical / string
                        mode_series = cleaned_df[col].mode()
                        mode_val = mode_series[0] if not mode_series.empty else "Unknown"
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                        imputed_details.append(f"`{col}`: {null_count} nulls ({pct}%) filled with mode ('{mode_val}')")

            if imputed_details:
                audit_log.append({
                    "action": "Missing Value Imputation",
                    "details": "; ".join(imputed_details),
                    "status": "Success"
                })
            else:
                audit_log.append({
                    "action": "Missing Value Check",
                    "details": "No missing values detected. All columns are 100% complete.",
                    "status": "Clean"
                })

        # 4. Outlier Handling (Optional clipping)
        if handle_outliers and outlier_strategy == "clip":
            outlier_actions = []
            num_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                q1 = cleaned_df[col].quantile(0.25)
                q3 = cleaned_df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower_bound = q1 - 2.5 * iqr
                    upper_bound = q3 + 2.5 * iqr
                    outliers = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)).sum()
                    if outliers > 0:
                        cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
                        outlier_actions.append(f"`{col}`: {outliers} extreme values clipped to [{round(lower_bound, 1)}, {round(upper_bound, 1)}]")

            if outlier_actions:
                audit_log.append({
                    "action": "Outlier Treatment (IQR Winsorization)",
                    "details": "; ".join(outlier_actions),
                    "status": "Success"
                })

        return cleaned_df, audit_log
