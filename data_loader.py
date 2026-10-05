import os
import io
import pandas as pd
import numpy as np
import duckdb

class DataLoader:
    """
    Handles robust ingestion of datasets across various formats (CSV, Excel, Parquet, JSON),
    with intelligent type inference and memory optimization.
    """

    @staticmethod
    def load_file(uploaded_file) -> pd.DataFrame:
        """Loads data from a Streamlit UploadedFile object or file path."""
        if uploaded_file is None:
            return None
        
        name = getattr(uploaded_file, "name", str(uploaded_file))
        ext = os.path.splitext(name)[1].lower()

        if ext in [".csv", ".txt"]:
            # Try utf-8 first, fallback to latin1
            try:
                if hasattr(uploaded_file, "read"):
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                if hasattr(uploaded_file, "read"):
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding="latin1")
                else:
                    df = pd.read_csv(uploaded_file, encoding="latin1")
        elif ext in [".xlsx", ".xls"]:
            if hasattr(uploaded_file, "read"):
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        elif ext == ".parquet":
            if hasattr(uploaded_file, "read"):
                uploaded_file.seek(0)
                df = pd.read_parquet(uploaded_file)
            else:
                df = pd.read_parquet(uploaded_file)
        elif ext == ".json":
            if hasattr(uploaded_file, "read"):
                uploaded_file.seek(0)
                df = pd.read_json(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Please upload CSV, Excel, Parquet, or JSON.")

        return DataLoader.optimize_dataframe(df)

    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Detect and convert date columns, strip string whitespace, and optimize memory."""
        df = df.copy()

        # Clean string whitespace in column names
        df.columns = [str(col).strip() for col in df.columns]

        # Detect potential date columns
        for col in df.columns:
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                # Check column name hints
                col_lower = str(col).lower()
                is_date_name = any(k in col_lower for k in ["date", "time", "timestamp", "created_at", "updated_at", "order_date", "dob", "day"])
                
                # Check sample non-null values
                sample = df[col].dropna().head(20)
                if len(sample) > 0 and (is_date_name or DataLoader._is_mostly_dates(sample)):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except Exception:
                        pass

        return df

    @staticmethod
    def _is_mostly_dates(series_sample) -> bool:
        """Check if sample strings look like dates without crashing."""
        success_count = 0
        for val in series_sample:
            if not isinstance(val, str):
                continue
            if len(val) < 6 or len(val) > 35:
                continue
            # If contains hyphens or slashes typical of dates
            if ('-' in val or '/' in val) and any(char.isdigit() for char in val):
                try:
                    pd.to_datetime(val)
                    success_count += 1
                except Exception:
                    pass
        return (success_count / max(1, len(series_sample))) > 0.7

    @staticmethod
    def inspect_columns(df: pd.DataFrame):
        """Returns categorizations of columns: date, numeric, categorical, text/id."""
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in date_cols]
        cat_cols = []
        text_id_cols = []

        for c in df.columns:
            if c not in date_cols and c not in num_cols:
                nunique = df[c].nunique()
                total = len(df)
                # If low cardinality or ratio, treat as categorical
                if nunique <= 50 or (total > 0 and (nunique / total) < 0.05):
                    cat_cols.append(c)
                else:
                    text_id_cols.append(c)

        return {
            "date_cols": date_cols,
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
            "text_id_cols": text_id_cols
        }

    @staticmethod
    def query_with_duckdb(df: pd.DataFrame, sql_query: str) -> pd.DataFrame:
        """Execute ultra-fast SQL directly against the dataframe using DuckDB."""
        con = duckdb.connect(database=":memory:")
        con.register("dataset", df)
        result_df = con.execute(sql_query).df()
        con.close()
        return result_df
