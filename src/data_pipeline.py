# src/data_pipeline.py
"""
Data loading, basic cleaning, and exploratory stuff.
These are the "upstream" steps.
"""

from __future__ import annotations
import pandas as pd
import unicodedata
from src.config import (
    PROVEEDORES_FULL,
    PROVEEDORES_SHORT,
    CORREO_KEYS,
)


class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> pd.DataFrame:
        print(f"Loading data from: {self.path}")
        df = pd.read_csv(self.path)
        return df


class Preprocessor:
    """
    Handles dataset cleaning and normalization.
    """

    def __init__(self, normalize_names: bool = True):
        self.normalize_names = normalize_names

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters nulls and keeps only delivered / not_delivered for 1st visit."""
        cleaned_df = df.copy()
        cleaned_df = cleaned_df.dropna()
        cleaned_df = cleaned_df[
            cleaned_df["Estado 1era Visita"].isin(["delivered", "not_delivered"])
        ].reset_index(drop=True)
        cleaned_df = cleaned_df[
            cleaned_df["Estado"].isin(["delivered", "not_delivered"])
        ].reset_index(drop=True)

        if self.normalize_names and "Provincia" in cleaned_df.columns:
            cleaned_df["Provincia"] = self.normalize_provincias(cleaned_df["Provincia"])

        return cleaned_df

    def clean_with_extra_dates(
        self,
        df: pd.DataFrame,
        extra_csv_path: str | None = None,
        key_main: str = "Order ID",
        key_extra: str = "id",
    ) -> pd.DataFrame:
        """
        Clean the dataset with extra date columns.

        Two modes:
        1) Classic: main CSV + extra CSV (extra_csv_path not None)
        - Read extra CSV
        - Merge raw main df with extra on key_main / key_extra
        - Apply the same cleaning as `clean`
        - Convert extra date columns to datetime
        - Drop rows where ALL extra date columns are NaT
        - Drop the extra key column (key_extra)

        2) Already-merged: df already contains some/all of:
            last_status_date, minimum_delivery, maximum_delivery
        (key_extra may or may not be present)
        - Apply the same cleaning as `clean`
        - For each of those columns that exists: ensure datetime (convert if needed)
        - Drop rows where ALL present extra date columns are NaT
        - Drop key_extra if present (to match old behavior)

        If none of the extra date columns are present, falls back to basic `clean(df)`.
        """

        # The date columns we care about when "already merged"
        merged_date_cols = ["last_status_date", "minimum_delivery", "maximum_delivery"]

        # --- ALREADY-MERGED MODE (no extra_csv_path) ---
        if extra_csv_path is None:
            present_date_cols = [c for c in merged_date_cols if c in df.columns]

            if not present_date_cols:
                # No extra date columns at all → just do basic cleaning
                print(
                    "No extra CSV provided and merged date columns not found. "
                    "Falling back to basic `clean()`."
                )
                return self.clean(df)

            print(
                "No extra CSV provided. Detected merged date columns in df; "
                "running cleaning + enforcing datetime conversion."
            )

            # 1) Apply same cleaning rules as `clean`
            cleaned_df = self.clean(df)

            # 2) Ensure datetime for each present date col (convert only if needed)
            for col in present_date_cols:
                if not pd.api.types.is_datetime64_any_dtype(cleaned_df[col]):
                    cleaned_df[col] = pd.to_datetime(
                        cleaned_df[col].astype(str).str.strip(),
                        errors="coerce",
                        format="mixed",
                    )

            # 3) Verify datetime dtypes after conversion
            bad_cols = [
                c for c in present_date_cols
                if not pd.api.types.is_datetime64_any_dtype(cleaned_df[c])
            ]
            if bad_cols:
                raise ValueError(
                    f"The following columns could not be converted to datetime: {bad_cols}"
                )

            # 4) Drop rows where ALL present date cols are NaT
            all_nat = cleaned_df[present_date_cols].isna().all(axis=1)
            cleaned_df = cleaned_df[~all_nat].reset_index(drop=True)

            # 5) Drop key_extra if present (optional in this mode)
            if key_extra in cleaned_df.columns:
                cleaned_df = cleaned_df.drop(columns=[key_extra])

            return cleaned_df

        # --- CLASSIC MODE: main CSV + extra CSV ---
        print(f"Loading extra date data from: {extra_csv_path}")
        extra_df = pd.read_csv(extra_csv_path)

        if key_extra not in extra_df.columns:
            raise KeyError(
                f"Expected key column '{key_extra}' in extra CSV, "
                f"got columns: {list(extra_df.columns)}"
            )

        # All non-key columns in extra_df are considered date columns
        extra_date_cols = [c for c in extra_df.columns if c != key_extra]
        if not extra_date_cols:
            raise ValueError(
                f"Extra CSV at {extra_csv_path} only has the key column '{key_extra}'. "
                "Expected additional date columns."
            )

        # 2) Merge raw main df with extra df
        merged_df = df.merge(
            extra_df,
            how="inner",
            left_on=key_main,
            right_on=key_extra,
        )

        # 3) Apply the same cleaning rules as `clean`, but on the merged df
        cleaned_df = self.clean(merged_df)

        # 4) Convert extra date columns to datetime in the cleaned merged df
        for col in extra_date_cols:
            if col in cleaned_df.columns:
                if not pd.api.types.is_datetime64_any_dtype(cleaned_df[col]):
                    cleaned_df[col] = pd.to_datetime(
                        cleaned_df[col].astype(str).str.strip(),
                        errors="coerce",
                        format="mixed",
                    )

        # 5) Verify they really are datetime dtypes
        bad_cols = [
            c
            for c in extra_date_cols
            if c in cleaned_df.columns
            and not pd.api.types.is_datetime64_any_dtype(cleaned_df[c])
        ]
        if bad_cols:
            raise ValueError(
                f"The following columns could not be converted to datetime: {bad_cols}"
            )

        # 6) Drop rows where ALL of the extra date conversions failed (all NaT)
        valid_date_cols = [c for c in extra_date_cols if c in cleaned_df.columns]
        if valid_date_cols:
            all_nat = cleaned_df[valid_date_cols].isna().all(axis=1)
            cleaned_df = cleaned_df[~all_nat].reset_index(drop=True)

        # 7) Drop the extra key column (e.g. "id") after the merge & cleaning
        if key_extra in cleaned_df.columns:
            cleaned_df = cleaned_df.drop(columns=[key_extra])

        return cleaned_df




    @staticmethod
    def _normalize_key(series: pd.Series) -> pd.Series:
        """
        Normalize key columns so merges actually match:

        - cast to string
        - strip spaces
        - remove trailing '.0' (typical from float IDs like 12345.0)
        """
        s = series.astype(str).str.strip()
        s = s.str.replace(r"\.0$", "", regex=True)
        return s

    @staticmethod
    def normalize_provincias(series: pd.Series) -> pd.Series:
        """Normalize province names: lowercase, no accents, trimmed."""
        return (
            series.astype(str)
            .str.strip()
            .str.lower()
            .apply(
                lambda x: ''.join(
                    c for c in unicodedata.normalize('NFKD', x)
                    if not unicodedata.combining(c)
                )
            )
        )



# --- rest of the file (other classes) unchanged ---



class ExploratoryAnalysis:
    def __init__(self, df: pd.DataFrame):
        """Store the DataFrame for analysis."""
        self.df = df

    def peso_range_summary(self) -> pd.DataFrame:
        """
        Returns a DataFrame showing the min and max 'Peso' for each 'Rango de Peso' value.
        """
        summary = (
            self.df.groupby("Rango de Peso")["Peso"]
            .agg(Peso_min="min", Peso_max="max")
            .reset_index()
            .set_index("Rango de Peso")
        )
        return summary

    def avg_price_per_kg_by_rango(
        self,
        provincia: str = None,
        correos: list[str] | None = None,
        location: str = "both",
        codigo_postal: str | None = None,
        rango_peso: str | None = None,
    ) -> pd.DataFrame:
        """
        Returns, for each 'Rango de Peso', the average (Presupuesto / Peso) per proveedor,
        optionally filtered by province, location, CP, and/or a specific range.
        """
        df_filtered = self.df.copy()

        # --- Apply filters ---
        if provincia:
            df_filtered = df_filtered[df_filtered["Provincia"] == provincia]
        if codigo_postal:
            df_filtered = df_filtered[df_filtered["Codigo Postal"] == codigo_postal]
        if rango_peso:
            df_filtered = df_filtered[df_filtered["Rango de Peso"] == rango_peso]

        if location == "capital":
            df_filtered = df_filtered[df_filtered["Capital/Interior"] == "CIUDAD"]
        elif location == "interior":
            df_filtered = df_filtered[df_filtered["Capital/Interior"] == "INTERIOR"]
        # if 'both' → no extra filter

        # Determine selected columns
        rename_dict = dict(zip(PROVEEDORES_FULL, PROVEEDORES_SHORT))
        if correos:
            # user gave short names → map back to full
            selected_cols = [full for full, short in rename_dict.items() if short in correos]
        else:
            selected_cols = PROVEEDORES_FULL

        # keep only positive weight rows
        df_valid = df_filtered[df_filtered["Peso"] > 0].copy()

        # compute ratios
        for col in selected_cols:
            df_valid[col + "_ratio"] = df_valid[col] / df_valid["Peso"]

        # group
        grouped = df_valid.groupby("Rango de Peso").agg(
            Peso_min=("Peso", "min"),
            Peso_max=("Peso", "max"),
            **{rename_dict[col]: (col + "_ratio", "mean") for col in selected_cols},
        )

        # reorder
        output_cols = ["Peso_min", "Peso_max"] + [rename_dict[col] for col in selected_cols]
        grouped = grouped[output_cols]

        return grouped
