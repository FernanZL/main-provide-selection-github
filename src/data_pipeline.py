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

        if self.normalize_names and "Provincia" in cleaned_df.columns:
            cleaned_df["Provincia"] = self.normalize_provincias(cleaned_df["Provincia"])

        return cleaned_df

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
