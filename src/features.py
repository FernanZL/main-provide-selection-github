from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Callable, Sequence, Dict

from src.config import (
    PROVEEDORES_FULL,
    PROVEEDORES_SHORT,
    CORREO_KEYS,
)


@dataclass
class FeatureConfig:
    """
    Metadata for a feature:
    - func: function that computes the feature
    - criteria_type: 'benefit' or 'cost' for MCDA
    """
    func: Callable[..., pd.Series | pd.DataFrame]
    criteria_type: str  # "benefit" or "cost"


class FeatureBuilder:
    def __init__(self):
        self.proveedores_short = PROVEEDORES_SHORT
        self.correo_keys = CORREO_KEYS

        # Registry now stores FeatureConfig instead of bare functions
        self._registry: Dict[str, FeatureConfig] = {
            "first_visit": FeatureConfig(
                func=self._feature_first_visit,
                criteria_type="benefit",
            ),
            "cost": FeatureConfig(
                func=self._feature_cost,
                criteria_type="cost",
            ),
            "cost_abs": FeatureConfig(
                func=self._feature_cost_abs,
                criteria_type="cost",
            ),
            "cost_actual": FeatureConfig(
                func=self._feature_cost_actual,
                criteria_type="cost",
            ),
            "coverage": FeatureConfig(
                func=self._feature_coverage,
                criteria_type="benefit",
            ),
            "price_std": FeatureConfig(
                func=self._feature_price_std,
                criteria_type="cost",
            ),
            "cheap_ratio": FeatureConfig(
                func=self._feature_cheap_ratio,
                criteria_type="benefit",
            ),
        }

    @property
    def available_features(self) -> list[str]:
        return list(self._registry.keys())

    def criteria_types_for(self, features: Sequence[str]) -> tuple[str, ...]:
        """
        Return the criteria_types ('benefit' / 'cost') for the given
        features, in order.
        """
        types: list[str] = []
        for name in features:
            if name not in self._registry:
                raise ValueError(f"Unknown feature '{name}'")
            types.append(self._registry[name].criteria_type)
        return tuple(types)

    def build(
        self,
        df: pd.DataFrame,
        features: list[str],
        drop_incomplete: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Build the requested feature set.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned input dataset.
        features : list[str]
            Names of features to compute.
        drop_incomplete : bool, default True
            If True, drop providers (rows) where ANY of the
            cost/coverage-type features ('cost', 'cost_abs',
            'cost_actual', 'coverage') that are present in `features`
            is NaN or 0.
        **kwargs :
            Extra arguments forwarded to each feature function
            (provincia, location, codigo_postal, rango_peso, ...).
        """
        results = []
        for name in features:
            if name not in self._registry:
                raise ValueError(f"Unknown feature '{name}'")

            cfg = self._registry[name]
            feat = cfg.func(df, **kwargs)

            if isinstance(feat, pd.DataFrame):
                feat = feat.squeeze()

            feat = feat.reindex(self.proveedores_short)
            feat.name = name
            results.append(feat)

        out = pd.concat(results, axis=1)
        out.index.name = "Proveedor"

        if drop_incomplete:
            # Only check the columns that are both in the output
            # and in the critical cost/coverage set
            critical = {"cost", "cost_abs", "cost_actual", "coverage"}
            cols_to_check = [c for c in out.columns if c in critical]

            if cols_to_check:
                # keep rows where all critical cols are non-null AND non-zero
                mask = out[cols_to_check].notna().all(axis=1)
                mask &= (out[cols_to_check] != 0).all(axis=1)
                out = out[mask]

        return out

    # -------------------- features --------------------

    def _feature_first_visit(
        self,
        df: pd.DataFrame,
        provincia: str | None = None,
        location: str = "both",
        codigo_postal: str | None = None,
        rango_peso: str | None = None,
        **_,
    ) -> pd.Series:
        df2 = df.copy()
        df2 = df2[
            [
                "Correo",
                "Estado 1era Visita",
                "Provincia",
                "Codigo Postal",
                "Capital/Interior",
                "Rango de Peso",
            ]
        ]
        df2["Estado 1era Visita"] = df2["Estado 1era Visita"].map(
            {"delivered": 1, "not_delivered": 0}
        )

        if provincia:
            df2 = df2[df2["Provincia"] == provincia]
        if codigo_postal:
            df2 = df2[df2["Codigo Postal"] == codigo_postal]
        if rango_peso:
            df2 = df2[df2["Rango de Peso"] == rango_peso]
        if location == "capital":
            df2 = df2[df2["Capital/Interior"] == "CIUDAD"]
        elif location == "interior":
            df2 = df2[df2["Capital/Interior"] == "INTERIOR"]

        out = {}
        for prov_short in self.proveedores_short:
            correo_key = self.correo_keys[prov_short]
            rows = df2[df2["Correo"] == correo_key]
            if len(rows) == 0:
                out[prov_short] = np.nan
            else:
                out[prov_short] = rows["Estado 1era Visita"].mean() * 100
        return pd.Series(out)

    def _feature_cost(
        self,
        df: pd.DataFrame,
        provincia: str | None = None,
        location: str = "both",
        codigo_postal: str | None = None,
        rango_peso: str | None = None,
        **_,
    ) -> pd.Series:
        rename = dict(zip(PROVEEDORES_FULL, PROVEEDORES_SHORT))
        df2 = df.copy()

        if provincia:
            df2 = df2[df2["Provincia"] == provincia]
        if codigo_postal:
            df2 = df2[df2["Codigo Postal"] == codigo_postal]
        if rango_peso:
            df2 = df2[df2["Rango de Peso"] == rango_peso]
        if location == "capital":
            df2 = df2[df2["Capital/Interior"] == "CIUDAD"]
        elif location == "interior":
            df2 = df2[df2["Capital/Interior"] == "INTERIOR"]

        ratios = {}
        for full, short in rename.items():
            valid = df2[(df2["Peso"] > 0) & (df2[full] > 0)]
            ratios[short] = (valid[full] / valid["Peso"]).mean() if len(valid) else np.nan
        return pd.Series(ratios)

    def _feature_cost_abs(
        self,
        df: pd.DataFrame,
        provincia: str | None = None,
        location: str = "both",
        codigo_postal: str | None = None,
        rango_peso: str | None = None,
        **_,
    ) -> pd.Series:
        rename = dict(zip(PROVEEDORES_FULL, PROVEEDORES_SHORT))
        df2 = df.copy()

        if provincia:
            df2 = df2[df2["Provincia"] == provincia]
        if codigo_postal:
            df2 = df2[df2["Codigo Postal"] == codigo_postal]
        if rango_peso:
            df2 = df2[df2["Rango de Peso"] == rango_peso]
        if location == "capital":
            df2 = df2[df2["Capital/Interior"] == "CIUDAD"]
        elif location == "interior":
            df2 = df2[df2["Capital/Interior"] == "INTERIOR"]

        totals = {}
        for full, short in rename.items():
            valid = df2[df2[full] > 0]
            totals[short] = valid[full].mean() if len(valid) else np.nan
        return pd.Series(totals)

    def _feature_cost_actual(
        self,
        df: pd.DataFrame,
        provincia: str | None = None,
        location: str = "both",
        codigo_postal: str | None = None,
        rango_peso: str | None = None,
        **_,
    ) -> pd.Series:
        df2 = df.copy()

        if provincia:
            df2 = df2[df2["Provincia"] == provincia]
        if codigo_postal:
            df2 = df2[df2["Codigo Postal"] == codigo_postal]
        if rango_peso:
            df2 = df2[df2["Rango de Peso"] == rango_peso]
        if location == "capital":
            df2 = df2[df2["Capital/Interior"] == "CIUDAD"]
        elif location == "interior":
            df2 = df2[df2["Capital/Interior"] == "INTERIOR"]

        df2 = df2[df2["Peso"] > 0]

        out = {}
        for short, correo_key in self.correo_keys.items():
            rows = df2[df2["Correo"] == correo_key]
            out[short] = (
                (rows["Precio de Envio"] / rows["Peso"]).mean() if len(rows) else np.nan
            )
        return pd.Series(out)

    def _feature_coverage(
        self,
        df: pd.DataFrame,
        provincia: str | None = None,
        location: str = "both",
        codigo_postal: str | None = None,
        rango_peso: str | None = None,
        **_,
    ) -> pd.Series:
        rename = dict(zip(PROVEEDORES_FULL, PROVEEDORES_SHORT))
        df2 = df.copy()
        df2 = df2[
            ["Provincia", "Codigo Postal", "Capital/Interior", "Rango de Peso"]
            + PROVEEDORES_FULL
        ]

        if provincia:
            df2 = df2[df2["Provincia"] == provincia]
        if codigo_postal:
            df2 = df2[df2["Codigo Postal"] == codigo_postal]
        if rango_peso:
            df2 = df2[df2["Rango de Peso"] == rango_peso]
        if location == "capital":
            df2 = df2[df2["Capital/Interior"] == "CIUDAD"]
        elif location == "interior":
            df2 = df2[df2["Capital/Interior"] == "INTERIOR"]

        for c in PROVEEDORES_FULL:
            df2[c] = (df2[c] != 0).astype(int)

        if codigo_postal:
            if len(df2) == 0:
                return pd.Series({short: 0 for short in PROVEEDORES_SHORT})
            row = df2.iloc[0]
            return pd.Series({rename[c]: int(row[c]) for c in PROVEEDORES_FULL})

        total_cp = df2["Codigo Postal"].nunique()
        out = {}
        for full, short in rename.items():
            if total_cp == 0:
                out[short] = np.nan
            else:
                covered = df2[df2[full] == 1]["Codigo Postal"].nunique()
                out[short] = covered / total_cp * 100
        return pd.Series(out)

    def _feature_price_std(
        self,
        df: pd.DataFrame,
        provincia: str | None = None,
        location: str = "both",
        codigo_postal: str | None = None,
        rango_peso: str | None = None,
        **_,
    ) -> pd.Series:
        rename = dict(zip(PROVEEDORES_FULL, PROVEEDORES_SHORT))
        df2 = df.copy()

        if provincia:
            df2 = df2[df2["Provincia"] == provincia]
        if codigo_postal:
            df2 = df2[df2["Codigo Postal"] == codigo_postal]
        if rango_peso:
            df2 = df2[df2["Rango de Peso"] == rango_peso]
        if location == "capital":
            df2 = df2[df2["Capital/Interior"] == "CIUDAD"]
        elif location == "interior":
            df2 = df2[df2["Capital/Interior"] == "INTERIOR"]

        out = {}
        for full, short in rename.items():
            valid = df2[(df2["Peso"] > 0) & (df2[full] > 0)]
            if len(valid):
                price_per_kg = valid[full] / valid["Peso"]
                out[short] = price_per_kg.std()
            else:
                out[short] = np.nan
        return pd.Series(out)

    def _feature_cheap_ratio(
        self,
        df: pd.DataFrame,
        provincia: str | None = None,
        location: str = "both",
        codigo_postal: str | None = None,
        rango_peso: str | None = None,
        **_,
    ) -> pd.Series:
        df2 = df.copy()

        if provincia:
            df2 = df2[df2["Provincia"] == provincia]
        if codigo_postal:
            df2 = df2[df2["Codigo Postal"] == codigo_postal]
        if rango_peso:
            df2 = df2[df2["Rango de Peso"] == rango_peso]
        if location == "capital":
            df2 = df2[df2["Capital/Interior"] == "CIUDAD"]
        elif location == "interior":
            df2 = df2[df2["Capital/Interior"] == "INTERIOR"]

        # filter only rows with positive weight
        df2 = df2[df2["Peso"] > 0].copy()

        pres_cols = PROVEEDORES_FULL
        df2[pres_cols] = df2[pres_cols].replace(0, np.nan)
        row_mins = df2[pres_cols].min(axis=1)  # (kept in case you use it later)

        counts = {short: 0 for short in PROVEEDORES_SHORT}
        total = 0

        for _, row in df2.iterrows():
            total += 1
            this_min = row[pres_cols].min()
            if pd.isna(this_min):
                continue
            for full, short in zip(PROVEEDORES_FULL, PROVEEDORES_SHORT):
                if pd.notna(row[full]) and np.isclose(row[full], this_min):
                    counts[short] += 1

        if total == 0:
            return pd.Series({short: np.nan for short in PROVEEDORES_SHORT})

        return pd.Series({short: cnt / total * 100 for short, cnt in counts.items()})
