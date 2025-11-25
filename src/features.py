from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Callable, Sequence, Dict, Optional  # <-- add Optional here

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

        # here we will store global means (rate 0..1)
        self.first_visit_mu_global: float | None = None
        self.delivery_mu_global: float | None = None  # <-- still available

        # Registry now stores FeatureConfig instead of bare functions
        self._registry: Dict[str, FeatureConfig] = {
            "first_visit": FeatureConfig(
                func=self._feature_first_visit,
                criteria_type="benefit",
            ),
            "delivery": FeatureConfig(  # uses "Estado"
                func=self._feature_delivery,
                criteria_type="benefit",
            ),
            "cost": FeatureConfig(
                func=self._feature_cost,
                criteria_type="cost",
            ),
            "coverage": FeatureConfig(
                func=self._feature_coverage,
                criteria_type="benefit",
            ),
            "sla": FeatureConfig(              
                func=self._feature_sla,
                criteria_type="benefit",
            ),
            "speed": FeatureConfig(
                func=self._feature_speed,
                criteria_type="cost",   # smaller delivery window = better
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
        features: Optional[Sequence[str]] = None,
        drop_incomplete: bool = True,
        update_first_visit_mu_global: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Build the requested feature set.

        If `features` is None, all registered features are computed.

        If update_first_visit_mu_global is True and 'first_visit' is in the
        selected features, we compute the GLOBAL mean first_visit rate (0..1)
        over ALL provincias and store it in self.first_visit_mu_global.
        """
        # ---- if no features are passed, use all registered ones ----
        if features is None:
            features = list(self._registry.keys())
        else:
            features = list(features)

        # --- compute global μ_first_visit if requested ---
        if update_first_visit_mu_global and "first_visit" in features:
            # We deliberately ignore provincia / cp filters here → global.
            s_global = self._feature_first_visit(
                df,
                provincia=None,
                location="both",
                codigo_postal=None,
                rango_peso=None,
            )
            r_global = (s_global.dropna() / 100.0).clip(0.0, 1.0)
            self.first_visit_mu_global = (
                float(r_global.mean()) if not r_global.empty else 0.8
            )

        # ---------- existing logic below ----------
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
            critical = {"cost", "cost_abs", "cost_actual", "coverage", "sla"}
            cols_to_check = [c for c in out.columns if c in critical]

            if cols_to_check:
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

        # keep only valid states
        df2 = df2[df2["Estado 1era Visita"].isin(["delivered", "not_delivered"])]

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

    def _feature_delivery(
        self,
        df: pd.DataFrame,
        provincia: str | None = None,
        location: str = "both",
        codigo_postal: str | None = None,
        rango_peso: str | None = None,
        **_,
    ) -> pd.Series:
        """
        Igual que first_visit pero usando la columna 'Estado'
        (delivered / not_delivered) → % entregas finales.
        """
        df2 = df.copy()
        df2 = df2[
            [
                "Correo",
                "Estado",
                "Provincia",
                "Codigo Postal",
                "Capital/Interior",
                "Rango de Peso",
            ]
        ]

        # keep only valid states
        df2 = df2[df2["Estado"].isin(["delivered", "not_delivered"])]

        df2["Estado"] = df2["Estado"].map(
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
                out[prov_short] = rows["Estado"].mean() * 100
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

    def _feature_sla(
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
                "Provincia",
                "Codigo Postal",
                "Capital/Interior",
                "Rango de Peso",
                "last_status_date",
                "minimum_delivery",
                "maximum_delivery",
            ]
        ]

        # Filters
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

        # Drop rows with any NaT dates
        date_cols = ["last_status_date", "minimum_delivery", "maximum_delivery"]
        df2 = df2[df2[date_cols].notna().all(axis=1)]

        # Scoring:
        # < minimum_delivery → ON TIME (1)
        # within window → ON TIME (1)
        # > maximum_delivery → LATE (0)
        df2["sla_ok"] = (
            df2["last_status_date"] <= df2["maximum_delivery"]
        ).astype(int)

        out = {}
        for prov_short in self.proveedores_short:
            correo_key = self.correo_keys[prov_short]
            rows = df2[df2["Correo"] == correo_key]
            if len(rows) == 0:
                out[prov_short] = np.nan
            else:
                out[prov_short] = rows["sla_ok"].mean() * 100.0

        return pd.Series(out)
    
    def _feature_speed(
        self,
        df: pd.DataFrame,
        provincia: str | None = None,
        location: str = "both",
        codigo_postal: str | None = None,
        rango_peso: str | None = None,
        **_,
    ) -> pd.Series:
        """
        'Speed' feature: average delivery window length (maximum_delivery - minimum_delivery)
        for each proveedor, after applying the usual filters.

        Smaller values = faster service (shorter promised window), in *days*.
        """

        df2 = df.copy()
        df2 = df2[
            [
                "Correo",
                "Provincia",
                "Codigo Postal",
                "Capital/Interior",
                "Rango de Peso",
                "minimum_delivery",
                "maximum_delivery",
            ]
        ]

        # Filters
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

        # Need valid dates to compute a difference
        date_cols = ["minimum_delivery", "maximum_delivery"]
        df2 = df2[df2[date_cols].notna().all(axis=1)]

        # Length of the SLA window in *days*
        # (if some dirty row has maximum < minimum, treat as NaN)
        window_days = (df2["maximum_delivery"] - df2["minimum_delivery"]).dt.total_seconds() / (24 * 3600)
        #window_days = (df2["maximum_delivery"] - df2["minimum_delivery"]).dt.days
        window_days = window_days.mask(window_days < 0, np.nan)
        df2["sla_window_days"] = window_days

        out: dict[str, float] = {}
        for prov_short in self.proveedores_short:
            correo_key = self.correo_keys[prov_short]
            rows = df2[df2["Correo"] == correo_key]

            if len(rows) == 0:
                out[prov_short] = np.nan
            else:
                out[prov_short] = rows["sla_window_days"].mean()

        # name is optional but sometimes nice for debugging
        return pd.Series(out, name="speed")


