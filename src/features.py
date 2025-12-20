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
        self.delivery_mu_global: float | None = None

        # NEW: global feature matrix (for normalization)
        self.global_feature_df: pd.DataFrame | None = None

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
        drop_features_with_missing: bool = False,
        update_first_visit_mu_global: bool = True,
        cache_global: bool = False,   # <- if True: ALSO build & cache GLOBAL feature_df
        **kwargs,
    ) -> pd.DataFrame:
        """
        Build the requested feature set.

        If `features` is None, all registered features are computed.

        Semantics with `cache_global=True`:
        -----------------------------------
        In a *single* call, the builder will:
        - Compute a GLOBAL feature_df (ignoring provincia/CP/location filters),
        cache it in `self.global_feature_df`, and (optionally) update
        `first_visit_mu_global` / `delivery_mu_global`.
        - Compute and return the feature_df for the *requested filters*
        (provincia, location, codigo_postal, rango_peso, etc).

        This way, the UI can simply call:
            fb.build(cleaned_df, provincia=..., ..., cache_global=True)
        and does NOT need a separate "global build" call.
        """

        # ---- if no features are passed, use all registered ones ----
        if features is None:
            features = list(self._registry.keys())
        else:
            features = list(features)

        # Helper: drop providers if:
        # - any critical feature is NaN
        # - OR (cost == 0) OR (coverage == 0) when those columns exist
        def _apply_drop_incomplete_nan_all__zero_cost_coverage(out: pd.DataFrame) -> pd.DataFrame:
            critical = {"first_visit", "delivery", "cost", "coverage", "sla", "speed"}
            cols_to_check = [c for c in out.columns if c in critical]
            if not cols_to_check:
                return out

            # 1) Drop if ANY critical is NaN
            mask = out[cols_to_check].notna().all(axis=1)

            # 2) Additionally drop if cost/coverage are zero (only those two)
            if "cost" in cols_to_check:
                mask &= (out["cost"] != 0)
            if "coverage" in cols_to_check:
                mask &= (out["coverage"] != 0)

            return out.loc[mask].copy()

        # ================== 1) GLOBAL BUILD (cache_global=True) ==================
        if cache_global:
            # We ignore provincia / CP / location filters for the GLOBAL matrix.
            # Start from kwargs but force global filters.
            global_kwargs = dict(kwargs)
            global_kwargs.update(
                {
                    "provincia": None,
                    "codigo_postal": None,
                    "rango_peso": None,
                    "location": "both",
                }
            )

            global_results: list[pd.Series] = []
            for name in features:
                if name not in self._registry:
                    raise ValueError(f"Unknown feature '{name}'")

                cfg = self._registry[name]
                g_feat = cfg.func(df, **global_kwargs)

                if isinstance(g_feat, pd.DataFrame):
                    g_feat = g_feat.squeeze()

                g_feat = g_feat.reindex(self.proveedores_short)
                g_feat.name = name
                global_results.append(g_feat)

            global_out = pd.concat(global_results, axis=1)
            global_out.index.name = "Proveedor"

            # ---- drop entire features (columns) with missing values, if requested ----
            if drop_features_with_missing:
                critical = {"first_visit", "delivery", "cost", "coverage", "sla", "speed"}
                cols_to_check = [c for c in global_out.columns if c in critical]

                cols_to_keep: list[str] = []
                for col in cols_to_check:
                    col_values = global_out[col]
                    if col_values.notna().all():
                        cols_to_keep.append(col)

                non_critical = [c for c in global_out.columns if c not in critical]
                global_out = global_out[cols_to_keep + non_critical]

            # ---- drop incomplete providers (NEW RULE) ----
            if drop_incomplete:
                global_out = _apply_drop_incomplete_nan_all__zero_cost_coverage(global_out)

            # Cache global feature matrix
            self.global_feature_df = global_out

            # Optional: compute GLOBAL μs from global_out
            if update_first_visit_mu_global:
                if "first_visit" in global_out.columns:
                    r_global = (global_out["first_visit"].dropna() / 100.0).clip(0.0, 1.0)
                    self.first_visit_mu_global = (
                        float(r_global.mean()) if not r_global.empty else 0.8
                    )

                if "delivery" in global_out.columns:
                    r_global_del = (global_out["delivery"].dropna() / 100.0).clip(0.0, 1.0)
                    self.delivery_mu_global = (
                        float(r_global_del.mean()) if not r_global_del.empty else 0.8
                    )

        # ================== 2) FILTERED BUILD (what we return) ==================
        results: list[pd.Series] = []
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

        # ---- drop entire features (columns) with missing values, if requested ----
        if drop_features_with_missing:
            critical = {"first_visit", "delivery", "cost", "coverage", "sla", "speed"}
            cols_to_check = [c for c in out.columns if c in critical]

            cols_to_keep: list[str] = []
            for col in cols_to_check:
                col_values = out[col]
                if col_values.notna().all():
                    cols_to_keep.append(col)

            non_critical = [c for c in out.columns if c not in critical]
            out = out[cols_to_keep + non_critical]
            return out

        # ---- drop incomplete providers (NEW RULE) ----
        if drop_incomplete:
            out = _apply_drop_incomplete_nan_all__zero_cost_coverage(out)

        return out





    # -------------------- features --------------------

    def _feature_first_visit(
        self,
        df: pd.DataFrame,
        provincia: str | None = None,
        location: str = "both",
        codigo_postal: str | None = None,
        rango_peso: str | None = None,
        shrink_small_samples: bool = True,    # <--- NEW
        shrink_m: float = 100.0,              # <--- NEW (prior strength)
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

        # First compute raw rate (0..1) and count per proveedor
        rates: dict[str, float] = {}
        counts: dict[str, int] = {}

        for prov_short in self.proveedores_short:
            correo_key = self.correo_keys[prov_short]
            rows = df2[df2["Correo"] == correo_key]

            n = len(rows)
            counts[prov_short] = n

            if n == 0:
                rates[prov_short] = np.nan
            else:
                rates[prov_short] = float(rows["Estado 1era Visita"].mean())

        # If we don't shrink, just return the raw % values
        if not shrink_small_samples:
            out = {
                prov_short: (rates[prov_short] * 100.0 if not np.isnan(rates[prov_short]) else np.nan)
                for prov_short in self.proveedores_short
            }
            return pd.Series(out)

        # ---- Shrinkage: credibility-weighted towards slice mean ----
        # Global mean μ over this filtered slice
        num = 0.0
        den = 0
        for prov_short in self.proveedores_short:
            p = rates[prov_short]
            n = counts[prov_short]
            if n > 0 and not np.isnan(p):
                num += p * n
                den += n

        if den == 0:
            # no data at all after filters
            return pd.Series({prov_short: np.nan for prov_short in self.proveedores_short})

        mu = num / den  # slice mean in [0,1]

        out: dict[str, float] = {}
        for prov_short in self.proveedores_short:
            p = rates[prov_short]
            n = counts[prov_short]

            if n == 0 or np.isnan(p):
                out[prov_short] = np.nan
            else:
                # shrunk rate: μ + n/(n+m) * (p - μ)
                w = n / (n + shrink_m)
                adj = mu + w * (p - mu)
                out[prov_short] = adj * 100.0  # back to %
        return pd.Series(out)

    def _feature_delivery(
        self,
        df: pd.DataFrame,
        provincia: str | None = None,
        location: str = "both",
        codigo_postal: str | None = None,
        rango_peso: str | None = None,
        shrink_small_samples: bool = True,    # <--- NEW
        shrink_m: float = 100.0,              # <--- NEW #how many envios are a good parameter to trust the proveedor
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

        # Raw rates and counts per proveedor
        rates: dict[str, float] = {}
        counts: dict[str, int] = {}

        for prov_short in self.proveedores_short:
            correo_key = self.correo_keys[prov_short]
            rows = df2[df2["Correo"] == correo_key]

            n = len(rows)
            counts[prov_short] = n

            if n == 0:
                rates[prov_short] = np.nan
            else:
                rates[prov_short] = float(rows["Estado"].mean())

        if not shrink_small_samples:
            out = {
                prov_short: (rates[prov_short] * 100.0 if not np.isnan(rates[prov_short]) else np.nan)
                for prov_short in self.proveedores_short
            }
            return pd.Series(out)

        # Shrinkage towards slice mean
        num = 0.0
        den = 0
        for prov_short in self.proveedores_short:
            p = rates[prov_short]
            n = counts[prov_short]
            if n > 0 and not np.isnan(p):
                num += p * n
                den += n

        if den == 0:
            return pd.Series({prov_short: np.nan for prov_short in self.proveedores_short})

        mu = num / den

        out: dict[str, float] = {}
        for prov_short in self.proveedores_short:
            p = rates[prov_short]
            n = counts[prov_short]

            if n == 0 or np.isnan(p):
                out[prov_short] = np.nan
            else:
                w = n / (n + shrink_m)
                adj = mu + w * (p - mu)
                out[prov_short] = adj * 100.0
        return pd.Series(out)


    def _feature_cost(
        self,
        df: pd.DataFrame,
        provincia: str | None = None,
        location: str = "both",
        codigo_postal: str | None = None,
        rango_peso: str | None = None,
        weight_by_range: bool = True,        # existing flag
        cell_sum_if_cp_and_range: bool = True,  # <<< NEW FLAG
        **_,
    ) -> pd.Series:
        """
        Compute cost per proveedor.

        Special case:
        -------------
        If cell_sum_if_cp_and_range=True (default) AND both
        `codigo_postal` and `rango_peso` are provided:

            cost = sum( Presupuesto_x ) over the filtered df
                   (only rows where that provider has tariff > 0).

        This is intended for the "single cell" case (one CP + one Rango de Peso),
        where we just care about total pesos per provider in that slice.

        General case (no CP+range slice or flag disabled):
        -------------------------------------------------
        If weight_by_range=False:
            cost = mean( price / Peso ) over all valid shipments.

        If weight_by_range=True:
            1) Compute mean(price / Peso) per Rango de Peso and proveedor.
            2) Compute a global distribution of Rango de Peso (within the filtered df).
            3) For each proveedor, take a weighted average of its per-range costs
               using the global range weights (renormalized over ranges where that
               proveedor has data).
        """
        rename = dict(zip(PROVEEDORES_FULL, PROVEEDORES_SHORT))
        df2 = df.copy()

        # ---- Filters ----
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

        # -------------------------------
        # Special case: CP + Rango de Peso
        # -------------------------------
        # If we are exactly in a "cell" (one CP + one Rango de Peso) and the flag
        # is enabled, cost is just the total Presupuesto per provider.
        if (
            cell_sum_if_cp_and_range
            and codigo_postal is not None
            and rango_peso is not None
        ):
            if df2.empty:
                return pd.Series({short: np.nan for short in PROVEEDORES_SHORT})

            totals: dict[str, float] = {}
            for full, short in rename.items():
                if full not in df2.columns:
                    totals[short] = np.nan
                    continue
                valid = df2[df2[full] > 0]
                if len(valid) == 0:
                    totals[short] = np.nan
                else:
                    totals[short] = float(valid[full].sum())
            return pd.Series(totals)

        # ------------------------------------------------
        # General case: cost per kg (old logic, unchanged)
        # ------------------------------------------------

        # Only rows with positive weight are meaningful
        df2_valid = df2[df2["Peso"] > 0].copy()

        ratios: dict[str, float] = {}

        # ----------------------------
        # Simple behavior (old logic)
        # ----------------------------
        if not weight_by_range:
            for full, short in rename.items():
                if full not in df2_valid.columns:
                    ratios[short] = np.nan
                    continue
                valid = df2_valid[df2_valid[full] > 0]
                if len(valid) == 0:
                    ratios[short] = np.nan
                else:
                    ratios[short] = (valid[full] / valid["Peso"]).mean()
            return pd.Series(ratios)

        # ---------------------------------------
        # New behavior: weight by Rango de Peso
        # ---------------------------------------

        if len(df2_valid) == 0:
            # No shipments at all after filters
            return pd.Series({short: np.nan for short in PROVEEDORES_SHORT})

        # Global distribution of Rango de Peso (for weighting)
        # Same for all proveedores, within the filtered df.
        global_counts = df2_valid["Rango de Peso"].value_counts()
        global_weights = global_counts / global_counts.sum()

        for full, short in rename.items():
            if full not in df2_valid.columns:
                ratios[short] = np.nan
                continue

            prov_valid = df2_valid[df2_valid[full] > 0].copy()
            if len(prov_valid) == 0:
                ratios[short] = np.nan
                continue

            # Cost per kg for this proveedor
            prov_valid["unit_cost"] = prov_valid[full] / prov_valid["Peso"]

            # Mean unit cost per Rango de Peso
            mean_per_range = prov_valid.groupby("Rango de Peso")["unit_cost"].mean()

            # Only ranges where both global_weights and this proveedor have data
            common_ranges = global_weights.index.intersection(mean_per_range.index)

            if len(common_ranges) == 0:
                ratios[short] = np.nan
                continue

            # Renormalize weights over the common ranges
            w = global_weights[common_ranges]
            w = w / w.sum()

            c = mean_per_range[common_ranges]

            ratios[short] = float((w * c).sum())

        return pd.Series(ratios)


    def _feature_coverage(
        self,
        df: pd.DataFrame,
        provincia: str | None = None,
        location: str = "both",
        codigo_postal: str | None = None,
        rango_peso: str | None = None,
        by_shipments: bool = True,   # <<< NEW FLAG
        **_,
    ) -> pd.Series:
        """
        Compute coverage per proveedor.

        If by_shipments=False (default), coverage = % of distinct postal codes covered.
        If by_shipments=True,  coverage = % of shipments covered.
        """
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

        # Binarize proveedor columns
        for c in PROVEEDORES_FULL:
            df2[c] = (df2[c] != 0).astype(int)

        # === Special case: single CP or capital rule ===
        if codigo_postal:
            if len(df2) == 0:
                return pd.Series({short: 0 for short in PROVEEDORES_SHORT})
            row = df2.iloc[0]
            return pd.Series({rename[c]: int(row[c]) for c in PROVEEDORES_FULL})

        if location == "capital":
            if len(df2) == 0:
                return pd.Series({short: 0 for short in PROVEEDORES_SHORT})
            row = df2.iloc[0]
            return pd.Series({rename[c]: int(row[c]) for c in PROVEEDORES_FULL})

        # === COVERAGE CALCULATION ===
        out = {}

        if not by_shipments:
            # OLD behavior: unique postal codes
            total_cp = df2["Codigo Postal"].nunique()

            for full, short in rename.items():
                if total_cp == 0:
                    out[short] = np.nan
                else:
                    covered_cp = df2[df2[full] == 1]["Codigo Postal"].nunique()
                    out[short] = covered_cp / total_cp * 100

        else:
            # NEW behavior: shipments
            total_shipments = len(df2)

            for full, short in rename.items():
                if total_shipments == 0:
                    out[short] = np.nan
                else:
                    covered_shipments = (df2[full] == 1).sum()
                    out[short] = covered_shipments / total_shipments * 100

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


