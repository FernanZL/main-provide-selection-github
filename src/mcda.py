"""
MCDA methods (WASPAS, VIKOR, TOPSIS, etc.) and the final recommender.
These should work on the df produced by FeatureBuilder.
"""

from __future__ import annotations

from typing import Sequence, Optional, TYPE_CHECKING

import pandas as pd
import numpy as np
from pymcdm.methods import WASPAS as PymWASPAS, VIKOR as PymVIKOR, TOPSIS as PymTOPSIS, WSM as PymWSM

if TYPE_CHECKING:
    from src.features import FeatureBuilder


class MCDAEngine:
    def __init__(self, methods=("weighted", "topsis", "vikor", "waspas"), first_visit_threshold: float = 80.0,):
        self.methods = set(methods)

        # Business rule: below this % first_visit is "bad".
        # Used as neutral imputation for missing first_visit values in MCDA.
        self.first_visit_threshold = float(first_visit_threshold)

        # Presets focused on (first_visit, cost, coverage)
        # You can tune these numbers later if you want.
        self.weight_presets: dict[str, dict[str, float]] = {
            # Everything equally important
            "balanced": {
                "first_visit": 1 / 3,
                "cost": 1 / 3,
                "coverage": 1 / 3,
            },
            # Similar to what you were using (0.5, 0.4, 0.1)
            "quality_over_cost": {
                "first_visit": 0.5,
                "cost": 0.4,
                "coverage": 0.1,
            },
            # Push cost harder, still keep some service and coverage
            "cost_focus": {
                "first_visit": 0.25,
                "cost": 0.55,
                "coverage": 0.20,
            },
            # Super strict on service quality, cost is secondary
            "service_first": {
                "first_visit": 0.6,
                "cost": 0.25,
                "coverage": 0.15,
            },
        }

    @property
    def available_weight_presets(self) -> list[str]:
        return list(self.weight_presets.keys())

    def score(
        self,
        df_metrics: pd.DataFrame,
        methods="waspas",
        weights: Optional[Sequence[float]] = None,
        criteria_types: Optional[Sequence[str]] = None,
        return_df: bool = False,
        sort: bool = True,
        avg: bool = False,
        *,
        features: Optional[Sequence[str]] = None,
        feature_builder: Optional["FeatureBuilder"] = None,
        weights_preset: Optional[str] = None,
    ):
        """
        Compute MCDA scores for the given DataFrame.

        Parameters
        ----------
        df_metrics : pd.DataFrame
            DataFrame with criteria columns (typically the output of FeatureBuilder.build).
        methods : str or sequence of str
            MCDA methods to apply (e.g. 'weighted', 'topsis', 'vikor', 'waspas').
        weights : sequence of float, optional
            Custom weights for each criterion (same order as the selected columns).
            Must sum to 1, otherwise an error is raised.
        criteria_types : sequence of {'benefit', 'cost'}, optional
            Type of each criterion. If None, and both `features` and
            `feature_builder` are provided, these are inferred from the
            FeatureBuilder. If still None, falls back to the heuristic
            "cost" if "cost" in column name, else "benefit".
        return_df : bool, default False
            If True, return the full DataFrame with original metrics + scores.
            If False, return only the first score Series.
        sort : bool, default True
            If True, sort by the (average or first) score in descending order.
        avg : bool, default False
            If True and multiple methods, add Score_avg (mean of Score_* columns).
        features : sequence of str, optional (keyword-only)
            Explicit list of feature/criteria column names to use (subset and/or ordering).
            If None, all df_metrics columns are used.
        feature_builder : FeatureBuilder, optional (keyword-only)
            FeatureBuilder instance used to infer criteria_types when criteria_types is None.
        weights_preset : str, optional (keyword-only)
            Name of a predefined weight preset (focused on first_visit, cost, coverage).
            Mutually exclusive with `weights`.
        """

        df_all = df_metrics.copy()

        # ---- which criteria columns to use (and in what order) ----
        if features is not None:
            cols = list(features)
            missing = [c for c in cols if c not in df_all.columns]
            if missing:
                raise ValueError(f"Columns {missing} not found in df_metrics")
            df = df_all[cols].copy()
        else:
            df = df_all.copy()
            cols = list(df.columns)

        # ---- normalize methods parameter ----
        if isinstance(methods, str):
            methods = [methods]
        methods = [m.lower() for m in methods]

        # ---- weights: custom (must sum to 1) or preset or equal ----
        if weights is not None and weights_preset is not None:
            raise ValueError("Provide either `weights` or `weights_preset`, not both.")

        if weights is not None:
            # manual weights – must match number of criteria and sum to 1
            weights_arr = np.array(weights, dtype=float)
            if len(weights_arr) != len(cols):
                raise ValueError(
                    f"weights length ({len(weights_arr)}) must match number of criteria ({len(cols)})."
                )
            if not np.isclose(weights_arr.sum(), 1.0, atol=1e-6):
                raise ValueError(
                    f"Sum of weights must be 1, got {weights_arr.sum():.6f}."
                )
        elif weights_preset is not None:
            if weights_preset not in self.weight_presets:
                raise ValueError(
                    f"Unknown weights_preset '{weights_preset}'. "
                    f"Available: {self.available_weight_presets}"
                )
            preset_map = self.weight_presets[weights_preset]
            # Map preset weights by column name; missing columns → 0
            base = np.array([preset_map.get(c, 0.0) for c in cols], dtype=float)
            if base.sum() <= 0:
                raise ValueError(
                    f"Preset '{weights_preset}' produced all-zero weights "
                    f"for the selected criteria {cols}."
                )
            weights_arr = base / base.sum()
        else:
            # default: equal weights
            weights_arr = np.ones(len(cols), dtype=float)
            weights_arr = weights_arr / weights_arr.sum()  # trivially 1

        # ---- criteria_types: explicit, inferred from FeatureBuilder, or heuristic ----
        if criteria_types is None:
            if feature_builder is not None and features is not None:
                # infer from FeatureBuilder metadata
                criteria_types = feature_builder.criteria_types_for(features)
            else:
                # fallback: infer from column names (old behavior)
                criteria_types = tuple(
                    "cost" if "cost" in c.lower() else "benefit" for c in cols
                )
        else:
            criteria_types = tuple(criteria_types)

        if len(criteria_types) != len(cols):
            raise ValueError("criteria_types must match the selected criteria columns")

        # ---- numeric conversion ----
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # ---- drop criteria that are all NaN ----
        col_valid_mask = ~df.isna().all(axis=0)  # bool Series indexed by cols
        df = df.loc[:, col_valid_mask]
        cols = list(df.columns)

        if len(cols) == 0:
            # nothing to evaluate → return dummy NaN scores
            return pd.DataFrame(np.nan, index=df_metrics.index, columns=["Score"])

        # align weights and criteria_types to remaining columns
        mask = col_valid_mask.to_numpy()
        weights_arr = weights_arr[mask]
        criteria_types = tuple(ct for ct, m in zip(criteria_types, mask) if m)

        # renormalize after dropping all-NaN criteria (we keep proportions)
        if weights_arr.sum() <= 0:
            raise ValueError("After dropping NaN-only criteria, all weights became zero.")
        weights_arr = weights_arr / weights_arr.sum()

        # ---- run each MCDA method ----
        results = {}
        for m in methods:
            if m not in self.methods:
                raise ValueError(f"Unknown method '{m}'")
            func = getattr(self, f"_{m}")
            score = func(df, cols, weights_arr, criteria_types)
            results[f"Score_{m}"] = score

        # ---- build output ----
        df_out = pd.concat([df_metrics.copy()] + [v for v in results.values()], axis=1)

        if avg and len(results) > 1:
            df_out["Score_avg"] = pd.concat(results.values(), axis=1).mean(axis=1)

        sort_col = "Score_avg" if avg else list(results.keys())[0]
        if sort:
            df_out = df_out.sort_values(sort_col, ascending=False)

        return df_out if return_df else results[list(results.keys())[0]]

    def _impute_with_business_rule(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """
        Return df[cols] with NaNs imputed and first_visit transformed to [0, 1].

        - 'first_visit' (assumed in %: 0..100):
            * Convert to rate r in [0,1].
            * Piecewise transform with threshold t (e.g. 0.8):
                - r <= t: y = low_band_max * (r / t)
                - r  > t: y = low_band_max + (1 - low_band_max) * (r - t) / (1 - t)
            where low_band_max is the max score for the "bad" region (e.g. 0.2).
            * NaNs are filled with `nan_fill` (in [0,1], e.g. 0.6 = mid of the good region).

        - Other criteria:
            * NaNs filled with midpoint (min + max) / 2, or 0 if the column is fully missing.
        """
        data = df[cols].copy()

        # --- Parameters for first_visit transform ---
        thr_pct = float(self.first_visit_threshold)  # e.g. 80.0
        t = thr_pct / 100.0                          # threshold in [0,1], e.g. 0.8
        low_band_max = 0.2                           # "bad" region lives in [0, 0.2]
        #nan_fill = (low_band_max + 1.0) / 2.0        # mid of good region [0.2,1] -> 0.6
        nan_fill = 0.2

        for col in cols:
            vals = data[col]

            if col == "first_visit":
                valid = vals.dropna()
                if not valid.empty:
                    # 1) convert % -> rate in [0,1]
                    r = (valid / 100.0).clip(0.0, 1.0)

                    y = pd.Series(index=valid.index, dtype=float)

                    # bad region: r <= t -> [0, low_band_max]
                    mask_bad = r <= t
                    if mask_bad.any():
                        y.loc[mask_bad] = low_band_max * (r.loc[mask_bad] / max(t, 1e-9))

                    # good region: r > t -> [low_band_max, 1]
                    mask_good = r > t
                    if mask_good.any():
                        y.loc[mask_good] = (
                            low_band_max
                            + (1.0 - low_band_max)
                            * ((r.loc[mask_good] - t) / max(1.0 - t, 1e-9))
                        )

                    data.loc[valid.index, col] = y

                # 2) NaNs in first_visit -> business neutral in the *transformed* [0,1] scale
                data[col] = data[col].astype(float).fillna(nan_fill)

            else:
                # old logic: midpoint neutral imputation for other criteria
                valid = vals.dropna()
                if not valid.empty:
                    neutral = (valid.min() + valid.max()) / 2.0
                else:
                    neutral = 0.0
                data[col] = vals.fillna(neutral)

        return data




    # -------------- MCDA methods --------------

    def _topsis(self, df, cols, weights, criteria_types):
        """
        TOPSIS via pymcdm.

        Automatically handles:
        - NaN imputation using:
            * first_visit → threshold-normalized [0,1] + NaNs→0.5
            * others → midpoint neutral
        - Benefit/cost mapping
        - Constant-criteria removal (no variation)
        """
        data = self._impute_with_business_rule(df, cols)

        X = data.to_numpy(dtype=float)
        w = np.asarray(weights, dtype=float)
        type_map = {"benefit": 1, "cost": -1}
        types = np.array([type_map[ct] for ct in criteria_types], dtype=int)

        # (rest of your existing TOPSIS code unchanged)
        const_mask = np.std(X, axis=0) > 1e-12
        if const_mask.sum() == 0:
            score = pd.Series(np.ones(X.shape[0]), index=df.index, name="Score_topsis")
            return score
        if not const_mask.all():
            X = X[:, const_mask]
            w = w[const_mask]
            types = types[const_mask]
            w = w / w.sum() if w.sum() else np.ones_like(w) / len(w)

        method = PymTOPSIS()
        prefs = method(X, w, types, validation=False)
        score = pd.Series(prefs, index=df.index, name="Score_topsis")
        return score


    def _weighted(self, df, cols, weights, criteria_types):
        """
        Weighted Sum (WSM / SAW) via pymcdm.

        Uses:
        - first_visit → threshold-normalized [0,1] + NaNs→0.5
        - others → midpoint neutral
        """
        data = self._impute_with_business_rule(df, cols)

        X = data.to_numpy(dtype=float)
        w = np.asarray(weights, dtype=float)
        type_map = {"benefit": 1, "cost": -1}
        types = np.array([type_map[ct] for ct in criteria_types], dtype=int)

        const_mask = np.std(X, axis=0) > 1e-12
        if const_mask.sum() == 0:
            score = pd.Series(np.ones(X.shape[0]), index=df.index, name="Score_weighted")
            return score
        if not const_mask.all():
            X = X[:, const_mask]
            w = w[const_mask]
            types = types[const_mask]
            w = w / w.sum() if w.sum() else np.ones_like(w) / len(w)

        method = PymWSM()
        prefs = method(X, w, types, validation=False)
        score = pd.Series(prefs, index=df.index, name="Score_weighted")
        return score


    def _vikor(
        self,
        df,
        cols,
        weights,
        criteria_types,
        return_details: bool = False,
        explain: bool = False,
    ):
        """
        VIKOR via pymcdm, with:
        - first_visit → threshold-normalized [0,1] + NaNs→0.5
        - others → midpoint neutral
        - automatic removal of constant criteria.
        """
        data = self._impute_with_business_rule(df, cols)

        X = data.to_numpy(dtype=float)
        w = np.asarray(weights, dtype=float)

        type_map = {"benefit": 1, "cost": -1}
        types = np.array([type_map[ct] for ct in criteria_types], dtype=int)

        # constant criteria detection
        const_idx = []
        for j in range(X.shape[1]):
            col_vals = X[:, j]
            finite = ~np.isnan(col_vals)
            if finite.sum() == 0:
                const_idx.append(j)
            else:
                vals = col_vals[finite]
                if np.allclose(vals, vals[0]):
                    const_idx.append(j)

        if const_idx:
            mask = np.ones(X.shape[1], dtype=bool)
            mask[const_idx] = False

            if mask.sum() == 0:
                score = pd.Series(
                    np.ones(X.shape[0]),
                    index=df.index,
                    name="Score_vikor",
                )
                if return_details:
                    return {"score": score}
                return score

            X = X[:, mask]
            w = w[mask]
            types = types[mask]

            if w.sum() != 0:
                w = w / w.sum()
            else:
                w = np.ones_like(w, dtype=float) / len(w)

        method = PymVIKOR()
        prefs = method(X, w, types, validation=False, verbose=False)
        prefs = 1.0 - prefs  # flip so higher is better

        score = pd.Series(prefs, index=df.index, name="Score_vikor")

        if explain:
            print(
                "VIKOR computed via pymcdm.VIKOR (0→best flipped to 1→best). "
                "NaNs handled with threshold-normalized first_visit."
            )

        if return_details:
            return {"score": score}

        return score


    def _waspas(self, df, cols, weights, criteria_types):
        """
        WASPAS via pymcdm.

        Uses:
        - first_visit → threshold-normalized [0,1] + NaNs→0.5
        - others → midpoint neutral
        """
        data = self._impute_with_business_rule(df, cols)

        X = data.to_numpy(dtype=float)
        w = np.asarray(weights, dtype=float)

        type_map = {"benefit": 1, "cost": -1}
        types = np.array([type_map[ct] for ct in criteria_types], dtype=int)

        method = PymWASPAS()
        prefs = method(X, w, types, validation=True, verbose=False)

        score = pd.Series(prefs, index=df.index, name="Score_waspas")
        return score
