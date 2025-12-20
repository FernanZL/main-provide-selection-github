"""
MCDA methods (WASPAS, VIKOR, TOPSIS, etc.) and the final recommender.
These should work on the df produced by FeatureBuilder.
"""

from __future__ import annotations

from typing import Sequence, Optional, TYPE_CHECKING

import pandas as pd
import numpy as np
from pymcdm.methods import WASPAS as PymWASPAS, VIKOR as PymVIKOR, TOPSIS as PymTOPSIS, WSM as PymWSM
from src.weight_presets import get_weight_presets
from sklearn.preprocessing import MinMaxScaler  # add at top of file

if TYPE_CHECKING:
    from src.features import FeatureBuilder


class MCDAEngine:
    def __init__(
        self,
        methods=("weighted", "topsis", "vikor", "waspas"),
        first_visit_threshold: float = 90.0,
        weight_presets: dict[str, dict[str, float]] | None = None,
    ):
        self.methods = set(methods)

        # Business rule: threshold, etc.
        self.first_visit_threshold = first_visit_threshold

        # Load presets from config file unless caller overrides
        if weight_presets is None:
            self.weight_presets = get_weight_presets()
        else:
            # Make a defensive copy so external code doesn't mutate internal state
            self.weight_presets = {
                name: p.copy() for name, p in weight_presets.items()
            }

        # DEBUG: last feature matrices used internally
        self.debug_last_imputed: pd.DataFrame | None = None
        self.debug_last_scaled: pd.DataFrame | None = None

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
        smooth_approximation: bool = False,  # <--- NEW FLAG
        normalize_globally: bool = False, 
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
            FeatureBuilder instance used to infer criteria_types and read
            global μ for first_visit / delivery when doing business-rule imputation.
        weights_preset : str, optional (keyword-only)
            Name of a predefined weight preset (focused on the 6 features).
            Mutually exclusive with `weights`.
        smooth_approximation : bool, optional (keyword-only)
            If True, use cached cubic smooth approximation for first_visit / delivery
            business-rule transform (no kink). If False (default), use the original
            piecewise-linear soft-ReLU with kink.
        """

        df_all = df_metrics.copy()

        # ---- which criteria columns to use (and in what order) ----
        if features is not None:
            # <<< NEW: intersect requested features with columns that actually exist
            requested = list(features)
            cols = [c for c in requested if c in df_all.columns]
            if not cols:
                raise ValueError(
                    f"None of the requested features {requested} are present "
                    f"in df_metrics columns {list(df_all.columns)}."
                )
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
            # manual weights – must match number of *current* criteria (cols)
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
            # (columns that didn't survive earlier dropping simply don't appear in `cols`)
            base = np.array([preset_map.get(c, 0.0) for c in cols], dtype=float)
            if base.sum() <= 0:
                raise ValueError(
                    f"Preset '{weights_preset}' produced all-zero weights "
                    f"for the selected criteria {cols}."
                )
            weights_arr = base / base.sum()   # <<< NEW: normalized for surviving cols
        else:
            # default: equal weights over the surviving cols
            weights_arr = np.ones(len(cols), dtype=float)
            weights_arr = weights_arr / weights_arr.sum()

        # ---- criteria_types: explicit, inferred from FeatureBuilder, or heuristic ----
        if criteria_types is None:
            if feature_builder is not None and features is not None:
                # infer from FeatureBuilder metadata, but only for columns that survived
                criteria_types = feature_builder.criteria_types_for(cols)  # <<< NEW: use `cols`
            else:
                # fallback: infer from column names (old behavior)
                criteria_types = []
                for c in cols:
                    name = c.lower()
                    if "cost" in name or name == "speed":
                        criteria_types.append("cost")
                    else:
                        criteria_types.append("benefit")
                criteria_types = tuple(criteria_types)

        else:
            # we trust the caller, but they must match surviving columns
            if len(criteria_types) != len(cols):
                raise ValueError(
                    "criteria_types must match the selected criteria columns "
                    f"(got {len(criteria_types)}, expected {len(cols)})."
                )
            criteria_types = tuple(criteria_types)

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

        # renormalize after dropping NaN-only criteria (we keep proportions)
        if weights_arr.sum() <= 0:
            raise ValueError("After dropping NaN-only criteria, all weights became zero.")
        weights_arr = weights_arr / weights_arr.sum()

        # (implicit but important: if only ONE column survived, weights_arr will be [1.0])

        # ---- NEW: get global μ_first_visit / μ_delivery from FeatureBuilder if available ----
        mu_global_first_visit = None
        mu_global_delivery = None
        if feature_builder is not None:
            if hasattr(feature_builder, "first_visit_mu_global"):
                mu_global_first_visit = feature_builder.first_visit_mu_global
            if hasattr(feature_builder, "delivery_mu_global"):
                mu_global_delivery = feature_builder.delivery_mu_global

                # ---- apply business-rule imputation ONCE (for first_visit & delivery) ----
        data_imputed = self._impute_with_business_rule(
            df,
            cols,
            mu_global_first_visit=mu_global_first_visit,
            mu_global_delivery=mu_global_delivery,
            smooth_approximation=smooth_approximation,
        )

        # DEBUG: store imputed (pre-scaling) matrix
        self.debug_last_imputed = data_imputed.copy()
                # ---- OPTIONAL: global normalization using sklearn scaler ----
        data_for_mcda = data_imputed

        if normalize_globally:
            if feature_builder is None:
                raise ValueError(
                    "normalize_globally=True but feature_builder is None. "
                    "Pass the FeatureBuilder used to build the features."
                )

            if (
                not hasattr(feature_builder, "global_feature_df")
                or feature_builder.global_feature_df is None
            ):
                raise ValueError(
                    "normalize_globally=True but feature_builder.global_feature_df is None. "
                    "Call FeatureBuilder.build(..., cache_global=True) once on the full dataset."
                )

            global_df = feature_builder.global_feature_df

            # All criteria actually used in this run:
            global_cols_all = [c for c in cols if c in global_df.columns]

            # We scale all of them globally
            scale_cols = [c for c in global_cols_all]

            # If nothing to scale, just use imputed data as-is
            if not scale_cols:
                data_for_mcda = data_imputed
            else:
                global_raw = global_df[scale_cols].copy()

                # Apply the SAME business-rule transform so the scaler "sees" the same kind of data
                global_imputed = self._impute_with_business_rule(
                    global_raw,
                    scale_cols,
                    mu_global_first_visit=mu_global_first_visit,
                    mu_global_delivery=mu_global_delivery,
                    smooth_approximation=smooth_approximation,
                )

                scaler = MinMaxScaler()
                scaler.fit(global_imputed[scale_cols])

                data_scaled = data_imputed.copy()

                # Transform and THEN CLIP to [0, 1]
                scaled_arr = scaler.transform(data_imputed[scale_cols])
                scaled_arr = np.clip(scaled_arr, 0.0, 1.0)
                data_scaled[scale_cols] = scaled_arr

                # 🔧 NEW: if in THIS FILTERED SLICE a column is all 1s,
                # we skip normalization for that column and keep it as 1s.
                for c in scale_cols:
                    local_vals = data_imputed[c].to_numpy(dtype=float)
                    # If everything (finite) is ~1.0 in this slice, restore it
                    finite = np.isfinite(local_vals)
                    if finite.any() and np.allclose(local_vals[finite], 1.0, atol=1e-9):
                        # keep the original imputed 1s instead of the scaled values (which might be 0)
                        data_scaled[c] = local_vals

                data_for_mcda = data_scaled

            # DEBUG: store scaled matrix
            self.debug_last_scaled = data_for_mcda.copy()

        else:
            self.debug_last_scaled = data_imputed.copy()
            data_for_mcda = data_imputed



        # ---- run each MCDA method (assume df already imputed & (optionally) scaled) ----

        # SAFETY: cost criteria cannot be zero for WASPAS (min/x normalization),
        # so we bump zeros to a tiny epsilon. This is negligible numerically
        # but avoids division by zero inside pymcdm.
        eps = 1e-6
        data_for_mcda = data_for_mcda.copy()
        for col, ct in zip(cols, criteria_types):
            if ct == "cost":
                col_vals = data_for_mcda[col].to_numpy(dtype=float)
                if np.any(col_vals <= 0):
                    col_vals[col_vals <= 0] = eps
                    data_for_mcda[col] = col_vals

        results = {}
        for m in methods:
            if m not in self.methods:
                raise ValueError(f"Unknown method '{m}'")
            func = getattr(self, f"_{m}")
            score = func(data_for_mcda, cols, weights_arr, criteria_types)
            results[f"Score_{m}"] = score



        # ---- build output ----
        df_out = pd.concat([df_metrics.copy()] + [v for v in results.values()], axis=1)

        if avg and len(results) > 1:
            df_out["Score_avg"] = pd.concat(results.values(), axis=1).mean(axis=1)

        sort_col = "Score_avg" if avg else list(results.keys())[0]
        if sort:
            df_out = df_out.sort_values(sort_col, ascending=False)

        return df_out if return_df else results[list(results.keys())[0]]


    def _impute_with_business_rule(
        self,
        df: pd.DataFrame,
        cols: list[str],
        mu_global_first_visit: float | None = None,
        mu_global_delivery: float | None = None,
        smooth_approximation: bool = False,
    ) -> pd.DataFrame:
        """
        Return df[cols] with NaNs imputed and first_visit / delivery
        transformed to [0, 1].

        Assumes df is the feature matrix for ONE provincia.

        For 'first_visit' and 'delivery' (assumed in %: 0..100):
            * Convert to rate r in [0,1].
            * t_global = self.first_visit_threshold / 100.
            * μ_prov = mean r in this df (per courier matrix).
            * If corresponding global μ is provided (from FeatureBuilder):
                  μ_global = mu_global_*
                  Δ = μ_global - t_global
                  t_prov = clip(μ_prov - Δ, [0.6, 0.95])
              else:
                  t_prov = t_global
            * If smooth_approximation is False (default):
                  Soft-ReLU with kink (piecewise linear) and low_band_max:
                      if r <= t_prov:
                          y = low_band_max * (r / t_prov)
                      else:
                          y = low_band_max + (1 - low_band_max) * (r - t_prov) / (1 - t_prov)
              NaNs are filled (in score space) with:
                  r_nan = t_prov + alpha * (μ_prov - t_prov), alpha = 0.25
                  nan_fill = soft_relu(r_nan, t_prov)
            * If smooth_approximation is True:
                  Use a cached cubic smoothing of the same soft-ReLU shape
                  (no kink) built once per provincia for each metric.

              If there is no info at all, fall back to 0.5.

        Other criteria:
            * NaNs filled with midpoint (min + max)/2, or 0 if the column is fully missing.
        """
        data = df[cols].copy()

        # --- Global params for transforms ---
        t_global = float(self.first_visit_threshold) / 100.0  # e.g. 0.9
        low_band_max = 0.35
        alpha = 0.25
        nan_fill_default = 0.5

        def _soft_relu_scalar(r_val: float, t_val: float) -> float:
            if pd.isna(r_val) or pd.isna(t_val):
                return nan_fill_default
            t_safe = min(max(float(t_val), 1e-6), 1.0 - 1e-6)
            r_safe = float(r_val)
            if r_safe <= t_safe:
                return low_band_max * (r_safe / t_safe)
            else:
                denom = max(1.0 - t_safe, 1e-9)
                return low_band_max + (1.0 - low_band_max) * ((r_safe - t_safe) / denom)

        def _build_soft_relu_cubic(t_val: float, band_width: float = 0.10):
            """
            Build a reusable smooth soft-ReLU (cached cubic) for a given threshold t_val.
            Returns a function f(r_array) -> scores, vectorized over r.
            """
            t_safe = min(max(float(t_val), 1e-6), 1.0 - 1e-6)
            L = float(low_band_max)

            # Slopes of the original linear pieces
            m_left = L / t_safe
            m_right = (1.0 - L) / (1.0 - t_safe)

            # Band around threshold where we smooth
            left = max(t_safe - band_width, 0.0)
            right = min(t_safe + band_width, 1.0)

            # Values of the original function at band edges
            y_left = L * (left / t_safe)
            y_right = L + (1.0 - L) * ((right - t_safe) / (1.0 - t_safe))

            # Solve for cubic S(r) = a r^3 + b r^2 + c r + d
            M = np.array([
                [left**3,    left**2,  left,  1.0],
                [3*left**2,  2*left,   1.0,   0.0],
                [right**3,   right**2, right, 1.0],
                [3*right**2, 2*right,  1.0,   0.0],
            ])
            v = np.array([y_left, m_left, y_right, m_right])
            a, b, c, d = np.linalg.solve(M, v)

            def _smooth_fn(r):
                r_arr = np.asarray(r, dtype=float)
                out = np.empty_like(r_arr)

                mask_left = r_arr <= left
                mask_right = r_arr >= right
                mask_mid = ~(mask_left | mask_right)

                # Left linear segment
                if mask_left.any():
                    out[mask_left] = L * (r_arr[mask_left] / t_safe)

                # Right linear segment
                if mask_right.any():
                    denom = max(1.0 - t_safe, 1e-9)
                    out[mask_right] = L + (1.0 - L) * (
                        (r_arr[mask_right] - t_safe) / denom
                    )

                # Cubic blend in the middle
                if mask_mid.any():
                    rm = r_arr[mask_mid]
                    out[mask_mid] = ((a * rm + b) * rm + c) * rm + d

                if np.isscalar(r):
                    return float(out)
                return out

            return _smooth_fn

        # --- FIRST_VISIT params per province ---
        t_prov_first = t_global
        nan_fill_first_visit = nan_fill_default
        soft_fn_first = None  # cached cubic smoother for first_visit (if used)

        if "first_visit" in cols:
            vals_all = data["first_visit"]
            valid_all = vals_all.dropna()

            if not valid_all.empty:
                r_all = (valid_all / 100.0).clip(0.0, 1.0)
                mu_prov = float(r_all.mean())

                if mu_global_first_visit is not None and not np.isnan(mu_global_first_visit):
                    mu_global = float(mu_global_first_visit)
                    delta = mu_global - t_global
                    t_prov_first = float(np.clip(mu_prov - delta, 0.6, 0.95))
                else:
                    t_prov_first = t_global

                r_nan = t_prov_first + alpha * (mu_prov - t_prov_first)

                if smooth_approximation:
                    # Build smoother once for this provincia / metric
                    soft_fn_first = _build_soft_relu_cubic(t_prov_first)
                    # Simple rule: use r_nan in rate space, then smooth it
                    nan_fill_first_visit = float(soft_fn_first(r_nan))
                else:
                    nan_fill_first_visit = _soft_relu_scalar(r_nan, t_prov_first)
            else:
                t_prov_first = t_global
                nan_fill_first_visit = nan_fill_default

        # --- DELIVERY params per province (similar to first_visit) ---
        t_prov_delivery = t_global
        nan_fill_delivery = nan_fill_default
        soft_fn_delivery = None  # cached cubic smoother for delivery (if used)

        if "delivery" in cols:
            vals_all = data["delivery"]
            valid_all = vals_all.dropna()

            if not valid_all.empty:
                r_all = (valid_all / 100.0).clip(0.0, 1.0)
                mu_prov = float(r_all.mean())

                if mu_global_delivery is not None and not np.isnan(mu_global_delivery):
                    mu_global = float(mu_global_delivery)
                    delta = mu_global - t_global
                    t_prov_delivery = float(np.clip(mu_prov - delta, 0.6, 0.95))
                else:
                    t_prov_delivery = t_global

                r_nan = t_prov_delivery + alpha * (mu_prov - t_prov_delivery)

                if smooth_approximation:
                    soft_fn_delivery = _build_soft_relu_cubic(t_prov_delivery)
                    nan_fill_delivery = float(soft_fn_delivery(r_nan))
                else:
                    nan_fill_delivery = _soft_relu_scalar(r_nan, t_prov_delivery)
            else:
                t_prov_delivery = t_global
                nan_fill_delivery = nan_fill_default

        # --- Main loop over columns ---
        for col in cols:
            vals = data[col]

            if col == "first_visit":
                valid = vals.dropna()

                if not valid.empty:
                    r = (valid / 100.0).clip(0.0, 1.0)

                    if smooth_approximation and soft_fn_first is not None:
                        # Use cached cubic smoother
                        y_vals = soft_fn_first(r.values)
                        y = pd.Series(y_vals, index=valid.index, dtype=float)
                        data.loc[valid.index, col] = y
                    else:
                        # Original piecewise (with kink)
                        y = pd.Series(index=valid.index, dtype=float)
                        t_safe = min(max(t_prov_first, 1e-6), 1.0 - 1e-6)

                        # bad region: r <= t_prov_first → [0, low_band_max]
                        mask_bad = r <= t_safe
                        if mask_bad.any():
                            y.loc[mask_bad] = low_band_max * (r[mask_bad] / t_safe)

                        # good region: r > t_prov_first → [low_band_max, 1]
                        mask_good = ~mask_bad
                        if mask_good.any():
                            denom_good = max(1.0 - t_safe, 1e-9)
                            y.loc[mask_good] = (
                                low_band_max
                                + (1.0 - low_band_max)
                                * ((r[mask_good] - t_safe) / denom_good)
                            )

                        data.loc[valid.index, col] = y

                # Fill NaNs with the province-specific neutral score
                data[col] = data[col].astype(float).fillna(nan_fill_first_visit)

            elif col == "delivery":
                valid = vals.dropna()

                if not valid.empty:
                    r = (valid / 100.0).clip(0.0, 1.0)

                    if smooth_approximation and soft_fn_delivery is not None:
                        # Use cached cubic smoother
                        y_vals = soft_fn_delivery(r.values)
                        y = pd.Series(y_vals, index=valid.index, dtype=float)
                        data.loc[valid.index, col] = y
                    else:
                        # Original piecewise (with kink)
                        y = pd.Series(index=valid.index, dtype=float)
                        t_safe = min(max(t_prov_delivery, 1e-6), 1.0 - 1e-6)

                        # bad region: r <= t_prov_delivery → [0, low_band_max]
                        mask_bad = r <= t_safe
                        if mask_bad.any():
                            y.loc[mask_bad] = low_band_max * (r[mask_bad] / t_safe)

                        # good region: r > t_prov_delivery → [low_band_max, 1]
                        mask_good = ~mask_bad
                        if mask_good.any():
                            denom_good = max(1.0 - t_safe, 1e-9)
                            y.loc[mask_good] = (
                                low_band_max
                                + (1.0 - low_band_max)
                                * ((r[mask_good] - t_safe) / denom_good)
                            )

                        data.loc[valid.index, col] = y

                # Fill NaNs with the province-specific neutral score
                data[col] = data[col].astype(float).fillna(nan_fill_delivery)

            else:
                # generic midpoint neutral for other criteria
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

        Assumes:
        - df[cols] already business-rule imputed (first_visit / delivery in [0,1])
          by _impute_with_business_rule.
        """
        data = df[cols].copy()

        X = data.to_numpy(dtype=float)
        w = np.asarray(weights, dtype=float)
        type_map = {"benefit": 1, "cost": -1}
        types = np.array([type_map[ct] for ct in criteria_types], dtype=int)

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

        Assumes:
        - df[cols] already business-rule imputed (first_visit / delivery in [0,1])
          by _impute_with_business_rule.
        """
        data = df[cols].copy()

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
        VIKOR via pymcdm.

        Assumes:
        - df[cols] already business-rule imputed (first_visit / delivery in [0,1])
          by _impute_with_business_rule.
        """
        data = df[cols].copy()

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
                "NaNs handled with business-rule transforms (first_visit / delivery)."
            )

        if return_details:
            return {"score": score}

        return score

    def _waspas(self, df, cols, weights, criteria_types):
        """
        WASPAS via pymcdm.

        Assumes:
        - df[cols] already business-rule imputed (first_visit / delivery in [0,1])
          by _impute_with_business_rule.
        """
        data = df[cols].copy()

        X = data.to_numpy(dtype=float)
        w = np.asarray(weights, dtype=float)

        type_map = {"benefit": 1, "cost": -1}
        types = np.array([type_map[ct] for ct in criteria_types], dtype=int)

        # ---- 1) If only ONE alternative, score = 1.0 by convention ----
        if X.shape[0] == 1:
            return pd.Series(
                np.ones(1, dtype=float),
                index=df.index,
                name="Score_waspas",
            )

        # ---- 2) Detect constant criteria (zero variance or all NaN) ----
        const_idx = []
        for j in range(X.shape[1]):
            col_vals = X[:, j]
            finite = ~np.isnan(col_vals)
            if finite.sum() == 0:
                # fully NaN → constant / useless
                const_idx.append(j)
            else:
                vals = col_vals[finite]
                if np.allclose(vals, vals[0]):
                    # zero variance
                    const_idx.append(j)

        if const_idx:
            mask = np.ones(X.shape[1], dtype=bool)
            mask[const_idx] = False

            # If *all* criteria are constant, everyone is equally good: score 1
            if mask.sum() == 0:
                return pd.Series(
                    np.ones(X.shape[0], dtype=float),
                    index=df.index,
                    name="Score_waspas",
                )

            # Drop constant columns (they don't affect ranking)
            X = X[:, mask]
            w = w[mask]
            types = types[mask]

            # Re-normalize weights
            if w.sum() != 0:
                w = w / w.sum()
            else:
                w = np.ones_like(w, dtype=float) / len(w)

        # ---- 3) Run WASPAS on the cleaned matrix ----
        method = PymWASPAS()
        # validation=False for consistency with TOPSIS/WSM/VIKOR calls
        prefs = method(X, w, types, validation=False, verbose=False)

        score = pd.Series(prefs, index=df.index, name="Score_waspas")
        return score
