# src/presets.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Optional, List, Any

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING

from src.config import PROVEEDORES_SHORT, CORREO_KEYS
from src.mcda import MCDAEngine

if TYPE_CHECKING:
    from src.features import FeatureBuilder


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


@dataclass
class WeightPreset:
    """
    Represents a single MCDA weight preset.

    Example
    -------
    WeightPreset(
        name='speed_heavy',
        weights={
            'first_visit': 0.6,
            'cost': 0.2,
            'coverage': 0.2,
        }
    )
    """
    name: str
    weights: Dict[str, float]


# ---------------------------------------------------------------------
# HistoricalPresetLearner
# ---------------------------------------------------------------------


class HistoricalPresetLearner:
    """
    Learns which MCDA weight presets best explain historical choices,
    starting from the *shipment-level* cleaned_df.

    High-level flow
    ---------------
    Input: cleaned_df (one row per shipment), with at least:
        - 'Correo' (actual chosen provider; e.g. 'urbano', 'oca', ...)
        - columns used for grouping (e.g. 'Provincia', 'Codigo Postal', ...)
        - all columns FeatureBuilder needs (Peso, Presupuesto URBANO, etc.)

    For each group (defined by group_cols):

        1. Extract group_df = cleaned_df for that group.

        2. Count historical shipments per provider in that group
           using the 'Correo' column, mapped to PROVEEDORES_SHORT
           via CORREO_KEYS.

              correo value "cruz_del_sur" -> "CDS"
              etc.

        3. Determine the "human" top provider in that group:
              the one with max shipments (after mapping to short names).

        4. Use FeatureBuilder.build(group_df, features=feature_names)
           to get a provider-level feature table, indexed by short provider
           names (Urbano, OCA, ...).

        5. Restrict the historical counts to the providers that survived
           in the feature table (FeatureBuilder may drop some rows).

        6. For each candidate preset:
             - Take the subset of features that preset defines weights for.
             - Build the aligned weight vector.
             - Run MCDAEngine.score(...) on that group feature table.
             - Take the top-ranked provider under that preset.
             - Check if it matches the human top provider.
             - Record the result.

    Output: a DataFrame with one row per (group, preset), containing:
        - group_cols ...
        - preset_name
        - human_provider
        - preset_top_provider
        - hit (0/1)
        - n_shipments_group
        - human_provider_volume
        - preset_top_score

    There is also summarize_best_preset(...) that picks a single
    "best" preset per group using a simple heuristic.
    """

    def __init__(
        self,
        mcda_engine: MCDAEngine,
        feature_builder: FeatureBuilder,
        candidate_presets: Mapping[str, Mapping[str, float]],
        correo_col: str = "Correo",
        correo_to_short_map: Optional[Mapping[str, str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        mcda_engine
            Instance of MCDAEngine used to compute scores given features + weights.
        feature_builder
            FeatureBuilder instance used to build provider-level features.
        candidate_presets
            Dict: preset_name -> {feature_name: weight}.
            These are the fixed "trial" presets we will test against history
            (e.g. balanced, speed-heavy, cost-heavy, coverage-heavy, etc.).
        correo_col
            Column in cleaned_df that encodes the chosen provider per shipment
            (e.g. 'Correo').
        correo_to_short_map
            Mapping from values in `correo_col` to short provider names
            (Urbano, OCA, CDS, ...). If None, it is inferred from CORREO_KEYS.
        """
        self.mcda_engine = mcda_engine
        self.feature_builder = feature_builder
        self.correo_col = correo_col

        # Inverse mapping: 'urbano' -> 'Urbano', etc.
        if correo_to_short_map is not None:
            self.correo_to_short = dict(correo_to_short_map)
        else:
            self.correo_to_short = {v: k for k, v in CORREO_KEYS.items()}

        # Normalize into WeightPreset objects for convenience.
        self._presets: Dict[str, WeightPreset] = {
            name: WeightPreset(name=name, weights=dict(wdict))
            for name, wdict in candidate_presets.items()
        }

        # Keep a copy of the canonical provider list (short names).
        self.proveedores_short: List[str] = list(PROVEEDORES_SHORT)

    # ----------------------- public API --------------------------------

    @property
    def presets(self) -> Dict[str, WeightPreset]:
        """Return the candidate presets."""
        return self._presets

    def learn_for_groups(
        self,
        cleaned_df: pd.DataFrame,
        group_cols: Sequence[str],
        feature_names: Optional[Sequence[str]] = None,
        mcda_method: str = "weighted",
        drop_incomplete: bool = True,
    ) -> pd.DataFrame:
        """
        Evaluate all candidate presets against historical choices,
        starting from the shipment-level cleaned_df.

        Parameters
        ----------
        cleaned_df
            Original cleaned dataset, one row per shipment. Must contain:
            - group_cols
            - self.correo_col (e.g. 'Correo')
            - all columns needed by FeatureBuilder for the chosen features.
        group_cols
            Columns that define the groups / segments
            (e.g. ['Provincia', 'Codigo Postal']).
        feature_names
            Feature names to compute with FeatureBuilder. If None, it defaults
            to the union of all feature names referenced in candidate_presets.
        mcda_method
            Which MCDA method to use for learning (default 'weighted').
            It should be one of the methods supported by MCDAEngine
            (e.g. 'weighted', 'waspas', 'topsis', 'vikor').
        drop_incomplete
            Passed down to FeatureBuilder.build (defaults to True).

        Returns
        -------
        results_df : pd.DataFrame
            One row per (group, preset) with columns:
            - group_cols ...
            - 'preset_name'
            - 'human_provider'
            - 'preset_top_provider'
            - 'hit' (0/1)
            - 'n_shipments_group'
            - 'human_provider_volume'
            - 'preset_top_score'
        """
        self._validate_input(cleaned_df, group_cols)

        # If no explicit feature_names, use the union of all features
        # referenced by the presets.
        if feature_names is None:
            feature_names = self._infer_feature_names_from_presets()
        feature_names = list(feature_names)

        results: List[Dict[str, Any]] = []

        grouped = cleaned_df.groupby(list(group_cols), dropna=False)

        for group_key, group_df in grouped:
            # normalize group_key into a dict {col_name: value}
            if len(group_cols) == 1:
                group_values = {group_cols[0]: group_key}
            else:
                group_values = {col: val for col, val in zip(group_cols, group_key)}

            if group_df.empty:
                continue

            # -----------------------
            # 1) Historical choice: count shipments per provider (short)
            # -----------------------
            counts_short = self._count_providers_short(group_df)
            if counts_short is None:
                # no recognizable providers in this group
                continue

            # We'll only consider providers that end up in the feature table,
            # so we don't pick a "human top" that later disappears.
            # For now we just get the total shipments in case we need it later.
            n_shipments_group_total = counts_short.sum()
            if n_shipments_group_total <= 0:
                continue

            # -----------------------
            # 2) Build provider-level features for this group
            # -----------------------
            try:
                features_df = self.feature_builder.build(
                    group_df,
                    features=feature_names,
                    drop_incomplete=drop_incomplete,
                )
            except Exception:
                # If something goes wrong for this group, skip it instead
                # of crashing the whole process.
                continue

            if features_df.empty:
                # No provider had complete features in this group.
                continue

            # align counts to the providers that survived in features_df
            counts_aligned = counts_short.reindex(features_df.index).fillna(0.0)

            if counts_aligned.sum() <= 0:
                # After alignment, none of the providers with features were
                # actually used historically (could happen in weird data).
                continue

            # human top provider (short name) among those with features
            human_provider = counts_aligned.idxmax()
            human_volume = float(counts_aligned.loc[human_provider])
            n_shipments_group = float(counts_aligned.sum())

            # -----------------------
            # 3) For each preset, run MCDA and compare with human choice
            # -----------------------
            for preset in self._presets.values():
                # use only the feature columns that this preset talks about
                used_cols = [
                    c for c in feature_names
                    if c in features_df.columns and preset.weights.get(c, 0.0) > 0
                ]

                if not used_cols:
                    # This preset does not apply to these features
                    continue

                sub_df = features_df[used_cols].copy()
                if sub_df.empty:
                    continue

                w = np.array([preset.weights[c] for c in used_cols], dtype=float)
                if w.sum() <= 0:
                    continue
                w = w / w.sum()

                scores_df = self.mcda_engine.score(
                    sub_df,
                    methods=mcda_method,
                    weights=w,
                    return_df=True,
                    sort=True,
                    features=used_cols,
                    feature_builder=self.feature_builder,
                )

                score_col = f"Score_{mcda_method.lower()}"
                if score_col not in scores_df.columns:
                    continue

                idx_top = scores_df[score_col].idxmax()
                preset_top_provider = idx_top
                preset_top_score = float(scores_df.loc[idx_top, score_col])

                hit = int(str(preset_top_provider) == str(human_provider))

                row: Dict[str, Any] = {}
                row.update(group_values)
                row.update(
                    dict(
                        preset_name=preset.name,
                        human_provider=str(human_provider),
                        preset_top_provider=str(preset_top_provider),
                        hit=hit,
                        n_shipments_group=n_shipments_group,
                        human_provider_volume=human_volume,
                        preset_top_score=preset_top_score,
                    )
                )
                results.append(row)

        if not results:
            return pd.DataFrame(
                columns=list(group_cols)
                + [
                    "preset_name",
                    "human_provider",
                    "preset_top_provider",
                    "hit",
                    "n_shipments_group",
                    "human_provider_volume",
                    "preset_top_score",
                ]
            )

        results_df = pd.DataFrame(results)
        return results_df

    def summarize_best_preset(
        self,
        results_df: pd.DataFrame,
        group_cols: Sequence[str],
    ) -> pd.DataFrame:
        """
        Given the detailed results of learn_for_groups, pick one "best" preset
        per group.

        Heuristic
        ---------
        - Prefer presets where hit == 1 (preset's top == historical top).
        - Among those, prefer groups with more n_shipments_group.
        - If no preset hits (all hit == 0), pick the one with highest
          preset_top_score.

        Parameters
        ----------
        results_df
            Output of learn_for_groups.
        group_cols
            Same grouping columns used there.

        Returns
        -------
        best_df : pd.DataFrame
            One row per group with:

            - group_cols ...
            - 'best_preset'
            - 'hit'
            - 'human_provider'
            - 'preset_top_provider'
            - 'n_shipments_group'
            - 'human_provider_volume'
            - 'preset_top_score'
        """
        if results_df.empty:
            return results_df.copy()

        sorted_df = results_df.sort_values(
            by=["hit", "n_shipments_group", "preset_top_score"],
            ascending=[False, False, False],
        )

        best_rows = (
            sorted_df.groupby(list(group_cols), as_index=False)
            .head(1)
            .reset_index(drop=True)
        )

        best_df = best_rows.copy()
        best_df = best_df.rename(columns={"preset_name": "best_preset"})
        return best_df

    # ----------------------- internal helpers --------------------------

    def _count_providers_short(self, group_df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Count shipments per short provider name within a group, based on
        the `Correo` column and CORREO_KEYS mapping.

        Returns a Series indexed by PROVEEDORES_SHORT with float counts,
        or None if no recognizable providers are present.
        """
        if self.correo_col not in group_df.columns:
            return None

        correo_series = group_df[self.correo_col].dropna().astype(str)

        if correo_series.empty:
            return None

        # map correo values ('urbano', 'oca', etc.) to short names ('Urbano', ...)
        mapped = correo_series.map(self.correo_to_short)
        mapped = mapped.dropna()

        if mapped.empty:
            return None

        counts = mapped.value_counts()
        # align to canonical provider list (even if some are missing)
        counts_all = pd.Series(0.0, index=self.proveedores_short, dtype=float)
        for prov, cnt in counts.items():
            if prov in counts_all.index:
                counts_all.loc[prov] = float(cnt)

        return counts_all

    def _infer_feature_names_from_presets(self) -> List[str]:
        """
        Union of all feature names referenced in candidate_presets.
        """
        names: set[str] = set()
        for preset in self._presets.values():
            names.update(preset.weights.keys())
        return sorted(names)

    @staticmethod
    def _validate_input(
        cleaned_df: pd.DataFrame,
        group_cols: Sequence[str],
    ) -> None:
        """Basic sanity checks on the input DataFrame."""
        missing = [c for c in group_cols if c not in cleaned_df.columns]
        if missing:
            raise ValueError(
                f"cleaned_df is missing required grouping columns: {missing}"
            )
