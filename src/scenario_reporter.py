# src/scenario_reporter.py
"""
ScenarioReporter: "what-if" cost scenarios on top of the shipping MCDA engine,
and matrix-style exports (CP x Rango de Peso) for each provincia + feature.

Typical questions:

- "How much would we have saved or lost this month if instead of the most-used
  provider we had used the #1 ranked by MCDA (with a given preset/weights)?"

- "How much more / less would we have paid using proveedor A vs proveedor B in
  Corrientes, interior, in a given period?"

- "For each provincia / CP / rango de peso, how do the metrics look for the
  most-used provider vs the MCDA-recommended one?"
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import os
import re

import numpy as np
import pandas as pd

from src.config import PROVEEDORES_FULL, PROVEEDORES_SHORT, CORREO_KEYS
from src.features import FeatureBuilder
from src.mcda import MCDAEngine


class ScenarioReporter:
    """
    Run cost "what-if" scenarios and matrix-style reports on top of the main
    shipping dataset.

    Assumptions
    -----------
    - The main DataFrame (df) has one row per shipment.
    - Each provider has a tariff column in PROVEEDORES_FULL
      (e.g. "Presupuesto URBANO", "Presupuesto OCA", ...).
    - FeatureBuilder.build(df, ...) produces a matrix whose index is
      PROVEEDORES_SHORT (["Urbano", "OCA", ...]).
    - MCDAEngine.score(...) returns scores indexed by the same short names.

    Parameters
    ----------
    df :
        Main cleaned DataFrame with one row per shipment and columns matching
        PROVEEDORES_FULL for tariffs.
    feature_builder :
        Instance of FeatureBuilder already configured for this project.
    mcda_engine :
        Instance of MCDAEngine already configured for this project.
    date_col :
        Name of the datetime column in df used for date filtering.
    real_provider_col :
        Optional column in df indicating which provider was actually used.
        Values may be full names ("Presupuesto URBANO") or short names
        ("Urbano"); both are resolved to the full tariff column name.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_builder: FeatureBuilder,
        mcda_engine: MCDAEngine,
        *,
        date_col: str = "Fecha de Despacho",
        real_provider_col: Optional[str] = "Correo",
        date_format: Optional[str] = "%d-%m-%y %H:%M",
        dayfirst: bool = True,
    ) -> None:
        self.df = df.copy()
        self.feature_builder = feature_builder
        self.mcda_engine = mcda_engine
        self.date_col = date_col
        self.real_provider_col = real_provider_col

        # NEW: store options for date parsing
        self.date_format = date_format
        self.dayfirst = dayfirst

        # From src.config
        self.proveedores_full: list[str] = list(PROVEEDORES_FULL)
        self.proveedores_short: list[str] = list(PROVEEDORES_SHORT)

        # full <-> short mappings
        self.full_to_short: Dict[str, str] = dict(
            zip(self.proveedores_full, self.proveedores_short)
        )
        self.short_to_full: Dict[str, str] = dict(
            zip(self.proveedores_short, self.proveedores_full)
        )

        # CORREO_KEYS mapping
        if isinstance(CORREO_KEYS, Mapping):
            self.correo_keys: Mapping[str, str] = CORREO_KEYS
        else:
            self.correo_keys = CORREO_KEYS[0] if CORREO_KEYS else {}

        # Ensure date column exists and is datetime
        if self.date_col not in self.df.columns:
            raise ValueError(
                f"date_col={self.date_col!r} not found in DataFrame columns."
            )

        if not np.issubdtype(self.df[self.date_col].dtype, np.datetime64):
            if self.date_format is not None:
                self.df[self.date_col] = pd.to_datetime(
                    self.df[self.date_col],
                    format=self.date_format,
                    errors="coerce",
                )
            else:
                self.df[self.date_col] = pd.to_datetime(
                    self.df[self.date_col],
                    errors="coerce",
                    dayfirst=self.dayfirst,
                )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _filtered(
        self,
        *,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        provincia: Optional[str] = None,
        location: str = "both",
        rango_peso: Optional[str] = None,
        codigo_postal: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Filter the main df by common selection parameters.
        """
        df = self.df

        # --- Default date range: min → max of dataset ---
        if date_from is None:
            date_from = self.df[self.date_col].min()
        if date_to is None:
            date_to = self.df[self.date_col].max()

        date_from = pd.to_datetime(date_from)
        date_to = pd.to_datetime(date_to)

        df = df[(df[self.date_col] >= date_from) & (df[self.date_col] <= date_to)]

        if provincia:
            df = df[df["Provincia"] == provincia]
        if codigo_postal:
            df = df[df["Codigo Postal"] == codigo_postal]
        if rango_peso:
            df = df[df["Rango de Peso"] == rango_peso]

        if location == "capital":
            df = df[df["Capital/Interior"] == "CIUDAD"]
        elif location == "interior":
            df = df[df["Capital/Interior"] == "INTERIOR"]

        return df
    
    def _feature_is_cost(self, feature: str) -> bool:
        """
        Decide if a feature is COST-type (lower is better).

        Primary source: FeatureBuilder metadata (preferred).
        Fallback: heuristic based on feature name (only if metadata not found).
        """
        fb = self.feature_builder

        # --- Try common metadata patterns on FeatureBuilder ---
        # Pattern A: dict like {"cost": "cost", "speed": "cost", "sla": "benefit", ...}
        for attr in ("criteria_types_map", "criteria_types_by_feature", "feature_criteria_types"):
            m = getattr(fb, attr, None)
            if isinstance(m, dict) and feature in m:
                val = str(m[feature]).lower().strip()
                return val.startswith("c") or val == "cost"

        # Pattern B: list aligned with available_features
        feats = getattr(fb, "available_features", None)
        cts = getattr(fb, "criteria_types", None)
        if isinstance(feats, (list, tuple)) and isinstance(cts, (list, tuple)):
            if feature in feats and len(cts) == len(feats):
                val = str(cts[feats.index(feature)]).lower().strip()
                return val.startswith("c") or val == "cost"

        # Pattern C: dict like {"cost": "min", "sla": "max"} or {"cost": -1, "sla": +1}
        for attr in ("directions", "feature_directions", "criteria_directions"):
            m = getattr(fb, attr, None)
            if isinstance(m, dict) and feature in m:
                v = m[feature]
                if isinstance(v, (int, float)):
                    return v < 0
                val = str(v).lower().strip()
                return val in ("min", "cost", "lower", "down", "negative")

        # --- Fallback heuristic (ONLY if metadata is missing) ---
        name = str(feature).lower()
        return ("cost" in name) or (name == "speed") or ("precio" in name) or ("tarifa" in name)

    def _resolve_full_name(self, name: str) -> str:
        """
        Resolve a provider name to the corresponding full tariff column name.

        Accepts:
        - Full:  "Presupuesto URBANO" (already a df column)
        - Short: "Urbano" (from PROVEEDORES_SHORT)
        - Slug:  "urbano", "oca", "cruz_del_sur", etc. (from CORREO_KEYS values)
        """
        # 1) Already a full column name
        if name in self.proveedores_full:
            return name

        # 2) Short name
        if name in self.short_to_full:
            return self.short_to_full[name]

        # 3) CORREO_KEYS key: "Urbano" -> treat as short
        if name in self.correo_keys:
            short = name
            if short in self.short_to_full:
                return self.short_to_full[short]

        # 4) CORREO_KEYS value: "urbano" -> find its key "Urbano"
        if name in self.correo_keys.values():
            for short, slug in self.correo_keys.items():
                if slug == name:
                    if short in self.short_to_full:
                        return self.short_to_full[short]
                    break

        # 5) Case-insensitive match to short names
        lower_name = name.lower()
        for short in self.proveedores_short:
            if short.lower() == lower_name:
                return self.short_to_full.get(short, short)

        # 6) Fallback
        return name

    def _compare_cost_two_providers_on_df(
        self,
        df: pd.DataFrame,
        current_provider_full: str,
        alternative_provider_full: str,
    ) -> Dict[str, Any]:
        """
        Core cost comparison logic on a pre-filtered DataFrame.
        """
        if current_provider_full not in df.columns:
            raise ValueError(
                f"Current provider column {current_provider_full!r} not found in df."
            )
        if alternative_provider_full not in df.columns:
            raise ValueError(
                f"Alternative provider column {alternative_provider_full!r} not found in df."
            )

        df_valid = df[
            (df[current_provider_full] > 0) & (df[alternative_provider_full] > 0)
        ]

        n_total = len(df)
        n_comparable = len(df_valid)
        n_unusable = n_total - n_comparable

        current_short = self.full_to_short.get(
            current_provider_full, current_provider_full
        )
        alternative_short = self.full_to_short.get(
            alternative_provider_full, alternative_provider_full
        )

        if n_comparable == 0:
            return {
                "n_total_shipments": int(n_total),
                "n_comparable": 0,
                "n_unusable": int(n_unusable),
                "current_provider": current_short,
                "alternative_provider": alternative_short,
                "current_total": np.nan,
                "alternative_total": np.nan,
                "absolute_delta": np.nan,
                "relative_delta_pct": np.nan,
            }

        current_total = float(df_valid[current_provider_full].sum())
        alternative_total = float(df_valid[alternative_provider_full].sum())
        absolute_delta = alternative_total - current_total
        relative_delta_pct = (
            absolute_delta / current_total * 100 if current_total > 0 else np.nan
        )

        return {
            "n_total_shipments": int(n_total),
            "n_comparable": int(n_comparable),
            "n_unusable": int(n_unusable),
            "current_provider": current_short,
            "alternative_provider": alternative_short,
            "current_total": current_total,
            "alternative_total": alternative_total,
            "absolute_delta": float(absolute_delta),
            "relative_delta_pct": float(relative_delta_pct),
        }

    def _most_used_provider_full(self, df: pd.DataFrame) -> Optional[str]:
        """
        Infer the "most used" provider for the given df.
        """
        # Case 1: explicit real provider column
        if self.real_provider_col is not None and self.real_provider_col in df.columns:
            col = df[self.real_provider_col].dropna()
            if len(col) > 0:
                candidate = str(col.mode().iloc[0])
                return self._resolve_full_name(candidate)

        # Case 2: approximate from tariff columns
        best_provider = None
        best_count = -1

        for full in self.proveedores_full:
            if full not in df.columns:
                continue
            count = int((df[full] > 0).sum())
            if count > best_count:
                best_count = count
                best_provider = full

        return best_provider

    # -------------------------------------------------------------------------
    # Pretty text report
    # -------------------------------------------------------------------------

    def pretty_cost_report(
        self,
        stats: Dict[str, Any],
        *,
        title_prefix: Optional[str] = None,
    ) -> str:
        """
        Turn a stats dict from a scenario into a human-readable Spanish report.
        """
        n_total = stats.get("n_total_shipments", 0)
        n_comp = stats.get("n_comparable", 0)
        n_unusable = stats.get("n_unusable", 0)
        current_provider = stats.get("current_provider", "Proveedor base")
        alternative_provider = stats.get("alternative_provider", "Proveedor alternativo")
        current_total = stats.get("current_total", np.nan)
        alternative_total = stats.get("alternative_total", np.nan)
        delta = stats.get("absolute_delta", np.nan)
        pct = stats.get("relative_delta_pct", np.nan)

        if (
            n_comp == 0
            or not np.isfinite(current_total)
            or not np.isfinite(alternative_total)
        ):
            msg = f"{title_prefix}\n\n" if title_prefix else ""
            msg += (
                "No hay envíos comparables donde ambos proveedores tengan tarifa > 0 "
                "para los filtros seleccionados.\n"
            )
            msg += f"Total de envíos en el período/filtros: {n_total}.\n"
            return msg

        sign_word = "más" if delta > 0 else "menos"
        direction = "gastar" if delta > 0 else "ahorrar"
        delta_abs = abs(delta)

        lines: list[str] = []

        if title_prefix:
            lines.append(title_prefix)
            lines.append("")

        lines.append(
            f"Sobre {n_total} envíos ({n_comp} comparables) en el período y filtros seleccionados:"
        )
        lines.append("")
        lines.append(
            f"- Con {current_provider} se hubieran pagado: {current_total:,.2f}"
        )
        lines.append(
            f"- Con {alternative_provider} se hubieran pagado: {alternative_total:,.2f}"
        )
        lines.append("")

        if np.isfinite(pct):
            lines.append(
                f"Eso implica {sign_word} costo total de {delta_abs:,.2f} "
                f"({pct:+.2f}%)."
            )
        else:
            lines.append(f"Eso implica {sign_word} costo total de {delta_abs:,.2f}.")

        lines.append(
            f"En otras palabras, la empresa habría llegado a {direction} "
            f"aproximadamente {delta_abs:,.2f} en ese conjunto de envíos."
        )

        if n_unusable > 0:
            lines.append("")
            lines.append(
                f"Nota: {n_unusable} envíos no se incluyeron en la comparación "
                f"porque al menos uno de los dos proveedores no tenía tarifa (> 0)."
            )

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Public API: direct provider vs provider
    # -------------------------------------------------------------------------

    def compare_two_providers_cost(
        self,
        current_provider: str,
        alternative_provider: str,
        *,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        provincia: Optional[str] = None,
        location: str = "both",
        rango_peso: Optional[str] = None,
        codigo_postal: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compare total cost using `current_provider` vs `alternative_provider`
        over a filtered subset of shipments.
        """
        df_f = self._filtered(
            date_from=date_from,
            date_to=date_to,
            provincia=provincia,
            location=location,
            rango_peso=rango_peso,
            codigo_postal=codigo_postal,
        )

        if len(df_f) == 0:
            return {
                "n_total_shipments": 0,
                "n_comparable": 0,
                "n_unusable": 0,
                "current_provider": self._resolve_full_name(current_provider),
                "alternative_provider": self._resolve_full_name(alternative_provider),
                "current_total": np.nan,
                "alternative_total": np.nan,
                "absolute_delta": np.nan,
                "relative_delta_pct": np.nan,
            }

        current_full = self._resolve_full_name(current_provider)
        alternative_full = self._resolve_full_name(alternative_provider)

        stats = self._compare_cost_two_providers_on_df(
            df_f, current_full, alternative_full
        )
        return stats

    # -------------------------------------------------------------------------
    # Public API: baseline (most used) vs MCDA #1
    # -------------------------------------------------------------------------

    def scenario_baseline_vs_mcda_best(
        self,
        *,
        # Filters
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        provincia: Optional[str] = None,
        location: str = "both",
        rango_peso: Optional[str] = None,
        codigo_postal: Optional[str] = None,
        # MCDA options (if None, let MCDAEngine use its own defaults)
        methods: Optional[Iterable[str] | str] = None,
        weights: Optional[Sequence[float]] = None,
        criteria_types: Optional[Sequence[str]] = None,
        features: Optional[Sequence[str]] = None,
        weights_preset: Optional[str] = None,
        smooth_approximation: Optional[bool] = None,
        normalize_globally: Optional[bool] = None,
        # FeatureBuilder aggregation options (if None, use its defaults)
        by_shipments: Optional[bool] = None,
        weight_by_range: Optional[bool] = None,
        # Baseline override
        baseline_provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Scenario: "baseline provider" vs "MCDA #1 provider" for the same filters.

        Note: parameters like methods, normalize_globally, by_shipments, etc.,
        are optional. If you leave them as None, this method does NOT override
        whatever defaults you configured inside MCDAEngine / FeatureBuilder.
        """
        # 1) Filter dataset
        df_f = self._filtered(
            date_from=date_from,
            date_to=date_to,
            provincia=provincia,
            location=location,
            rango_peso=rango_peso,
            codigo_postal=codigo_postal,
        )

        n_filtered = len(df_f)
        if n_filtered == 0:
            return {
                "n_total_shipments": 0,
                "n_comparable": 0,
                "n_unusable": 0,
                "current_provider": None,
                "alternative_provider": None,
                "current_total": np.nan,
                "alternative_total": np.nan,
                "absolute_delta": np.nan,
                "relative_delta_pct": np.nan,
                "mcda_best_provider_short": None,
                "mcda_best_provider_full": None,
                "mcda_methods": methods,
                "mcda_weights_preset": weights_preset,
            }

        # 2) Baseline provider
        if baseline_provider is not None:
            baseline_full = self._resolve_full_name(baseline_provider)
        else:
            baseline_full = self._most_used_provider_full(df_f)

        if baseline_full is None:
            return {
                "n_total_shipments": int(n_filtered),
                "n_comparable": 0,
                "n_unusable": int(n_filtered),
                "current_provider": None,
                "alternative_provider": None,
                "current_total": np.nan,
                "alternative_total": np.nan,
                "absolute_delta": np.nan,
                "relative_delta_pct": np.nan,
                "mcda_best_provider_short": None,
                "mcda_best_provider_full": None,
                "mcda_methods": methods,
                "mcda_weights_preset": weights_preset,
            }

        # 3) Build features & run MCDA on the same filtered df
        build_kwargs: Dict[str, Any] = dict(
            provincia=provincia,
            location=location,
            codigo_postal=codigo_postal,
            rango_peso=rango_peso,
        )
        if features is not None:
            build_kwargs["features"] = list(features)
        if by_shipments is not None:
            build_kwargs["by_shipments"] = by_shipments
        if weight_by_range is not None:
            build_kwargs["weight_by_range"] = weight_by_range

        df_metrics = self.feature_builder.build(df_f, **build_kwargs)

        if df_metrics is None or len(df_metrics) == 0:
            return {
                "n_total_shipments": int(n_filtered),
                "n_comparable": 0,
                "n_unusable": int(n_filtered),
                "current_provider": baseline_full,
                "alternative_provider": None,
                "current_total": np.nan,
                "alternative_total": np.nan,
                "absolute_delta": np.nan,
                "relative_delta_pct": np.nan,
                "mcda_best_provider_short": None,
                "mcda_best_provider_full": None,
                "mcda_methods": methods,
                "mcda_weights_preset": weights_preset,
            }

        score_kwargs: Dict[str, Any] = dict(
            return_df=True,
            sort=True,
            avg=False,
            feature_builder=self.feature_builder,
        )

        if methods is not None:
            score_kwargs["methods"] = methods
        if weights is not None:
            score_kwargs["weights"] = weights
        if criteria_types is not None:
            score_kwargs["criteria_types"] = criteria_types
        if features is not None:
            score_kwargs["features"] = list(features)
        if weights_preset is not None:
            score_kwargs["weights_preset"] = weights_preset
        if smooth_approximation is not None:
            score_kwargs["smooth_approximation"] = smooth_approximation
        if normalize_globally is not None:
            score_kwargs["normalize_globally"] = normalize_globally

        scores_df = self.mcda_engine.score(df_metrics, **score_kwargs)

        if scores_df is None or len(scores_df) == 0:
            return {
                "n_total_shipments": int(n_filtered),
                "n_comparable": 0,
                "n_unusable": int(n_filtered),
                "current_provider": baseline_full,
                "alternative_provider": None,
                "current_total": np.nan,
                "alternative_total": np.nan,
                "absolute_delta": np.nan,
                "relative_delta_pct": np.nan,
                "mcda_best_provider_short": None,
                "mcda_best_provider_full": None,
                "mcda_methods": methods,
                "mcda_weights_preset": weights_preset,
            }

        best_short = str(scores_df.index[0])
        best_full = self._resolve_full_name(best_short)

        stats = self._compare_cost_two_providers_on_df(df_f, baseline_full, best_full)

        stats["mcda_best_provider_short"] = best_short
        stats["mcda_best_provider_full"] = best_full
        stats["mcda_methods"] = methods
        stats["mcda_weights_preset"] = weights_preset

        return stats

    # -------------------------------------------------------------------------
    # Most-used provider summary
    # -------------------------------------------------------------------------

    def most_used_provider_summary(
        self,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
        provincia: str | None = None,
        location: str = "both",
        rango_peso: str | None = None,
        codigo_postal: str | None = None,
    ) -> dict:
        """
        Return which provider was the most used in the filtered subset.
        """
        df_f = self._filtered(
            date_from=date_from,
            date_to=date_to,
            provincia=provincia,
            location=location,
            rango_peso=rango_peso,
            codigo_postal=codigo_postal,
        )

        n_filtered = len(df_f)
        if n_filtered == 0:
            return {
                "n_total_shipments": 0,
                "most_used_full": None,
                "most_used_short": None,
                "count": 0,
            }

        full = self._most_used_provider_full(df_f)
        if full is None:
            return {
                "n_total_shipments": n_filtered,
                "most_used_full": None,
                "most_used_short": None,
                "count": 0,
            }

        short = self.full_to_short.get(full, full)

        if self.real_provider_col is not None and self.real_provider_col in df_f.columns:
            col = df_f[self.real_provider_col].dropna().astype(str)
            resolved = col.map(self._resolve_full_name)
            count = int((resolved == full).sum())
        else:
            if full in df_f.columns:
                count = int((df_f[full] > 0).sum())
            else:
                count = 0

        return {
            "n_total_shipments": int(n_filtered),
            "most_used_full": full,
            "most_used_short": short,
            "count": count,
        }

    # -------------------------------------------------------------------------
    # Cost scenario: most used vs all
    # -------------------------------------------------------------------------

    def scenario_most_used_vs_all(
        self,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
        provincia: str | None = None,
        location: str = "both",
        rango_peso: str | None = None,
        codigo_postal: str | None = None,
    ) -> str:
        """
        Texto resumen de costos: proveedor más usado vs todos los demás.
        """
        df_f = self._filtered(
            date_from=date_from,
            date_to=date_to,
            provincia=provincia,
            location=location,
            rango_peso=rango_peso,
            codigo_postal=codigo_postal,
        )

        n_filtered = len(df_f)
        if n_filtered == 0:
            return (
                "No hay envíos para los filtros seleccionados; no se puede "
                "calcular el proveedor más usado ni escenarios alternativos."
            )

        baseline_full = self._most_used_provider_full(df_f)
        if baseline_full is None:
            return (
                f"Hay {n_filtered} envíos con estos filtros, pero no se pudo "
                "inferir un proveedor más usado (no hay tarifas > 0 en ninguna columna)."
            )

        baseline_short = self.full_to_short.get(baseline_full, baseline_full)

        if baseline_full in df_f.columns:
            baseline_mask = df_f[baseline_full] > 0
            baseline_used_count = int(baseline_mask.sum())
            baseline_total_cost = float(df_f.loc[baseline_mask, baseline_full].sum())
        else:
            baseline_used_count = 0
            baseline_total_cost = float("nan")

        alt_stats = []
        for alt_full in self.proveedores_full:
            if alt_full == baseline_full:
                continue

            stats = self._compare_cost_two_providers_on_df(
                df_f, baseline_full, alt_full
            )
            if stats["n_comparable"] == 0:
                continue

            alt_short = self.full_to_short.get(alt_full, alt_full)

            alt_stats.append(
                {
                    "alt_short": alt_short,
                    "n_comparable": stats["n_comparable"],
                    "current_total": stats["current_total"],
                    "alternative_total": stats["alternative_total"],
                    "absolute_delta": stats["absolute_delta"],
                    "relative_delta_pct": stats["relative_delta_pct"],
                }
            )

        if not alt_stats:
            return (
                f"El proveedor más usado con estos filtros fue {baseline_short}, "
                f"pero no hay ningún otro proveedor con tarifas comparables (tarifa > 0) "
                f"en el mismo conjunto de envíos."
            )

        alt_stats.sort(key=lambda d: d["absolute_delta"])

        lines: list[str] = []

        lines.append(f"En total hay {n_filtered} envíos con los filtros seleccionados.")
        lines.append(
            f"El proveedor más usado fue **{baseline_short}**, con tarifa disponible "
            f"en aproximadamente {baseline_used_count} envíos."
        )
        lines.append("")
        lines.append(
            f"El costo total usando {baseline_short} en esos envíos fue: {baseline_total_cost:,.2f}"
        )
        lines.append("")
        lines.append("Si en lugar de ese proveedor hubiéramos usado otros:")
        lines.append("")

        for s in alt_stats:
            alt_short = s["alt_short"]
            n_comp = s["n_comparable"]
            delta = s["absolute_delta"]
            pct = s["relative_delta_pct"]
            delta_abs = abs(delta)

            if delta < 0:
                line = (
                    f"- Con **{alt_short}** se habría ahorrado aproximadamente "
                    f"{delta_abs:,.2f}"
                )
                if np.isfinite(pct):
                    line += f" ({pct:+.2f}%)"
                line += f" sobre {n_comp} envíos comparables."
            else:
                line = (
                    f"- Con **{alt_short}** se habría gastado aproximadamente "
                    f"{delta_abs:,.2f}"
                )
                if np.isfinite(pct):
                    line += f" ({pct:+.2f}%)"
                line += f" sobre {n_comp} envíos comparables."

            lines.append(line)

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Feature-based report: most used vs all
    # -------------------------------------------------------------------------

    def feature_report_most_used_vs_all(
        self,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
        provincia: str | None = None,
        location: str = "both",
        rango_peso: str | None = None,
        codigo_postal: str | None = None,
        features: Optional[Sequence[str]] = None,
        by_shipments: Optional[bool] = None,
        weight_by_range: Optional[bool] = None,
    ) -> str:
        """
        Reporte textual basado en las métricas del FeatureBuilder.
        """
        df_f = self._filtered(
            date_from=date_from,
            date_to=date_to,
            provincia=provincia,
            location=location,
            rango_peso=rango_peso,
            codigo_postal=codigo_postal,
        )

        n_filtered = len(df_f)
        if n_filtered == 0:
            return (
                "No hay envíos para los filtros seleccionados; no se pueden "
                "calcular métricas de proveedores."
            )

        baseline_full = self._most_used_provider_full(df_f)
        if baseline_full is None:
            return (
                f"Hay {n_filtered} envíos con estos filtros, pero no se pudo "
                "inferir un proveedor más usado (no hay tarifas > 0 en ninguna columna)."
            )

        baseline_short = self.full_to_short.get(baseline_full, baseline_full)

        build_kwargs: Dict[str, Any] = dict(
            provincia=provincia,
            location=location,
            codigo_postal=codigo_postal,
            rango_peso=rango_peso,
        )
        if features is not None:
            build_kwargs["features"] = list(features)
        if by_shipments is not None:
            build_kwargs["by_shipments"] = by_shipments
        if weight_by_range is not None:
            build_kwargs["weight_by_range"] = weight_by_range

        df_metrics = self.feature_builder.build(df_f, **build_kwargs)

        if df_metrics is None or len(df_metrics) == 0:
            return (
                f"El FeatureBuilder no devolvió métricas para los filtros seleccionados. "
                f"No se puede elaborar el reporte de métricas por proveedor."
            )

        if baseline_short not in df_metrics.index:
            if baseline_full in df_metrics.index:
                baseline_idx = baseline_full
            else:
                return (
                    f"El proveedor más usado ({baseline_short}) no aparece en la matriz "
                    f"de métricas; no se puede comparar contra el resto."
                )
        else:
            baseline_idx = baseline_short

        # Features a usar
        if features is None:
            features_used = [
                c
                for c in df_metrics.columns
                if np.issubdtype(df_metrics[c].dtype, np.number)
            ]
        else:
            features_used = [c for c in features if c in df_metrics.columns]

        if not features_used:
            return (
                "No hay features numéricas disponibles en la matriz de métricas "
                "para elaborar el reporte."
            )

        def is_cost_like(col_name: str) -> bool:
            name = col_name.lower()
            return ("cost" in name) or (name == "speed")

        def is_percentage_like(col_name: str) -> bool:
            name = col_name.lower()
            return any(
                key in name
                for key in ("first_visit", "delivery", "coverage", "sla", "sla_")
            )

        def format_value(col_name: str, val: float) -> str:
            if pd.isna(val):
                return "N/A"
            if is_percentage_like(col_name):
                return f"{val:.1f}%"
            if is_cost_like(col_name):
                return f"{val:.3f}"
            return f"{val:.2f}"

        def format_delta(col_name: str, delta: float) -> str:
            if pd.isna(delta):
                return "N/A"
            if is_percentage_like(col_name):
                signo = "+" if delta >= 0 else ""
                return f"{signo}{delta:.1f} pts"
            if is_cost_like(col_name):
                signo = "+" if delta >= 0 else ""
                return f"{signo}{delta:.3f}"
            signo = "+" if delta >= 0 else ""
            return f"{signo}{delta:.2f}"

        def feature_label(col_name: str) -> str:
            mapping = {
                "first_visit": "porcentaje de primera visita",
                "delivery": "porcentaje de entrega",
                "coverage": "cobertura de envíos",
                "SLA": "cumplimiento de SLA",
                "sla": "cumplimiento de SLA",
                "cost": "costo",
                "speed": "ventana de entrega (días)",
            }
            for k, v in mapping.items():
                if col_name.lower() == k.lower():
                    return v
            return col_name

        baseline_row = df_metrics.loc[baseline_idx, features_used]

        lines: list[str] = []

        lines.append(f"Hay {n_filtered} envíos con los filtros seleccionados.")
        lines.append(
            f"El proveedor más usado (según tarifas disponibles) es **{baseline_short}**."
        )
        lines.append("")
        lines.append("Indicadores promedio para el proveedor más usado en este conjunto:")

        for col in features_used:
            val = baseline_row[col]
            lines.append(f"- {feature_label(col)}: {format_value(col, val)}")

        lines.append("")
        lines.append(
            "Comparación con otros proveedores (sobre sus métricas promedio):"
        )
        lines.append("")

        for alt_short in df_metrics.index:
            if alt_short == baseline_idx:
                continue

            alt_row = df_metrics.loc[alt_short, features_used]

            better = []
            worse = []
            similar = []

            for col in features_used:
                base_val = baseline_row[col]
                alt_val = alt_row[col]

                if pd.isna(base_val) or pd.isna(alt_val):
                    continue

                delta = float(alt_val - base_val)
                tol = 1e-6 if not is_percentage_like(col) else 0.1

                if abs(delta) <= tol:
                    similar.append((col, delta))
                else:
                    if is_cost_like(col):
                        if delta < 0:
                            better.append((col, delta))
                        else:
                            worse.append((col, delta))
                    else:
                        if delta > 0:
                            better.append((col, delta))
                        else:
                            worse.append((col, delta))

            if not (better or worse):
                lines.append(
                    f"- **{alt_short}**: presenta métricas muy similares a {baseline_short} "
                    f"en las features consideradas."
                )
                continue

            desc_parts = []

            if better:
                txt = ", ".join(
                    f"{feature_label(col)} ({format_delta(col, d)})"
                    for col, d in better
                )
                desc_parts.append(f"mejor en {txt}")

            if worse:
                txt = ", ".join(
                    f"{feature_label(col)} ({format_delta(col, d)})"
                    for col, d in worse
                )
                desc_parts.append(f"peor en {txt}")

            desc = "; ".join(desc_parts)

            lines.append(f"- **{alt_short}**: {desc}.")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Combined report (metrics + costs)
    # -------------------------------------------------------------------------

    def combined_most_used_report(
        self,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
        provincia: str | None = None,
        location: str = "both",
        rango_peso: str | None = None,
        codigo_postal: str | None = None,
        features: Optional[Sequence[str]] = None,
        by_shipments: Optional[bool] = None,
        weight_by_range: Optional[bool] = None,
    ) -> str:
        """
        Reporte combinado que junta:
        1) feature_report_most_used_vs_all(...)
        2) scenario_most_used_vs_all(...)
        """
        metrics_text = self.feature_report_most_used_vs_all(
            date_from=date_from,
            date_to=date_to,
            provincia=provincia,
            location=location,
            rango_peso=rango_peso,
            codigo_postal=codigo_postal,
            features=features,
            by_shipments=by_shipments,
            weight_by_range=weight_by_range,
        )

        cost_text = self.scenario_most_used_vs_all(
            date_from=date_from,
            date_to=date_to,
            provincia=provincia,
            location=location,
            rango_peso=rango_peso,
            codigo_postal=codigo_postal,
        )

        lines: list[str] = []

        lines.append("=== REPORTE DE MÉTRICAS (FeatureBuilder) ===")
        lines.append("")
        lines.append(metrics_text.strip())
        lines.append("")
        lines.append("=== REPORTE DE COSTOS (Presupuestos) ===")
        lines.append("")
        lines.append(cost_text.strip())

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Helpers for CSV export (CP x Rango de Peso per provincia)
    # -------------------------------------------------------------------------

    def _sanitize_name(self, s: str) -> str:
        s = str(s)
        s = s.strip()
        s = s.replace(" ", "_")
        s = re.sub(r"[^0-9A-Za-z_\-]+", "", s)
        if not s:
            s = "unknown"
        return s

    def build_feature_matrices_for_provincia(
        self,
        provincia: str,
        feature: str,
        *,
        min_shipments: int = 1,
        methods: Optional[Iterable[str] | str] = None,
        weights: Optional[Sequence[float]] = None,
        criteria_types: Optional[Sequence[str]] = None,
        all_features_for_mcda: Optional[Sequence[str]] = None,
        weights_preset: Optional[str] = None,
        smooth_approximation: Optional[bool] = None,
        normalize_globally: Optional[bool] = None,
        by_shipments: Optional[bool] = None,
        weight_by_range: Optional[bool] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Construye matrices CP x Rango de Peso para una combinación (provincia, feature).

        Ahora soporta provincia="Todas" (o None) para construir la matriz global.
        Para evitar colisiones de CP entre provincias, en ese caso el índice es "Provincia | CP".

        Delta (para colores consistentes):
          - COST:    delta = real - mcda   (positivo => mejora/ahorro)
          - BENEFIT: delta = mcda - real   (positivo => mejora)
        """
        prov_in = provincia
        is_all = (prov_in is None) or (str(prov_in).strip().lower() in ("todas", "all", "*"))

        # Subset base
        if is_all:
            df_base = self.df.copy()
        else:
            df_base = self.df[self.df["Provincia"] == prov_in].copy()

        if df_base.empty:
            return {}

        if "Codigo Postal" not in df_base.columns or "Rango de Peso" not in df_base.columns:
            return {}

        # Decide delta convention
        is_cost = self._feature_is_cost(feature)

        # Build row labels (avoid CP collisions if "Todas")
        if is_all:
            df_base = df_base.dropna(subset=["Provincia", "Codigo Postal", "Rango de Peso"])
            row_keys = (
                df_base[["Provincia", "Codigo Postal"]]
                .drop_duplicates()
                .sort_values(["Provincia", "Codigo Postal"])
            )
            rows = [f"{p} | {cp}" for p, cp in row_keys.to_numpy()]
            # Map label -> (prov, cp)
            label_to_pair = {f"{p} | {cp}": (p, cp) for p, cp in row_keys.to_numpy()}
        else:
            cps = sorted(df_base["Codigo Postal"].dropna().unique())
            rows = list(cps)
            label_to_pair = {cp: (prov_in, cp) for cp in cps}

        rangos = sorted(df_base["Rango de Peso"].dropna().unique())
        if not rows or not rangos:
            return {}

        shape = (len(rows), len(rangos))
        mat_real_val = np.full(shape, np.nan, dtype=float)
        mat_mcda_val = np.full(shape, np.nan, dtype=float)
        mat_delta = np.full(shape, np.nan, dtype=float)
        mat_real_short = np.full(shape, "", dtype=object)
        mat_mcda_short = np.full(shape, "", dtype=object)

        for i, row_label in enumerate(rows):
            prov_cell, cp = label_to_pair[row_label]
            for j, rp in enumerate(rangos):
                df_cell = df_base[
                    (df_base["Provincia"] == prov_cell)
                    & (df_base["Codigo Postal"] == cp)
                    & (df_base["Rango de Peso"] == rp)
                ]

                if len(df_cell) < min_shipments:
                    continue

                # --- proveedor "real" (más usado) ---
                baseline_full = self._most_used_provider_full(df_cell)
                if baseline_full is None:
                    continue
                baseline_short = self.full_to_short.get(baseline_full, baseline_full)

                # --- construir métricas para esta celda ---
                build_kwargs: Dict[str, Any] = dict(
                    location="both",
                    codigo_postal=cp,
                    rango_peso=rp,
                )
                # Solo pasar provincia si NO es "Todas"
                if not is_all:
                    build_kwargs["provincia"] = prov_in
                else:
                    # para el builder, si le sirve, pasamos la provincia real de la celda
                    build_kwargs["provincia"] = prov_cell

                if by_shipments is not None:
                    build_kwargs["by_shipments"] = by_shipments
                if weight_by_range is not None:
                    build_kwargs["weight_by_range"] = weight_by_range

                df_metrics = self.feature_builder.build(df_cell, **build_kwargs)
                if df_metrics is None or df_metrics.empty:
                    continue

                if feature not in df_metrics.columns:
                    continue

                if baseline_short in df_metrics.index:
                    baseline_idx = baseline_short
                elif baseline_full in df_metrics.index:
                    baseline_idx = baseline_full
                else:
                    continue

                try:
                    real_val = float(df_metrics.loc[baseline_idx, feature])
                except Exception:
                    continue

                # --- MCDA ranking para esta celda ---
                all_feats_for_mcda_local = (
                    list(all_features_for_mcda)
                    if all_features_for_mcda is not None
                    else list(df_metrics.columns)
                )

                score_kwargs: Dict[str, Any] = dict(
                    return_df=True,
                    sort=True,
                    avg=False,
                    feature_builder=self.feature_builder,
                    features=all_feats_for_mcda_local,
                )
                if methods is not None:
                    score_kwargs["methods"] = methods
                if weights is not None:
                    score_kwargs["weights"] = weights
                if criteria_types is not None:
                    score_kwargs["criteria_types"] = criteria_types
                if weights_preset is not None:
                    score_kwargs["weights_preset"] = weights_preset
                if smooth_approximation is not None:
                    score_kwargs["smooth_approximation"] = smooth_approximation
                if normalize_globally is not None:
                    score_kwargs["normalize_globally"] = normalize_globally

                try:
                    scores_df = self.mcda_engine.score(df_metrics, **score_kwargs)
                except Exception:
                    continue

                if scores_df is None or scores_df.empty:
                    continue

                mcda_short = str(scores_df.index[0])
                if mcda_short not in df_metrics.index:
                    continue

                try:
                    mcda_val = float(df_metrics.loc[mcda_short, feature])
                except Exception:
                    continue

                # --- Delta convention for consistent colors ---
                if is_cost:
                    delta_val = real_val - mcda_val
                else:
                    delta_val = mcda_val - real_val

                mat_real_val[i, j] = real_val
                mat_mcda_val[i, j] = mcda_val
                mat_delta[i, j] = delta_val
                mat_real_short[i, j] = baseline_short
                mat_mcda_short[i, j] = mcda_short

        idx = pd.Index(rows, name="Codigo Postal" if not is_all else "Provincia | Codigo Postal")
        cols = pd.Index(rangos, name="Rango de Peso")

        df_real_val = pd.DataFrame(mat_real_val, index=idx, columns=cols)
        df_mcda_val = pd.DataFrame(mat_mcda_val, index=idx, columns=cols)
        df_delta = pd.DataFrame(mat_delta, index=idx, columns=cols)
        df_real_short = pd.DataFrame(mat_real_short, index=idx, columns=cols)
        df_mcda_short = pd.DataFrame(mat_mcda_short, index=idx, columns=cols)

        return {
            "valores_real": df_real_val,
            "valores_mcda": df_mcda_val,
            # Backward compatible key:
            "delta_mcda_menos_real": df_delta,
            # Neutral key (recommended going forward):
            "delta": df_delta,
            "proveedor_real": df_real_short,
            "proveedor_mcda": df_mcda_short,
        }



    def export_feature_matrices_for_provincias(
        self,
        provincias: Optional[Union[str, Sequence[str]]] = None,
        features: Optional[Sequence[str]] = None,
        *,
        out_dir: str = "report_matrices",
        min_shipments: int = 1,
        methods: Optional[Iterable[str] | str] = None,
        weights: Optional[Sequence[float]] = None,
        criteria_types: Optional[Sequence[str]] = None,
        weights_preset: Optional[str] = None,
        smooth_approximation: Optional[bool] = None,
        normalize_globally: Optional[bool] = None,
        by_shipments: Optional[bool] = None,
        weight_by_range: Optional[bool] = None,
    ) -> Dict[str, list[str]]:
        """
        Genera CSVs con filas CP x Rango de Peso y columnas:

        - Provincia
        - Codigo Postal
        - Rango de Peso
        - proveedor_real         (más usado en la celda)
        - proveedor_mcda         (top según MCDA)
        - Para cada feature f en `features`:
            f"_real"   -> valor de f para proveedor_real
            f"_mcda"   -> valor de f para proveedor_mcda
            f"_delta"  -> convención consistente:
                COST:    delta = real - mcda   (positivo => mejora/ahorro)
                BENEFIT: delta = mcda - real   (positivo => mejora)

        Comportamiento de `provincias`:
        - None / "Todas" / "all" / "*": exporta UN SOLO CSV global (todas las provincias),
        agregando la columna "Provincia".
        - str (p.ej. "Corrientes"): exporta un CSV para esa provincia.
        - Sequence[str]: exporta un CSV por provincia en la lista (comportamiento histórico).

        Devuelve:
            { clave: [ruta_csv_creada, ...] }
            - Para global: {"Todas": [ruta]}
            - Para una provincia: {provincia: [ruta]}
            - Para lista: {prov: [ruta], ...}
        """
        if "Provincia" not in self.df.columns:
            raise ValueError("La columna 'Provincia' no está en el DataFrame.")

        # --- provincias a procesar ---
        def _is_all_token(x: Optional[Union[str, Sequence[str]]]) -> bool:
            if x is None:
                return True
            if isinstance(x, str):
                return str(x).strip().lower() in ("todas", "all", "*")
            return False

        export_global = _is_all_token(provincias)

        if export_global:
            provs = sorted(self.df["Provincia"].dropna().unique())
        elif isinstance(provincias, str):
            provs = [provincias]
        else:
            provs = list(provincias) if provincias is not None else sorted(self.df["Provincia"].dropna().unique())

        # --- features a procesar ---
        if features is None:
            feats = list(getattr(self.feature_builder, "available_features", []))
        else:
            feats = list(features)

        if not feats:
            raise ValueError("No hay features para procesar (lista vacía).")

        # carpeta de salida
        os.makedirs(out_dir, exist_ok=True)

        # Resultado
        if export_global:
            result: Dict[str, list[str]] = {"Todas": []}
        else:
            result = {p: [] for p in provs}

        # ------------------------------------------------------------------
        # MODO GLOBAL: UN SOLO CSV (todas las provincias juntas)
        # ------------------------------------------------------------------
        if export_global:
            df_base = self.df.copy()
            if df_base.empty:
                return result

            if "Codigo Postal" not in df_base.columns or "Rango de Peso" not in df_base.columns:
                return result

            # Para evitar combinaciones inválidas, iteramos por provincia y luego por CP/RP
            rows: list[dict] = []

            for prov in provs:
                df_prov = df_base[df_base["Provincia"] == prov].copy()
                if df_prov.empty:
                    continue

                cps = sorted(df_prov["Codigo Postal"].dropna().unique())
                rangos = sorted(df_prov["Rango de Peso"].dropna().unique())

                for cp in cps:
                    for rp in rangos:
                        df_cell = df_prov[
                            (df_prov["Codigo Postal"] == cp)
                            & (df_prov["Rango de Peso"] == rp)
                        ]

                        n_rows = len(df_cell)
                        if n_rows < min_shipments:
                            continue

                        # --- proveedor "real" (más usado) ---
                        baseline_full = self._most_used_provider_full(df_cell)
                        if baseline_full is None:
                            continue
                        baseline_short = self.full_to_short.get(baseline_full, baseline_full)

                        # --- construir métricas para esta celda ---
                        build_kwargs: Dict[str, Any] = dict(
                            provincia=prov,
                            location="both",
                            codigo_postal=cp,
                            rango_peso=rp,
                        )
                        if by_shipments is not None:
                            build_kwargs["by_shipments"] = by_shipments
                        if weight_by_range is not None:
                            build_kwargs["weight_by_range"] = weight_by_range

                        df_metrics = self.feature_builder.build(df_cell, **build_kwargs)
                        if df_metrics is None or df_metrics.empty:
                            continue

                        # índice del proveedor real
                        if baseline_short in df_metrics.index:
                            baseline_idx = baseline_short
                        elif baseline_full in df_metrics.index:
                            baseline_idx = baseline_full
                        else:
                            continue

                        # --- MCDA ranking para esta celda ---
                        all_feats_for_mcda = list(df_metrics.columns)

                        score_kwargs: Dict[str, Any] = dict(
                            return_df=True,
                            sort=True,
                            avg=False,
                            feature_builder=self.feature_builder,
                            features=all_feats_for_mcda,
                        )
                        if methods is not None:
                            score_kwargs["methods"] = methods
                        if weights is not None:
                            score_kwargs["weights"] = weights
                        if criteria_types is not None:
                            score_kwargs["criteria_types"] = criteria_types
                        if weights_preset is not None:
                            score_kwargs["weights_preset"] = weights_preset
                        if smooth_approximation is not None:
                            score_kwargs["smooth_approximation"] = smooth_approximation
                        if normalize_globally is not None:
                            score_kwargs["normalize_globally"] = normalize_globally

                        try:
                            scores_df = self.mcda_engine.score(df_metrics, **score_kwargs)
                        except Exception:
                            continue

                        if scores_df is None or scores_df.empty:
                            continue

                        mcda_short = str(scores_df.index[0])
                        if mcda_short not in df_metrics.index:
                            continue

                        # --- construir la fila de salida ---
                        row: Dict[str, Any] = {
                            "Provincia": prov,
                            "Codigo Postal": cp,
                            "Rango de Peso": rp,
                            "proveedor_real": baseline_short,
                            "proveedor_mcda": mcda_short,
                        }

                        for ft in feats:
                            real_col = f"{ft}_real"
                            mcda_col = f"{ft}_mcda"
                            delta_col = f"{ft}_delta"

                            if ft in df_metrics.columns:
                                try:
                                    real_val = float(df_metrics.loc[baseline_idx, ft])
                                    mcda_val = float(df_metrics.loc[mcda_short, ft])
                                except Exception:
                                    real_val = np.nan
                                    mcda_val = np.nan

                                if np.isfinite(real_val) and np.isfinite(mcda_val):
                                    if self._feature_is_cost(ft):
                                        delta_val = real_val - mcda_val
                                    else:
                                        delta_val = mcda_val - real_val
                                else:
                                    delta_val = np.nan
                            else:
                                real_val = np.nan
                                mcda_val = np.nan
                                delta_val = np.nan

                            row[real_col] = real_val
                            row[mcda_col] = mcda_val
                            row[delta_col] = delta_val

                        rows.append(row)

            if not rows:
                return result

            df_out = pd.DataFrame(rows)

            # orden de columnas
            base_cols = ["Provincia", "Codigo Postal", "Rango de Peso", "proveedor_real", "proveedor_mcda"]
            real_cols = [f"{ft}_real" for ft in feats]
            mcda_cols = [f"{ft}_mcda" for ft in feats]
            delta_cols = [f"{ft}_delta" for ft in feats]

            col_order = [c for c in (base_cols + real_cols + mcda_cols + delta_cols) if c in df_out.columns]
            df_out = df_out[col_order]

            filename = "report_matrices_todas.csv"
            filepath = os.path.join(out_dir, filename)
            df_out.to_csv(filepath, index=False, encoding="utf-8")

            result["Todas"].append(filepath)
            return result

        # ------------------------------------------------------------------
        # MODO HISTÓRICO: UN CSV POR PROVINCIA (para str o lista)
        # ------------------------------------------------------------------
        for prov in provs:
            df_prov = self.df[self.df["Provincia"] == prov].copy()
            if df_prov.empty:
                continue

            if "Codigo Postal" not in df_prov.columns or "Rango de Peso" not in df_prov.columns:
                continue

            cps = sorted(df_prov["Codigo Postal"].dropna().unique())
            rangos = sorted(df_prov["Rango de Peso"].dropna().unique())

            rows: list[dict] = []

            for cp in cps:
                for rp in rangos:
                    df_cell = df_prov[
                        (df_prov["Codigo Postal"] == cp)
                        & (df_prov["Rango de Peso"] == rp)
                    ]

                    n_rows = len(df_cell)
                    if n_rows < min_shipments:
                        continue

                    baseline_full = self._most_used_provider_full(df_cell)
                    if baseline_full is None:
                        continue
                    baseline_short = self.full_to_short.get(baseline_full, baseline_full)

                    build_kwargs: Dict[str, Any] = dict(
                        provincia=prov,
                        location="both",
                        codigo_postal=cp,
                        rango_peso=rp,
                    )
                    if by_shipments is not None:
                        build_kwargs["by_shipments"] = by_shipments
                    if weight_by_range is not None:
                        build_kwargs["weight_by_range"] = weight_by_range

                    df_metrics = self.feature_builder.build(df_cell, **build_kwargs)
                    if df_metrics is None or df_metrics.empty:
                        continue

                    if baseline_short in df_metrics.index:
                        baseline_idx = baseline_short
                    elif baseline_full in df_metrics.index:
                        baseline_idx = baseline_full
                    else:
                        continue

                    all_feats_for_mcda = list(df_metrics.columns)

                    score_kwargs: Dict[str, Any] = dict(
                        return_df=True,
                        sort=True,
                        avg=False,
                        feature_builder=self.feature_builder,
                        features=all_feats_for_mcda,
                    )
                    if methods is not None:
                        score_kwargs["methods"] = methods
                    if weights is not None:
                        score_kwargs["weights"] = weights
                    if criteria_types is not None:
                        score_kwargs["criteria_types"] = criteria_types
                    if weights_preset is not None:
                        score_kwargs["weights_preset"] = weights_preset
                    if smooth_approximation is not None:
                        score_kwargs["smooth_approximation"] = smooth_approximation
                    if normalize_globally is not None:
                        score_kwargs["normalize_globally"] = normalize_globally

                    try:
                        scores_df = self.mcda_engine.score(df_metrics, **score_kwargs)
                    except Exception:
                        continue

                    if scores_df is None or scores_df.empty:
                        continue

                    mcda_short = str(scores_df.index[0])
                    if mcda_short not in df_metrics.index:
                        continue

                    row: Dict[str, Any] = {
                        "Codigo Postal": cp,
                        "Rango de Peso": rp,
                        "proveedor_real": baseline_short,
                        "proveedor_mcda": mcda_short,
                    }

                    for ft in feats:
                        real_col = f"{ft}_real"
                        mcda_col = f"{ft}_mcda"
                        delta_col = f"{ft}_delta"

                        if ft in df_metrics.columns:
                            real_val = float(df_metrics.loc[baseline_idx, ft])
                            mcda_val = float(df_metrics.loc[mcda_short, ft])

                            if self._feature_is_cost(ft):
                                delta_val = real_val - mcda_val
                            else:
                                delta_val = mcda_val - real_val
                        else:
                            real_val = np.nan
                            mcda_val = np.nan
                            delta_val = np.nan

                        row[real_col] = real_val
                        row[mcda_col] = mcda_val
                        row[delta_col] = delta_val

                    rows.append(row)

            if not rows:
                continue

            df_out = pd.DataFrame(rows)

            base_cols = ["Codigo Postal", "Rango de Peso", "proveedor_real", "proveedor_mcda"]
            real_cols = [f"{ft}_real" for ft in feats]
            mcda_cols = [f"{ft}_mcda" for ft in feats]
            delta_cols = [f"{ft}_delta" for ft in feats]

            col_order = [c for c in (base_cols + real_cols + mcda_cols + delta_cols) if c in df_out.columns]
            df_out = df_out[col_order]

            prov_slug = self._sanitize_name(prov)
            filename = f"report_matrices_{prov_slug}.csv"
            filepath = os.path.join(out_dir, filename)

            df_out.to_csv(filepath, index=False, encoding="utf-8")

            result[prov].append(filepath)

        return result


