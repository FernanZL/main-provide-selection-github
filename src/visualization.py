from __future__ import annotations
from typing import Tuple, Optional, Sequence

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.features import FeatureBuilder  # adjust import if needed
from src.mcda import MCDAEngine


class ShippingVisualizer:
    """
    Helper class for plotting shipping metrics and MCDA scores.
    """

    @staticmethod
    def plot_features_company_province(
        cleaned_df: pd.DataFrame,
        fb: FeatureBuilder,
        carrier: str,
        x_feature: str = "first_visit",
        y_feature: str = "cost",
        provincias: list[str] | None = None,
        location: str = "both",
        figsize: Tuple[float, float] = (12, 8),
        dpi: int = 120,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Scatter plot of two features (x_feature, y_feature) for a single carrier,
        with one dot per Provincia.

        Features are computed via FeatureBuilder.build() for each Provincia.

        Parameters
        ----------
        cleaned_df : pd.DataFrame
            Cleaned dataset (the same one you pass to FeatureBuilder).
        fb : FeatureBuilder
            An instance of FeatureBuilder already configured.
        carrier : str
            Short provider name (one of fb.proveedores_short).
            This is the index label in the feature DataFrames.
        x_feature : str, default 'first_visit'
            Feature name for the X axis (must be in fb.available_features).
        y_feature : str, default 'cost'
            Feature name for the Y axis (must be in fb.available_features).
        provincias : list[str], optional
            List of provincias to include. If None, uses all distinct
            provincias in cleaned_df["Provincia"].
        location : {'both', 'capital', 'interior'}, default 'both'
            Passed through to FeatureBuilder.build().
        figsize : (float, float), default (12, 8)
            Figure size in inches.
        dpi : int, default 120
            Figure DPI.
        ax : matplotlib.axes.Axes, optional
            Existing axis to draw on. If None, a new figure/axis is created.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axis with the plot drawn on it.
        """

        # --- Validate requested features ---
        for feat in (x_feature, y_feature):
            if feat not in fb.available_features:
                raise ValueError(
                    f"Feature '{feat}' not available. "
                    f"Available: {fb.available_features}"
                )

        if provincias is None:
            provincias = (
                cleaned_df["Provincia"]
                .dropna()
                .unique()
                .tolist()
            )

        xs: list[float] = []
        ys: list[float] = []
        labels: list[str] = []

        # --- For each provincia, compute features and pick the carrier's values ---
        for prov in provincias:
            try:
                feat_df = fb.build(
                    df=cleaned_df,
                    features=[x_feature, y_feature],
                    drop_incomplete=False,   # keep incomplete data
                    provincia=prov,
                    location=location,
                )
            except Exception:
                # if FeatureBuilder throws any error for this province, skip
                continue

            if feat_df is None or feat_df.empty:
                continue

            # If this carrier is not present (all NaN etc.), skip
            if carrier not in feat_df.index:
                continue

            row = feat_df.loc[carrier]

            x_val = row.get(x_feature, np.nan)
            y_val = row.get(y_feature, np.nan)

            # Skip invalid or NaN values
            if pd.isna(x_val) or pd.isna(y_val):
                continue

            xs.append(float(x_val))
            ys.append(float(y_val))
            labels.append(str(prov))

        # --- Handle case with no valid data ---
        if not xs:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax.text(
                0.5, 0.5,
                f"No valid data for '{carrier}'",
                ha="center", va="center",
                fontsize=12, color="gray"
            )
            ax.set_axis_off()
            return ax

        # --- Create axis/figure if needed ---
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        ax.scatter(xs, ys, s=80)

        # Annotate each point with the provincia name
        for x, y, label in zip(xs, ys, labels):
            ax.text(
                x,
                y,
                label,
                fontsize=10,
                ha="center",
                va="bottom",
            )

        # Pretty labels
        pretty_names = {
            "cost": "Average cost (ratio)",
            "cost_abs": "Average absolute cost",
            "cost_actual": "Actual cost / kg",
            "first_visit": "First visit success (%)",
            "coverage": "Coverage (%)",
            "price_std": "Std dev price/kg",
            "cheap_ratio": "% of times cheapest",
        }
        x_label = pretty_names.get(x_feature, x_feature)
        y_label = pretty_names.get(y_feature, y_feature)

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(
            f"{carrier}: {y_label} vs {x_label} by Provincia",
            fontsize=14,
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return ax

    @staticmethod
    def plot_scores_by_weight_preset(
        cleaned_df: pd.DataFrame,
        fb: FeatureBuilder,
        mcda: MCDAEngine,
        provincia: str | None = None,
        location: str = "both",
        features: Sequence[str] | None = None,
        method: str = "waspas",
        weight_presets: Sequence[str] | None = None,
        figsize: tuple[float, float] = (10, 6),
        dpi: int = 120,
        title: str | None = None,
        legend: bool = True,
    ) -> plt.Figure:
        """
        Plot evolution of each carrier's MCDA score across different weight presets.

        X-axis: weight presets (e.g. 'balanced', 'quality_over_cost', ...)
        Y-axis: score from the chosen MCDA method (default: WASPAS).

        Metrics are computed via FeatureBuilder for a single provincia (optional)
        and location, then MCDAEngine.score is run once per weight preset.

        Parameters
        ----------
        cleaned_df : pd.DataFrame
            Cleaned dataset (input to FeatureBuilder).
        fb : FeatureBuilder
            FeatureBuilder instance.
        mcda : MCDAEngine
            MCDAEngine instance (with weight_presets defined).
        provincia : str, optional
            Provincia to filter on when computing features. If None, use all provincias.
        location : {'both', 'capital', 'interior'}, default 'both'
            Passed through to FeatureBuilder.build().
        features : sequence of str, optional
            Criteria/features to use in MCDA. If None, defaults to
            ['first_visit', 'cost', 'coverage'].
        method : str, default 'waspas'
            MCDA method name to use ('weighted', 'topsis', 'vikor', 'waspas').
        weight_presets : sequence of str, optional
            Subset of mcda.available_weight_presets to plot. If None, use all.
        figsize : tuple, default (10, 6)
            Figure size in inches.
        dpi : int, default 120
            Resolution of the figure.
        title : str, optional
            Custom plot title. If None, an automatic title is used.
        legend : bool, default True
            Whether to display the legend.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure.
        """

        # --- default features ---
        if features is None:
            features = ["first_visit", "cost", "coverage"]
        features = list(features)

        # --- build metrics for given provincia / location ---
        df_metrics = fb.build(
            df=cleaned_df,
            features=features,
            drop_incomplete=True,
            provincia=provincia,
            location=location,
        )

        if df_metrics.empty:
            raise ValueError("No metrics available for the specified filters.")

        # --- which presets to use ---
        if weight_presets is None:
            presets = mcda.available_weight_presets
        else:
            presets = list(weight_presets)

        if not presets:
            raise ValueError("No weight presets provided/available.")

        # validate presets
        for p in presets:
            if p not in mcda.weight_presets:
                raise ValueError(
                    f"Unknown weight_preset '{p}'. "
                    f"Available: {mcda.available_weight_presets}"
                )

        # --- compute scores for each preset ---
        method = method.lower()
        score_col = f"Score_{method}"

        score_table = pd.DataFrame(index=df_metrics.index, columns=presets, dtype=float)

        for p in presets:
            df_scores = mcda.score(
                df_metrics=df_metrics,
                methods=method,
                features=features,
                feature_builder=fb,
                weights_preset=p,
                return_df=True,
                sort=False,   # keep index order consistent across presets
            )

            if score_col not in df_scores.columns:
                raise ValueError(
                    f"Expected column '{score_col}' not found in MCDA output for preset '{p}'."
                )

            score_table[p] = df_scores[score_col]

        # --- plot ---
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for proveedor in score_table.index:
            ax.plot(
                presets,
                score_table.loc[proveedor],
                marker="o",
                label=proveedor,
            )

        if title is None:
            base_title = f"Score {method.upper()} por preset de pesos"
            if provincia is not None:
                base_title += f" – Provincia: {provincia}"
            #base_title += f" (location={location})"
            title = base_title

        ax.set_ylabel("Score (higher = better)", fontsize=12)
        ax.set_xlabel("Preset de pesos", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(alpha=0.3)

        if legend:
            ax.legend(title="Proveedor", bbox_to_anchor=(1.05, 1), loc="upper left")

        fig.tight_layout()
        return fig

       

    @staticmethod
    def plot_scores_by_method(
        cleaned_df: pd.DataFrame,
        fb: FeatureBuilder,
        mcda: MCDAEngine,
        provincia: str | None = None,
        location: str = "both",
        features: Sequence[str] | None = None,
        methods: Sequence[str] | None = None,
        weights_preset: str | None = None,
        figsize: tuple[float, float] = (10, 6),
        dpi: int = 120,
        title: str | None = None,
        legend: bool = True,
    ) -> plt.Figure:
        """
        Plot evolution of each carrier's scores across MCDA methods.

        X-axis: MCDA methods (weighted, topsis, vikor, waspas, ...)
        Y-axis: score for each method, for each provider.

        Metrics are computed via FeatureBuilder for a single provincia (optional)
        and location, then MCDAEngine.score is run once per method.

        Parameters
        ----------
        cleaned_df : pd.DataFrame
            Cleaned dataset (input to FeatureBuilder).
        fb : FeatureBuilder
            FeatureBuilder instance.
        mcda : MCDAEngine
            MCDAEngine instance.
        provincia : str, optional
            Provincia to filter on when computing features. If None, use all provincias.
        location : {'both', 'capital', 'interior'}, default 'both'
            Passed through to FeatureBuilder.build().
        features : sequence of str, optional
            Criteria/features to use in MCDA. If None, defaults to
            ['first_visit', 'cost', 'coverage'].
        methods : sequence of str, optional
            MCDA methods to compare. If None, uses mcda.methods in sorted order.
        weights_preset : str, optional
            Name of a weight preset to use for all methods
            (e.g. 'balanced', 'quality_over_cost'). If None, MCDAEngine
            falls back to equal weights.
        figsize : tuple, default (10, 6)
            Figure size in inches.
        dpi : int, default 120
            Resolution.
        title : str, optional
            Custom title. If None, an automatic title is used.
        legend : bool, default True
            Whether to show the legend.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure.
        """

        # --- default features ---
        if features is None:
            features = ["first_visit", "cost", "coverage"]
        features = list(features)

        # --- build metrics for given provincia / location ---
        df_metrics = fb.build(
            df=cleaned_df,
            features=features,
            drop_incomplete=True,
            provincia=provincia,
            location=location,
        )

        if df_metrics.empty:
            raise ValueError("No metrics available for the specified filters.")

        # --- which methods to use ---
        if methods is None:
            # use the engine's configured methods, in a stable order
            methods = sorted(list(mcda.methods))
        else:
            methods = [m.lower() for m in methods]

        if not methods:
            raise ValueError("No MCDA methods provided/available.")

        # validate against MCDAEngine.methods
        for m in methods:
            if m not in mcda.methods:
                raise ValueError(
                    f"Unknown method '{m}'. "
                    f"Available in MCDAEngine: {mcda.methods}"
                )

        # --- compute scores for each method ---
        score_table = pd.DataFrame(index=df_metrics.index, columns=methods, dtype=float)

        for m in methods:
            df_scores = mcda.score(
                df_metrics=df_metrics,
                methods=m,
                features=features,
                feature_builder=fb,
                weights_preset=weights_preset,
                return_df=True,
                sort=False,   # keep index order consistent across methods
            )

            score_col = f"Score_{m}"
            if score_col not in df_scores.columns:
                raise ValueError(
                    f"Expected column '{score_col}' not found in MCDA output for method '{m}'."
                )

            score_table[m] = df_scores[score_col]

        # --- plot ---
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for proveedor in score_table.index:
            ax.plot(
                methods,
                score_table.loc[proveedor],
                marker="o",
                label=proveedor,
            )

        # default title
        if title is None:
            base_title = "Evolución de los scores por método MCDA"
            if provincia is not None:
                base_title += f" – Provincia: {provincia}"
            #base_title += f" (location={location})"
            if weights_preset is not None:
                base_title += f" – preset: {weights_preset}"
            title = base_title

        ax.set_ylabel("Score (higher = better)", fontsize=12)
        ax.set_xlabel("Método MCDA", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(alpha=0.3)

        if legend:
            ax.legend(title="Proveedor", bbox_to_anchor=(1.05, 1), loc="upper left")

        fig.tight_layout()
        return fig
   

    

    @staticmethod
    def plot_scores_bar(
        cleaned_df: pd.DataFrame,
        fb: FeatureBuilder,
        mcda: MCDAEngine,
        provincia: str | None = None,
        location: str = "both",
        features: Sequence[str] | None = None,
        methods: Sequence[str] | None = None,
        weights_preset: str | None = None,
        figsize: tuple[float, float] = (10, 6),
        dpi: int = 120,
        title: str | None = None,
        ylabel: str = "Score (higher = better)",
        legend: bool = True,
        rotation: int = 45,
    ) -> plt.Figure:
        """
        Bar plot comparing MCDA method scores for each carrier (provider).

        Uses the full pipeline:
        - FeatureBuilder.build(...) to compute metrics for a given provincia/location
        - MCDAEngine.score(...) to compute scores for one or more methods

        Parameters
        ----------
        cleaned_df : pd.DataFrame
            Cleaned dataset (input to FeatureBuilder).
        fb : FeatureBuilder
            FeatureBuilder instance.
        mcda : MCDAEngine
            MCDAEngine instance.
        provincia : str, optional
            Provincia to filter on when computing features. If None, use all provincias.
        location : {'both', 'capital', 'interior'}, default 'both'
            Passed through to FeatureBuilder.build().
        features : sequence of str, optional
            Criteria/features to use in MCDA. If None, defaults to
            ['first_visit', 'cost', 'coverage'].
        methods : sequence of str, optional
            MCDA methods to compare. If None, uses mcda.methods in sorted order.
        weights_preset : str, optional
            Name of a weight preset to use for all methods
            (e.g. 'balanced', 'quality_over_cost'). If None, MCDAEngine
            falls back to equal weights.
        figsize : tuple, default (10, 6)
            Size of the figure in inches.
        dpi : int, default 120
            Resolution of the figure.
        title : str, optional
            Title of the plot. If None, an automatic title is used.
        ylabel : str, default "Score (higher = better)"
            Label for the Y-axis.
        legend : bool, default True
            Whether to display the legend.
        rotation : int, default 45
            Rotation angle for X-tick labels.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        """

        # --- default features ---
        if features is None:
            features = ["first_visit", "cost", "coverage"]
        features = list(features)

        # --- build metrics for given provincia / location ---
        df_metrics = fb.build(
            df=cleaned_df,
            features=features,
            drop_incomplete=True,
            provincia=provincia,
            location=location,
        )

        if df_metrics.empty:
            raise ValueError("No metrics available for the specified filters.")

        # --- which methods to use ---
        if methods is None:
            methods = sorted(list(mcda.methods))
        else:
            methods = [m.lower() for m in methods]

        if not methods:
            raise ValueError("No MCDA methods provided/available.")

        for m in methods:
            if m not in mcda.methods:
                raise ValueError(
                    f"Unknown method '{m}'. "
                    f"Available in MCDAEngine: {mcda.methods}"
                )

        # --- compute scores (single call for all methods) ---
        df_scores = mcda.score(
            df_metrics=df_metrics,
            methods=methods,
            features=features,
            feature_builder=fb,
            weights_preset=weights_preset,
            return_df=True,
            sort=False,  # keep provider order
        )

        # --- get score columns in the same order as 'methods' ---
        score_cols = [f"Score_{m}" for m in methods if f"Score_{m}" in df_scores.columns]
        if not score_cols:
            raise ValueError(
                f"No 'Score_*' columns found for methods {methods} in MCDA output."
            )

        # --- plot ---
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        df_scores[score_cols].plot(kind="bar", ax=ax)

        ax.set_ylabel(ylabel, fontsize=12)

        # automatic title
        if title is None:
            base_title = "Comparación de métodos MCDA por proveedor"
            if provincia is not None:
                base_title += f" – Provincia: {provincia}"
            #base_title += f" (location={location})"
            if weights_preset is not None:
                base_title += f" – preset: {weights_preset}"
            title = base_title

        ax.set_title(title, fontsize=14)
        ax.set_xticks(range(len(df_scores.index)))
        ax.set_xticklabels(df_scores.index, rotation=rotation, ha="right")

        if legend:
            ax.legend(title="Método", bbox_to_anchor=(1.05, 1), loc="upper left")

        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        return fig

    @staticmethod
    def first_visit_barplot(
        cleaned_df: pd.DataFrame,
        fb: FeatureBuilder,
        groupby: str = "provincia",  # 'provincia', 'proveedor', 'provincia_proveedor', 'proveedor_provincia'
        provincia: str | None = None,
        provincias: Sequence[str] | None = None,
        proveedores: Sequence[str] | None = None,
        location: str = "both",
        figsize: Tuple[float, float] = (20, 10),
        dpi: int = 120,
        rotation: int = 45,
        title: str | None = None,
    ) -> plt.Figure:
        """
        Bar plot for the 'first_visit' feature.

        Modes controlled by `groupby`:

        - groupby='provincia' (default):
            X-axis: one bar per provincia.
            Y-axis: average first_visit across all providers in that provincia.

        - groupby='proveedor':
            X-axis: one bar per proveedor.
            Y-axis: first_visit for each provider, optionally filtered by `provincia`.

        - groupby='provincia_proveedor':
            X-axis: provincias.
            For each provincia, a group of bars (one per proveedor).

        - groupby='proveedor_provincia':
            X-axis: proveedores.
            For each proveedor, a group of bars (one per provincia).

        In all cases, the numeric value is annotated on top of each bar (in % if
        data look like ratios 0–1).

        Parameters
        ----------
        cleaned_df : pd.DataFrame
            Cleaned dataset (input to FeatureBuilder).
        fb : FeatureBuilder
            FeatureBuilder instance.
        groupby : {'provincia', 'proveedor', 'provincia_proveedor',
                   'proveedor_provincia'}, default 'provincia'
            Grouping mode for the X-axis.
        provincia : str, optional
            When groupby='proveedor', restrict metrics to a single provincia.
            Ignored in other modes.
        provincias : sequence of str, optional
            When groupby in {'provincia', 'provincia_proveedor',
            'proveedor_provincia'}, subset of provincias to include.
            If None, uses all provincias in cleaned_df["Provincia"].
        proveedores : sequence of str, optional
            Subset of providers (index names in FeatureBuilder output).
            If None, uses all that appear in the data.
        location : {'both', 'capital', 'interior'}, default 'both'
            Passed to FeatureBuilder.build().
        figsize : (float, float), default (12, 6)
            Figure size in inches.
        dpi : int, default 120
            Figure DPI.
        rotation : int, default 45
            Rotation of X-axis labels.
        title : str, optional
            Custom title. If None, an automatic title is used.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        """

        groupby = groupby.lower()
        if groupby not in {
            "provincia",
            "proveedor",
            "provincia_proveedor",
            "proveedor_provincia",
        }:
            raise ValueError(
                "groupby must be 'provincia', 'proveedor', "
                "'provincia_proveedor' or 'proveedor_provincia'."
            )

        # Helper to scale to percentage if values look like ratios in [0,1]
        def _scale_values_to_pct(raw_vals: Sequence[float]) -> tuple[list[float], bool]:
            valid_vals = [v for v in raw_vals if v is not None and not pd.isna(v)]
            if not valid_vals:
                return [], False
            max_val = max(valid_vals)
            scale_to_pct = max_val <= 1.0 + 1e-9
            if scale_to_pct:
                return [float(v) * 100.0 for v in raw_vals], True
            else:
                return [float(v) for v in raw_vals], False

        # ------------------ groupby = 'provincia' ------------------
        if groupby == "provincia":
            if provincias is None:
                provincias = (
                    cleaned_df["Provincia"]
                    .dropna()
                    .unique()
                    .tolist()
                )
                provincias = sorted(provincias)

            categories: list[str] = []
            raw_values: list[float] = []

            for prov in provincias:
                df_metrics = fb.build(
                    df=cleaned_df,
                    features=["first_visit"],
                    drop_incomplete=True,
                    provincia=prov,
                    location=location,
                )

                if df_metrics is None or df_metrics.empty:
                    continue

                col = df_metrics["first_visit"].dropna()
                if col.empty:
                    continue

                val = float(col.mean())
                categories.append(str(prov))
                raw_values.append(val)

            if not categories:
                raise ValueError(
                    "No valid 'first_visit' data found for any provincia "
                    "with the given filters."
                )

            plot_values, scaled = _scale_values_to_pct(raw_values)

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            x_pos = np.arange(len(categories))
            ax.bar(x_pos, plot_values)

            if title is None:
                title = "First visit por provincia (promedio proveedores)"

            ax.set_title(title, fontsize=14)
            ax.set_ylabel(
                "First visit success (%)" if scaled else "First visit",
                fontsize=12,
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels(categories, rotation=rotation, ha="right")

            # annotate
            for x, val in zip(x_pos, plot_values):
                ax.text(
                    x,
                    val,
                    f"{val:.1f}%" if scaled else f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            return fig

        # ------------------ groupby = 'proveedor' ------------------
        if groupby == "proveedor":
            df_metrics = fb.build(
                df=cleaned_df,
                features=["first_visit"],
                drop_incomplete=True,
                provincia=provincia,
                location=location,
            )

            if df_metrics is None or df_metrics.empty:
                raise ValueError(
                    "No metrics available for the specified filters "
                    "(provincia/location)."
                )

            series = df_metrics["first_visit"].dropna()

            if proveedores is not None:
                proveedores = list(proveedores)
                valid_prov = [p for p in proveedores if p in series.index]
                series = series.loc[valid_prov].dropna()

            if series.empty:
                raise ValueError(
                    "No valid 'first_visit' data for the requested proveedores."
                )

            categories = list(series.index.astype(str))
            raw_values = [float(v) for v in series.values]
            plot_values, scaled = _scale_values_to_pct(raw_values)

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            x_pos = np.arange(len(categories))
            ax.bar(x_pos, plot_values)

            if title is None:
                base = "First visit por proveedor"
                if provincia is not None:
                    base += f" – Provincia: {provincia}"
                title = base

            ax.set_title(title, fontsize=14)
            ax.set_ylabel(
                "First visit success (%)" if scaled else "First visit",
                fontsize=12,
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels(categories, rotation=rotation, ha="right")

            for x, val in zip(x_pos, plot_values):
                ax.text(
                    x,
                    val,
                    f"{val:.1f}%" if scaled else f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            return fig

        # ----------------- common prep for 2D modes -----------------
        # Provincias to consider
        if provincias is None:
            provincias = (
                cleaned_df["Provincia"]
                .dropna()
                .unique()
                .tolist()
            )
            provincias = sorted(provincias)

        # For each provincia, compute metrics and collect providers
        metrics_by_prov: dict[str, pd.Series] = {}
        all_proveedores: set[str] = set()

        for prov in provincias:
            df_metrics = fb.build(
                df=cleaned_df,
                features=["first_visit"],
                drop_incomplete=True,
                provincia=prov,
                location=location,
            )

            if df_metrics is None or df_metrics.empty:
                continue

            s = df_metrics["first_visit"].dropna()
            if s.empty:
                continue

            metrics_by_prov[str(prov)] = s
            all_proveedores.update(s.index.astype(str))

        if not metrics_by_prov:
            raise ValueError(
                "No 'first_visit' metrics available for any provincia "
                "with the given filters."
            )

        # Determine proveedores universe
        if proveedores is not None:
            proveedores_all = [p for p in proveedores if p in all_proveedores]
        else:
            proveedores_all = sorted(all_proveedores)

        if not proveedores_all:
            raise ValueError("No proveedores found with valid 'first_visit' data.")

        provincias_final = list(metrics_by_prov.keys())

        # ---------------- groupby = 'provincia_proveedor' ----------------
        if groupby == "provincia_proveedor":
            # Build matrix of values: rows = provincias, cols = proveedores
            values_matrix: list[list[float | float]] = []
            flat_vals: list[float] = []

            for prov in provincias_final:
                row_vals: list[float | float] = []
                s = metrics_by_prov[prov]
                for proveedor in proveedores_all:
                    if proveedor in s.index:
                        v = float(s.loc[proveedor])
                        row_vals.append(v)
                        flat_vals.append(v)
                    else:
                        row_vals.append(np.nan)
                values_matrix.append(row_vals)

            # Scale to pct
            if flat_vals:
                max_val = max(flat_vals)
                scale_to_pct = max_val <= 1.0 + 1e-9
            else:
                scale_to_pct = False

            def _scale(v: float | float) -> float:
                if pd.isna(v):
                    return np.nan
                return float(v) * 100.0 if scale_to_pct else float(v)

            plot_matrix: list[list[float]] = [
                [_scale(v) for v in row] for row in values_matrix
            ]

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            n_prov = len(provincias_final)
            n_carriers = len(proveedores_all)
            x_base = np.arange(n_prov)
            bar_width = 0.8 / max(n_carriers, 1)

            for j, proveedor in enumerate(proveedores_all):
                x_pos = x_base - 0.4 + bar_width * (j + 0.5)
                vals = [row[j] for row in plot_matrix]
                ax.bar(x_pos, vals, width=bar_width, label=proveedor)

                for x, val in zip(x_pos, vals):
                    if pd.isna(val):
                        continue
                    ax.text(
                        x,
                        val,
                        f"{val:.1f}%" if scale_to_pct else f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

            if title is None:
                title = "First visit por provincia y proveedor"

            ax.set_title(title, fontsize=14)
            ax.set_ylabel(
                "First visit success (%)" if scale_to_pct else "First visit",
                fontsize=12,
            )
            ax.set_xticks(x_base)
            ax.set_xticklabels(provincias_final, rotation=rotation, ha="right")

            ax.legend(title="Proveedor", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            return fig

        # -------------- groupby = 'proveedor_provincia' --------------
        # Build matrix of values: rows = proveedores, cols = provincias
        values_matrix_pp: list[list[float | float]] = []
        flat_vals_pp: list[float] = []

        for proveedor in proveedores_all:
            row_vals: list[float | float] = []
            for prov in provincias_final:
                s = metrics_by_prov[prov]
                if proveedor in s.index:
                    v = float(s.loc[proveedor])
                    row_vals.append(v)
                    flat_vals_pp.append(v)
                else:
                    row_vals.append(np.nan)
            values_matrix_pp.append(row_vals)

        if flat_vals_pp:
            max_val = max(flat_vals_pp)
            scale_to_pct_pp = max_val <= 1.0 + 1e-9
        else:
            scale_to_pct_pp = False

        def _scale_pp(v: float | float) -> float:
            if pd.isna(v):
                return np.nan
            return float(v) * 100.0 if scale_to_pct_pp else float(v)

        plot_matrix_pp: list[list[float]] = [
            [_scale_pp(v) for v in row] for row in values_matrix_pp
        ]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        n_prov = len(provincias_final)
        n_carriers = len(proveedores_all)
        x_base = np.arange(n_carriers)
        bar_width = 0.8 / max(n_prov, 1)

        for j, prov in enumerate(provincias_final):
            x_pos = x_base - 0.4 + bar_width * (j + 0.5)
            vals = [row[j] for row in plot_matrix_pp]
            ax.bar(x_pos, vals, width=bar_width, label=prov)

            for x, val in zip(x_pos, vals):
                if pd.isna(val):
                    continue
                ax.text(
                    x,
                    val,
                    f"{val:.1f}%" if scale_to_pct_pp else f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        if title is None:
            title = "First visit por proveedor y provincia"

        ax.set_title(title, fontsize=14)
        ax.set_ylabel(
            "First visit success (%)" if scale_to_pct_pp else "First visit",
            fontsize=12,
        )
        ax.set_xticks(x_base)
        ax.set_xticklabels(proveedores_all, rotation=rotation, ha="right")

        ax.legend(title="Provincia", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        return fig


