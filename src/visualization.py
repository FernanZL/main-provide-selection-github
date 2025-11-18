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
        courier: str,
        x_feature: str = "first_visit",
        y_feature: str = "cost",
        provincias: list[str] | None = None,
        location: str = "both",
        figsize: Tuple[float, float] = (12, 8),
        dpi: int = 120,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Scatter plot of two features (x_feature, y_feature) for a single courier,
        with one dot per Provincia.

        Features are computed via FeatureBuilder.build() for each Provincia.

        Parameters
        ----------
        cleaned_df : pd.DataFrame
            Cleaned dataset (the same one you pass to FeatureBuilder).
        fb : FeatureBuilder
            An instance of FeatureBuilder already configured.
        courier : str
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

        # --- For each provincia, compute features and pick the courier's values ---
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

            # If this courier is not present (all NaN etc.), skip
            if courier not in feat_df.index:
                continue

            row = feat_df.loc[courier]

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
                f"No valid data for '{courier}'",
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
            f"{courier}: {y_label} vs {x_label} by Provincia",
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
        Plot evolution of each courier's MCDA score across different weight presets.

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
        Plot evolution of each courier's scores across MCDA methods.

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
        Bar plot comparing MCDA method scores for each courier (provider).

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
