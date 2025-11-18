import pandas as pd
from src.features import FeatureBuilder
from src.mcda import MCDAEngine


class Recommender:
    def __init__(self, feature_builder: FeatureBuilder, mcda_engine: MCDAEngine):
        self.fb = feature_builder
        self.mcda = mcda_engine

    def recommend(
        self,
        df: pd.DataFrame,
        group_by=("Codigo Postal", "Rango de Peso"),
        features=("first_visit", "cost", "coverage"),
        mcda_methods=("weighted",),
        weights=None,
        criteria_types=None,
        location="both",
        output_col_prefix="best_",
        fill_no_data=None,
        sort=True,
        avg=False,
    ) -> pd.DataFrame:
        df_out = df.copy()

        for col in group_by:
            if col not in df_out.columns:
                raise ValueError(f"group_by column '{col}' not in dataframe")

        if isinstance(mcda_methods, str):
            mcda_methods = [mcda_methods]

        best_cols = [f"{output_col_prefix}{m}" for m in mcda_methods]
        if avg and len(mcda_methods) > 1:
            best_cols.append(f"{output_col_prefix}avg")
        for c in best_cols:
            df_out[c] = fill_no_data

        grouped = df_out.groupby(list(group_by), dropna=False)

        for keys, subdf in grouped:
            key_kwargs = self._keys_to_kwargs(group_by, keys)

            feat_df = self.fb.build(
                df,
                features=list(features),
                provincia=key_kwargs.get("Provincia"),
                codigo_postal=key_kwargs.get("Codigo Postal"),
                rango_peso=key_kwargs.get("Rango de Peso"),
                location=location,
            )

            if feat_df.isna().all().all():
                df_out.loc[subdf.index, best_cols] = fill_no_data
                continue

            mcda_df = self.mcda.score(
                feat_df,
                methods=mcda_methods,
                weights=weights,
                criteria_types=criteria_types,
                return_df=True,
                sort=sort,
                avg=avg,
            )

            for m in mcda_methods:
                score_col = f"Score_{m}"
                if score_col not in mcda_df.columns or mcda_df[score_col].dropna().empty:
                    best_provider = fill_no_data
                else:
                    best_provider = mcda_df[score_col].idxmax()
                df_out.loc[subdf.index, f"{output_col_prefix}{m}"] = best_provider

            if avg and len(mcda_methods) > 1 and "Score_avg" in mcda_df.columns:
                best_avg = mcda_df["Score_avg"].idxmax()
                df_out.loc[subdf.index, f"{output_col_prefix}avg"] = best_avg

        return df_out

    @staticmethod
    def _keys_to_kwargs(group_by, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        return dict(zip(group_by, keys))
