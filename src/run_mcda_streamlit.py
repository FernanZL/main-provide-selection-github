""" Put this stuff in a jupyter notebook to run this

import subprocess, time, webbrowser

# Run streamlit as a background process
proc = subprocess.Popen(
    ["streamlit", "run", "src/run_mcda_streamlit.py", "--server.headless", "true"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)

# Wait a few seconds for the app to start
time.sleep(4)

# Open it in the browser automatically
webbrowser.open("http://localhost:8501")

print("✅ Streamlit app running at http://localhost:8501")
print("Stop it with: proc.terminate()")

#proc.terminate() 
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import streamlit as st

from src.data_pipeline import Preprocessor
from src.features import FeatureBuilder
from src.mcda import MCDAEngine


st.set_page_config(page_title="MCDA Shipping Recommender", layout="wide")

st.title("📦 MCDA Shipping Recommender")


# ---------- 1) Upload CSV ----------
uploaded_file = st.file_uploader("Subí el archivo CSV de envíos", type=["csv"])

if uploaded_file is not None:
    # Load raw df directly from the uploaded file
    raw_df = pd.read_csv(uploaded_file)

    st.subheader("Vista rápida del dataset original")
    st.dataframe(raw_df.head())

    # ---------- 2) Clean with Preprocessor ----------
    pre = Preprocessor(normalize_names=True)
    cleaned_df = pre.clean(raw_df)

    st.write(
        f"Filas originales: {len(raw_df)} | "
        f"Filas después de limpiar: {len(cleaned_df)}"
    )

    if cleaned_df.empty:
        st.error("Después de la limpieza no quedaron filas. Revisá el archivo de entrada.")
        st.stop()

    fb = FeatureBuilder()
    mcda = MCDAEngine()

    # ---------- 3) Sidebar controls ----------
    st.sidebar.header("Filtros")

    # Provincia (normalizada por Preprocessor)
    provincias = sorted(cleaned_df["Provincia"].dropna().unique())
    provincia_opt = ["Todas"] + provincias
    provincia_sel = st.sidebar.selectbox("Provincia", provincia_opt, index=0)
    provincia = None if provincia_sel == "Todas" else provincia_sel

    # Zona
    location = st.sidebar.radio("Zona", ["both", "capital", "interior"], index=0)

    # Código postal (text input)
    codigo_postal_input = st.sidebar.text_input("Código postal (vacío = todos)", value="")
    codigo_postal = codigo_postal_input.strip() or None

    # Rango de peso
    rango_opts = sorted(cleaned_df["Rango de Peso"].dropna().unique())
    rango_opt_labels = ["Todos"] + list(rango_opts)
    rango_sel = st.sidebar.selectbox("Rango de peso", rango_opt_labels, index=0)
    rango_peso = None if rango_sel == "Todos" else rango_sel

    drop_incomplete = st.sidebar.checkbox(
        "Excluir proveedores con funciones de Costo/Peso y % Cobertura con valores nulos",
        value=True,
    )

    st.sidebar.markdown("---")

    # Features
    st.sidebar.header("Features & MCDA")
    default_feats = ["first_visit", "cost", "coverage"]
    features = st.sidebar.multiselect(
        "Features",
        fb.available_features,
        default=default_feats,
    )

    available_methods = ["weighted", "topsis", "vikor", "waspas"]
    methods = st.sidebar.multiselect(
        "Métodos MCDA",
        available_methods,
        default="waspas",
    )

    # Weights: text input, comma-separated, or blank for equal weights
    weights_text = st.sidebar.text_input(
        "Pesos (coma separada, vacío = iguales)",
        value="",
        help="Ejemplo: 0.5, 0.4, 0.1 para 3 criterios",
    )

    # ---------- 4) Run button ----------
    run_btn = st.button("Calcular ranking")

    if run_btn:
        if not features:
            st.error("Tenés que elegir al menos una feature.")
            st.stop()
        if not methods:
            st.error("Tenés que elegir al menos un método MCDA.")
            st.stop()

        # 4.1 Build feature table
        st.subheader("Métricas por proveedor")
        df_metrics = fb.build(
            cleaned_df,
            features=features,
            provincia=provincia,
            location=location,
            codigo_postal=codigo_postal,
            rango_peso=rango_peso,
            drop_incomplete=drop_incomplete,
        )

        if df_metrics.empty:
            st.error("No hay datos para ese filtro (df_metrics está vacío).")
            st.stop()

        st.dataframe(df_metrics.round(3))

        # 4.2 Prepare weights
        n_criteria = len(df_metrics.columns)
        if weights_text.strip():
            try:
                w_vals = [float(x) for x in weights_text.split(",") if x.strip()]
                if len(w_vals) != n_criteria:
                    st.warning(
                        f"Se esperaban {n_criteria} pesos, pero se recibieron {len(w_vals)}. "
                        "Usando pesos iguales."
                    )
                    weights = None
                else:
                    weights = np.array(w_vals, dtype=float)
            except ValueError:
                st.warning("No se pudieron parsear los pesos. Usando pesos iguales.")
                weights = None
        else:
            weights = None  # MCDAEngine maneja el None como pesos iguales

        # 4.3 criteria_types: None → MCDAEngine infiere ("cost" si nombre contiene "cost")
        criteria_types = None

        # 4.4 Run MCDAEngine
        st.subheader("Ranking de proveedores")

        avg_flag = len(methods) > 1  # solo computar Score_avg si hay más de un método

        scores_df = mcda.score(
            df_metrics,
            methods=methods,
            weights=weights,
            criteria_types=criteria_types,
            return_df=True,
            sort=True,
            avg=avg_flag,
        )

        st.dataframe(scores_df.round(4))

        # Optional: download as CSV
        csv = scores_df.to_csv(index=True).encode("utf-8")
        st.download_button(
            "Descargar resultados como CSV",
            data=csv,
            file_name="mcda_scores.csv",
            mime="text/csv",
        )

else:
    st.info("Subí un archivo CSV para empezar.")


