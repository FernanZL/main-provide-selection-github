# shipping_recommender_st.py
# ---- MCDA Shipping Recommender - Streamlit UI (contextual sidebar) ----

import sys
import os
import io
import json
import re
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import streamlit as st

from src.data_pipeline import Preprocessor
from src.features import FeatureBuilder
from src.mcda import MCDAEngine
from src.weight_presets import get_weight_presets  # default presets source
from src.scenario_reporter import ScenarioReporter  # ScenarioReporter


# ============ PRESETS STORE (defaults + current) ============
PRESETS_JSON_PATH = "current_weight_presets.json"  # stored in current working dir

# === Rutas de los CSV (cambiar a lo que uses) ===
MAIN_CSV_PATH = "datasets\\shipments_july_sla.csv"      # ruta al CSV principal
EXTRA_CSV_PATH = None                                   # o algo como "data/envios_sla.csv"


# ---------- UI (CSS) ----------

def _inject_dark_dashboard_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }

:root{
  --bg0: #0b0f17;
  --bg1: #0e1420;
  --border: rgba(255,255,255,0.10);
  --text: rgba(255,255,255,0.86);
  --muted: rgba(255,255,255,0.60);
  --accent: #7c3aed; /* violet */
}

.stApp {
  background: radial-gradient(1200px 600px at 10% -10%, rgba(124,58,237,0.22), transparent 60%),
              radial-gradient(1000px 500px at 110% 10%, rgba(34,197,94,0.14), transparent 55%),
              linear-gradient(180deg, var(--bg0), var(--bg1));
  color: var(--text);
}

.block-container { padding-top: 1.0rem; }

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.015));
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container{ padding-top: 1.0rem; }

div.stButton > button {
  border-radius: 12px !important;
  border: 1px solid var(--border) !important;
  background: rgba(255,255,255,0.06) !important;
  color: var(--text) !important;
  padding: 0.55rem 0.85rem !important;
  transition: 120ms ease;
}
div.stButton > button:hover {
  border-color: rgba(255,255,255,0.20) !important;
  transform: translateY(-1px);
  background: rgba(255,255,255,0.09) !important;
}
div.stButton > button:active { transform: translateY(0px); }

div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, textarea {
  border-radius: 12px !important;
  border-color: rgba(255,255,255,0.14) !important;
  background: rgba(255,255,255,0.05) !important;
  color: var(--text) !important;
}

[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.02);
}

div[data-testid="stAlert"] {
  border-radius: 14px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.04);
}

._chipline{
  margin: 6px 0 10px 0;
  display:flex; flex-wrap:wrap; gap:8px;
}
._chip{
  display:inline-flex; align-items:center; gap:8px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.04);
  border-radius: 999px;
  padding: 6px 10px;
  font-size: 12px;
  color: rgba(255,255,255,0.72);
}
._dot{ width:8px; height:8px; border-radius:999px; background: var(--accent); display:inline-block; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_filter_chips(filter_text: str) -> None:
    if not filter_text:
        return
    content = filter_text.replace(" — ", "").strip()
    st.markdown(
        f"""
<div class="_chipline">
  <span class="_chip"><span class="_dot"></span>{content}</span>
</div>
        """,
        unsafe_allow_html=True,
    )


# ---------- Presets persistence ----------

def _load_current_presets(path: str = PRESETS_JSON_PATH) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        clean: Dict[str, Dict[str, float]] = {}
        for name, weights in data.items():
            if isinstance(weights, dict):
                clean[name] = {str(k): float(v) for k, v in weights.items()}
        return clean
    except Exception:
        return {}


def _save_current_presets(
    current_presets: Dict[str, Dict[str, float]],
    path: str = PRESETS_JSON_PATH,
) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(current_presets, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"⚠️ No se pudieron guardar los presets: {e}")


_default_presets: Dict[str, Dict[str, float]] = get_weight_presets()
_loaded_current = _load_current_presets()
if _loaded_current:
    _current_presets: Dict[str, Dict[str, float]] = _loaded_current
else:
    _current_presets = {name: preset.copy() for name, preset in _default_presets.items()}
    _save_current_presets(_current_presets)


def _rebuild_candidate_presets() -> Dict[str, Dict[str, float]]:
    return dict(_current_presets)


# >>> FIX: apply preset BEFORE widgets are instantiated
def _apply_pending_preset_if_any(
    candidate_presets: Dict[str, Dict[str, float]],
    feats_all: List[str],
) -> None:
    pending_name = st.session_state.get("_pending_apply_preset_name", None)
    if not pending_name:
        return

    st.session_state["_pending_apply_preset_name"] = None

    preset = candidate_presets.get(pending_name)
    if not isinstance(preset, dict) or not preset:
        st.session_state["_preset_flash"] = f"❌ Preset inválido: '{pending_name}'."
        return

    selected_feats = st.session_state.get("selected_feats", feats_all) or feats_all
    if not selected_feats:
        st.session_state["_preset_flash"] = "❌ No hay features seleccionadas para aplicar el preset."
        return

    raw_vals = [preset.get(f, 0.0) for f in selected_feats]
    arr = np.array(raw_vals, dtype=float)
    if arr.sum() <= 0:
        st.session_state["_preset_flash"] = (
            f"❌ El preset '{pending_name}' no aplica a las features actuales (suma <= 0)."
        )
        return

    arr = arr / arr.sum()

    for f, v in zip(selected_feats, arr):
        st.session_state[f"weight_{f}"] = float(v)

    # ✅ Force reset of "inspeccionar celda" widgets after preset apply
    st.session_state["_inspect_reset_token"] = int(st.session_state.get("_inspect_reset_token", 0)) + 1

    # Clear any older keys you may have used before
    for k in ("inspect_row", "inspect_col", "inspect_row_label", "inspect_col_label"):
        st.session_state.pop(k, None)

    st.session_state["_preset_flash"] = f"✅ Preset '{pending_name}' aplicado."


# >>> FIX: normalize weights BEFORE widgets are instantiated
def _apply_pending_normalize_if_any(
    feats_all: List[str],
) -> None:
    if not st.session_state.get("_pending_normalize_weights", False):
        return

    st.session_state["_pending_normalize_weights"] = False

    selected_feats = st.session_state.get("selected_feats", feats_all)
    if not selected_feats:
        return

    vals = []
    for f in selected_feats:
        v = st.session_state.get(f"weight_{f}", 0.0)
        vals.append(float(v))

    arr = np.array(vals, dtype=float)
    s = arr.sum()
    if not np.isfinite(s) or s <= 0:
        st.session_state["_normalize_flash"] = "❌ No se puede normalizar: suma = 0."
        return

    arr = arr / s
    for f, v in zip(selected_feats, arr):
        st.session_state[f"weight_{f}"] = float(v)

    st.session_state["_normalize_flash"] = "✅ Pesos normalizados."


# ---------- Helpers (non-UI) ----------

def _filter_suffix(
    provincia: Optional[str],
    location: str,
    codigo_postal: Optional[int],
    rango_peso: Optional[str],
    date_from,
    date_to,
    date_min,
    date_max,
) -> str:
    parts: List[str] = []

    if provincia and provincia not in ("Todas", "(subí un archivo)"):
        parts.append(f"Provincia={provincia}")

    if codigo_postal is not None:
        parts.append(f"CP={codigo_postal}")

    if location and location != "both":
        loc_label = {"capital": "Capital", "interior": "Interior"}.get(location, location)
        parts.append(f"Zona={loc_label}")

    if rango_peso not in (None, "Todos", "(subí un archivo)", "(no disponible)"):
        parts.append(f"Rango={rango_peso}")

    show_date = False
    if date_from is not None and date_to is not None and date_min is not None and date_max is not None:
        if not (date_from <= date_min and date_to >= date_max):
            show_date = True
    elif date_from is not None or date_to is not None:
        show_date = True

    if show_date:
        if date_from is not None:
            parts.append(f"Desde={date_from.isoformat()}")
        if date_to is not None:
            parts.append(f"Hasta={date_to.isoformat()}")

    if not parts:
        return ""
    return " — Filtros: " + ", ".join(parts)


def _report_text_to_html(text: str) -> str:
    esc = (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    esc = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc)

    lines = esc.split("\n")
    html_parts: List[str] = []
    in_list = False

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped == "":
            html_parts.append("<div style='height:6px;'></div>")
            continue

        if stripped.startswith("==="):
            title = stripped.strip("= ").strip()
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(
                f"<div style='margin:10px 0 6px 0; font-size:13px; font-weight:700; "
                f"color:rgba(255,255,255,0.90); padding-bottom:6px; "
                f"border-bottom:1px solid rgba(255,255,255,0.10);'>{title}</div>"
            )
            continue

        if stripped.startswith("- "):
            content = stripped[2:].strip()
            if not in_list:
                html_parts.append("<ul style='margin:6px 0 6px 18px; padding-left:14px;'>")
                in_list = True
            html_parts.append(
                f"<li style='margin-bottom:4px; color:rgba(255,255,255,0.78); "
                f"font-size:12.5px;'>{content}</li>"
            )
            continue

        if in_list:
            html_parts.append("</ul>")
            in_list = False

        html_parts.append(
            f"<div style='margin:2px 0; color:rgba(255,255,255,0.78); font-size:12.5px;'>{line}</div>"
        )

    if in_list:
        html_parts.append("</ul>")

    inner_html = "\n".join(html_parts)

    return f"""
    <div style="
        border-radius:16px;
        border:1px solid rgba(255,255,255,0.12);
        padding:12px 14px;
        background: linear-gradient(135deg, rgba(255,255,255,0.055), rgba(255,255,255,0.02));
        box-shadow: 0 18px 40px rgba(0,0,0,0.22);
    ">
        {inner_html}
    </div>
    """


def _detect_date_range(
    df: pd.DataFrame,
    scenario_reporter: Optional[ScenarioReporter],
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[str]]:
    if df is None or df.empty:
        return None, None, None

    date_col = None

    if scenario_reporter is not None and hasattr(scenario_reporter, "date_col"):
        dc = scenario_reporter.date_col
        if dc in df.columns:
            date_col = dc

    if date_col is None:
        for cand in ["Fecha de Despacho", "last_status_date"]:
            if cand in df.columns:
                date_col = cand
                break

    if date_col is None:
        return None, None, None

    col = df[date_col]
    if np.issubdtype(col.dtype, np.datetime64):
        s = col.dropna()
    else:
        s = pd.to_datetime(col, errors="coerce", dayfirst=True).dropna()

    if s.empty:
        return None, None, date_col

    return s.min().date(), s.max().date(), date_col


def _filter_df_by_dates_streamlit(
    df: pd.DataFrame,
    date_from,
    date_to,
    date_col: Optional[str],
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    if date_from is None and date_to is None:
        return df

    if date_col is None or date_col not in df.columns:
        return df

    s = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    valid = ~s.isna()

    if not valid.any():
        return df

    s_min = s[valid].min().date()
    s_max = s[valid].max().date()

    if ((date_from is None or date_from <= s_min) and
        (date_to is None or date_to >= s_max)):
        return df

    mask = pd.Series(True, index=df.index)

    if date_from is not None:
        dfrom_ts = pd.to_datetime(date_from)
        mask &= (~valid) | (s >= dfrom_ts)

    if date_to is not None:
        end_dt = pd.to_datetime(date_to) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        mask &= (~valid) | (s <= end_dt)

    return df.loc[mask].copy()


# =========================
# NEW: Dataset source logic
# =========================

def _read_csv_any(obj) -> pd.DataFrame:
    """
    Read a CSV from:
      - a filesystem path (str)
      - an UploadedFile / file-like object
      - raw bytes
    """
    if obj is None:
        raise ValueError("CSV source is None")

    if isinstance(obj, (bytes, bytearray)):
        return pd.read_csv(io.BytesIO(obj))

    if isinstance(obj, str):
        return pd.read_csv(obj)

    # Streamlit UploadedFile behaves like a file-like object
    try:
        return pd.read_csv(obj)
    except Exception:
        # last resort: try bytes
        try:
            b = obj.getvalue()
            return pd.read_csv(io.BytesIO(b))
        except Exception as e:
            raise e


def _load_data_from_raw(
    raw: pd.DataFrame,
    pre: Preprocessor,
    fb: FeatureBuilder,
    mcda: MCDAEngine,
    extra_csv_path: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[ScenarioReporter], List[str]]:
    messages: List[str] = []

    if raw is None or raw.empty:
        messages.append("❌ El CSV está vacío o no se pudo leer.")
        return None, None, None, messages

    try:
        if extra_csv_path and os.path.exists(extra_csv_path):
            cleaned = pre.clean_with_extra_dates(
                raw,
                extra_csv_path=extra_csv_path,
                key_main="Order ID",
                key_extra="id",
            )
        else:
            cleaned = pre.clean_with_extra_dates(raw, extra_csv_path=None)

        messages.append(
            f"Filas originales: {len(raw)} | "
            f"Filas después de limpiar/mergear: {len(cleaned)}"
        )

        if cleaned.empty:
            messages.append("❌ Después de la limpieza/merge no quedaron filas.")
            return None, raw, None, messages

        scenario_reporter = ScenarioReporter(
            df=cleaned,
            feature_builder=fb,
            mcda_engine=mcda,
            date_col="Fecha de Despacho",
        )
        return cleaned, raw, scenario_reporter, messages

    except Exception as e:
        messages.append(f"❌ Error limpiando/mergeando el CSV: {e}")
        return None, raw, None, messages


def _get_active_dataset(
    pre: Preprocessor,
    fb: FeatureBuilder,
    mcda: MCDAEngine,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[ScenarioReporter], List[str], str]:
    """
    Returns (cleaned_df, raw_df, scenario_reporter, messages, source_label)
    source_label: "default" or "uploaded"
    """
    messages: List[str] = []

    # state init
    if "dataset_source" not in st.session_state:
        st.session_state["dataset_source"] = "default"  # "default" | "uploaded"
    if "uploaded_csv_bytes" not in st.session_state:
        st.session_state["uploaded_csv_bytes"] = None
    if "uploaded_csv_name" not in st.session_state:
        st.session_state["uploaded_csv_name"] = None

    source = st.session_state.get("dataset_source", "default")

    # If user chose uploaded but bytes are missing, fallback to default
    if source == "uploaded" and not st.session_state.get("uploaded_csv_bytes"):
        source = "default"
        st.session_state["dataset_source"] = "default"

    if source == "uploaded":
        try:
            raw = _read_csv_any(st.session_state["uploaded_csv_bytes"])
        except Exception as e:
            messages.append(f"❌ No se pudo leer el CSV subido: {e}")
            # hard fallback to default
            st.session_state["dataset_source"] = "default"
            source = "default"
        else:
            cleaned_df, raw_df, scenario_reporter, msgs = _load_data_from_raw(
                raw=raw, pre=pre, fb=fb, mcda=mcda, extra_csv_path=None  # uploaded dataset => no EXTRA_CSV_PATH
            )
            messages.extend(msgs)
            return cleaned_df, raw_df, scenario_reporter, messages, "uploaded"

    # default path
    if not os.path.exists(MAIN_CSV_PATH):
        messages.append(f"❌ MAIN_CSV_PATH no existe: {MAIN_CSV_PATH}")
        return None, None, None, messages, "default"

    try:
        raw = _read_csv_any(MAIN_CSV_PATH)
    except Exception as e:
        messages.append(f"❌ Error leyendo MAIN_CSV_PATH: {e}")
        return None, None, None, messages, "default"

    cleaned_df, raw_df, scenario_reporter, msgs = _load_data_from_raw(
        raw=raw, pre=pre, fb=fb, mcda=mcda, extra_csv_path=EXTRA_CSV_PATH
    )
    messages.extend(msgs)
    return cleaned_df, raw_df, scenario_reporter, messages, "default"


def _render_dataset_source_controls(active_source_label: str) -> None:
    """
    UI controls (buttons + uploader) for picking default vs uploaded dataset.
    Requirement:
      - default dataset exists by default
      - user can upload a new one (with a button)
      - once uploaded, entire app uses uploaded dataset and default "shouldn't be a thing"
    """
    name = st.session_state.get("uploaded_csv_name")
    if active_source_label == "uploaded":
        st.info(f"📌 Dataset activo: **SUBIDO**{f' — {name}' if name else ''}")
    else:
        st.info("📌 Dataset activo: **POR DEFECTO**")

    c1, c2 = st.columns([1, 2])

    with c1:
        if st.button("↩️ Volver al dataset por defecto", use_container_width=True):
            st.session_state["dataset_source"] = "default"
            st.session_state["uploaded_csv_bytes"] = None
            st.session_state["uploaded_csv_name"] = None
            # optional: reset date widgets so they don't keep an invalid range
            for k in ("date_from", "date_to", "date_from_ui", "date_to_ui"):
                st.session_state.pop(k, None)
            st.rerun()

    with c2:
        up = st.file_uploader(
            "Subir CSV (reemplaza el dataset por defecto para TODA la app)",
            type=["csv"],
            accept_multiple_files=False,
            key="dataset_uploader",
        )
        cc1, cc2 = st.columns([1, 1])
        with cc1:
            load_clicked = st.button("✅ Cargar dataset subido", use_container_width=True)
        with cc2:
            # small “safety” button: clears the uploader selection UI
            clear_clicked = st.button("🧹 Limpiar selección", use_container_width=True)

        if clear_clicked:
            st.session_state.pop("dataset_uploader", None)
            st.rerun()

        if load_clicked:
            if up is None:
                st.warning("Subí un CSV primero.")
            else:
                try:
                    b = up.getvalue()
                    # quick validation read (so we fail fast)
                    _ = _read_csv_any(b)
                except Exception as e:
                    st.error(f"❌ Ese archivo no parece un CSV válido: {e}")
                else:
                    st.session_state["dataset_source"] = "uploaded"
                    st.session_state["uploaded_csv_bytes"] = b
                    st.session_state["uploaded_csv_name"] = getattr(up, "name", None)
                    # reset date widgets (range changes with dataset)
                    for k in ("date_from", "date_to", "date_from_ui", "date_to_ui"):
                        st.session_state.pop(k, None)
                    st.rerun()


def _apply_current_filters_to_df(
    df: pd.DataFrame,
    provincia: Optional[str],
    location: str,
    codigo_postal: Optional[int],
    rango_peso: Optional[str],
    date_from,
    date_to,
    date_col: Optional[str],
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    filtered = df

    if provincia and provincia not in ("Todas", "(subí un archivo)"):
        if "Provincia" in filtered.columns:
            filtered = filtered[filtered["Provincia"] == provincia]

    if location == "capital" and "Capital/Interior" in filtered.columns:
        filtered = filtered[filtered["Capital/Interior"] == "CIUDAD"]
    elif location == "interior" and "Capital/Interior" in filtered.columns:
        filtered = filtered[filtered["Capital/Interior"] == "INTERIOR"]

    if codigo_postal is not None and "Codigo Postal" in filtered.columns:
        filtered = filtered[filtered["Codigo Postal"] == codigo_postal]

    if (
        rango_peso not in (None, "Todos", "(subí un archivo)", "(no disponible)")
        and "Rango de Peso" in filtered.columns
    ):
        filtered = filtered[filtered["Rango de Peso"] == rango_peso]

    filtered = _filter_df_by_dates_streamlit(filtered, date_from, date_to, date_col)

    return filtered


# ---------- NEW: stable labels for matrices / inspect ----------

def _label_no_float(x):
    """Make labels stable across reruns: if it's 3400.0 -> 3400 (int)."""
    if x is None:
        return x
    try:
        if pd.isna(x):
            return x
    except Exception:
        pass

    try:
        fx = float(x)
        if np.isfinite(fx) and abs(fx - round(fx)) < 1e-9:
            return int(round(fx))
    except Exception:
        pass
    return x


def _sanitize_matrix_axes(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Return a copy with non-float-ish index/columns (postal codes etc.)."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    out.index = [_label_no_float(v) for v in out.index]
    out.columns = [_label_no_float(v) for v in out.columns]
    return out


def _sanitize_provider_matrix(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Provider matrices: keep values but fix axes too."""
    return _sanitize_matrix_axes(df)


# ---------- MCDA core actions (ranking & presets) ----------

def _run_mcda_ranking(
    cleaned_df: pd.DataFrame,
    fb: FeatureBuilder,
    mcda: MCDAEngine,
    provincia: Optional[str],
    location: str,
    codigo_postal: Optional[int],
    rango_peso: Optional[str],
    date_from,
    date_to,
    date_col: Optional[str],
    feats_selected: List[str],
    drop_incomplete: bool,
    drop_features_with_missing: bool,
    metric_method: str,
    weights_map: Dict[str, float],
):
    info_lines: List[str] = []

    df_for_mcda = _apply_current_filters_to_df(
        cleaned_df,
        provincia,
        location,
        codigo_postal,
        rango_peso,
        date_from,
        date_to,
        date_col,
    )

    if df_for_mcda is None or df_for_mcda.empty:
        return None, ["❌ No hay datos para ese filtro (después de aplicar todos los filtros)."]

    provincia_for_fb = None if provincia in ("Todas", "(subí un archivo)") else provincia
    rango_peso_for_fb = None
    if rango_peso not in (None, "Todos", "(subí un archivo)", "(no disponible)"):
        rango_peso_for_fb = rango_peso

    try:
        df_metrics = fb.build(
            df_for_mcda,
            features=feats_selected,
            provincia=provincia_for_fb,
            location=location,
            codigo_postal=codigo_postal,
            rango_peso=rango_peso_for_fb,
            drop_incomplete=drop_incomplete,
            drop_features_with_missing=drop_features_with_missing,
        )
    except Exception as e:
        return None, [f"❌ Error calculando features: {e}"]

    if df_metrics.empty:
        return None, [
            "❌ No hay datos para ese filtro luego de construir las features "
            "(puede ser por drop_incomplete / NaN)."
        ]

    feats_effective = [f for f in feats_selected if f in df_metrics.columns]
    if not feats_effective:
        return None, [
            "❌ Ninguna de las features seleccionadas quedó disponible "
            "después de aplicar filtros/drop."
        ]

    if set(feats_effective) != set(feats_selected):
        dropped = sorted(set(feats_selected) - set(feats_effective))
        info_lines.append(
            "ℹ️ Se excluyeron estas features porque no están en df_metrics "
            f"(posiblemente por NaN completos o filtros): {', '.join(dropped)}"
        )

    weights_list = [float(weights_map[f]) for f in feats_effective]
    sum_w = float(sum(weights_list))
    if not np.isfinite(sum_w) or sum_w <= 0:
        return None, ["❌ Los pesos deben ser válidos (suma > 0)."]

    if abs(sum_w - 1.0) > 1e-6:
        arr = np.array(weights_list, dtype=float)
        arr = arr / arr.sum()
        for f, v in zip(feats_effective, arr):
            weights_map[f] = float(v)
        info_lines.append("⚠️ Normalicé los pesos (suma original distinta de 1).")

    weights = np.array([weights_map[f] for f in feats_effective], dtype=float)

    info_lines.append("📊 Pesos efectivos por feature:")
    for f, w_val in zip(feats_effective, weights):
        info_lines.append(f"   - {f}: {float(w_val):.4f}")

    try:
        scores_df = mcda.score(
            df_metrics,
            methods=[metric_method],
            weights=weights,
            criteria_types=None,
            return_df=True,
            sort=True,
            avg=False,
            features=feats_effective,
            feature_builder=fb,
        )
    except Exception as e:
        return None, [f"❌ Error en MCDAEngine.score: {e}"]

    return scores_df, info_lines


def _evaluate_all_presets(
    cleaned_df: pd.DataFrame,
    fb: FeatureBuilder,
    mcda: MCDAEngine,
    provincia: Optional[str],
    location: str,
    codigo_postal: Optional[int],
    rango_peso: Optional[str],
    date_from,
    date_to,
    date_col: Optional[str],
    feats_selected: List[str],
    metric_method: str,
    drop_incomplete: bool,
    drop_features_with_missing: bool,
    candidate_presets: Dict[str, Dict[str, float]],
):
    info_lines: List[str] = []

    if not feats_selected:
        return None, None, ["❌ Tenés que elegir al menos una feature antes de evaluar presets."]

    df_for_mcda = _apply_current_filters_to_df(
        cleaned_df,
        provincia,
        location,
        codigo_postal,
        rango_peso,
        date_from,
        date_to,
        date_col,
    )

    if df_for_mcda is None or df_for_mcda.empty:
        return None, None, [
            "❌ No hay datos para ese filtro (después de aplicar todos los filtros) "
            "al evaluar presets."
        ]

    provincia_for_fb = None if provincia in ("Todas", "(subí un archivo)") else provincia
    rango_peso_for_fb = None
    if rango_peso not in (None, "Todos", "(subí un archivo)", "(no disponible)"):
        rango_peso_for_fb = rango_peso

    try:
        df_metrics = fb.build(
            df_for_mcda,
            features=feats_selected,
            provincia=provincia_for_fb,
            location=location,
            codigo_postal=codigo_postal,
            rango_peso=rango_peso_for_fb,
            drop_incomplete=drop_incomplete,
            drop_features_with_missing=drop_features_with_missing,
            cache_global=True,
        )
    except Exception as e:
        return None, None, [f"❌ Error calculando features para presets: {e}"]

    if df_metrics.empty:
        return None, None, [
            "❌ No hay datos para ese filtro luego de construir las features "
            "al evaluar presets."
        ]

    feats_effective = [f for f in feats_selected if f in df_metrics.columns]
    if not feats_effective:
        return None, None, [
            "❌ Ninguna de las features seleccionadas quedó disponible para evaluar presets."
        ]

    winners = []
    for preset_name, preset_weights in candidate_presets.items():
        raw_vals = [preset_weights.get(f, 0.0) for f in feats_effective]
        arr = np.array(raw_vals, dtype=float)

        if arr.sum() <= 0:
            continue

        arr = arr / arr.sum()

        try:
            scores_df = mcda.score(
                df_metrics,
                methods=[metric_method],
                weights=arr,
                criteria_types=None,
                return_df=True,
                sort=True,
                avg=False,
                features=feats_effective,
                feature_builder=fb,
            )
        except Exception as e:
            info_lines.append(f"❌ Error evaluando preset '{preset_name}': {e}")
            continue

        if scores_df.empty:
            continue

        winner = scores_df.index[0]
        winners.append({"preset": preset_name, "winner": winner})

    if not winners:
        return None, None, info_lines + ["❌ Ningún preset se pudo evaluar con las features seleccionadas."]

    winners_series = pd.Series([w["winner"] for w in winners], name="Proveedor")
    summary_prov = winners_series.value_counts().reset_index()
    summary_prov.columns = ["Proveedor", "Veces_ganador"]
    summary_prov = summary_prov.sort_values("Veces_ganador", ascending=False)

    summary_preset = pd.DataFrame(winners).rename(
        columns={"preset": "Preset", "winner": "Proveedor_ganador"}
    )

    return summary_prov, summary_preset, info_lines


# ---------- Matrix styling (delta only) ----------

def _style_delta_red_green(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """Green for >0, red for <0, gray for ~0, none for NaN."""
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for r in df.index:
        for c in df.columns:
            v = df.loc[r, c]
            if pd.isna(v):
                continue
            try:
                x = float(v)
            except Exception:
                continue

            if abs(x) < eps:
                styles.loc[r, c] = "background-color: rgba(148,163,184,0.25);"  # gray
            elif x > 0:
                styles.loc[r, c] = "background-color: rgba(34,197,94,0.35);"   # green
            else:
                styles.loc[r, c] = "background-color: rgba(239,68,68,0.35);"   # red
    return styles


def _fmt_delta(v, eps: float = 1e-12) -> str:
    """Format delta: hide NaN, show + for non-zero, and handle ~0 nicely."""
    if pd.isna(v):
        return ""
    try:
        x = float(v)
    except Exception:
        return str(v)

    if abs(x) < eps:
        return "0.000"

    return f"{x:+.3f}"


def _delta_badge_html(v, eps: float = 1e-12) -> str:
    """Small colored pill for a delta value (consistent with the matrix)."""
    if pd.isna(v):
        return "<span style='opacity:0.6;'>—</span>"
    try:
        x = float(v)
    except Exception:
        return f"<span style='opacity:0.85;'>{v}</span>"

    if abs(x) < eps:
        bg = "rgba(148,163,184,0.25)"
        txt = "rgba(255,255,255,0.78)"
    elif x > 0:
        bg = "rgba(34,197,94,0.35)"
        txt = "rgba(255,255,255,0.92)"
    else:
        bg = "rgba(239,68,68,0.35)"
        txt = "rgba(255,255,255,0.92)"

    return (
        f"<span style='display:inline-block; padding:6px 10px; border-radius:999px; "
        f"background:{bg}; color:{txt}; border:1px solid rgba(255,255,255,0.14); "
        f"font-variant-numeric: tabular-nums; font-size:12.5px;'>"
        f"{_fmt_delta(x, eps=eps)}"
        f"</span>"
    )


def _feature_is_cost_fallback(feature: str) -> bool:
    name = str(feature).lower().strip()
    return ("cost" in name) or (name == "speed") or ("precio" in name) or ("tarifa" in name)


# ---------- Sidebar builders ----------

def _sidebar_navigation() -> str:
    st.sidebar.markdown("## 🧭 Navegación")

    # ✅ Dataset is now the default page (index=0)
    page = st.sidebar.radio(
        "Ir a",
        options=["🗂️ Dataset", "🏁 Ranking", "🧾 Reportes", "🧩 Matrices"],
        index=0,
        key="nav_page",
    )
    st.sidebar.markdown("---")
    return page


def _sidebar_filters(
    prov_options: List[str],
    rango_options: List[str],
    date_min,
    date_max,
    date_col_name: Optional[str],
) -> Tuple[Optional[str], str, Optional[int], Optional[str], Optional[object], Optional[object], Optional[str], bool, bool]:
    st.sidebar.markdown("### 🔎 Filtros")

    provincia = st.sidebar.selectbox("Provincia", prov_options, key="provincia")

    zona_label_to_value = {"Ambas": "both", "Capital": "capital", "Interior": "interior"}
    zona_value_to_label = {v: k for k, v in zona_label_to_value.items()}
    zona_label_default = zona_value_to_label.get(st.session_state.get("zona_value", "both"), "Ambas")

    zona_label = st.sidebar.radio(
        "Zona",
        options=list(zona_label_to_value.keys()),
        index=list(zona_label_to_value.keys()).index(zona_label_default),
        horizontal=True,
    )
    location = zona_label_to_value[zona_label]
    st.session_state["zona_value"] = location

    cp_str = st.sidebar.text_input("Cód. postal", key="cp_str")
    codigo_postal: Optional[int] = None
    if cp_str.strip() != "":
        try:
            codigo_postal = int(cp_str.strip())
        except ValueError:
            st.sidebar.warning("❌ El código postal debe ser numérico. Ignorando filtro de CP.")

    rango_peso = st.sidebar.selectbox("Rango de peso", rango_options, key="rango_peso")

    # --- DATE FILTER (stable UX): two pickers + form "Aplicar" + UI auto-swap ---
    if date_min is not None and date_max is not None and date_col_name is not None:

        if "date_from" not in st.session_state:
            st.session_state["date_from"] = date_min
        if "date_to" not in st.session_state:
            st.session_state["date_to"] = date_max

        pending = st.session_state.pop("_pending_apply_dates", None)
        if isinstance(pending, dict):
            dfrom = pending.get("from") or date_min
            dto = pending.get("to") or date_max

            if dfrom < date_min:
                dfrom = date_min
            if dfrom > date_max:
                dfrom = date_max
            if dto < date_min:
                dto = date_min
            if dto > date_max:
                dto = date_max

            if dfrom > dto:
                dfrom, dto = dto, dfrom

            st.session_state["date_from"] = dfrom
            st.session_state["date_to"] = dto

            st.session_state["date_from_ui"] = dfrom
            st.session_state["date_to_ui"] = dto

        dfrom_applied = st.session_state["date_from"] or date_min
        dto_applied = st.session_state["date_to"] or date_max

        if dfrom_applied < date_min:
            dfrom_applied = date_min
        if dfrom_applied > date_max:
            dfrom_applied = date_max
        if dto_applied < date_min:
            dto_applied = date_min
        if dto_applied > date_max:
            dto_applied = date_max
        if dfrom_applied > dto_applied:
            dfrom_applied, dto_applied = dto_applied, dfrom_applied

        st.session_state["date_from"] = dfrom_applied
        st.session_state["date_to"] = dto_applied

        if "date_from_ui" not in st.session_state:
            st.session_state["date_from_ui"] = st.session_state["date_from"]
        if "date_to_ui" not in st.session_state:
            st.session_state["date_to_ui"] = st.session_state["date_to"]

        with st.sidebar.form("date_filter_form", clear_on_submit=False):
            st.markdown("**Rango de fechas**")

            cdf1, cdf2 = st.columns(2)
            with cdf1:
                ui_from = st.date_input(
                    "Desde",
                    min_value=date_min,
                    max_value=date_max,
                    key="date_from_ui",
                )
            with cdf2:
                ui_to = st.date_input(
                    "Hasta",
                    min_value=date_min,
                    max_value=date_max,
                    key="date_to_ui",
                )

            cbtn1, cbtn2 = st.columns(2)
            with cbtn1:
                apply_dates = st.form_submit_button("Aplicar")
            with cbtn2:
                reset_dates = st.form_submit_button("Reset")

        if reset_dates:
            st.session_state["date_from"] = date_min
            st.session_state["date_to"] = date_max
            st.session_state["date_from_ui"] = date_min
            st.session_state["date_to_ui"] = date_max
            st.rerun()

        if apply_dates:
            st.session_state["_pending_apply_dates"] = {"from": ui_from, "to": ui_to}
            st.rerun()

        date_from = st.session_state["date_from"]
        date_to = st.session_state["date_to"]
        DATE_COL = date_col_name

    else:
        date_from, date_to, DATE_COL = None, None, None

    drop_incomplete = st.sidebar.checkbox(
        "Excluir proveedores con valores nulos en las features",
        value=True,
        key="drop_incomplete",
    )
    drop_features_nan = st.sidebar.checkbox(
        "Excluir features con columnas totalmente nulas (NaN)",
        value=False,
        key="drop_features_nan",
    )

    st.sidebar.markdown("---")
    return provincia, location, codigo_postal, rango_peso, date_from, date_to, DATE_COL, drop_incomplete, drop_features_nan


def _sidebar_mcda_settings(feats_all: List[str]) -> Tuple[List[str], Dict[str, float], str]:
    st.sidebar.markdown("### 🧠 Métrica (features & pesos)")

    flash = st.session_state.get("_normalize_flash", None)
    if flash:
        if flash.startswith("✅"):
            st.sidebar.success(flash)
        else:
            st.sidebar.warning(flash)
        st.session_state["_normalize_flash"] = None

    selected_feats = st.sidebar.multiselect(
        "Features",
        options=feats_all,
        default=st.session_state.get("selected_feats", feats_all),
        key="selected_feats",
    )
    if not selected_feats:
        st.sidebar.warning("Elegí al menos una feature.")

    weights_map: Dict[str, float] = {}
    for feat in selected_feats:
        key = f"weight_{feat}"
        if key not in st.session_state:
            st.session_state[key] = 1.0 / float(len(selected_feats)) if selected_feats else 0.0
        weights_map[feat] = st.sidebar.number_input(
            f"Peso: {feat}",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key=key,
        )

    sum_weights = sum(weights_map.values()) if weights_map else 0.0
    st.sidebar.markdown(f"Suma de pesos: **{sum_weights:.6f}**")

    if st.sidebar.button("Normalizar pesos"):
        if sum_weights > 0:
            st.session_state["_pending_normalize_weights"] = True
            st.rerun()
        else:
            st.sidebar.warning("No se puede normalizar: la suma actual es 0.")

    metric_label_to_value = {
        "WASPAS (default)": "waspas",
        "Weighted": "weighted",
        "TOPSIS": "topsis",
        "VIKOR": "vikor",
    }
    metric_label = st.sidebar.selectbox(
        "Métrica",
        list(metric_label_to_value.keys()),
        key="metric_label",
    )
    metric_method = metric_label_to_value[metric_label]

    st.sidebar.markdown("---")
    return selected_feats, weights_map, metric_method


def _sidebar_presets(
    candidate_presets: Dict[str, Dict[str, float]],
    selected_feats: List[str],
    weights_map: Dict[str, float],
) -> None:
    st.sidebar.markdown("### 💾 Presets")

    flash = st.session_state.get("_preset_flash", None)
    if flash:
        if flash.startswith("✅"):
            st.sidebar.success(flash)
        else:
            st.sidebar.warning(flash)
        st.session_state["_preset_flash"] = None

    preset_options = ["(sin preset)"] + list(candidate_presets.keys())
    current_preset_name = st.sidebar.selectbox(
        "Preset",
        options=preset_options,
        index=0,
        key="preset_select",
    )

    preset_name_text = st.sidebar.text_input(
        "Nombre (guardar como nuevo preset)",
        key="preset_name_text",
        placeholder="nuevo_preset",
    )

    cols_p = st.sidebar.columns(3)
    with cols_p[0]:
        if st.button("Aplicar"):
            if current_preset_name not in candidate_presets:
                st.warning("Seleccioná un preset válido para aplicar.")
            else:
                st.session_state["_pending_apply_preset_name"] = current_preset_name
                st.rerun()

    with cols_p[1]:
        if st.button("Guardar nuevo"):
            name = preset_name_text.strip()
            if not name:
                st.warning("Escribí un nombre para el nuevo preset.")
            elif not selected_feats:
                st.warning("Tenés que elegir al menos una feature para guardar un preset.")
            else:
                weights_to_save = {f: float(weights_map.get(f, 0.0)) for f in selected_feats}
                if not any(v != 0.0 for v in weights_to_save.values()):
                    st.warning("No hay pesos (distintos de cero) para guardar.")
                else:
                    _current_presets[name] = weights_to_save
                    _save_current_presets(_current_presets)
                    st.success(f"Preset '{name}' guardado/actualizado.")
                    st.rerun()

    with cols_p[2]:
        if st.button("Sobrescribir"):
            if current_preset_name not in candidate_presets:
                st.warning("Seleccioná un preset existente para sobrescribirlo.")
            elif not selected_feats:
                st.warning("Tenés que elegir al menos una feature para guardar un preset.")
            else:
                weights_to_save = {f: float(weights_map.get(f, 0.0)) for f in selected_feats}
                if not any(v != 0.0 for v in weights_to_save.values()):
                    st.warning("No hay pesos para guardar.")
                else:
                    _current_presets[current_preset_name] = weights_to_save
                    _save_current_presets(_current_presets)
                    st.success(f"Preset '{current_preset_name}' sobrescrito.")
                    st.rerun()

    cols_p2 = st.sidebar.columns(3)
    with cols_p2[0]:
        if st.button("Eliminar"):
            if current_preset_name not in _current_presets:
                st.warning("Seleccioná un preset existente para eliminar.")
            else:
                del _current_presets[current_preset_name]
                _save_current_presets(_current_presets)
                st.success(f"Preset '{current_preset_name}' eliminado.")
                st.rerun()

    with cols_p2[1]:
        if st.button("Restaurar preset"):
            if current_preset_name not in _default_presets:
                st.warning("No existe versión por defecto para este preset.")
            else:
                _current_presets[current_preset_name] = _default_presets[current_preset_name].copy()
                _save_current_presets(_current_presets)
                st.success(f"Preset '{current_preset_name}' restaurado.")
                st.rerun()

    with cols_p2[2]:
        if st.button("Restaurar TODOS"):
            _current_presets.clear()
            for n, p in _default_presets.items():
                _current_presets[n] = p.copy()
            _save_current_presets(_current_presets)
            st.success("Presets restaurados por defecto.")
            st.rerun()

    st.sidebar.markdown("---")


# ---------- Main ----------

def main():
    st.set_page_config(
        page_title="Recomendador de Envíos",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_dark_dashboard_css()

    st.title("📦 Recomendador de Envíos")

    pre = Preprocessor(normalize_names=True)
    fb = FeatureBuilder()
    mcda = MCDAEngine()

    # ✅ Load active dataset (default OR uploaded) for the entire app
    cleaned_df, raw_df, scenario_reporter, messages, active_source = _get_active_dataset(pre, fb, mcda)

    for msg in messages:
        if msg.startswith("❌") or msg.startswith("Error"):
            st.error(msg)
        else:
            st.info(msg)

    if cleaned_df is None or cleaned_df.empty or scenario_reporter is None:
        st.stop()

    DATE_MIN, DATE_MAX, DATE_COL = _detect_date_range(cleaned_df, scenario_reporter)

    prov_options = ["Todas"]
    if "Provincia" in cleaned_df.columns:
        provs = sorted(cleaned_df["Provincia"].dropna().unique().tolist())
        prov_options += provs

    if "Rango de Peso" in cleaned_df.columns:
        rango_options = ["Todos"] + sorted(cleaned_df["Rango de Peso"].dropna().unique().tolist())
    else:
        rango_options = ["(no disponible)"]

    feats_all = list(getattr(fb, "available_features", []))
    if not feats_all:
        st.error("❌ FeatureBuilder.available_features está vacío. Revisá la configuración de FeatureBuilder.")
        st.stop()

    candidate_presets = _rebuild_candidate_presets()

    _apply_pending_preset_if_any(candidate_presets=candidate_presets, feats_all=feats_all)
    _apply_pending_normalize_if_any(feats_all=feats_all)

    page = _sidebar_navigation()

    provincia = "Todas"
    location = "both"
    codigo_postal = None
    rango_peso = "Todos"
    date_from, date_to = DATE_MIN, DATE_MAX
    drop_incomplete = True
    drop_features_nan = False

    selected_feats = st.session_state.get("selected_feats", feats_all)
    weights_map: Dict[str, float] = {}
    metric_method = st.session_state.get("metric_label", "WASPAS (default)")
    metric_method = {"WASPAS (default)": "waspas", "Weighted": "weighted", "TOPSIS": "topsis", "VIKOR": "vikor"}.get(metric_method, "waspas")

    if page in ("🏁 Ranking", "🧾 Reportes", "🗂️ Dataset"):
        (
            provincia,
            location,
            codigo_postal,
            rango_peso,
            date_from,
            date_to,
            DATE_COL,
            drop_incomplete,
            drop_features_nan,
        ) = _sidebar_filters(
            prov_options=prov_options,
            rango_options=rango_options,
            date_min=DATE_MIN,
            date_max=DATE_MAX,
            date_col_name=DATE_COL,
        )

    if page in ("🏁 Ranking", "🧾 Reportes", "🧩 Matrices"):
        selected_feats, weights_map, metric_method = _sidebar_mcda_settings(feats_all)
        _sidebar_presets(candidate_presets, selected_feats, weights_map)

    filter_suffix = _filter_suffix(
        provincia=provincia,
        location=location,
        codigo_postal=codigo_postal,
        rango_peso=rango_peso,
        date_from=date_from,
        date_to=date_to,
        date_min=DATE_MIN,
        date_max=DATE_MAX,
    )

    # =========================
    # PAGE: Dataset  (NOW DEFAULT)
    # =========================
    if page == "🗂️ Dataset":
        st.markdown("### 🗂️ Dataset")
        _render_filter_chips(filter_suffix)

        with st.expander("📥 Fuente de dataset (por defecto vs subido)", expanded=True):
            _render_dataset_source_controls(active_source_label=active_source)

        col_o1, col_o2 = st.columns([1, 1])
        with col_o1:
            apply_filters_view = st.checkbox("Aplicar filtros actuales a la vista", value=False)
        with col_o2:
            rows_label_to_value = {
                "Primeras 10 filas": 10,
                "Primeras 50 filas": 50,
                "Primeras 500 filas": 500,
                "Todas las filas": None,
            }
            rows_label = st.selectbox("Mostrar", list(rows_label_to_value.keys()), index=0)
            rows_value = rows_label_to_value[rows_label]

        df_original = cleaned_df
        total_rows = len(df_original)

        if apply_filters_view:
            df_original = _apply_current_filters_to_df(
                df_original,
                provincia,
                location,
                codigo_postal,
                rango_peso,
                date_from,
                date_to,
                DATE_COL,
            )

        filtered_rows = len(df_original)
        df_show = df_original.head(rows_value) if rows_value is not None else df_original
        shown_rows = len(df_show)

        if apply_filters_view:
            msg = (
                f"Filas totales (dataset limpio): **{total_rows}**  \n"
                f"Filas después de aplicar filtros: **{filtered_rows}**  \n"
            )
        else:
            msg = (
                f"Filas totales (dataset limpio): **{total_rows}**  \n"
                f"Filas en la vista (sin filtros adicionales): **{filtered_rows}**  \n"
            )

        if shown_rows < filtered_rows:
            msg += f"Mostrando **{shown_rows}** de **{filtered_rows}** filas."
        else:
            msg += f"Mostrando todas las **{shown_rows}** filas."

        st.markdown(msg)
        st.dataframe(df_show)

    # =========================
    # PAGE: Ranking
    # =========================
    elif page == "🏁 Ranking":
        st.markdown("### 🏆 Ranking de proveedores")
        _render_filter_chips(filter_suffix)

        col_b1, col_b2 = st.columns([1, 1])
        with col_b1:
            run_ranking_clicked = st.button("⚡ Calcular ranking")
        with col_b2:
            presets_clicked = st.button("🧪 Evaluar todos los presets")

        if run_ranking_clicked:
            if not selected_feats:
                st.session_state["ranking_info_lines"] = ["❌ Tenés que elegir al menos una feature."]
                st.session_state["scores_df"] = None
            else:
                scores_df, info_lines = _run_mcda_ranking(
                    cleaned_df=cleaned_df,
                    fb=fb,
                    mcda=mcda,
                    provincia=provincia,
                    location=location,
                    codigo_postal=codigo_postal,
                    rango_peso=rango_peso,
                    date_from=date_from,
                    date_to=date_to,
                    date_col=DATE_COL,
                    feats_selected=selected_feats,
                    drop_incomplete=drop_incomplete,
                    drop_features_with_missing=drop_features_nan,
                    metric_method=metric_method,
                    weights_map=weights_map,
                )
                st.session_state["ranking_info_lines"] = info_lines
                st.session_state["scores_df"] = scores_df

        scores_df = st.session_state.get("scores_df", None)
        info_lines = st.session_state.get("ranking_info_lines", [])

        if info_lines:
            st.text_area("Log (ranking)", value="\n".join(info_lines), height=150)

        if scores_df is not None and not scores_df.empty:
            st.dataframe(scores_df.head(20).round(4))
            csv_data = scores_df.to_csv(index=True).encode("utf-8")
            st.download_button(
                "⬇️ Descargar CSV de scores",
                data=csv_data,
                file_name="mcda_scores.csv",
                mime="text/csv",
            )

        if presets_clicked:
            if not selected_feats:
                st.session_state["presets_info_lines"] = ["❌ Tenés que elegir al menos una feature antes de evaluar presets."]
                st.session_state["summary_prov"] = None
                st.session_state["summary_preset"] = None
            else:
                summary_prov, summary_preset, p_info = _evaluate_all_presets(
                    cleaned_df=cleaned_df,
                    fb=fb,
                    mcda=mcda,
                    provincia=provincia,
                    location=location,
                    codigo_postal=codigo_postal,
                    rango_peso=rango_peso,
                    date_from=date_from,
                    date_to=date_to,
                    date_col=DATE_COL,
                    feats_selected=selected_feats,
                    metric_method=metric_method,
                    drop_incomplete=drop_incomplete,
                    drop_features_with_missing=drop_features_nan,
                    candidate_presets=candidate_presets,
                )
                st.session_state["summary_prov"] = summary_prov
                st.session_state["summary_preset"] = summary_preset
                st.session_state["presets_info_lines"] = p_info

        summary_prov = st.session_state.get("summary_prov", None)
        summary_preset = st.session_state.get("summary_preset", None)
        p_info = st.session_state.get("presets_info_lines", [])

        if p_info:
            st.text_area("Log (presets)", value="\n".join(p_info), height=150)

        if summary_prov is not None and not summary_prov.empty:
            st.markdown("🏆 **Proveedores más robustos:**")
            st.dataframe(summary_prov)

        if summary_preset is not None and not summary_preset.empty:
            st.markdown("📋 **Ganador por preset:**")
            st.dataframe(summary_preset)

    # =========================
    # PAGE: Reportes
    # =========================
    elif page == "🧾 Reportes":
        st.markdown("### 🧾 Reportes de escenarios")
        _render_filter_chips(filter_suffix)

        shorts = list(getattr(scenario_reporter, "proveedores_short", [])) or []

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            prov_a = st.selectbox(
                "Proveedor A",
                options=["(sin datos)"] + shorts if shorts else ["(sin datos)"],
                index=1 if shorts else 0,
                key="prov_a_dd",
            )
        with col_r2:
            prov_b = st.selectbox(
                "Proveedor B",
                options=["(sin datos)"] + shorts if shorts else ["(sin datos)"],
                index=1 if shorts else 0,
                key="prov_b_dd",
            )

        col_rb1, col_rb2, col_rb3 = st.columns(3)
        with col_rb1:
            report_combined_clicked = st.button("🧩 Reporte combinado: métricas + costos")
        with col_rb2:
            report_mcda_clicked = st.button("🏁 Comparación: histórico vs #1 ranking")
        with col_rb3:
            report_two_clicked = st.button("🆚 Escenario: proveedor A vs B")

        def get_report_filters():
            return {
                "provincia": None if provincia in ("Todas", "(subí un archivo)") else provincia,
                "location": location,
                "codigo_postal": codigo_postal,
                "rango_peso": None
                if rango_peso in ("Todos", "(subí un archivo)", "(no disponible)")
                else rango_peso,
                "date_from": date_from.isoformat() if date_from is not None else None,
                "date_to": date_to.isoformat() if date_to is not None else None,
            }

        if report_combined_clicked:
            filters = get_report_filters()
            try:
                text = scenario_reporter.combined_most_used_report(
                    date_from=filters["date_from"],
                    date_to=filters["date_to"],
                    provincia=filters["provincia"],
                    location=filters["location"],
                    rango_peso=filters["rango_peso"],
                    codigo_postal=filters["codigo_postal"],
                    features=None,
                )
                st.session_state["scenario_report_text"] = text
            except Exception as e:
                st.session_state["scenario_report_text"] = f"❌ Error generando el reporte combinado: {e}"

        if report_mcda_clicked:
            filters = get_report_filters()
            if not selected_feats:
                st.session_state["scenario_report_text"] = "❌ Tenés que elegir al menos una feature para el escenario MCDA."
            else:
                raw_weights = [float(weights_map.get(f, 0.0)) for f in selected_feats]
                sum_w = float(sum(raw_weights))
                if not np.isfinite(sum_w) or sum_w <= 0:
                    st.session_state["scenario_report_text"] = "❌ Los pesos actuales no son válidos (suma <= 0)."
                else:
                    weights_arr = np.array(raw_weights, dtype=float) / sum_w
                    methods = [metric_method] if metric_method else ["waspas"]
                    try:
                        stats = scenario_reporter.scenario_baseline_vs_mcda_best(
                            date_from=filters["date_from"],
                            date_to=filters["date_to"],
                            provincia=filters["provincia"],
                            location=filters["location"],
                            rango_peso=filters["rango_peso"],
                            codigo_postal=filters["codigo_postal"],
                            methods=methods,
                            weights=weights_arr,
                            criteria_types=None,
                            features=selected_feats,
                            weights_preset=None,
                            baseline_provider=None,
                        )
                        text = scenario_reporter.pretty_cost_report(
                            stats,
                            title_prefix="Escenario: proveedor más usado vs mejor proveedor según simulación",
                        )
                        st.session_state["scenario_report_text"] = text
                    except Exception as e:
                        st.session_state["scenario_report_text"] = f"❌ Error generando el escenario MCDA: {e}"

        if report_two_clicked:
            filters = get_report_filters()
            if prov_a in (None, "(sin datos)") or prov_b in (None, "(sin datos)"):
                st.session_state["scenario_report_text"] = "❌ Elegí ambos proveedores A y B."
            elif prov_a == prov_b:
                st.session_state["scenario_report_text"] = "❌ Elegí dos proveedores distintos para comparar."
            else:
                try:
                    stats = scenario_reporter.compare_two_providers_cost(
                        current_provider=prov_a,
                        alternative_provider=prov_b,
                        date_from=filters["date_from"],
                        date_to=filters["date_to"],
                        provincia=filters["provincia"],
                        location=filters["location"],
                        rango_peso=filters["rango_peso"],
                        codigo_postal=filters["codigo_postal"],
                    )
                    title = f"Escenario: {prov_a} (base) vs {prov_b} (alternativo)"
                    text = scenario_reporter.pretty_cost_report(stats, title_prefix=title)
                    st.session_state["scenario_report_text"] = text
                except Exception as e:
                    st.session_state["scenario_report_text"] = f"❌ Error comparando proveedores '{prov_a}' y '{prov_b}': {e}"

        report_text = st.session_state.get("scenario_report_text", None)
        if report_text:
            st.markdown(_report_text_to_html(report_text), unsafe_allow_html=True)

    # =========================
    # PAGE: Matrices
    # =========================
    elif page == "🧩 Matrices":
        st.markdown("### 🧩 Matrices")
        st.subheader("Matriz visual por provincia y feature")

        viz_prov = st.selectbox(
            "Provincia para matriz visual",
            options=prov_options,  # ✅ includes "Todas"
            index=0,
            key="viz_prov",
        )

        viz_feat = st.selectbox(
            "Feature para matriz visual",
            options=feats_all,
            key="viz_feat",
        )

        # Decide cost/benefit using ScenarioReporter (preferred)
        try:
            feat_is_cost = bool(getattr(scenario_reporter, "_feature_is_cost")(viz_feat))
        except Exception:
            feat_is_cost = _feature_is_cost_fallback(viz_feat)

        # "Sheet-style" selector: default Delta
        sheet_label_to_key = {
            "Comparación (simulación vs histórico) ✅": "delta",
            "Valores históricos": "real",
            "Valores simulados": "mcda",
        }
        sheet_label = st.radio(
            "Hoja",
            options=list(sheet_label_to_key.keys()),
            index=0,
            horizontal=True,
            key="matrix_sheet_label",
        )
        sheet_key = sheet_label_to_key[sheet_label]

        min_shipments = st.number_input(
            "Mínimo de envíos (por celda)",
            min_value=1,
            step=1,
            value=1,
            key="matrix_min_shipments",
        )

        if st.button("✨ Generar / actualizar matriz"):
            # Choose weights/features for MCDA used inside the matrix builder
            if selected_feats:
                mcda_feats_for_matrix = selected_feats
                raw_vals = [weights_map.get(f, 0.0) for f in mcda_feats_for_matrix]
                arr = np.array(raw_vals, dtype=float)
                if arr.sum() <= 0:
                    arr = np.ones_like(arr, dtype=float)
                arr = arr / arr.sum()
                weights_for_matrix = arr.tolist()
            else:
                mcda_feats_for_matrix = feats_all
                arr = np.ones(len(mcda_feats_for_matrix), dtype=float)
                arr = arr / arr.sum()
                weights_for_matrix = arr.tolist()

            try:
                mats = scenario_reporter.build_feature_matrices_for_provincia(
                    provincia=viz_prov,
                    feature=viz_feat,
                    min_shipments=int(min_shipments),
                    methods=[metric_method],
                    weights=weights_for_matrix,
                    criteria_types=None,
                    all_features_for_mcda=mcda_feats_for_matrix,
                    weights_preset=None,
                    smooth_approximation=None,
                    normalize_globally=None,
                    by_shipments=None,
                    weight_by_range=None,
                )
            except AttributeError:
                st.error(
                    "❌ ScenarioReporter no tiene el método "
                    "'build_feature_matrices_for_provincia'. "
                    "Asegurate de haber actualizado scenario_reporter.py."
                )
                mats = {}
            except Exception as e:
                st.error(f"❌ Error construyendo matrices: {e}")
                mats = {}

            if not mats:
                st.info("ℹ️ No se pudo construir la matriz para esa provincia/feature (quizás no hay datos suficientes).")
                st.session_state["last_mats"] = None
                st.session_state["last_mats_sig"] = None
            else:
                # Sanitize axes BEFORE saving
                if isinstance(mats, dict):
                    mats = mats.copy()
                    mats["valores_real"] = _sanitize_matrix_axes(mats.get("valores_real"))
                    mats["valores_mcda"] = _sanitize_matrix_axes(mats.get("valores_mcda"))

                    # ✅ Prefer new neutral "delta" if present
                    if "delta" in mats and isinstance(mats.get("delta"), pd.DataFrame):
                        mats["delta"] = _sanitize_matrix_axes(mats.get("delta"))
                    mats["delta_mcda_menos_real"] = _sanitize_matrix_axes(mats.get("delta_mcda_menos_real"))

                    mats["proveedor_real"] = _sanitize_provider_matrix(mats.get("proveedor_real"))
                    mats["proveedor_mcda"] = _sanitize_provider_matrix(mats.get("proveedor_mcda"))

                st.session_state["last_mats"] = mats
                st.session_state["last_mats_meta"] = {
                    "provincia": viz_prov,
                    "feature": viz_feat,
                    "metric": metric_method,
                    "min_shipments": int(min_shipments),
                }

                # Signature to reset inspection selection
                df_delta_tmp = mats.get("delta") if isinstance(mats.get("delta"), pd.DataFrame) else mats.get("delta_mcda_menos_real")
                shape = df_delta_tmp.shape if isinstance(df_delta_tmp, pd.DataFrame) else None
                st.session_state["last_mats_sig"] = (viz_prov, viz_feat, metric_method, int(min_shipments), shape)

                st.session_state.pop("inspect_row", None)
                st.session_state.pop("inspect_col", None)

        mats = st.session_state.get("last_mats", None)

        if mats is None:
            st.info("ℹ️ Generá una matriz para poder verla.")
        else:
            df_real = mats.get("valores_real")
            df_mcda = mats.get("valores_mcda")

            # ✅ prefer mats["delta"], fallback mats["delta_mcda_menos_real"]
            df_delta = mats.get("delta")
            if df_delta is None:
                df_delta = mats.get("delta_mcda_menos_real")

            prov_real = mats.get("proveedor_real")
            prov_mcda = mats.get("proveedor_mcda")

            if df_real is None or df_mcda is None or df_delta is None:
                st.warning("⚠️ La salida de matrices no tiene todas las tablas esperadas (histórico/mcda/delta).")
            else:
                df_real = _sanitize_matrix_axes(df_real)
                df_mcda = _sanitize_matrix_axes(df_mcda)
                df_delta = _sanitize_matrix_axes(df_delta)
                prov_real = _sanitize_provider_matrix(prov_real)
                prov_mcda = _sanitize_provider_matrix(prov_mcda)

                # Reset inspect widgets if signature changed
                cur_sig = (
                    st.session_state.get("last_mats_meta", {}).get("provincia"),
                    st.session_state.get("last_mats_meta", {}).get("feature"),
                    st.session_state.get("last_mats_meta", {}).get("metric"),
                    st.session_state.get("last_mats_meta", {}).get("min_shipments"),
                    df_delta.shape if isinstance(df_delta, pd.DataFrame) else None,
                )
                prev_sig = st.session_state.get("_inspect_sig")
                if prev_sig != cur_sig:
                    st.session_state["_inspect_sig"] = cur_sig
                    st.session_state.pop("inspect_row", None)
                    st.session_state.pop("inspect_col", None)

                # --- COST summary above delta matrix ---
                if sheet_key == "delta" and feat_is_cost:
                    total = float(np.nansum(df_delta.to_numpy(dtype=float)))
                    if np.isfinite(total):
                        if total > 0:
                            st.success(f"🟢 Ahorro total (sumatoria de celdas): **{total:,.3f}**")
                        elif total < 0:
                            st.error(f"🔴 Pérdida total (sumatoria de celdas): **{abs(total):,.3f}**")
                        else:
                            st.info("⚪ Ahorro/Pérdida total: **0.000**")

                # Select which sheet to show
                if sheet_key == "real":
                    st.markdown("**Valores históricos (proveedor más elegido)**")
                    sty = df_real.style.format(lambda v: "" if pd.isna(v) else f"{float(v):.3f}")
                    st.dataframe(sty, use_container_width=True)
                elif sheet_key == "mcda":
                    st.markdown("**Valores simulados (mejor proveedor según métricas)**")
                    sty = df_mcda.style.format(lambda v: "" if pd.isna(v) else f"{float(v):.3f}")
                    st.dataframe(sty, use_container_width=True)
                else:
                    # ✅ wording now matches sign convention (always positive=mejora)
                    st.markdown("**Delta (mejora vs histórico; positivo = mejor)**")
                    sty = (
                        df_delta.style
                        .apply(lambda _: _style_delta_red_green(df_delta), axis=None)
                        .format(_fmt_delta)
                    )
                    st.dataframe(sty, use_container_width=True)

                # ---- inspection ----
                with st.expander("🔍 Inspeccionar una celda", expanded=True):
                    rows = list(df_delta.index)
                    cols = list(df_delta.columns)

                    if not rows or not cols:
                        st.info("ℹ️ No hay celdas para inspeccionar.")
                    else:
                        tok = int(st.session_state.get("_inspect_reset_token", 0))

                        c1, c2 = st.columns(2)
                        with c1:
                            sel_r = st.selectbox("Código Postal", options=rows, index=0, key=f"inspect_row_{tok}")
                        with c2:
                            sel_c = st.selectbox("Rango de Peso", options=cols, index=0, key=f"inspect_col_{tok}")

                        try:
                            v_real = df_real.loc[sel_r, sel_c]
                        except Exception:
                            v_real = np.nan
                        try:
                            v_mcda = df_mcda.loc[sel_r, sel_c]
                        except Exception:
                            v_mcda = np.nan
                        try:
                            v_delta = df_delta.loc[sel_r, sel_c]
                        except Exception:
                            v_delta = np.nan

                        try:
                            p_real = prov_real.loc[sel_r, sel_c] if prov_real is not None else None
                        except Exception:
                            p_real = None
                        try:
                            p_mcda = prov_mcda.loc[sel_r, sel_c] if prov_mcda is not None else None
                        except Exception:
                            p_mcda = None

                        def fmt_plain(x):
                            if pd.isna(x):
                                return "—"
                            try:
                                return f"{float(x):.3f}"
                            except Exception:
                                return str(x)

                        st.markdown("#### Detalle de celda")
                        cold1, cold2, cold3 = st.columns(3)

                        with cold1:
                            st.markdown("**Histórico**")
                            st.write(f"Proveedor: {p_real if p_real not in (None, '', np.nan) else '—'}")
                            st.write(f"Valor: {fmt_plain(v_real)}")

                        with cold2:
                            st.markdown("**Simulado**")
                            st.write(f"Proveedor: {p_mcda if p_mcda not in (None, '', np.nan) else '—'}")
                            st.write(f"Valor: {fmt_plain(v_mcda)}")

                        with cold3:
                            st.markdown("**Comparación (mejora vs histórico)**")
                            st.markdown(_delta_badge_html(v_delta), unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Reportes de matrices (CSV)")

        matrix_features = st.multiselect(
            "Features para matrices",
            options=feats_all,
            default=feats_all,
            key="matrix_features",
        )
        matrix_provs = st.multiselect(
            "Provincias para matrices",
            options=prov_options[1:],  # export: provinces only
            default=[],
            key="matrix_provs",
        )

        if st.button("⬇️ Exportar matrices a CSV"):
            selected_provs = matrix_provs or None
            selected_feats_for_export = matrix_features or None
            try:
                result = scenario_reporter.export_feature_matrices_for_provincias(
                    provincias=selected_provs,
                    features=selected_feats_for_export,
                    out_dir="report_matrices",
                )
                if not result:
                    st.info("ℹ️ No se generó ningún archivo (quizás no había datos suficientes).")
                else:
                    total_files = sum(len(paths) for paths in result.values())
                    st.success(f"✅ Exportación completada. Archivos generados: {total_files}")

                    for prov, paths in result.items():
                        if not paths:
                            continue
                        st.markdown(f"**Provincia: {prov}**")
                        for p in paths:
                            st.write(f"- {p}")
            except AttributeError:
                st.error(
                    "❌ ScenarioReporter no tiene el método "
                    "'export_feature_matrices_for_provincias'. "
                    "Asegurate de haber actualizado scenario_reporter.py."
                )
            except Exception as e:
                st.error(f"❌ Error exportando matrices: {e}")


if __name__ == "__main__":
    main()
