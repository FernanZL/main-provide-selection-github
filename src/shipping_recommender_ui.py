# ---- MCDA Shipping Recommender UI ----
import sys, os, io, json, re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "..")))

import numpy as np
import pandas as pd
from IPython.display import display, HTML
import ipywidgets as w

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

_BASE_DIR = None  # set from notebook

def set_base_dir(path: str):
    global _BASE_DIR
    _BASE_DIR = os.path.abspath(path)

def get_datasets_dir() -> str:
    base = _BASE_DIR or os.getcwd()
    return os.path.join(base, "datasets")

# ---------- Colab detection (local-safe) ----------
def _is_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False


def _load_current_presets(path: str = PRESETS_JSON_PATH) -> dict[str, dict[str, float]]:
    """
    Load current presets from JSON file, or {} if not found/invalid.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        clean = {}
        for name, weights in data.items():
            if isinstance(weights, dict):
                clean[name] = {str(k): float(v) for k, v in weights.items()}
        return clean
    except Exception:
        return {}


def _save_current_presets(current_presets: dict[str, dict[str, float]],
                          path: str = PRESETS_JSON_PATH) -> None:
    """
    Save current presets to JSON file.
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(current_presets, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ No se pudieron guardar los presets: {e}")


# defaults from the Python file (factory defaults)
_default_presets: dict[str, dict[str, float]] = get_weight_presets()

# try loading current presets from JSON; if empty, start from defaults
_loaded_current = _load_current_presets()
if _loaded_current:
    _current_presets: dict[str, dict[str, float]] = _loaded_current
else:
    # first run: start from defaults
    _current_presets = {name: preset.copy() for name, preset in _default_presets.items()}
    _save_current_presets(_current_presets)

# candidate_presets is what UI uses (mirror of _current_presets)
candidate_presets: dict[str, dict[str, float]] = {}


# ---------- Widgets ----------
title = w.HTML("<h2>📦 Recomendador de Envíos</h2>")

info_box = w.Output()

# === Dataset source (default vs upload) ===
data_source_title = w.HTML("<b>Dataset</b>")
data_source_mode = w.ToggleButtons(
    options=[("Usar dataset por defecto", "default"), ("Subir CSV", "upload")],
    value="default",
    layout=w.Layout(width="430px")
)

data_source_status = w.HTML(
    f"<small>Fuente actual: <b>por defecto</b> ({MAIN_CSV_PATH})</small>"
)

load_default_btn = w.Button(
    description="Cargar dataset por defecto",
    icon="refresh",
    button_style="",
    layout=w.Layout(width="430px")
)

# --- Local/Jupyter upload widgets (not reliable in Colab) ---
upload_widget = w.FileUpload(
    accept=".csv",
    multiple=False,
    description="Subir CSV",
    layout=w.Layout(width="430px")
)

load_uploaded_btn = w.Button(
    description="Cargar dataset subido",
    icon="upload",
    button_style="success",
    layout=w.Layout(width="430px"),
    disabled=True
)

# --- Colab upload (reliable) ---
colab_upload_btn = w.Button(
    description="Subir CSV (Colab)",
    icon="upload",
    button_style="success",
    layout=w.Layout(width="430px"),
)
colab_upload_out = w.Output(layout=w.Layout(width="430px"))

# hidden by default; enabled in _apply_data_source_mode_ui()
colab_upload_btn.layout.display = "none"
colab_upload_out.layout.display = "none"

prov_dropdown = w.Dropdown(
    options=["(subí un archivo)"], value="(subí un archivo)",
    description="Provincia:", disabled=True, layout=w.Layout(width="350px")
)
zona_radio = w.RadioButtons(
    options=[("Ambas", "both"), ("Capital", "capital"), ("Interior", "interior")],
    value="both", description="Zona:", layout=w.Layout(width="350px")
)
cp_text = w.Text(value='', description='Cód. postal:', layout=w.Layout(width="350px"))
rango_dropdown = w.Dropdown(
    options=["(subí un archivo)"], value="(subí un archivo)",
    description="Rango peso:", disabled=True, layout=w.Layout(width="350px")
)

# --- Date filters ---
date_from_picker = w.DatePicker(
    description="Desde:",
    disabled=True,
    layout=w.Layout(width="200px")
)
date_to_picker = w.DatePicker(
    description="Hasta:",
    disabled=True,
    layout=w.Layout(width="200px")
)

date_row = w.HBox(
    [date_from_picker, date_to_picker],
    layout=w.Layout(width="430px", justify_content="space-between")
)

# Wrapped checkbox label row (so it never ellipsizes)
drop_incomplete_cb = w.Checkbox(value=True, description='', indent=False)
drop_incomplete_label = w.HTML(
    "<span>Excluir proveedores con valores nulos en las features</span>"
)
drop_incomplete_row = w.HBox(
    [drop_incomplete_cb, drop_incomplete_label],
    layout=w.Layout(width="430px", align_items="center", margin="5px 0 10px 0")
)

# Wrapped checkbox label row para excluir features con NaN
exclude_nan_feats_cb = w.Checkbox(value=False, description='', indent=False)
exclude_nan_feats_label = w.HTML(
    "<span>Excluir features con columnas totalmente nulas (NaN)</span>"
)
exclude_nan_feats_row = w.HBox(
    [exclude_nan_feats_cb, exclude_nan_feats_label],
    layout=w.Layout(width="430px", align_items="center", margin="0 0 10px 0")
)

# NEW: botón para limpiar todos los filtros
clear_filters_btn = w.Button(
    description="Limpiar filtros",
    icon="trash",
    button_style='warning',
    layout=w.Layout(width="350px")
)

# --- Features + weights side by side ---
features_title = w.HTML("<b>Features & Pesos:</b>")
features_box = w.VBox(layout=w.Layout(width="420px"))
_feature_checks = {}
_feature_weight_boxes = {}  # right-hand cells for weight widgets
_available_features = []

metric_select = w.Dropdown(
    options=[
        ("WASPAS (default)", "waspas"),
        ("Weighted", "weighted"),
        ("TOPSIS", "topsis"),
        ("VIKOR", "vikor"),
    ],
    value="waspas",
    description="Métrica:",
    layout=w.Layout(width="350px"),
)

# --- Dynamic weights ---
weights_title = w.HTML("<b>Pesos por feature (deben sumar 1.0)</b>")
weights_sum_label = w.HTML("Suma: 0.0")
weight_widgets = {}

# Buttons
run_btn = w.Button(description='Calcular ranking', button_style='primary', icon='calculator')
presets_btn = w.Button(description="Evaluar todos los presets", icon="search", button_style='')

run_btn.layout = w.Layout(width="180px")
presets_btn.layout = w.Layout(width="230px")

# Preset UI
preset_status = w.HTML("<i>Presets sin evaluar para estos filtros.</i>")

# preset management widgets
preset_select = w.Dropdown(
    options=[("(sin preset)", None)],
    value=None,
    description="Preset:",
    layout=w.Layout(width="350px")
)

preset_name_text = w.Text(
    value="",
    description="Nombre:",
    placeholder="nuevo_preset",
    layout=w.Layout(width="350px")
)

save_as_new_preset_btn = w.Button(
    description="Guardar como nuevo preset",
    icon="save",
    button_style='success',
    layout=w.Layout(width="350px")
)

save_over_preset_btn = w.Button(
    description="Guardar cambios en preset actual",
    icon="save",
    button_style='',
    layout=w.Layout(width="350px")
)

delete_preset_btn = w.Button(
    description="Eliminar preset actual",
    icon="trash",
    button_style='danger',
    layout=w.Layout(width="350px")
)

restore_preset_btn = w.Button(
    description="Restaurar preset actual desde defaults",
    icon="undo",
    button_style='',
    layout=w.Layout(width="350px")
)

reset_presets_btn = w.Button(
    description="Restaurar presets por defecto",
    icon="refresh",
    button_style='warning',
    layout=w.Layout(width="350px")
)

# --- Section headers and outputs ---
hdr_original = w.HTML("<h3>Vista rápida del dataset original</h3>")

# NEW: controles para la vista rápida
original_filters_cb = w.Checkbox(
    value=False,
    description="Aplicar filtros actuales a la vista",
    indent=False,
    layout=w.Layout(width="260px")
)
original_rows_dd = w.Dropdown(
    options=[
        ("Primeras 10 filas", 10),
        ("Primeras 50 filas", 50),
        ("Primeras 500 filas", 500),
        ("Todas las filas", None),
    ],
    value=10,
    description="Mostrar:",
    layout=w.Layout(width="260px")
)
original_update_btn = w.Button(
    description="Actualizar vista",
    icon="refresh",
    layout=w.Layout(width="180px")
)

original_controls_row = w.HBox(
    [original_filters_cb, original_rows_dd, original_update_btn],
    layout=w.Layout(width="100%", justify_content="flex-start", gap="10px", margin="5px 0 5px 0")
)

out_original = w.Output(
    layout=w.Layout(width="100%")
)

hdr_scores = w.HTML("<h3>Ranking de proveedores</h3>")
out_scores_main = w.Output(layout=w.Layout(width="50%"))
out_scores_presets = w.Output(layout=w.Layout(width="50%"))

out_scores_main.layout.width = "100%"
out_scores_presets.layout.width = "100%"

scores_row = w.VBox(
    [
        out_scores_main,   # Ranking de proveedores (arriba)
        out_scores_presets # Evaluación de todos los presets (abajo)
    ],
    layout=w.Layout(width="100%")
)

section_original = w.VBox(
    [hdr_original, original_controls_row, out_original],
    layout=w.Layout(display='none')
)
section_scores   = w.VBox([hdr_scores,   scores_row],   layout=w.Layout(display=''))

# === Report widgets (ScenarioReporter) ===
hdr_report = w.HTML("<h3>Reportes de escenarios</h3>")

report_output = w.Output(
    layout=w.Layout(
        border="none",
        max_height="400px",
        overflow_y="auto",
        width="100%",
        margin="0 0 10px 0",
    )
)

# Dropdowns para comparación directa de proveedores
providers_a_dd = w.Dropdown(
    options=[("(sin datos)", None)],
    value=None,
    description="Proveedor A:",
    layout=w.Layout(width="260px")
)
providers_b_dd = w.Dropdown(
    options=[("(sin datos)", None)],
    value=None,
    description="Proveedor B:",
    layout=w.Layout(width="260px")
)

report_combined_btn = w.Button(
    description="Reporte combinado: métricas + costos",
    icon="file-text",
    layout=w.Layout(width="280px")
)

report_mcda_btn = w.Button(
    description="Comparar: más elegido vs #1 en simulación",
    icon="search",
    layout=w.Layout(width="260px")
)

report_two_btn = w.Button(
    description="Escenario: proveedor A vs B",
    icon="exchange",
    layout=w.Layout(width="260px")
)

report_provider_row = w.HBox(
    [providers_a_dd, providers_b_dd],
    layout=w.Layout(width="100%", justify_content="flex-start", gap="10px")
)

report_buttons_row = w.HBox(
    [report_combined_btn, report_mcda_btn, report_two_btn],
    layout=w.Layout(width="100%", justify_content="flex-start", gap="10px", margin="5px 0 10px 0")
)

# === Matrix reporter widgets (CSV) ===
matrix_hdr = w.HTML("<h4>Reportes de matrices (CSV)</h4>")

matrix_features_select = w.SelectMultiple(
    options=[],
    value=(),
    description="Features:",
    layout=w.Layout(width="260px", height="130px")
)

matrix_provinces_select = w.SelectMultiple(
    options=[],
    value=(),
    description="Provincias:",
    layout=w.Layout(width="260px", height="130px")
)

matrix_report_btn = w.Button(
    description="Exportar matrices a CSV",
    icon="download",
    layout=w.Layout(width="260px")
)

matrix_report_output = w.Output(
    layout=w.Layout(
        border="none",
        max_height="200px",
        overflow_y="auto",
        width="100%",
        margin="5px 0 0 0"
    )
)

matrix_controls_row = w.HBox(
    [matrix_features_select, matrix_provinces_select, matrix_report_btn],
    layout=w.Layout(width="100%", justify_content="flex-start", gap="10px")
)

# --- Separate containers for scenario reports and matrix reports (to avoid overlap) ---
report_box = w.VBox(
    [
        report_provider_row,
        report_buttons_row,
        report_output,
    ],
    layout=w.Layout(width="100%")
)

matrix_box = w.VBox(
    [
        matrix_hdr,
        matrix_controls_row,
        matrix_report_output,
    ],
    layout=w.Layout(width="100%", margin="10px 0 0 0")
)

section_report = w.VBox(
    [
        hdr_report,
        report_box,
        w.HTML("<hr>"),
        matrix_box,
    ],
    layout=w.Layout(display='none')
)

# Toggles (solo original, ranking, reportes)
toggle_original = w.Checkbox(value=False, description="Mostrar 'Vista rápida del dataset'")
toggle_scores   = w.Checkbox(value=True,  description="Mostrar 'Ranking de proveedores'")
toggle_report   = w.Checkbox(value=False, description="Mostrar 'Reportes de escenarios'")

for t in (toggle_original, toggle_scores, toggle_report):
    t.layout = w.Layout(width="100%")

toggles_grid = w.GridBox(
    children=[toggle_original, toggle_scores, toggle_report],
    layout=w.Layout(
        width="100%",
        grid_template_columns="repeat(3, minmax(260px, 1fr))",
        grid_gap="10px 20px",
    ),
)

try:
    FileDownload = w.FileDownload
    dl_btn = FileDownload(data=b"", filename="mcda_scores.csv",
                          description="Descargar CSV", disabled=True)
except Exception:
    dl_btn = None


# ---------- State ----------
raw_df = None           # CSV principal crudo
cleaned_df = None       # dataset limpio (y mergeado si hay extra)
pre = Preprocessor(normalize_names=True)
fb = FeatureBuilder()
mcda = MCDAEngine()  # already loading defaults from weight_presets.py internally

DATE_MIN = None
DATE_MAX = None

last_run_signature = None
last_presets_signature = None

scenario_reporter: ScenarioReporter | None = None


def _make_filter_signature(
    feats_effective,
    provincia,
    location,
    codigo_postal,
    rango_peso,
    date_from,
    date_to,
):
    """
    Build a hashable signature of the current filters/feature set.
    """
    return (
        tuple(sorted(feats_effective)),
        provincia,
        location,
        codigo_postal,
        rango_peso,
        date_from,
        date_to,
        bool(drop_incomplete_cb.value),
        bool(exclude_nan_feats_cb.value),
        metric_select.value,
    )

# ---------- Helpers ----------

def _rebuild_candidate_presets_and_dropdown():
    candidate_presets.clear()
    candidate_presets.update(_current_presets)

    options = [("(sin preset)", None)] + [(name, name) for name in candidate_presets.keys()]
    preset_select.options = options
    if preset_select.value not in [None] + list(candidate_presets.keys()):
        preset_select.value = None


def show_scrollable(df: pd.DataFrame, height: int = 450):
    """
    Muestra el DataFrame dentro de un div scrollable
    (funciona bien tanto en vertical como horizontal).
    """
    html = df.to_html()
    html = f"""
    <div style="
        max-height:{height}px;
        overflow:auto;
        border:1px solid #ccc;
        padding:5px;
    ">
        {html}
    </div>
    """
    display(HTML(html))


def _safe_display_df(df, head=None, round_decimals=3, height=450):
    """
    Muestra un DataFrame usando show_scrollable.

    - head=None  -> muestra todas las filas
    - head=N     -> muestra las primeras N filas
    """
    if df is None or df.empty:
        display(pd.DataFrame({"Mensaje": ["(vacío)"]}))
        return

    data = df.round(round_decimals)
    if head is not None:
        data = data.head(head)

    show_scrollable(data, height=height)


def _get_selected_features():
    return [f for f in _available_features if _feature_checks.get(f, w.Checkbox()).value]


def _filter_suffix():
    parts = []
    prov = prov_dropdown.value
    if prov and prov not in ("Todas", "(subí un archivo)"):
        parts.append(f"Provincia={prov}")
    cp = cp_text.value.strip()
    if cp:
        parts.append(f"CP={cp}")
    zona = zona_radio.value
    if zona and zona != "both":
        parts.append(f"Zona={zona}")
    rp = rango_dropdown.value
    if (not rango_dropdown.disabled) and rp not in (None, "Todos", "(subí un archivo)", "(no disponible)"):
        parts.append(f"Rango={rp}")
    # Fechas
    if date_from_picker.value is not None:
        parts.append(f"Desde={date_from_picker.value.isoformat()}")
    if date_to_picker.value is not None:
        parts.append(f"Hasta={date_to_picker.value.isoformat()}")

    if not parts:
        return ""
    return " — Filtros: " + ", ".join(parts)


def _update_titles():
    hdr_scores.value  = f"<h3>Ranking de proveedores{_filter_suffix()}</h3>"
    hdr_report.value  = f"<h3>Reportes de escenarios{_filter_suffix()}</h3>"


def _update_weights_sum_label():
    s = sum(ft.value for ft in weight_widgets.values()) if weight_widgets else 0.0
    weights_sum_label.value = f"Suma: {s:.6f}"


def _rebuild_weight_inputs():
    selected = _get_selected_features()

    for box in _feature_weight_boxes.values():
        box.children = []
    weight_widgets.clear()

    if not selected:
        weights_sum_label.value = "Suma: 0.0"
        return

    n = len(selected)
    base = round(1.0 / n, 6)
    vals = [base] * n
    vals[-1] = round(1.0 - sum(vals[:-1]), 6)

    for feat, v in zip(selected, vals):
        ft = w.BoundedFloatText(
            value=v, min=0, max=1, step=0.01,
            description="",
            layout=w.Layout(width="120px")
        )
        ft.observe(lambda *_: _update_weights_sum_label(), names='value')
        weight_widgets[feat] = ft

        weight_box = _feature_weight_boxes.get(feat)
        if weight_box is not None:
            weight_box.children = [ft]

    _update_weights_sum_label()


def _on_feature_checkbox_change(change):
    if change["name"] == "value":
        _rebuild_weight_inputs()


def _build_feature_checkboxes(options, defaults):
    features_box.children = []
    _feature_checks.clear()
    _feature_weight_boxes.clear()
    _available_features.clear()
    _available_features.extend(options)

    rows = []
    for f in options:
        cb = w.Checkbox(
            value=(f in defaults),
            description=f,
            indent=False,
            layout=w.Layout(width="260px")
        )
        cb.observe(_on_feature_checkbox_change, names='value')
        _feature_checks[f] = cb

        weight_cell = w.Box(layout=w.Layout(width="130px"))
        _feature_weight_boxes[f] = weight_cell

        row = w.HBox(
            [cb, weight_cell],
            layout=w.Layout(align_items="center")
        )
        rows.append(row)

    features_box.children = rows
    _rebuild_weight_inputs()


def _update_filters_from_df(df):
    # Provincias
    provs = sorted(df["Provincia"].dropna().unique())
    prov_dropdown.options = ["Todas"] + provs
    prov_dropdown.value = "Todas"
    prov_dropdown.disabled = False

    # Rangos de peso
    if "Rango de Peso" in df.columns:
        rangos = sorted(df["Rango de Peso"].dropna().unique())
        rango_dropdown.options = ["Todos"] + list(rangos)
        rango_dropdown.value = "Todos"
        rango_dropdown.disabled = False
    else:
        rango_dropdown.options = ["(no disponible)"]
        rango_dropdown.value = "(no disponible)"
        rango_dropdown.disabled = True

    # Features disponibles según el FeatureBuilder
    feats_all = list(fb.available_features)
    default_feats = feats_all  # todas por defecto
    _build_feature_checkboxes(feats_all, default_feats)
    _update_titles()

    # Matrix reporter widgets
    matrix_features_select.options = feats_all
    # ✅ default: all selected
    matrix_features_select.value = tuple(feats_all) if feats_all else ()
    matrix_provinces_select.options = ["todas"] + provs
    matrix_provinces_select.value = ()


def _init_date_widgets_from_df(df):
    """
    Inicializa date_from/date_to al mínimo y máximo de la columna de fechas.
    Si no se encuentra la columna o no hay fechas válidas, se deshabilitan.
    """
    global scenario_reporter, DATE_MIN, DATE_MAX

    date_from_picker.disabled = True
    date_to_picker.disabled = True
    date_from_picker.value = None
    date_to_picker.value = None
    DATE_MIN = None
    DATE_MAX = None

    if df is None or df.empty:
        return

    date_col = None

    # 1) Usar el date_col del ScenarioReporter si está disponible
    if scenario_reporter is not None and hasattr(scenario_reporter, "date_col"):
        dc = scenario_reporter.date_col
        if dc in df.columns:
            date_col = dc

    # 2) Candidatos
    if date_col is None:
        for cand in ["Fecha de Despacho", "last_status_date"]:
            if cand in df.columns:
                date_col = cand
                break

    if date_col is None:
        return

    col = df[date_col]

    if np.issubdtype(col.dtype, np.datetime64):
        s = col.dropna()
    else:
        date_format = getattr(scenario_reporter, "date_format", None)
        dayfirst = getattr(scenario_reporter, "dayfirst", True)

        if date_format is not None:
            s = pd.to_datetime(col, format=date_format, errors="coerce")
        else:
            s = pd.to_datetime(col, errors="coerce", dayfirst=dayfirst)

        s = s.dropna()

    if s.empty:
        return

    DATE_MIN = s.min().date()
    DATE_MAX = s.max().date()

    date_from_picker.disabled = False
    date_to_picker.disabled = False
    date_from_picker.value = DATE_MIN
    date_to_picker.value = DATE_MAX


def _on_date_picker_change(change):
    global DATE_MIN, DATE_MAX
    if change["name"] != "value":
        return
    val = change["new"]
    if val is None or DATE_MIN is None or DATE_MAX is None:
        return

    if val < DATE_MIN:
        change["owner"].value = DATE_MIN
    elif val > DATE_MAX:
        change["owner"].value = DATE_MAX


date_from_picker.observe(_on_date_picker_change, names="value")
date_to_picker.observe(_on_date_picker_change, names="value")


def _get_date_bounds():
    """
    Devuelve (date_from, date_to) como objetos date o (None, None) si no se usan.
    """
    if date_from_picker.disabled and date_to_picker.disabled:
        return None, None
    return date_from_picker.value, date_to_picker.value


def _filter_df_by_dates(df):
    """
    Aplica el filtro de fechas al DataFrame que se le pasa (para MCDA y vistas).

    Versión robusta y suave:
      - Si el rango del picker cubre TODO el rango local de fechas válidas,
        no filtra por fecha.
      - No elimina filas con fechas inválidas (NaT): esas filas siempre pasan.
      - Usa dayfirst=True para mantener consistencia con el resto del sistema.
    """
    if df is None or df.empty:
        return df

    dfrom, dto = _get_date_bounds()
    if dfrom is None and dto is None:
        return df

    # Determinar columna de fecha
    date_col = None
    if scenario_reporter is not None and hasattr(scenario_reporter, "date_col"):
        dc = scenario_reporter.date_col
        if dc in df.columns:
            date_col = dc

    if date_col is None:
        for cand in ["Fecha de Despacho"]:
            if cand in df.columns:
                date_col = cand
                break

    if date_col is None:
        return df

    s = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    valid = ~s.isna()

    if not valid.any():
        return df

    s_min = s[valid].min().date()
    s_max = s[valid].max().date()

    if ((dfrom is None or dfrom <= s_min) and
        (dto   is None or dto   >= s_max)):
        return df

    mask = pd.Series(True, index=df.index)

    if dfrom is not None:
        dfrom_ts = pd.to_datetime(dfrom)
        mask &= (~valid) | (s >= dfrom_ts)

    if dto is not None:
        end_dt = pd.to_datetime(dto) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        mask &= (~valid) | (s <= end_dt)

    return df.loc[mask].copy()


def reset_after_upload():
    global last_run_signature, last_presets_signature, DATE_MIN, DATE_MAX

    out_original.clear_output()
    out_scores_main.clear_output()
    out_scores_presets.clear_output()
    report_output.clear_output()
    matrix_report_output.clear_output()
    info_box.clear_output()
    preset_status.value = "<i>Presets sin evaluar para estos filtros.</i>"
    _rebuild_candidate_presets_and_dropdown()
    preset_select.value = None
    preset_name_text.value = ""
    last_run_signature = None
    last_presets_signature = None

    date_from_picker.disabled = True
    date_to_picker.disabled = True
    date_from_picker.value = None
    date_to_picker.value = None
    DATE_MIN = None
    DATE_MAX = None


def _apply_visibility():
    section_original.layout.display = ''  if toggle_original.value else 'none'
    section_scores.layout.display   = ''  if toggle_scores.value   else 'none'
    section_report.layout.display   = ''  if toggle_report.value   else 'none'


def _apply_preset_weights(preset_name: str | None):
    if preset_name is None:
        return
    if preset_name not in candidate_presets:
        return
    feats = _get_selected_features()
    if not feats:
        with info_box:
            print("⚠️ Elegí al menos una feature antes de aplicar un preset.")
        return

    preset = candidate_presets[preset_name]
    raw_vals = [preset.get(f, 0.0) for f in feats]
    arr = np.array(raw_vals, dtype=float)

    if arr.sum() <= 0:
        with info_box:
            print("⚠️ El preset no aplica a las features seleccionadas (no hay pesos > 0).")
        return

    arr = arr / arr.sum()

    for f, v in zip(feats, arr):
        if f in weight_widgets:
            weight_widgets[f].value = round(float(v), 6)

    _update_weights_sum_label()


def _report_text_to_html(text: str) -> str:
    """
    Make the ScenarioReporter text look nicer.
    """
    esc = (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    esc = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc)

    lines = esc.split("\n")
    html_parts = []
    in_list = False

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped == "":
            html_parts.append("<br>")
            continue

        if stripped.startswith("==="):
            title = stripped.strip("= ").strip()
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(
                f"<h4 style='margin:10px 0 4px 0; "
                f"font-size:14px; font-weight:600; "
                f"color:#333; border-bottom:1px solid #e0e0e0;'>{title}</h4>"
            )
            continue

        if stripped.startswith("- "):
            content = stripped[2:].strip()
            if not in_list:
                html_parts.append("<ul style='margin:4px 0 4px 18px; padding-left:14px;'>")
                in_list = True
            html_parts.append(f"<li style='margin-bottom:2px;'>{content}</li>")
            continue

        if in_list:
            html_parts.append("</ul>")
            in_list = False

        html_parts.append(
            f"<p style='margin:2px 0 2px 0; font-size:13px;'>{line}</p>"
        )

    if in_list:
        html_parts.append("</ul>")

    inner_html = "\n".join(html_parts)

    return f"""
    <div style="
        font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size:13px;
        line-height:1.4;
        background:linear-gradient(135deg, #fafafa, #f4f7fb);
        border-radius:8px;
        border:1px solid #dde3ef;
        padding:10px 12px;
        box-shadow:0 1px 3px rgba(0,0,0,0.06);
        color:#222;
    ">
        {inner_html}
    </div>
    """


# helper to update original dataset view (cleaned_df)
def _update_original_view():
    out_original.clear_output()

    if cleaned_df is None or cleaned_df.empty:
        with out_original:
            print("No hay datos cargados todavía.")
        return

    df = cleaned_df
    total_rows = len(cleaned_df)

    if original_filters_cb.value:
        provincia = None if prov_dropdown.value in ("Todas", "(subí un archivo)") else prov_dropdown.value
        location = zona_radio.value
        cp_val = cp_text.value.strip()

        if provincia is not None and "Provincia" in df.columns:
            df = df[df["Provincia"] == provincia]

        if location == "capital" and "Capital/Interior" in df.columns:
            df = df[df["Capital/Interior"] == "CIUDAD"]
        elif location == "interior" and "Capital/Interior" in df.columns:
            df = df[df["Capital/Interior"] == "INTERIOR"]

        if cp_val and "Codigo Postal" in df.columns:
            df = df[df["Codigo Postal"] == int(cp_val)]

        if (
            not rango_dropdown.disabled
            and rango_dropdown.value not in ("Todos", "(subí un archivo)", "(no disponible)")
            and "Rango de Peso" in df.columns
        ):
            df = df[df["Rango de Peso"] == rango_dropdown.value]

        df = _filter_df_by_dates(df)

    filtered_rows = len(df)

    n = original_rows_dd.value
    if isinstance(n, int):
        df_to_show = df.head(n)
    else:
        df_to_show = df

    shown_rows = len(df_to_show)

    with out_original:
        if original_filters_cb.value:
            msg = (
                f"Filas totales (dataset limpio): <b>{total_rows}</b><br>"
                f"Filas después de aplicar filtros: <b>{filtered_rows}</b><br>"
            )
        else:
            msg = (
                f"Filas totales (dataset limpio): <b>{total_rows}</b><br>"
                f"Filas en la vista (sin filtros adicionales): <b>{filtered_rows}</b><br>"
            )

        if shown_rows < filtered_rows:
            msg += f"Mostrando <b>{shown_rows}</b> de <b>{filtered_rows}</b> filas."
        else:
            msg += f"Mostrando todas las <b>{shown_rows}</b> filas."

        display(HTML(f"<p>{msg}</p>"))
        _safe_display_df(df_to_show)


# ---------- Dataset load core ----------

def _finalize_loaded_dataset(raw: pd.DataFrame, cleaned: pd.DataFrame, source_label: str):
    """
    Centraliza la inicialización post-carga:
      - set raw_df/cleaned_df
      - init ScenarioReporter
      - init filtros/features/fechas
      - init vista rápida
      - init dropdowns A/B
      - actualiza label de fuente
    """
    global raw_df, cleaned_df, scenario_reporter

    if cleaned is None or cleaned.empty:
        with info_box:
            print("❌ Después de la limpieza/merge no quedaron filas.")
        return

    raw_df = raw
    cleaned_df = cleaned

    scenario_reporter = ScenarioReporter(
        df=cleaned_df,
        feature_builder=fb,
        mcda_engine=mcda,
        date_col="Fecha de Despacho"
    )

    _update_filters_from_df(cleaned_df)
    _init_date_widgets_from_df(cleaned_df)
    _update_original_view()

    shorts = list(getattr(scenario_reporter, "proveedores_short", []))
    if shorts:
        providers_a_dd.options = shorts
        providers_b_dd.options = shorts
        providers_a_dd.value = shorts[0]
        providers_b_dd.value = shorts[0]
    else:
        providers_a_dd.options = [("(sin datos)", None)]
        providers_b_dd.options = [("(sin datos)", None)]
        providers_a_dd.value = None
        providers_b_dd.value = None

    data_source_status.value = f"<small>Fuente actual: <b>{source_label}</b></small>"


def load_data_from_paths():
    """
    Carga MAIN_CSV_PATH (y opcionalmente EXTRA_CSV_PATH),
    limpia con Preprocessor y deja raw_df / cleaned_df listos.
    """
    reset_after_upload()

    try:
        if not os.path.exists(MAIN_CSV_PATH):
            with info_box:
                print(f"❌ MAIN_CSV_PATH no existe: {MAIN_CSV_PATH}")
            return

        raw = pd.read_csv(MAIN_CSV_PATH)
        if raw is None or raw.empty:
            with info_box:
                print("❌ El CSV principal está vacío o no se pudo leer.")
            return

        if EXTRA_CSV_PATH and os.path.exists(EXTRA_CSV_PATH):
            cleaned = pre.clean_with_extra_dates(
                raw,
                extra_csv_path=EXTRA_CSV_PATH,
                key_main="Order ID",
                key_extra="id",
            )
        else:
            cleaned = pre.clean_with_extra_dates(
                raw,
                extra_csv_path=None,
            )

        with info_box:
            print(
                f"Fuente: dataset por defecto\n"
                f"Filas originales (principal): {len(raw)} | "
                f"Filas después de limpiar/mergear: {len(cleaned)}"
            )

        _finalize_loaded_dataset(raw, cleaned, f"por defecto ({MAIN_CSV_PATH})")

    except Exception as e:
        with info_box:
            print(f"❌ Error leyendo/limpiando/mergeando los CSV: {e}")


def load_data_from_uploaded_bytes(file_bytes: bytes, filename: str = "(subido)"):
    """
    Lee CSV subido (bytes), lo limpia y lo deja como dataset activo.
    """
    reset_after_upload()

    try:
        raw = pd.read_csv(io.BytesIO(file_bytes))
        if raw is None or raw.empty:
            with info_box:
                print("❌ El CSV subido está vacío o no se pudo leer.")
            return

        # Nota: no usamos EXTRA_CSV_PATH cuando subís CSV
        cleaned = pre.clean_with_extra_dates(
            raw,
            extra_csv_path=None,
        )

        with info_box:
            print(
                f"Fuente: CSV subido ({filename})\n"
                f"Filas originales: {len(raw)} | "
                f"Filas después de limpiar/mergear: {len(cleaned)}"
            )

        _finalize_loaded_dataset(raw, cleaned, f"CSV subido ({filename})")

    except Exception as e:
        with info_box:
            print(f"❌ Error leyendo/limpiando el CSV subido: {e}")


# ---------- Robust FileUpload extraction (works across ipywidgets versions) ----------

def _extract_uploaded_file(upload_value):
    """
    Returns (filename, file_bytes) from FileUpload.value for both:
      - dict style: {"file.csv": {"content": b"...", "metadata": {...}}}
      - tuple/list style: ({"name":"file.csv","content":b"...", ...},)
    """
    if not upload_value:
        return None, None

    # tuple/list style
    if isinstance(upload_value, (list, tuple)):
        item = upload_value[0] if len(upload_value) else None
        if isinstance(item, dict):
            name = item.get("name") or item.get("metadata", {}).get("name") or "(subido)"
            content = item.get("content", b"")
            return name, content

    # dict style
    if isinstance(upload_value, dict):
        first_key = next(iter(upload_value.keys()), None)
        if first_key is None:
            return None, None
        item = upload_value[first_key]
        if isinstance(item, dict):
            content = item.get("content", b"")
            name = item.get("metadata", {}).get("name") or first_key or "(subido)"
            return name, content

    return None, None


# ---------- Events ----------

def on_any_filter_change(_):
    _update_titles()
    preset_status.value = "<i>Presets sin evaluar para estos filtros.</i>"

for wdg in (prov_dropdown, zona_radio, cp_text, rango_dropdown,
            date_from_picker, date_to_picker):
    wdg.observe(on_any_filter_change, names='value')


def on_toggle_change(_):
    _apply_visibility()

for tg in (toggle_original, toggle_scores, toggle_report):
    tg.observe(on_toggle_change, names='value')


def on_preset_change(change):
    if change["name"] != "value":
        return
    new = change["new"]
    info_box.clear_output()
    _apply_preset_weights(new)
    if new is not None:
        preset_name_text.value = new

preset_select.observe(on_preset_change, names='value')


# ---------- Colab upload handler (local-safe) ----------

def on_colab_upload_clicked(_):
    info_box.clear_output()
    colab_upload_out.clear_output()

    if not _is_colab():
        with info_box:
            print("⚠️ Este botón es solo para Colab.")
        return

    try:
        from google.colab import files  # type: ignore
    except Exception as e:
        with info_box:
            print(f"❌ No pude importar google.colab.files: {e}")
        return

    with colab_upload_out:
        print("📥 Elegí un CSV para subir...")

    uploaded = files.upload()
    if not uploaded:
        with colab_upload_out:
            print("ℹ️ No se subió ningún archivo.")
        return

    filename, content = next(iter(uploaded.items()))
    if not content:
        with colab_upload_out:
            print("❌ El archivo subido está vacío.")
        return

    # ✅ Ensure datasets folder exists (relative to your PROJECT_DIR / cwd)
    datasets_dir = os.path.join(os.getcwd(), "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    # ✅ Save the uploaded file into datasets
    save_path = os.path.join(datasets_dir, filename)
    with open(save_path, "wb") as f:
        f.write(content)

    # ✅ Remove the runtime temp copy created by Colab (in /content)
    runtime_copy = os.path.join("/content", filename)
    if os.path.exists(runtime_copy):
        os.remove(runtime_copy)

    with colab_upload_out:
        print(f"✅ Subido: {filename} (bytes={len(content)})")
        print(f"📁 Guardado en: {save_path}")
        print("🔄 Cargando dataset...")

    # ✅ Key fix: run loader while cwd=datasets to avoid extra copy in colab/
    prev_cwd = os.getcwd()
    try:
        os.chdir(datasets_dir)
        load_data_from_uploaded_bytes(content, filename=filename)
    finally:
        os.chdir(prev_cwd)

colab_upload_btn.on_click(on_colab_upload_clicked)




# ========= RANKING HANDLER =========

def on_run_clicked(_):
    global last_run_signature, last_presets_signature

    info_box.clear_output()
    out_scores_main.clear_output()
    if dl_btn:
        dl_btn.disabled = True
        dl_btn.data = b""

    if cleaned_df is None or cleaned_df.empty:
        with info_box:
            print("❌ Cargá un dataset válido (por defecto o subido) antes de calcular.")
        return

    provincia = None if prov_dropdown.value in ("Todas", "(subí un archivo)") else prov_dropdown.value
    location = zona_radio.value
    cp_val = cp_text.value.strip()
    if cp_val == "":
        codigo_postal = None
    else:
        try:
            codigo_postal = int(cp_val)
        except ValueError:
            with info_box:
                print("❌ El código postal debe ser numérico.")
            return

    rango_peso = None
    if not rango_dropdown.disabled and rango_dropdown.value not in ("Todos", "(subí un archivo)", "(no disponible)"):
        rango_peso = rango_dropdown.value

    dfrom, dto = _get_date_bounds()

    feats = _get_selected_features()
    if not feats:
        with info_box:
            print("❌ Tenés que elegir al menos una feature.")
        return

    methods = [metric_select.value] if metric_select.value is not None else []
    if not methods:
        with info_box:
            print("❌ Tenés que elegir al menos una métrica.")
        return

    # filtro fechas antes del FeatureBuilder
    df_for_mcda = _filter_df_by_dates(cleaned_df)

    try:
        df_metrics = fb.build(
            df_for_mcda,
            features=feats,
            provincia=provincia,
            location=location,
            codigo_postal=codigo_postal,
            rango_peso=rango_peso,
            drop_incomplete=drop_incomplete_cb.value,
            drop_features_with_missing=exclude_nan_feats_cb.value,
        )
    except Exception as e:
        with info_box:
            print(f"❌ Error calculando features: {e}")
        return

    if df_metrics.empty:
        with info_box:
            print("❌ No hay datos para ese filtro.")
        return

    _update_titles()

    feats_effective = [f for f in feats if f in df_metrics.columns]
    if not feats_effective:
        with info_box:
            print("❌ Ninguna de las features seleccionadas quedó disponible después de aplicar filtros/drop.")
        return

    current_sig = _make_filter_signature(
        feats_effective=feats_effective,
        provincia=provincia,
        location=location,
        codigo_postal=codigo_postal,
        rango_peso=rango_peso,
        date_from=dfrom.isoformat() if dfrom else None,
        date_to=dto.isoformat() if dto else None,
    )

    if (last_presets_signature is not None) and (last_presets_signature != current_sig):
        out_scores_presets.clear_output()
        preset_status.value = "<i>Presets sin evaluar para estos filtros.</i>"

    last_run_signature = current_sig

    if set(feats_effective) != set(feats):
        dropped = sorted(set(feats) - set(feats_effective))
        with info_box:
            print(
                "ℹ️ Se excluyeron estas features porque no están en df_metrics "
                f"(posiblemente por NaN completos o filtros): {', '.join(dropped)}"
            )

    weights_list = [float(weight_widgets[f].value) for f in feats_effective]
    sum_w = float(sum(weights_list))
    if not np.isfinite(sum_w) or sum_w <= 0:
        with info_box:
            print("❌ Los pesos deben ser válidos.")
        return

    if abs(sum_w - 1.0) > 1e-6:
        w_norm = (np.array(weights_list, dtype=float) / sum_w).tolist()
        for f, v in zip(feats_effective, w_norm):
            weight_widgets[f].value = round(float(v), 6)
        _update_weights_sum_label()
        with info_box:
            print(f"⚠️ Normalicé los pesos (suma original: {sum_w:.6f})")
        weights = np.array([weight_widgets[f].value for f in feats_effective], dtype=float)
    else:
        weights = np.array(weights_list, dtype=float)

    with info_box:
        print("📊 Pesos efectivos por feature:")
        for f, w_val in zip(feats_effective, weights):
            print(f"   - {f}: {float(w_val):.4f}")

    try:
        avg_flag = len(methods) > 1
        scores_df = mcda.score(
            df_metrics,
            methods=methods,
            weights=weights,
            criteria_types=None,
            return_df=True,
            sort=True,
            avg=avg_flag,
            features=feats_effective,
            feature_builder=fb,
        )
    except Exception as e:
        with info_box:
            print(f"❌ Error en MCDAEngine.score: {e}")
        return

    with out_scores_main:
        _safe_display_df(scores_df, head=20, round_decimals=4)

    if dl_btn and not scores_df.empty:
        dl_btn.data = scores_df.to_csv(index=True).encode("utf-8")
        dl_btn.disabled = False


run_btn.on_click(on_run_clicked)


# ========= PRESETS EVALUATION HANDLER =========

def on_presets_clicked(_):
    global last_run_signature, last_presets_signature

    info_box.clear_output()
    out_scores_presets.clear_output()

    if cleaned_df is None or cleaned_df.empty:
        with info_box:
            print("❌ Cargá un dataset válido antes de evaluar presets.")
        return

    provincia = None if prov_dropdown.value in ("Todas", "(subí un archivo)") else prov_dropdown.value
    location = zona_radio.value
    cp_val = cp_text.value.strip()
    if cp_val == "":
        codigo_postal = None
    else:
        try:
            codigo_postal = int(cp_val)
        except ValueError:
            with info_box:
                print("❌ El código postal debe ser numérico.")
            return

    rango_peso = None
    if not rango_dropdown.disabled and rango_dropdown.value not in ("Todos", "(subí un archivo)", "(no disponible)"):
        rango_peso = rango_dropdown.value

    dfrom, dto = _get_date_bounds()

    feats = _get_selected_features()
    if not feats:
        with info_box:
            print("❌ Tenés que elegir al menos una feature antes de evaluar presets.")
        return

    metric_method = metric_select.value or "waspas"

    df_for_mcda = _filter_df_by_dates(cleaned_df)

    try:
        df_metrics = fb.build(
            df_for_mcda,
            features=feats,
            provincia=provincia,
            location=location,
            codigo_postal=codigo_postal,
            rango_peso=rango_peso,
            drop_incomplete=drop_incomplete_cb.value,
            drop_features_with_missing=exclude_nan_feats_cb.value,
            cache_global=True,
        )
    except Exception as e:
        with info_box:
            print(f"❌ Error calculando features para presets: {e}")
        return

    if df_metrics.empty:
        with info_box:
            print("❌ No hay datos para ese filtro al evaluar presets.")
        return

    feats_effective = [f for f in feats if f in df_metrics.columns]
    if not feats_effective:
        with info_box:
            print("❌ Ninguna de las features seleccionadas quedó disponible para evaluar presets.")
        return

    current_sig = _make_filter_signature(
        feats_effective=feats_effective,
        provincia=provincia,
        location=location,
        codigo_postal=codigo_postal,
        rango_peso=rango_peso,
        date_from=dfrom.isoformat() if dfrom else None,
        date_to=dto.isoformat() if dto else None,
    )

    if (last_run_signature is not None) and (last_run_signature != current_sig):
        out_scores_main.clear_output()
        if dl_btn:
            dl_btn.disabled = True
            dl_btn.data = b""

    last_presets_signature = current_sig

    if set(feats_effective) != set(feats):
        dropped = sorted(set(feats) - set(feats_effective))
        with info_box:
            print(
                "ℹ️ Al evaluar presets se excluyeron features que no están en df_metrics: "
                f"{', '.join(dropped)}"
            )

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
            with info_box:
                print(f"❌ Error evaluando preset '{preset_name}': {e}")
            continue

        if scores_df.empty:
            continue

        winner = scores_df.index[0]
        winners.append({"preset": preset_name, "winner": winner})

    if not winners:
        with info_box:
            print("❌ Ningún preset se pudo evaluar con las features seleccionadas.")
        return

    winners_series = pd.Series([w_["winner"] for w_ in winners], name="Proveedor")
    summary_prov = winners_series.value_counts().reset_index()
    summary_prov.columns = ["Proveedor", "Veces_ganador"]
    summary_prov = summary_prov.sort_values("Veces_ganador", ascending=False)

    summary_preset = pd.DataFrame(winners)
    summary_preset = summary_preset.rename(columns={"preset": "Preset", "winner": "Proveedor_ganador"})

    with out_scores_presets:
        print("🏆 Proveedores más robustos a través de todos los presets:")
        display(summary_prov.style.hide(axis='index'))

        print("\n📋 Ganador por preset:")
        display(summary_preset.style.hide(axis='index'))

    preset_status.value = "<i>Presets evaluados para estos filtros (ver tablas de ganadores).</i>"


presets_btn.on_click(on_presets_clicked)


# ========= PRESET MANAGEMENT HANDLERS =========

def on_save_as_new_preset_clicked(_):
    info_box.clear_output()
    name = preset_name_text.value.strip()
    if not name:
        with info_box:
            print("❌ Escribí un nombre para el nuevo preset.")
        return

    feats = _get_selected_features()
    if not feats:
        with info_box:
            print("❌ Tenés que elegir al menos una feature para guardar un preset.")
        return

    weights = {}
    for f in feats:
        if f in weight_widgets:
            weights[f] = float(weight_widgets[f].value)

    if not weights:
        with info_box:
            print("❌ No hay pesos para guardar.")
        return

    _current_presets[name] = weights
    _save_current_presets(_current_presets)
    _rebuild_candidate_presets_and_dropdown()
    preset_select.value = name

    with info_box:
        print(f"✅ Preset '{name}' guardado/actualizado.")


def on_save_over_preset_clicked(_):
    info_box.clear_output()
    current = preset_select.value
    if current is None:
        with info_box:
            print("❌ Seleccioná un preset en la lista para sobrescribirlo.")
        return

    feats = _get_selected_features()
    if not feats:
        with info_box:
            print("❌ Tenés que elegir al menos una feature para guardar un preset.")
        return

    weights = {}
    for f in feats:
        if f in weight_widgets:
            weights[f] = float(weight_widgets[f].value)

    if not weights:
        with info_box:
            print("❌ No hay pesos para guardar.")
        return

    _current_presets[current] = weights
    _save_current_presets(_current_presets)
    _rebuild_candidate_presets_and_dropdown()
    preset_select.value = current

    with info_box:
        print(f"✅ Preset '{current}' sobrescrito.")


def on_delete_preset_clicked(_):
    info_box.clear_output()
    current = preset_select.value
    if current is None:
        with info_box:
            print("❌ Seleccioná un preset para eliminar.")
        return

    if current in _current_presets:
        del _current_presets[current]
        _save_current_presets(_current_presets)
        _rebuild_candidate_presets_and_dropdown()
        preset_select.value = None
        preset_name_text.value = ""
        with info_box:
            print(f"🗑️ Preset '{current}' eliminado de la configuración actual.")
    else:
        with info_box:
            print("⚠️ Ese preset ya no existe en la configuración actual.")


def on_reset_presets_clicked(_):
    info_box.clear_output()
    _current_presets.clear()
    for name, preset in _default_presets.items():
        _current_presets[name] = preset.copy()
    _save_current_presets(_current_presets)
    _rebuild_candidate_presets_and_dropdown()
    preset_select.value = None
    preset_name_text.value = ""
    with info_box:
        print("🔄 Presets restaurados a los valores por defecto (lista completa).")


def on_restore_preset_clicked(_):
    info_box.clear_output()
    current = preset_select.value
    if current is None:
        with info_box:
            print("❌ Seleccioná un preset para restaurar desde defaults.")
        return

    if current not in _default_presets:
        with info_box:
            print("⚠️ No existe una versión por defecto para este preset (es solo de usuario).")
        return

    _current_presets[current] = _default_presets[current].copy()
    _save_current_presets(_current_presets)
    _rebuild_candidate_presets_and_dropdown()
    preset_select.value = current
    _apply_preset_weights(current)

    with info_box:
        print(f"↩️ Preset '{current}' restaurado a su versión por defecto.")


save_as_new_preset_btn.on_click(on_save_as_new_preset_clicked)
save_over_preset_btn.on_click(on_save_over_preset_clicked)
delete_preset_btn.on_click(on_delete_preset_clicked)
reset_presets_btn.on_click(on_reset_presets_clicked)
restore_preset_btn.on_click(on_restore_preset_clicked)


# ========= REPORT HANDLERS (ScenarioReporter) =========

def _get_current_filters_for_report():
    if cleaned_df is None or cleaned_df.empty:
        return None

    provincia = None if prov_dropdown.value in ("Todas", "(subí un archivo)") else prov_dropdown.value
    location = zona_radio.value
    cp_val = cp_text.value.strip()
    if cp_val == "":
        codigo_postal = None
    else:
        try:
            codigo_postal = int(cp_val)
        except ValueError:
            with info_box:
                print("❌ El código postal debe ser numérico.")
            return None

    rango_peso = None
    if not rango_dropdown.disabled and rango_dropdown.value not in ("Todos", "(subí un archivo)", "(no disponible)"):
        rango_peso = rango_dropdown.value

    dfrom, dto = _get_date_bounds()

    return {
        "provincia": provincia,
        "location": location,
        "codigo_postal": codigo_postal,
        "rango_peso": rango_peso,
        "date_from": dfrom.isoformat() if dfrom else None,
        "date_to": dto.isoformat() if dto else None,
    }


def on_report_combined_clicked(_):
    report_output.clear_output()

    if scenario_reporter is None:
        with report_output:
            print("❌ No hay datos cargados todavía (ScenarioReporter no está inicializado).")
        return

    filters = _get_current_filters_for_report()
    if filters is None:
        return

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
    except Exception as e:
        with report_output:
            print(f"❌ Error generando el reporte combinado: {e}")
        return

    html = _report_text_to_html(text)
    with report_output:
        display(HTML(html))


def on_report_mcda_clicked(_):
    report_output.clear_output()

    if scenario_reporter is None:
        with report_output:
            print("❌ No hay datos cargados todavía (ScenarioReporter no está inicializado).")
        return

    if cleaned_df is None or cleaned_df.empty:
        with report_output:
            print("❌ Cargá un dataset válido antes de generar el escenario.")
        return

    filters = _get_current_filters_for_report()
    if filters is None:
        return

    feats = _get_selected_features()
    if not feats:
        with report_output:
            print("❌ Tenés que elegir al menos una feature para el escenario MCDA.")
        return

    raw_weights = []
    for f in feats:
        if f not in weight_widgets:
            with report_output:
                print(f"❌ Falta widget de peso para la feature '{f}'.")
            return
        raw_weights.append(float(weight_widgets[f].value))

    sum_w = float(sum(raw_weights))
    if not np.isfinite(sum_w) or sum_w <= 0:
        with report_output:
            print("❌ Los pesos actuales no son válidos (suma <= 0).")
        return

    weights_arr = np.array(raw_weights, dtype=float) / sum_w

    methods = [metric_select.value] if metric_select.value is not None else ["waspas"]

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
            features=feats,
            weights_preset=None,
            baseline_provider=None,
        )

        text = scenario_reporter.pretty_cost_report(
            stats,
            title_prefix="Escenario: proveedor histórico vs mejor rankeado",
        )
    except Exception as e:
        with report_output:
            print(f"❌ Error generando el escenario: {e}")
        return

    html = _report_text_to_html(text)
    with report_output:
        display(HTML(html))


def on_report_two_clicked(_):
    report_output.clear_output()

    if scenario_reporter is None:
        with report_output:
            print("❌ No hay datos cargados todavía (ScenarioReporter no está inicializado).")
        return

    filters = _get_current_filters_for_report()
    if filters is None:
        return

    prov_a = providers_a_dd.value
    prov_b = providers_b_dd.value

    if not prov_a or not prov_b:
        with report_output:
            print("❌ Elegí ambos proveedores A y B.")
        return

    if prov_a == prov_b:
        with report_output:
            print("❌ Elegí dos proveedores distintos para comparar.")
        return

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

        title_txt = f"Escenario: {prov_a} (base) vs {prov_b} (alternativo)"
        text = scenario_reporter.pretty_cost_report(
            stats,
            title_prefix=title_txt,
        )
    except Exception as e:
        with report_output:
            print(f"❌ Error comparando proveedores '{prov_a}' y '{prov_b}': {e}")
        return

    html = _report_text_to_html(text)
    with report_output:
        display(HTML(html))


report_combined_btn.on_click(on_report_combined_clicked)
report_mcda_btn.on_click(on_report_mcda_clicked)
report_two_btn.on_click(on_report_two_clicked)


# ========= MATRIX REPORTER HANDLER =========

def on_matrix_report_clicked(_):
    """
    Ejecuta el export de matrices CP x Rango de Peso usando ScenarioReporter.

    - Provincias: se toman de matrix_provinces_select.
        * Si no hay nada seleccionado -> None => todas las provincias.
    - Features: se toman de matrix_features_select.
        * Si no hay nada seleccionado -> None => todas las features disponibles
          (pero por default dejamos todas seleccionadas al cargar el dataset).

    Los archivos se guardan en la carpeta 'report_matrices' del cwd.
    """
    matrix_report_output.clear_output()

    if scenario_reporter is None:
        with matrix_report_output:
            print("❌ No hay datos cargados todavía (ScenarioReporter no está inicializado).")
        return

    selected_provs = list(matrix_provinces_select.value) if matrix_provinces_select.value else []

    # Si el usuario elige "Todas", ignoramos el filtro de provincias
    if not selected_provs or "todas" in selected_provs:
        provincias_arg = None
    else:
        provincias_arg = selected_provs

    selected_feats = list(matrix_features_select.value) if matrix_features_select.value else []
    features_arg = selected_feats or None

    try:
        # export_feature_matrices_for_provincias devuelve:
        # { provincia: [ruta_csv_1, ruta_csv_2, ...], ... }
        result = scenario_reporter.export_feature_matrices_for_provincias(
            provincias=provincias_arg,
            features=features_arg,
            out_dir="report_matrices",
        )
    except AttributeError:
        with matrix_report_output:
            print(
                "❌ ScenarioReporter no tiene el método "
                "'export_feature_matrices_for_provincias'.\n"
                "Asegurate de haber actualizado scenario_reporter.py "
                "a la versión con el matrix reporter."
            )
        return
    except Exception as e:
        with matrix_report_output:
            print(f"❌ Error exportando matrices: {e}")
        return

    with matrix_report_output:
        if not result:
            print("ℹ️ No se generó ningún archivo (quizás no había datos suficientes).")
            return

        total_files = sum(len(paths) for paths in result.values())

        if provincias_arg is None:
            prov_msg = "todas las provincias del dataset"
        else:
            prov_msg = ", ".join(str(p) for p in provincias_arg)

        if features_arg is None:
            feat_msg = "todas las features disponibles del FeatureBuilder"
        else:
            feat_msg = ", ".join(str(f) for f in features_arg)

        print("✅ Exportación de matrices completada (CSV).")
        print(f"   Provincias usadas: {prov_msg}")
        print(f"   Features usadas: {feat_msg}")
        print("   Carpeta destino: report_matrices")
        print(f"   Archivos generados: {total_files}\n")

        for prov, paths in result.items():
            if not paths:
                continue
            print(f"Provincia: {prov}")
            for p in paths:
                print(f"  • {p}")
            print("")


matrix_report_btn.on_click(on_matrix_report_clicked)


# ---------- Clear filters handler ----------

def on_clear_filters_clicked(_):
    """Limpiar TODOS los filtros a sus valores por defecto razonables."""
    info_box.clear_output()

    if "Todas" in prov_dropdown.options:
        prov_dropdown.value = "Todas"

    zona_radio.value = "both"
    cp_text.value = ""

    if not rango_dropdown.disabled and "Todos" in rango_dropdown.options:
        rango_dropdown.value = "Todos"

    if DATE_MIN is not None and DATE_MAX is not None:
        date_from_picker.disabled = False
        date_to_picker.disabled = False
        date_from_picker.value = DATE_MIN
        date_to_picker.value = DATE_MAX
    else:
        date_from_picker.value = None
        date_to_picker.value = None

    drop_incomplete_cb.value = True
    exclude_nan_feats_cb.value = False

    _update_titles()
    preset_status.value = "<i>Presets sin evaluar para estos filtros.</i>"


clear_filters_btn.on_click(on_clear_filters_clicked)


# ---------- Original view update handler ----------

def on_original_update_clicked(_):
    _update_original_view()

original_update_btn.on_click(on_original_update_clicked)


# ---------- Dataset source mode UI ----------

def _apply_data_source_mode_ui():
    colab = _is_colab()

    if data_source_mode.value == "default":
        # hide all upload UIs
        upload_widget.layout.display = "none"
        load_uploaded_btn.layout.display = "none"
        colab_upload_btn.layout.display = "none"
        colab_upload_out.layout.display = "none"

        # show default loader
        load_default_btn.layout.display = ""
    else:
        # hide default loader
        load_default_btn.layout.display = "none"

        if colab:
            # Colab: use the reliable uploader
            upload_widget.layout.display = "none"
            load_uploaded_btn.layout.display = "none"

            colab_upload_btn.layout.display = ""
            colab_upload_out.layout.display = ""
        else:
            # Local/Jupyter: use ipywidgets upload
            upload_widget.layout.display = ""
            load_uploaded_btn.layout.display = ""
            colab_upload_btn.layout.display = "none"
            colab_upload_out.layout.display = "none"


def on_data_source_mode_change(change):
    if change["name"] != "value":
        return
    _apply_data_source_mode_ui()

data_source_mode.observe(on_data_source_mode_change, names="value")


def on_load_default_clicked(_):
    load_data_from_paths()

load_default_btn.on_click(on_load_default_clicked)


def on_upload_widget_change(change):
    if change["name"] != "value":
        return
    name, content = _extract_uploaded_file(upload_widget.value)
    load_uploaded_btn.disabled = not bool(content)

upload_widget.observe(on_upload_widget_change, names="value")


def on_load_uploaded_clicked(_):
    info_box.clear_output()

    name, content = _extract_uploaded_file(upload_widget.value)

    with info_box:
        if not content:
            print("❌ No pude leer el contenido del archivo subido.")
            print("   Tip: probá subir de nuevo el CSV.")
            print("   Tipo upload_widget.value:", type(upload_widget.value))
            return
        print(f"📥 Cargando CSV subido: {name} (bytes={len(content)})")

    load_data_from_uploaded_bytes(content, filename=name)

load_uploaded_btn.on_click(on_load_uploaded_clicked)


# ---------- Layout ----------
buttons_row = w.HBox(
    [run_btn, presets_btn],
    layout=w.Layout(width="100%", justify_content="space-between", margin="10px 0 0 0")
)

preset_buttons_box = w.VBox([
    save_as_new_preset_btn,
    save_over_preset_btn,
    delete_preset_btn,
    restore_preset_btn,
    reset_presets_btn,
], layout=w.Layout(width="350px"))

dataset_box = w.VBox(
    [
        data_source_title,
        data_source_mode,
        data_source_status,
        load_default_btn,
        # Local uploader
        upload_widget,
        load_uploaded_btn,
        # Colab uploader
        colab_upload_btn,
        colab_upload_out,
    ],
    layout=w.Layout(width="430px")
)

controls_left = w.VBox([
    dataset_box,
    w.HTML("<hr>"), w.HTML("<b>Filtros</b>"),
    prov_dropdown,
    zona_radio,
    cp_text,
    rango_dropdown,
    date_row,
    drop_incomplete_row,
    exclude_nan_feats_row,
    clear_filters_btn,
], layout=w.Layout(width="430px"))

controls_right = w.VBox([
    features_title, features_box,
    weights_title, weights_sum_label,
    metric_select,
    w.HTML("<hr>"),
    w.HTML("<b>Presets</b>"),
    preset_status,
    preset_select,
    preset_name_text,
    preset_buttons_box,
    buttons_row,
    dl_btn or w.Box(layout=w.Layout(display="none"))
], layout=w.Layout(width="460px"))

ui = w.VBox([
    title, info_box,
    w.HBox([controls_left, w.Box(layout=w.Layout(width="30px")), controls_right]),
    w.HTML("<hr>"),
    toggles_grid,
    section_original,
    section_scores,
    section_report,
])


def launch_shipping_recommender_ui(auto_load: bool = True) -> w.VBox:
    """
    Construye y lanza la interfaz del MCDA Shipping Recommender en un notebook.

    Parámetros
    ----------
    auto_load : bool, por defecto True
        Si es True, carga automáticamente el dataset por defecto.

    Devuelve
    --------
    ui : ipywidgets.VBox
        El widget raíz para que puedas guardarlo en una variable si querés.
    """
    _rebuild_candidate_presets_and_dropdown()
    _apply_visibility()
    _apply_data_source_mode_ui()

    if auto_load:
        data_source_mode.value = "default"
        load_data_from_paths()

    display(ui)
    return ui


__all__ = [
    "launch_shipping_recommender_ui",
    "load_data_from_paths",
    "load_data_from_uploaded_bytes",
    "MAIN_CSV_PATH",
    "EXTRA_CSV_PATH",
    "ui",
]
