import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# 1. LA FUNCIÓN OBLIGATORIA DEL PREPROCESADOR (No la quites, tu modelo la usa)
def arreglar_pdays(X):
    X_nuevo = X.copy()
    if 'pdays' in X_nuevo.columns:
        X_nuevo['pdays_contacted'] = (X_nuevo['pdays'] != -1).astype(int)
        X_nuevo['pdays'] = X_nuevo['pdays'].replace(-1, np.nan)
    return X_nuevo

st.set_page_config(page_title="Despliegue del Modelo", page_icon="🚀", layout="centered")
st.title("Predicción con Nuevos Datos")
st.write("Esta aplicación utiliza un `Pipeline` completo de `scikit-learn` cargado desde un archivo `joblib`.")

@st.cache_resource
def load_pack():
    # 2. Cargamos TU modelo real directamente
    modelo_real = load("modelo_final.joblib")
    
    # 3. Si el modelo se guardó como lista, extraemos el pipeline
    pipeline = modelo_real[0] if isinstance(modelo_real, list) else modelo_real
    
    # 4. Creamos los metadatos para que el código del profe pinte el formulario
    feature_metadata = {
        "age": {"type": "numerical", "min": 18, "max": 100, "median": 30},
        "balance": {"type": "numerical", "min": -10000, "max": 100000, "median": 1000},
        "duration": {"type": "numerical", "min": 0, "max": 5000, "median": 150},
        "day": {"type": "numerical", "min": 1, "max": 31, "median": 15},
        "campaign": {"type": "numerical", "min": 1, "max": 50, "median": 1},
        "pdays": {"type": "numerical", "min": -1, "max": 1000, "median": -1},
        "previous": {"type": "numerical", "min": 0, "max": 50, "median": 0},
        "job": {"type": "categorical", "options": ["admin.", "blue-collar", "technician", "management", "services", "retired", "student", "unemployed", "self-employed", "entrepreneur", "housemaid", "unknown"]},
        "marital": {"type": "categorical", "options": ["divorced", "married", "single", "unknown"]},
        "education": {"type": "categorical", "options": ["primary", "secondary", "tertiary", "unknown"]},
        "default": {"type": "categorical", "options": ["no", "yes", "unknown"]},
        "housing": {"type": "categorical", "options": ["no", "yes", "unknown"]},
        "loan": {"type": "categorical", "options": ["no", "yes", "unknown"]},
        "contact": {"type": "categorical", "options": ["cellular", "telephone", "unknown"]},
        "month": {"type": "categorical", "options": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]},
        "poutcome": {"type": "categorical", "options": ["failure", "nonexistent", "success", "unknown", "other"]}
    }
    
    # 5. Devolvemos el diccionario 
    return {
        "pipeline": pipeline,
        "feature_metadata": feature_metadata,
        "classes_": pipeline.classes_ if hasattr(pipeline, "classes_") else ["No", "Sí"]
    }

try:
    pack = load_pack()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

pipeline = pack["pipeline"]
feature_metadata = pack["feature_metadata"]
classes_ = pack.get("classes_", [])

st.markdown("### Introduce los valores de las variables:")

with st.form("prediction_form"):
    inputs = {}
    
    st.subheader("Variables Numéricas")
    num_cols = st.columns(2)
    num_idx = 0
    
    for feat, meta in feature_metadata.items():
        if meta["type"] == "numerical":
            with num_cols[num_idx % 2]:
                med = float(meta.get("median", 0.0))
                inputs[feat] = st.number_input(
                    label=feat,
                    min_value=float(meta.get("min", -1e9)),
                    max_value=float(meta.get("max", 1e9)),
                    value=med,
                    step=1.0 if med.is_integer() else 0.1
                )
            num_idx += 1
            
    st.subheader("Variables Categóricas")
    cat_cols = st.columns(2)
    cat_idx = 0
    
    for feat, meta in feature_metadata.items():
        if meta["type"] == "categorical":
            with cat_cols[cat_idx % 2]:
                opts = meta.get("options", [])
                inputs[feat] = st.selectbox(
                    label=feat,
                    options=opts,
                    index=0 if opts else None
                )
            cat_idx += 1
            
    st.markdown("---")
    submitted = st.form_submit_button("Predecir", use_container_width=True)

if submitted:
    # Convertimos a DataFrame para predecir
    X_new = pd.DataFrame([inputs])
    
    try:
        y_pred = pipeline.predict(X_new)[0]
        st.success(f"### Predicción: **{y_pred}**")
        
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X_new)[0]
            st.markdown("#### Probabilidades de la Predicción:")
            cols = st.columns(len(classes_))
            for i, (cls, p) in enumerate(zip(classes_, proba)):
                cols[i].metric(label=f"Clase {cls}", value=f"{p*100:.1f}%")
            
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")