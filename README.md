# 📜 Areté - Clasificador Semántico de Textos Clásicos

Sistema de Inteligencia Artificial para la clasificación semántica de textos clásicos mediante Fine-Tuning de modelos de lenguaje (BERT). Desarrollado para investigadores en humanidades, con interfaz gráfica web e integración de base de datos SQL.

## 🤗 Modelo en Hugging Face

El modelo entrenado está disponible públicamente en Hugging Face Hub:

🔗 **[martinforger/arete](https://huggingface.co/martinforger/arete)**

La aplicación web descarga automáticamente el modelo desde la nube, por lo que no necesitas tener los archivos del modelo localmente.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Cargar el modelo desde Hugging Face
tokenizer = AutoTokenizer.from_pretrained("martinforger/arete")
model = AutoModelForSequenceClassification.from_pretrained("martinforger/arete")
```

## 🎯 Categorías de Clasificación

El modelo clasifica fragmentos de textos clásicos en tres ejes temáticos:

| Categoría | Descripción |
|-----------|-------------|
| **Areté** | Excelencia, virtud y desarrollo del carácter |
| **Política y poder** | Estructuras de gobierno, poder y organización social |
| **Relación entre dioses y hombres** | Interacción entre lo divino y lo humano |

## 📁 Estructura del Proyecto

```
arete/
├── data/
│   ├── datos.xlsx              # Datos originales en Excel
│   └── corpus_clasico.db       # Base de datos SQLite
├── migration.py                # Migración Excel → SQLite
├── preprocessing.py            # Limpieza y balanceo de datos
├── training.py                 # Entrenamiento del modelo BERT
├── evaluation.py               # Evaluación y métricas
└── app.py                      # Interfaz web con Streamlit
```

> **Nota:** El modelo entrenado se almacena en Hugging Face Hub, no en este repositorio.

## ⚙️ Requisitos

- Python 3.10 o superior

### Instalación
```bash
pip install -r requirements.txt
```

## 🚀 Guía de Ejecución

### 1. Ejecutar la aplicación web (rápido)

Si solo quieres usar la aplicación, ejecuta directamente:

```bash
streamlit run app.py
```

El modelo se descargará automáticamente desde Hugging Face Hub la primera vez.

Abre tu navegador en `http://localhost:8501` para usar la interfaz.

---

### 2. Re-entrenar el modelo (opcional)

Si deseas entrenar el modelo desde cero:

#### 2.1 Migrar datos (solo primera vez)

```bash
python migration.py
```

#### 2.2 Entrenar el modelo

```bash
python training.py
```

- Preprocesamiento y balanceo de datos
- Entrenamiento del modelo BERT (15 épocas, ~10 minutos)
- Modelo guardado en `./modelo_final/`

#### 2.3 Evaluar el modelo

```bash
python evaluation.py
```

## 📊 Métricas del Modelo

| Categoría | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Areté | 0.98 | 0.98 | 0.98 |
| Política y poder | 0.98 | 0.98 | 0.98 |
| Relación entre dioses y hombres | 0.95 | 0.95 | 0.95 |
| **Promedio ponderado** | **0.97** | **0.97** | **0.97** |

## 🔧 Configuración del Entrenamiento

```python
TrainingArguments(
    num_train_epochs=15,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    warmup_ratio=0.1,
    weight_decay=0.01,
)
```

## 🖥️ Uso de la Interfaz Web

1. Ejecuta `streamlit run app.py`
2. Pega un fragmento de texto clásico en el área de texto
3. Haz clic en "Analizar Fragmento"
4. Obtén la categoría predicha con su nivel de confianza
