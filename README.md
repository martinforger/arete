# 📜 Areté - Clasificador Semántico de Textos Clásicos

Sistema de Inteligencia Artificial para la clasificación semántica de textos clásicos mediante Fine-Tuning de modelos de lenguaje (BERT). Desarrollado para investigadores en humanidades, con interfaz gráfica web e integración de base de datos SQL.

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
├── modelo_final/               # Modelo entrenado (generado)
├── resultados/                 # Checkpoints de entrenamiento
├── migration.py                # Migración Excel → SQLite
├── preprocessing.py            # Limpieza y balanceo de datos
├── training.py                 # Entrenamiento del modelo BERT
├── evaluation.py               # Evaluación y métricas
└── app.py                      # Interfaz web con Streamlit
```

## ⚙️ Requisitos

### Python
- Python 3.10 o superior

### Dependencias
```bash
pip install torch transformers datasets pandas numpy scikit-learn imbalanced-learn matplotlib seaborn streamlit sqlalchemy openpyxl
```

O crear un archivo `requirements.txt`:
```
torch
transformers
datasets
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
streamlit
sqlalchemy
openpyxl
```

E instalar con:
```bash
pip install -r requirements.txt
```

## 🚀 Guía de Ejecución

### 1. Migrar datos (solo primera vez)

Si tienes los datos en Excel (`data/datos.xlsx`), migra a SQLite:

```bash
python migration.py
```

Esto creará `data/corpus_clasico.db` con la tabla `textos_clasicos`.

### 2. Entrenar el modelo

```bash
python training.py
```

**Salida esperada:**
- Preprocesamiento y balanceo de datos
- Entrenamiento del modelo BERT (15 épocas, ~10 minutos)
- Modelo guardado en `./modelo_final/`

### 3. Evaluar el modelo

```bash
python evaluation.py
```

**Salida esperada:**
- Informe de clasificación (precision, recall, F1-score)
- Matriz de confusión visual

### 4. Ejecutar la aplicación web

```bash
streamlit run app.py
```

Abre tu navegador en `http://localhost:8501` para usar la interfaz.

## 📊 Métricas del Modelo

El modelo actual alcanza las siguientes métricas:

| Categoría | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Areté | 0.98 | 0.98 | 0.98 |
| Política y poder | 0.98 | 0.98 | 0.98 |
| Relación entre dioses y hombres | 0.95 | 0.95 | 0.95 |
| **Promedio ponderado** | **0.97** | **0.97** | **0.97** |

## 🔧 Configuración del Entrenamiento

Los hiperparámetros actuales en `training.py`:

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
