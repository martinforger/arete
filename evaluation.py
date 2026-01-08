import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocessing import preprocess_and_balance

def evaluar_modelo(nombre_bd, ruta_modelo):
    # 1. Cargar datos y preparar el set de prueba
    df = preprocess_and_balance(nombre_bd)
    
    # Usamos la misma lógica de etiquetas
    label2id = {"Areté": 0, "Política y poder": 1, "Relación entre dioses y hombres": 2}
    id2label = {v: k for k, v in label2id.items()}
    
    # 2. Cargar el modelo entrenado y el tokenizador
    print("📂 Cargando modelo desde:", ruta_modelo)
    tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)
    model = AutoModelForSequenceClassification.from_pretrained(ruta_modelo)
    model.eval() # Modo evaluación

    # 3. Realizar predicciones
    citas = df['cita_limpia'].tolist()
    etiquetas_reales = df['etiqueta'].map(label2id).tolist()
    predicciones = []

    print("🧪 Evaluando citas...")
    for cita in citas:
        inputs = tokenizer(cita, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=1).item()
        predicciones.append(pred_id)

    # 4. Generar Métricas
    print("\n--- INFORME DE CLASIFICACIÓN ---")
    reporte = classification_report(etiquetas_reales, predicciones, target_names=list(label2id.keys()))
    print(reporte)

    f1 = f1_score(etiquetas_reales, predicciones, average='weighted')
    if f1 >= 0.8:
        print(f"✅ ¡Objetivo cumplido! F1-Score: {f1:.2f}")
    else:
        print(f"⚠️ El F1-Score es {f1:.2f}. Podrías necesitar más épocas de entrenamiento.")

    # 5. Visualizar Matriz de Confusión
    cm = confusion_matrix(etiquetas_reales, predicciones)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label2id.keys(), yticklabels=label2id.keys(), cmap='Blues')
    plt.xlabel('Predicción del Modelo')
    plt.ylabel('Etiqueta Real (Experto)')
    plt.title('Matriz de Confusión: Clasificación de Textos Clásicos')
    plt.show()

if __name__ == "__main__":
    evaluar_modelo("./data/corpus_clasico.db", "./modelo_final")