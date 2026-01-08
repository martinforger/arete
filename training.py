from preprocessing import preprocess_and_balance

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import numpy as np
import os

def entrenar_modelo(df_balanceado):
    # 1. Preparar las etiquetas (Convertir texto a números: 0, 1, 2)
    label2id = {"Areté": 0, "Política y poder": 1, "Relación entre dioses y hombres": 2}
    id2label = {v: k for k, v in label2id.items()}
    
    df_balanceado['label'] = df_balanceado['etiqueta'].map(label2id)
    
    # 2. Convertir a formato Dataset de Hugging Face
    dataset = Dataset.from_pandas(df_balanceado[['cita_limpia', 'label']])
    dataset = dataset.train_test_split(test_size=0.2)

    # 3. Cargar el Tokenizador y el Modelo
    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["cita_limpia"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 4. Configurar el modelo para 3 categorías
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

    # 5. Argumentos de entrenamiento (optimizados para mejor F1-Score)
    training_args = TrainingArguments(
        output_dir="./resultados",
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=15,                    
        learning_rate=2e-5,                     
        per_device_train_batch_size=8,          
        per_device_eval_batch_size=8,
        warmup_ratio=0.1,                       
        weight_decay=0.01,                      
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",      
        logging_steps=10,                       
    )

    # 6. Iniciar el entrenador
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    print("--- Iniciando Entrenamiento ---")
    trainer.train()
    
    # Guardar el modelo para usarlo en la aplicación web después
    model.save_pretrained("./modelo_final")
    tokenizer.save_pretrained("./modelo_final")
    print("¡Modelo guardado con éxito en './modelo_final'!")

if __name__ == "__main__":
    # Llamamos a la función de preprocessing.py pasándole el nombre de la BD
    print("📦 Paso 1: Obteniendo datos balanceados de la base de datos...")
    datos_listos = preprocess_and_balance("./data/corpus_clasico.db")
    
    # Le pasamos esos datos a la función de entrenamiento
    print("🧠 Paso 2: Entrenando la red neuronal...")
    entrenar_modelo(datos_listos)