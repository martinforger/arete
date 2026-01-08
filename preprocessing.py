import sqlite3
import pandas as pd
import re
from imblearn.over_sampling import RandomOverSampler

def preprocess_and_balance(db_name):
    # Nos conectamos a la BD en sqlite
    conn = sqlite3.connect(db_name)
    query = "SELECT cita, etiqueta FROM textos_clasicos"
    df = pd.read_sql_query(query, conn)
    conn.close()

    print("--- Distribucion Original")
    print(df['etiqueta'].value_counts())

    # Limpieza de texto
    def limpiar_texto(texto):
        texto = texto.lower()
        texto = re.sub(r'\s+', ' ', texto).strip()
        return texto

    df['cita_limpia'] = df['cita'].apply(limpiar_texto)

    # Random Oversampling

    ros = RandomOverSampler(random_state=42)

    X = df[['cita_limpia']]
    y = df['etiqueta']

    X_resampled, y_resampled = ros.fit_resample(X, y)

    # nuevo df con los datos balanceados
    df_balanceado = pd.DataFrame(X_resampled, columns=['cita_limpia'])
    df_balanceado['etiqueta'] = y_resampled

    print("\n--- Distribucion despues del balanceo ---")
    print(df_balanceado['etiqueta'].value_counts())

    return df_balanceado

if __name__ == "__main__":
    df_ready = preprocess_and_balance("./data/corpus_clasico.db")