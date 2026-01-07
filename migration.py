import pandas as pd
from sqlalchemy import create_engine

def migrate_excel_to_sqlite(excel_file, sqlite_db):

    try: 
        # Lectura del archivo donde unifique tooodos los demas en
        # una sola tabla
        print(f" --- Lectura del archivo {excel_file} ---")
        df = pd.read_excel(excel_file, sheet_name='Informacion')

        # Limpieza
        df.columns = [col.strip() for col in df.columns]

        # Conexion con sqlite
        engine = create_engine(f'sqlite:///{sqlite_db}')

        # Pasar los archivos a la base de datos
        df.to_sql('textos_clasicos', con=engine, if_exists='replace', index=False)

        print(f'Se han migrado {len(df)} registros a la tabla \'textos clasicos\'.')
    except Exception as e:
        print(f'Ocurrio un error {e}')

# Ejecucion
if __name__ == "__main__":
    archivo_origen = "./data/datos.xlsx"
    base_de_datos = "./data/corpus_clasico.db"

    migrate_excel_to_sqlite(archivo_origen, base_de_datos)