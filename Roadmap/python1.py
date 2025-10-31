import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys

# Configuración para mejorar visualización de errores
pd.set_option('display.max_columns', None)

# Manejo adecuado de errores para la carga del archivo
try:
    # Prueba primero con separador punto y coma (común en archivos europeos)
    df = pd.read_csv("fake_news_dataset.csv", sep=";", encoding="latin1")
    # Si no hay error, informa del éxito
    print("✅ Archivo cargado correctamente con separador ';'")
except Exception as e1:
    try:
        # Si falla, intenta con separador coma (más común internacionalmente)
        df = pd.read_csv("fake_news_dataset.csv", sep=",", encoding="latin1")
        print("✅ Archivo cargado correctamente con separador ','")
    except Exception as e2:
        # Si ambos intentos fallan, muestra un error informativo y finaliza
        print(f"❌ Error al cargar el archivo: {str(e2)}")
        sys.exit(1)

# Limpieza de nombres de columnas
df.columns = df.columns.str.strip()
print(f"📊 Columnas disponibles: {', '.join(df.columns.tolist())}")

# Verifica si existen las columnas necesarias
required_columns = ['author', 'title']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"❌ Faltan columnas requeridas: {', '.join(missing_columns)}")
    sys.exit(1)

# Manejo adecuado de la columna de etiquetas
label_column = None
if 'label' in df.columns:
    label_column = 'label'
    print("✅ Columna 'label' encontrada.")
elif 'type' in df.columns:
    label_column = 'type'
    print("⚠️ Columna 'label' no encontrada. Usando 'type' como columna de etiquetas.")
else:
    print("❌ No se encontró columna para clasificar noticias ('label' o 'type').")
    sys.exit(1)

# Verificar valores únicos en la columna de etiquetas
unique_labels = df[label_column].unique()
print(f"📋 Valores únicos en columna '{label_column}': {', '.join(map(str, unique_labels))}")

# Manejar posibles valores vacíos o nulos en autor
df['author'] = df['author'].fillna('Unknown')

# Filtrar noticias falsas (considerar variaciones en mayúsculas/minúsculas)
fake_labels = ['fake', 'FAKE', 'Fake', 'FALSE', 'false', 'False']
fake_news_df = df[df[label_column].isin(fake_labels)]

if fake_news_df.empty:
    print(f"❌ No se encontraron noticias con etiquetas de falsedad en {fake_labels}")
    sys.exit(1)

print(f"📊 Se encontraron {len(fake_news_df)} noticias falsas para analizar.")

# Agrupar por autor y contar títulos
author_counts = fake_news_df.groupby('author')['title'].count().reset_index()
author_counts.rename(columns={'title': 'fake_news_count'}, inplace=True)
author_counts = author_counts.sort_values(by='fake_news_count', ascending=False)

# Validar que hay suficientes autores para mostrar
max_authors = min(10, len(author_counts))
if max_authors < 10:
    print(f"⚠️ Solo se encontraron {max_authors} autores con noticias falsas.")

top_authors = author_counts.head(max_authors)

# Crear directorio para resultados si no existe
os.makedirs("docs", exist_ok=True)

# Guardar datos en JSON
try:
    json_data = top_authors.to_json(orient="records", indent=4)
    
    with open("docs/datos_fake_news.json", "w", encoding="utf-8") as json_file:
        json_file.write(json_data)
    
    print("✅ Archivo JSON generado exitosamente: docs/datos_fake_news.json")
except Exception as e:
    print(f"❌ Error al guardar el archivo JSON: {str(e)}")

# Configurar visualización
try:
    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 6))
    
    # Usar una paleta de colores más profesional
    colors = sns.color_palette("coolwarm", max_authors)
    
    # Crear la gráfica
    bars = plt.barh(top_authors["author"], top_authors["fake_news_count"], 
                    color=colors, edgecolor="black", linewidth=0.8)
    
    # Añadir etiquetas con los valores en las barras
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.3, 
                 bar.get_y() + bar.get_height()/2, 
                 str(top_authors["fake_news_count"].iloc[i]),
                 va='center')
    
    # Mejorar la presentación
    plt.xlabel("Número de noticias falsas publicadas", fontsize=12, fontweight="bold")
    plt.ylabel("Autor", fontsize=12, fontweight="bold")
    plt.title("Autores con mayor cantidad de noticias falsas", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.gca().invert_yaxis()  # Invertir para que el mayor valor aparezca arriba
    
    # Guardar la imagen con alta calidad
    plt.savefig("docs/grafico_fake_news.png", dpi=300, bbox_inches="tight")
    print("✅ Gráfico guardado correctamente en docs/grafico_fake_news.png")
    
    # Mostrar el gráfico si se ejecuta en un entorno interactivo
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"❌ Error al generar el gráfico: {str(e)}")

print("✅ Análisis completado correctamente.")