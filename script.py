from elasticsearch import Elasticsearch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Leer CSV
df = pd.read_csv("fake_news_dataset.csv")

df.columns 
# Convertir columnas con tipos complejos a texto plano
for col in ['title', 'text', 'date', 'source', 'author', 'category', 'label']:
    if col in df.columns:
        df[col] = df[col].astype(str)

# Reemplazar NaNs
df = df.fillna("")

# Conectarse a Elasticsearch (si está corriendo en Docker sin SSL)
es = Elasticsearch("http://elasticsearch:9200", verify_certs=False)

# Crear índice si no existe
index_name = "fake_news"
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name)

# Indexar documentos
for _, row in df.iterrows():
    try:
        doc = row.to_dict()
        es.index(index=index_name, document=doc)
    except Exception as e:
        print(f"Error al indexar documento: {e}")

# Graficar: promedio de votos por idioma original
summary = df.groupby("label")["author"].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=summary.values, y=summary.index)
plt.xlabel("Promedio de Voto")
plt.ylabel("Idioma Original")
plt.title("Top 10 idiomas con mayor promedio de voto")
plt.tight_layout()
plt.savefig("grafica.png")