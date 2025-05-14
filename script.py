import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configurar estilo
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

print("Cargando el dataset...")
# Cargar el dataset
data = pd.read_csv('fake_news_dataset.csv')

# Mostrar información básica
print("Shape del dataset:", data.shape)
print("\nPrimeras 5 filas:")
print(data.head())

# Preprocesamiento básico
print("\nRealizando preprocesamiento básico...")
# Convertir label a formato numérico
data['label_num'] = data['label'].map({'real': 1, 'fake': 0})

# Agregar longitud de título y texto como características
data['title_length'] = data['title'].fillna('').apply(len)
data['text_length'] = data['text'].fillna('').apply(len)

# Procesamiento de categorías
# Tomar solo las top 10 categorías
top_categories = data['category'].value_counts().head(10).index.tolist()
data['category_filtered'] = data['category'].apply(lambda x: x if x in top_categories else 'Other')

# Calcular métricas por categoría
category_metrics = data.groupby('category_filtered').agg({
    'label_num': ['mean', 'count'],
    'text_length': 'mean',
    'title_length': 'mean'
}).reset_index()

category_metrics.columns = ['category', 'fake_ratio', 'count', 'avg_text_length', 'avg_title_length']
# Convertir el ratio de fake a porcentaje de noticias reales
category_metrics['real_ratio'] = 1 - category_metrics['fake_ratio']
category_metrics['fake_ratio'] = category_metrics['fake_ratio'] * 100
category_metrics['real_ratio'] = category_metrics['real_ratio'] * 100

# Ordenar por conteo
category_metrics = category_metrics.sort_values('count', ascending=False)

print("\nGenerando gráfica interactiva...")

# Crear una gráfica interactiva con Plotly
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "bar"}, {"type": "pie"}],
           [{"type": "scatter", "colspan": 2}, None]],
    subplot_titles=(
        "Distribución de Noticias por Categoría",
        "Proporción de Noticias Reales vs Falsas",
        "Relación entre Longitud del Texto, Título y Veracidad por Categoría"
    ),
    vertical_spacing=0.1,
    horizontal_spacing=0.05
)

# 1. Gráfico de barras para la distribución de noticias por categoría
colors = px.colors.qualitative.Plotly[:len(category_metrics)]
fig.add_trace(
    go.Bar(
        x=category_metrics['category'],
        y=category_metrics['count'],
        marker_color=colors,
        text=category_metrics['count'],
        textposition='auto',
        hoverinfo='text',
        hovertext=[
            f'Categoría: {cat}<br>'
            f'Cantidad: {count}<br>'
            f'Noticias reales: {real:.1f}%<br>'
            f'Noticias falsas: {fake:.1f}%'
            for cat, count, real, fake in 
            zip(category_metrics['category'], category_metrics['count'], 
                category_metrics['real_ratio'], category_metrics['fake_ratio'])
        ],
        name='Cantidad'
    ),
    row=1, col=1
)

# 2. Gráfico de pastel para la proporción total de noticias reales vs falsas
labels = ['Noticias Reales', 'Noticias Falsas']
values = [
    data[data['label'] == 'real'].shape[0],
    data[data['label'] == 'fake'].shape[0]
]
fig.add_trace(
    go.Pie(
        labels=labels,
        values=values,
        marker_colors=['#3D9970', '#FF4136'],
        textinfo='percent+label',
        hole=0.4,
        hoverinfo='label+value+percent',
    ),
    row=1, col=2
)

# 3. Gráfico de dispersión para la relación entre longitud del texto, título y veracidad
for i, category in enumerate(category_metrics['category']):
    subset = data[data['category_filtered'] == category]
    
    # Crear un gráfico para noticias reales
    fig.add_trace(
        go.Scatter(
            x=subset[subset['label'] == 'real']['text_length'],
            y=subset[subset['label'] == 'real']['title_length'],
            mode='markers',
            marker=dict(
                color=colors[i],
                size=10,
                opacity=0.7,
                symbol='circle'
            ),
            name=f"{category} (Real)",
            legendgroup=category,
            showlegend=True,
            hoverinfo='text',
            hovertext=[
                f'Categoría: {category}<br>'
                f'Longitud del texto: {text_len}<br>'
                f'Longitud del título: {title_len}<br>'
                f'Tipo: Real'
                for text_len, title_len in 
                zip(subset[subset['label'] == 'real']['text_length'], 
                    subset[subset['label'] == 'real']['title_length'])
            ]
        ),
        row=2, col=1
    )
    
    # Crear un gráfico para noticias falsas
    fig.add_trace(
        go.Scatter(
            x=subset[subset['label'] == 'fake']['text_length'],
            y=subset[subset['label'] == 'fake']['title_length'],
            mode='markers',
            marker=dict(
                color=colors[i],
                size=10,
                opacity=0.7,
                symbol='x'
            ),
            name=f"{category} (Fake)",
            legendgroup=category,
            showlegend=True,
            hoverinfo='text',
            hovertext=[
                f'Categoría: {category}<br>'
                f'Longitud del texto: {text_len}<br>'
                f'Longitud del título: {title_len}<br>'
                f'Tipo: Fake'
                for text_len, title_len in 
                zip(subset[subset['label'] == 'fake']['text_length'], 
                    subset[subset['label'] == 'fake']['title_length'])
            ]
        ),
        row=2, col=1
    )

# Actualizar diseño de la gráfica
fig.update_layout(
    title_text="Análisis Completo de Fake News Dataset",
    height=1000,
    width=1200,
    legend_title="Categorías y Tipos",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
)

# Actualizar etiquetas de ejes
fig.update_xaxes(title_text="Categoría", row=1, col=1)
fig.update_yaxes(title_text="Cantidad de Noticias", row=1, col=1)

fig.update_xaxes(title_text="Longitud del Texto", row=2, col=1)
fig.update_yaxes(title_text="Longitud del Título", row=2, col=1)

# Ajustar límites de ejes para el gráfico de dispersión
fig.update_xaxes(range=[0, data['text_length'].quantile(0.98)], row=2, col=1)
fig.update_yaxes(range=[0, data['title_length'].quantile(0.98)], row=2, col=1)

# Guardar como un archivo HTML independiente
fig.write_html("fake_news_dashboard.html")

# Crear un archivo HTML para GitHub Pages
with open("index.html", "w") as f:
    f.write("""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Análisis de Fake News</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            header {
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                text-align: center;
                border-radius: 5px 5px 0 0;
            }
            h1 {
                margin: 0;
                font-size: 2.5em;
            }
            .subtitle {
                font-style: italic;
                margin-top: 5px;
            }
            .dashboard-container {
                background-color: white;
                border-radius: 0 0 5px 5px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-bottom: 20px;
            }
            .dashboard-description {
                margin-bottom: 20px;
                text-align: justify;
            }
            iframe {
                width: 100%;
                height: 1000px;
                border: none;
                overflow: hidden;
            }
            footer {
                text-align: center;
                margin-top: 20px;
                font-size: 0.9em;
                color: #666;
            }
            .key-insights {
                background-color: #f8f9fa;
                border-left: 4px solid #2c3e50;
                padding: 15px;
                margin: 20px 0;
            }
            .key-insights h3 {
                margin-top: 0;
                color: #2c3e50;
            }
            .key-insights ul {
                padding-left: 20px;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>Análisis de Fake News</h1>
            <p class="subtitle">Visualización interactiva del dataset de noticias falsas</p>
        </header>
        
        <div class="dashboard-container">
            <div class="dashboard-description">
                <p>Este dashboard presenta un análisis completo del dataset de fake news, mostrando la distribución de noticias por categoría, la proporción de noticias reales vs falsas, y la relación entre la longitud del texto, título y veracidad por categoría.</p>
                
                <div class="key-insights">
                    <h3>Principales hallazgos:</h3>
                    <ul>
                        <li>Distribución de categorías: Muestra cuántas noticias hay en cada categoría temática.</li>
                        <li>Proporción de noticias reales vs falsas: Visualiza el balance general entre noticias verdaderas y falsas.</li>
                        <li>Relación texto-título: Explora cómo la longitud del texto y del título puede relacionarse con la veracidad de la noticia.</li>
                        <li>Patrones por categoría: Identifica características específicas de las noticias falsas en diferentes categorías.</li>
                    </ul>
                </div>
            </div>
            
            <iframe src="fake_news_dashboard.html" title="Fake News Dashboard"></iframe>
        </div>
        
        <footer>
            <p>© 2025 Análisis de Fake News. Desarrollado con Plotly y Python.</p>
        </footer>
    </body>
    </html>
    """)

print("\nArchivos generados con éxito:")
print("1. fake_news_dashboard.html - Dashboard interactivo")
print("2. index.html - Página principal para GitHub Pages")
print("\nPuedes subir estos archivos a GitHub para visualizarlos en GitHub Pages.")