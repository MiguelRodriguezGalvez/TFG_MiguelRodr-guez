import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Definir parámetros
NUM_SERIES = 10000  # Número de series
LONGITUD_SERIE = 200  # Longitud de las series
PROBABILIDAD_ESCALON = 0.5  # Probabilidad de tener un escalón por serie
PROBABILIDAD_ANOMALIA = 0.5  # Probabilidad de tener un valor anómalo por serie
MAGNITUD_ANOMALIA = 200  # Magnitud del valor anómalo
K_FACTOR = 0.1  # Factor para controlar la desviación estándar del ruido

# Fijar la semilla para reproducibilidad
np.random.seed(42)

# Función para calcular la energía promedio de una serie
def calcular_energia_promedio(serie):
    return np.mean(serie ** 2)

# Función para aplicar escalones a una serie
def aplicar_escalones(serie):
    if np.random.rand() < PROBABILIDAD_ESCALON:
        indice_escalón = np.random.randint(1, len(serie) - 5)  # Evitar índices cercanos al final
        duracion_escalón = np.random.randint(1, 30)  # Duración del escalón de 1 a 5 muestras
        escalon = np.random.randint(25, 55) * np.random.choice([-1, 1])  # Aumento de la magnitud del escalón
        serie[indice_escalón:indice_escalón + duracion_escalón] += escalon
        serie_info = "Escalon"  # Indicar que la serie tiene un escalón
    else:
        serie_info = "NoEscalon"  # Indicar que la serie no tiene un escalón
    return serie, serie_info

# Función para aplicar anomalías a una serie
def aplicar_anomalia(serie):
    if np.random.rand() < PROBABILIDAD_ANOMALIA:
        indice_anomalia = np.random.randint(1, len(serie) - 5)  # Evitar índices cercanos al final
        valor_anomalia = np.random.randint(MAGNITUD_ANOMALIA, 2 * MAGNITUD_ANOMALIA) * np.random.choice([-1, 1])
        serie[indice_anomalia] += valor_anomalia
        serie_info = "Anomalo"  # Indicar que la serie tiene un valor anómalo
    else:
        serie_info = "NoAnomalo"  # Indicar que la serie no tiene un valor anómalo
    return serie, serie_info

# Función para generar una serie aleatoria
def generar_serie_aleatoria(longitud):
    tipo = np.random.choice(['ascendente', 'descendente', 'estable', 'cambiante'])
    ruido = np.random.normal(0, 0.5, size=longitud)
    if tipo == 'ascendente':
        pendiente = np.random.randint(1, 5)  # Pendiente aleatoria positiva
        serie = np.cumsum(np.random.randint(1, 5, size=longitud)) * pendiente + ruido
    elif tipo == 'descendente':
        pendiente = np.random.randint(1, 5)  # Pendiente aleatoria negativa
        serie = np.cumsum(np.random.randint(1, 5, size=longitud))[::-1] * pendiente + ruido
    elif tipo == 'estable':
        serie = np.full(longitud, np.random.randint(10)) + ruido
    elif tipo == 'cambiante':
        cambios = np.random.choice([-1, 0, 1], size=longitud)
        serie = np.cumsum(cambios) + ruido

    # Aplicar un escalón aleatorio
    serie, escalon_info = aplicar_escalones(serie)
    
    # Aplicar una anomalía aleatoria
    serie, anomalia_info = aplicar_anomalia(serie)
    
    serie_info = f"{tipo.capitalize()}_{escalon_info}_{anomalia_info}"  # Concatenar la información de la serie
    
    return serie, serie_info

# Generar series aleatorias sintéticas con VAs introducidos artificialmente
series_originales = [generar_serie_aleatoria(LONGITUD_SERIE) for _ in range(NUM_SERIES)]

# Calcular la energía promedio de cada serie
energias_promedio = [calcular_energia_promedio(serie) for serie, _ in series_originales]

# Construir el autoencoder para la comparación
input_dim = LONGITUD_SERIE
encoding_dim = 10
input_layer = Input(shape=(input_dim,))
encoder_layer = Dense(encoding_dim, activation='relu')(input_layer)
decoder_layer = Dense(input_dim, activation='sigmoid')(encoder_layer)
autoencoder = Model(input_layer, decoder_layer)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Normalizar datos
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(np.array([serie for serie, _ in series_originales]))

# Aplicar ruido proporcional a la energía promedio
noisy_data = np.array([serie + np.random.normal(0, K_FACTOR * np.sqrt(energia_promedio), size=len(serie)) for (serie, _), energia_promedio in zip(series_originales, energias_promedio)])

# Entrenar el autoencoder
autoencoder.fit(normalized_data, normalized_data, epochs=50, batch_size=16, shuffle=True, verbose=2)

# Reconstruir las series originales usando el autoencoder
reconstructed_data = autoencoder.predict(normalized_data)

# Desnormalizar los datos reconstruidos
denormalized_data = scaler.inverse_transform(reconstructed_data)

# Crear DataFrames para datos originales y reconstruidos
df_originales = pd.DataFrame({f"{idx+1}_{tipo}": serie for idx, (serie, tipo) in enumerate(series_originales)})
df_reconstruidos = pd.DataFrame({f"{idx+1}_{tipo}": serie for idx, ((_, tipo), serie) in enumerate(zip(series_originales, denormalized_data))})

# Exportar los DataFrames a archivos CSV
df_originales.to_csv('originales.csv', index=False)
df_reconstruidos.to_csv('reconstruidos.csv', index=False)
