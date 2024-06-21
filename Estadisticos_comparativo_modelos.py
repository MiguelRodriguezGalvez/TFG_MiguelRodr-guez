import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
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
        duracion_escalón = np.random.randint(1, 30)  # Duración del escalón de 1 a 30 muestras
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

# ---- Autoencoder Convolucional ----

# Construir el autoencoder convolucional
input_dim = (LONGITUD_SERIE, 1)
input_layer = Input(shape=input_dim)
encoded = Conv1D(32, 3, activation='relu', padding='same')(input_layer)
encoded = MaxPooling1D(2, padding='same')(encoded)
decoded = Conv1D(32, 3, activation='relu', padding='same')(encoded)
decoded = UpSampling1D(2)(decoded)
decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(decoded)
autoencoder_cae = Model(input_layer, decoded)
autoencoder_cae.compile(optimizer='adam', loss='mean_squared_error')

# Normalizar datos
scaler_cae = MinMaxScaler()
normalized_data_cae = scaler_cae.fit_transform(np.array([serie for serie, _ in series_originales]))

# Reshape para datos de entrada convolucionales
normalized_data_cae = normalized_data_cae.reshape(-1, LONGITUD_SERIE, 1)

# Aplicar ruido proporcional a la energía promedio
noisy_data_cae = np.array([serie + np.random.normal(0, K_FACTOR * np.sqrt(energia_promedio), size=len(serie)) for (serie, _), energia_promedio in zip(series_originales, energias_promedio)])

# Entrenar el autoencoder convolucional
autoencoder_cae.fit(normalized_data_cae, normalized_data_cae, epochs=50, batch_size=16, shuffle=True, verbose=2)

# Reconstruir las series originales usando el autoencoder convolucional
reconstructed_data_cae = autoencoder_cae.predict(normalized_data_cae)

# Desnormalizar los datos reconstruidos
denormalized_data_cae = scaler_cae.inverse_transform(reconstructed_data_cae.reshape(-1, LONGITUD_SERIE))

# ---- Autoencoder Denso ----

# Construir el autoencoder denso
input_dim = LONGITUD_SERIE
encoding_dim = 10
input_layer = Input(shape=(input_dim,))
encoder_layer = Dense(encoding_dim, activation='relu')(input_layer)
decoder_layer = Dense(input_dim, activation='sigmoid')(encoder_layer)
autoencoder_dae = Model(input_layer, decoder_layer)
autoencoder_dae.compile(optimizer='adam', loss='mean_squared_error')

# Normalizar datos
scaler_dae = MinMaxScaler()
normalized_data_dae = scaler_dae.fit_transform(np.array([serie for serie, _ in series_originales]))

# Aplicar ruido proporcional a la energía promedio
noisy_data_dae = np.array([serie + np.random.normal(0, K_FACTOR * np.sqrt(energia_promedio), size=len(serie)) for (serie, _), energia_promedio in zip(series_originales, energias_promedio)])

# Entrenar el autoencoder denso
autoencoder_dae.fit(normalized_data_dae, normalized_data_dae, epochs=50, batch_size=16, shuffle=True, verbose=2)

# Reconstruir las series originales usando el autoencoder denso
reconstructed_data_dae = autoencoder_dae.predict(normalized_data_dae)

# Desnormalizar los datos reconstruidos
denormalized_data_dae = scaler_dae.inverse_transform(reconstructed_data_dae)

# ---- Evaluación de los Autoencoders ----

# Función para calcular métricas de error
def calcular_metricas(originales, reconstruidos):
    mse = mean_squared_error(originales, reconstruidos)
    mae = mean_absolute_error(originales, reconstruidos)
    r2 = r2_score(originales, reconstruidos)
    return mse, mae, r2

# Calcular métricas para el autoencoder convolucional
mse_cae, mae_cae, r2_cae = calcular_metricas(np.array([serie for serie, _ in series_originales]), denormalized_data_cae)

# Calcular métricas para el autoencoder denso
mse_dae, mae_dae, r2_dae = calcular_metricas(np.array([serie for serie, _ in series_originales]), denormalized_data_dae)

# Imprimir métricas
print(f"Autoencoder Convolucional - MSE: {mse_cae}, MAE: {mae_cae}, R2: {r2_cae}")
print(f"Autoencoder Denso - MSE: {mse_dae}, MAE: {mae_dae}, R2: {r2_dae}")

# Visualización de algunas series originales y reconstruidas
def plot_series(original, reconstruida, title):
    plt.figure(figsize=(10, 4))
    plt.plot(original, label='Original')
    plt.plot(reconstruida, label='Reconstruida')
    plt.title(title)
    plt.legend()
    plt.show()

# Visualizar algunas series
num_series_to_plot = 5
for i in range(num_series_to_plot):
    plot_series(np.array(series_originales[i][0]), denormalized_data_cae[i], f'Serie {i+1} - Convolucional')
    plot_series(np.array(series_originales[i][0]), denormalized_data_dae[i], f'Serie {i+1} - Denso')
