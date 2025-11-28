
# 📈 Proyecto de Pronóstico con Modelos No Lineales (LSTM)

Este proyecto implementa un modelo de pronóstico basado en redes neuronales **LSTM**, siguiendo punto por punto la rúbrica oficial del examen final. El análisis incluye: preparación del dataset, ingeniería de características, modelado, evaluación, pronóstico futuro y conclusiones.

---

# 🧭 1. Introducción

## 📌 Serie seleccionada  
Se utilizó una serie histórica de precios (columna **Close**) proveniente de un dataset financiero. Es relevante porque presenta variabilidad temporal, tendencia y posibles patrones no lineales que justifican el uso de un modelo LSTM.

## 🎯 Motivación  
El pronóstico de precios es crucial en entornos económicos, industriales y financieros, ya que permite:
- Identificar tendencias.
- Anticipar cambios futuros.
- Apoyar decisiones estratégicas.

## 🧩 Objetivo del proyecto  
- Construir una serie de tiempo limpia y analizada.  
- Preparar ventanas deslizantes sin data leakage.  
- Entrenar un modelo LSTM.  
- Evaluar su capacidad predictiva.  
- Generar un pronóstico a futuro.

---

# 🧹 2. Preparación del Dataset

## 📥 Fuente de datos  
Se utilizó un archivo CSV cargado en el notebook:


df = pd.read_csv('archivo.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df.set_index('DateTime')
````

##  Limpieza

* Conversión del índice temporal.
* Reemplazo/eliminación de valores faltantes.
* Asegurar consistencia temporal y ausencia de duplicados.

##  Exploración inicial

Se generaron gráficas interactivas con Plotly para identificar:

* Tendencia general
* Variabilidad
* Picos u outliers



---

# 🔧 3. Ingeniería de Características

## 🔍 Escalamiento

Se utilizó **MinMaxScaler**, entrenado únicamente con los datos de entrenamiento:


## 🪟 Construcción de ventanas

Para evitar fuga de información, se generaron secuencias con una función de ventaneo:


def crear_ventanas(datos, window_size):
    X, y = [], []
    for i in range(len(datos) - window_size):
        X.append(datos[i:i+window_size])
        y.append(datos[i+window_size])
    return np.array(X), np.array(y)


## ➕ Variables adicionales

El modelo es **univariado**, utilizando únicamente la columna `Close`.

---

# 🧠 4. Modelado

## 🏗️ Modelo utilizado: LSTM

La arquitectura seleccionada fue:


model = Sequential([
    Input(shape=(window_size, 1)),
    LSTM(64, return_sequences=False),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')


## ⚙️ Justificación

* **LSTM:** adecuado para relaciones temporales no lineales.
* **64 unidades:** buen balance entre capacidad y sobreajuste.
* **Adam:** optimizador robusto para gradientes ruidosos.
* **MSE:** métrica estándar para regresión.

## 🏋️ Entrenamiento


hist = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)


Incluye validación para monitorear desempeño.

---

# 📏 5. Evaluación del Modelo

## 📐 Métricas


* **MAE**
* **RMSE**
* **MAPE**



## 📉 Gráfica real vs predicho




Se observa el nivel de ajuste del modelo sobre el conjunto de prueba.

---

# 🔮 6. Pronóstico Futuro

Se utilizó la última ventana del conjunto para generar predicciones futuras paso a paso:

* Se mantiene el escalamiento.
* Cada predicción alimenta la siguiente ventana.
* Se revierte el escalamiento para interpretar resultados.

Se presentan:

* Tabla final con valores estimados.
* Gráfica de tendencia futura.


