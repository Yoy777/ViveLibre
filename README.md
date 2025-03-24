# ViveLibre
Análisis de Actividades Diarias a Partir de Datos de Wearable
Este proyecto tiene como objetivo detectar y clasificar patrones de actividad diaria en un entorno doméstico, a partir de datos recolectados por un reloj inteligente. Se utilizan técnicas de preprocesamiento, imputación y clustering para transformar datos en bruto en información relevante que permita inferir actividades como dormir, trabajar, desplazarse, ver televisión, comer o ir al baño.

Características Principales
Fusión de Datos:
Se integran múltiples archivos CSV provenientes de diferentes sensores (por ejemplo, frecuencia cardíaca, acelerometría, posición) en un único DataFrame, usando la columna de tiempo como clave.

Preprocesamiento y Extracción de Características:

Conversión y extracción temporal: Se convierte la columna de tiempo a formato datetime y se extraen componentes (hora, minuto, segundo).

Agrupación en ventanas: Se agrupan los registros en intervalos de 5 minutos para obtener la variable derivada time_bin_numeric (minutos transcurridos desde la medianoche).

Cálculo de la movilidad: Se calcula dist_travelled a partir de las coordenadas x e y, lo que permite evaluar la movilidad del usuario dentro del domicilio.

Filtrado de Datos de Calidad:
Solo se utilizan los registros donde onskin == 3, lo que indica que el sensor está en contacto óptimo con la piel, garantizando así la calidad y fiabilidad de las mediciones.

Imputación de Valores Faltantes:
Se implementa un forward fill recursivo para las coordenadas "x" e "y", imputando de forma conjunta cuando ambos valores están ausentes (0 o nulos). Esto asegura la continuidad de la serie espacial y mejora el cálculo de distancias.

Detección y Eliminación de Outliers:
Se utiliza el algoritmo Local Outlier Factor (LOF) para identificar y eliminar registros atípicos, basándose en la densidad local de los datos. Esto es especialmente útil en series de tiempo con alta frecuencia de muestreo.

Clustering con K-means:
Se aplica el algoritmo K-means para segmentar los datos en clusters. La elección de K se optimiza de forma automática utilizando el silhouette score, lo que permite identificar de forma objetiva el número de clusters óptimo. Cada cluster se interpreta en función de sus características y se asocia, de forma tentativa, a actividades diarias específicas (por ejemplo, dormir, trabajar, desplazarse).

Visualización y Exportación:
Se generan gráficos interactivos (barras, tarta, series de tiempo, box plots) para explorar las variables del dataset y se permite exportar tanto el DataFrame final como resúmenes y la información de los centroides.

Reclusterización Iterativa:
Se incluye la posibilidad de reclusterizar el conjunto de datos eliminando outliers o registros de clusters problemáticos, permitiendo refinar los resultados de forma iterativa.

Tecnologías Utilizadas
Python 3: Lenguaje principal para el procesamiento y análisis de datos.

Pandas & NumPy: Manejo y transformación de datos.

Matplotlib: Visualización gráfica de datos.

scikit-learn: Implementación de algoritmos de clustering (K-means) y detección de outliers (LOF).

Tkinter: Interfaz gráfica para interacción del usuario (carga de ficheros, selección de gráficas, etc.).

Aplicaciones y Resultados
El proyecto está orientado a ayudar a entender y clasificar el comportamiento diario de un usuario a través de datos de sensores wearable. Los resultados permiten:

Detectar actividades diarias clave.

Identificar periodos de actividad y reposo.

Mejorar la calidad de la información eliminando datos atípicos y rellenando datos faltantes.

Exportar análisis y gráficos para posteriores estudios o aplicaciones en salud, monitoreo de bienestar o gestión de rutinas diarias.

