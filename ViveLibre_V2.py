import os
import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import filedialog, scrolledtext, messagebox, simpledialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor

# Variables globales para almacenar los DataFrames cargados y el merge final
loaded_files = {}  # Clave: ruta del fichero; Valor: DataFrame procesado (sin duplicados)
merged_data = None
merged_stats = None
loaded_folder = None


def cargar_fichero():
    """Carga un fichero CSV individual y lo procesa, ofreciendo opciones de visualización."""
    global loaded_files
    global loaded_folder
    ruta_fichero = filedialog.askopenfilename(
        title="Selecciona un fichero CSV",
        filetypes=[("Archivos CSV", "*.csv")]
    )
    
    if ruta_fichero:
        loaded_folder = os.path.dirname(ruta_fichero)
        try:
            df = pd.read_csv(ruta_fichero, dtype=str)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el fichero: {e}")
            return
        
        log = "Primeros 5 registros (ANTES de conversión):\n"
        log += df.head().to_string(index=False) + "\n\n"
        log += f"Número de filas: {df.shape[0]}\n\n"
        
        # Se asume que la primera columna es el campo de tiempo
        columna_tiempo = df.columns[0]
        df[columna_tiempo] = pd.to_datetime(df[columna_tiempo], errors='coerce')
        df[columna_tiempo] = df[columna_tiempo].dt.strftime("%d/%m/%Y %H:%M:%S")
        df.rename(columns={columna_tiempo: "tiempo"}, inplace=True)
        
        # Convertir las demás columnas a numérico
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        log += "Primeros 5 registros (DESPUÉS de conversión):\n"
        log += df.head().to_string(index=False) + "\n\n"
        
        # Listar duplicados (por fila completa)
        duplicados_todos = df[df.duplicated(keep=False)]
        log += "**Listado de registros duplicados:**\n"
        if not duplicados_todos.empty:
            for grupo, sub_df in df.groupby(list(df.columns)):
                indices = sub_df.index.tolist()
                if len(indices) > 1:
                    log += f"Registro duplicado: {grupo} en filas: {indices}\n"
        else:
            log += "No se encontraron registros duplicados.\n"
        
        filas_iniciales = df.shape[0]
        df_sin_duplicados = df.drop_duplicates()
        filas_finales = df_sin_duplicados.shape[0]
        log += f"\nDuplicados removidos: {filas_iniciales - filas_finales}\n"
        log += f"Registros después de eliminar duplicados: {filas_finales}\n\n"
        
        # Estadísticas descriptivas para variables numéricas
        stats = {}
        numeric_cols = df_sin_duplicados.columns[1:]  # asumiendo la primera columna es 'tiempo'
        for col in numeric_cols:
            if pd.api.types.is_numeric_dtype(df_sin_duplicados[col]):
                stats[col] = {
                    "Mínimo": df_sin_duplicados[col].min(),
                    "Máximo": df_sin_duplicados[col].max(),
                    "Promedio": df_sin_duplicados[col].mean(),
                    "Q1": df_sin_duplicados[col].quantile(0.25),
                    "Q3": df_sin_duplicados[col].quantile(0.75),
                    "Nulos": df_sin_duplicados[col].isnull().sum(),
                    "Ceros": (df_sin_duplicados[col] == 0).sum()
                }
        stats_df = pd.DataFrame(stats).T
        if not stats_df.empty:
            log += "Estadísticas descriptivas (variables numéricas):\n" + stats_df.to_string() + "\n\n"
        else:
            log += "No se encontraron columnas numéricas para estadísticas descriptivas.\n\n"
        
        loaded_files[ruta_fichero] = df_sin_duplicados
        log += f"Fichero cargado y guardado: {ruta_fichero}\n"
        
        txt_log.delete(1.0, tk.END)
        txt_log.insert(tk.END, log)
        
        # ─────────────────────────────────────────────────────────────────────────
        # Opcional: Ofrecer opciones de visualización mediante una ventana de selección
        # ─────────────────────────────────────────────────────────────────────────
        if messagebox.askyesno("Explorar datos", "¿Deseas explorar algunas variables con gráficas?"):
            # Obtener lista de columnas
            all_cols = df_sin_duplicados.columns.tolist()
            
            # Crear ventana de selección de gráfica
            ventana_seleccion = tk.Toplevel()
            ventana_seleccion.title("Seleccionar tipo de gráfica")
            
            # Variable para almacenar el tipo de gráfica seleccionada
            tipo_grafica = tk.StringVar(value="hist")
            tk.Label(ventana_seleccion, text="Seleccione el tipo de gráfica:").pack(anchor="w", padx=10, pady=5)
            opciones = [("Histograma", "hist"), ("Gráfico de tarta", "pie"), ("Serie de tiempo", "line"), ("Box plot", "box")]
            for texto, valor in opciones:
                tk.Radiobutton(ventana_seleccion, text=texto, variable=tipo_grafica, value=valor).pack(anchor="w", padx=10)
            
            # Variable para la columna a graficar
            col_seleccion = tk.StringVar(value=all_cols[0] if all_cols else "")
            tk.Label(ventana_seleccion, text="Seleccione la columna:").pack(anchor="w", padx=10, pady=5)
            if all_cols:
                opcion_menu = tk.OptionMenu(ventana_seleccion, col_seleccion, *all_cols)
                opcion_menu.pack(anchor="w", padx=10, pady=5)
            else:
                tk.Label(ventana_seleccion, text="No hay columnas disponibles.").pack(anchor="w", padx=10, pady=5)
            
            def generar_grafica():
                ventana_seleccion.destroy()
                tipo = tipo_grafica.get()
                columna = col_seleccion.get()
                # Verificar que se seleccionó una columna
                if columna not in all_cols:
                    messagebox.showinfo("Información", "Columna no válida.")
                    return
                if tipo == "hist":
                    if pd.api.types.is_numeric_dtype(df_sin_duplicados[columna]):
                        plt.figure()
                        plt.hist(df_sin_duplicados[columna].dropna(), bins=30, color="skyblue", edgecolor="black")
                        plt.title(f"Histograma de {columna}")
                        plt.xlabel(columna)
                        plt.ylabel("Frecuencia")
                        plt.show()
                    else:
                        messagebox.showinfo("Información", "La columna seleccionada no es numérica para un histograma.")
                elif tipo == "pie":
                    counts = df_sin_duplicados[columna].value_counts(dropna=False)
                    if counts.empty:
                        messagebox.showinfo("Información", "No hay datos para graficar en tarta.")
                    else:
                        plt.figure()
                        counts.plot.pie(autopct='%1.1f%%')
                        plt.title(f"Distribución de {columna}")
                        plt.ylabel("")
                        plt.show()
                elif tipo == "line":
                    if "tiempo" in df_sin_duplicados.columns:
                        df_sin_duplicados["tiempo_dt"] = pd.to_datetime(df_sin_duplicados["tiempo"], format="%d/%m/%Y %H:%M:%S", errors='coerce')
                        df_time = df_sin_duplicados.dropna(subset=["tiempo_dt", columna]).copy()
                        df_time = df_time.sort_values("tiempo_dt")
                        if pd.api.types.is_numeric_dtype(df_time[columna]):
                            plt.figure()
                            plt.plot(df_time["tiempo_dt"], df_time[columna], marker='o', linestyle='-')
                            plt.title(f"Serie de tiempo de {columna}")
                            plt.xlabel("Tiempo")
                            plt.ylabel(columna)
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            plt.show()
                        else:
                            messagebox.showinfo("Información", "La columna seleccionada no es numérica para una serie de tiempo.")
                    else:
                        messagebox.showinfo("Información", "No se encontró la columna 'tiempo' para serie de tiempo.")
                elif tipo == "box":
                    if pd.api.types.is_numeric_dtype(df_sin_duplicados[columna]):
                        plt.figure()
                        df_sin_duplicados.boxplot(column=columna)
                        plt.title(f"Box plot de {columna}")
                        plt.ylabel(columna)
                        plt.show()
                    else:
                        messagebox.showinfo("Información", "La columna seleccionada no es numérica para un box plot.")
                else:
                    messagebox.showinfo("Información", "Tipo de gráfica no reconocido (usa hist, pie, line, box).")
            
            tk.Button(ventana_seleccion, text="Generar Gráfica", command=generar_grafica).pack(pady=10)
            ventana_seleccion.grab_set()
            ventana_seleccion.wait_window()
            
def auto_load_files():
    """
    Si no se han cargado ficheros, se permite seleccionar automáticamente
    los CSV: interruptions, location, heart_rate y onskin.
    """
    global loaded_files
    paths = filedialog.askopenfilenames(
        title="Selecciona los CSV: interruptions, location, heart_rate y onskin",
        filetypes=[("Archivos CSV", "*.csv")]
    )
    if not paths:
        messagebox.showwarning("Advertencia", "No se seleccionaron ficheros.")
        return False
    for path in paths:
        try:
            df = pd.read_csv(path, dtype=str)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el fichero {path}: {e}")
            continue
        columna_tiempo = df.columns[0]
        df[columna_tiempo] = pd.to_datetime(df[columna_tiempo], errors='coerce')
        df[columna_tiempo] = df[columna_tiempo].dt.strftime("%d/%m/%Y %H:%M:%S")
        df.rename(columns={columna_tiempo: "tiempo"}, inplace=True)
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df_sin_duplicados = df.drop_duplicates()
        loaded_files[path] = df_sin_duplicados
    return True

def popup_merge_options():
    """Muestra un pop up con las dos opciones de merge."""
    popup = tk.Toplevel()
    popup.title("Selecciona el método de merge")
    popup.geometry("300x150")
    
    tk.Label(popup, text="Elige el método de merge:").pack(pady=10)
    
    btn_sin = tk.Button(popup, text="Sin imputacion", width=20,
                         command=lambda: [popup.destroy(), merge_sin_imputacion()])
    btn_sin.pack(pady=5)
    
    btn_imp = tk.Button(popup, text="Imputacion", width=20,
                         command=lambda: [popup.destroy(), merge_con_imputacion()])
    btn_imp.pack(pady=5)

def merge_sin_imputacion():
    """
    Fusiona los ficheros cargados utilizando la PK 'tiempo' del fichero 'interruptions'
    (proceso actual, sin imputación) y almacena el resultado en la variable global merged_data.
    Al finalizar, se abre un diálogo para guardar el DataFrame resultante en Excel.
    """
    global loaded_files, merged_data, merged_stats
    if not loaded_files:
        if not auto_load_files():
            messagebox.showerror("Error", "No se pudieron cargar los ficheros automáticamente.")
            return
    main_key = None
    for key in loaded_files:
        if "interruptions" in key.lower():
            main_key = key
            break
    if not main_key:
        messagebox.showerror("Error", "No se encontró el fichero 'interruptions' entre los cargados.")
        return
    main_df = loaded_files[main_key]
    tamaño_interruptions = main_df.shape[0]
    log = f"Tamaño de la base 'interruptions' (sin duplicados): {tamaño_interruptions} registros\n\n"
    
    merged_df = main_df.copy()
    # Para cada fichero secundario, se garantiza una única fila por 'tiempo'
    for key, df in loaded_files.items():
        if key == main_key:
            continue
        df_sec = df.drop_duplicates(subset=["tiempo"])
        merged_df = merged_df.merge(df_sec, on="tiempo", how="left")
    
    log += "Primeros 5 registros del DataFrame fusionado:\n"
    log += merged_df.head().to_string(index=False) + "\n\n"
    
    duplicados_merged = merged_df[merged_df.duplicated(keep=False)]
    log += "**Listado de duplicados en el merge:**\n"
    if not duplicados_merged.empty:
        for grupo, sub_df in merged_df.groupby(list(merged_df.columns)):
            indices = sub_df.index.tolist()
            if len(indices) > 1:
                log += f"Registro duplicado: {grupo} en filas: {indices}\n"
    else:
        log += "No se encontraron duplicados en el merge.\n"
    
    filas_iniciales = merged_df.shape[0]
    merged_df_sin_dup = merged_df.drop_duplicates()
    filas_finales = merged_df_sin_dup.shape[0]
    log += f"\nDuplicados removidos en el merge: {filas_iniciales - filas_finales}\n"
    log += f"Registros después del merge: {filas_finales}\n\n"
    
    stats = {}
    numeric_cols = merged_df_sin_dup.columns.drop("tiempo", errors='ignore')
    for col in numeric_cols:
        stats[col] = {
            "Mínimo": merged_df_sin_dup[col].min(),
            "Máximo": merged_df_sin_dup[col].max(),
            "Promedio": merged_df_sin_dup[col].mean(),
            "Q1": merged_df_sin_dup[col].quantile(0.25),
            "Q3": merged_df_sin_dup[col].quantile(0.75),
            "Nulos": merged_df_sin_dup[col].isnull().sum(),
            "Ceros": (merged_df_sin_dup[col] == 0).sum()
        }
    merged_stats_df = pd.DataFrame(stats).T
    log += "Estadísticas descriptivas del merge:\n" + merged_stats_df.to_string() + "\n\n"
    
    merged_data = merged_df_sin_dup.copy()
    merged_stats = merged_stats_df.copy()
    
    log += "El DataFrame fusionado se ha guardado en 'merged_data' y sus estadísticas en 'merged_stats'.\n"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Diálogo para exportar la base fusionada
    # ─────────────────────────────────────────────────────────────────────────
    if messagebox.askyesno("Exportar Base", "¿Desea exportar el DataFrame sin imputación?"):
        ruta_excel = asksaveasfilename(
            title="Guardar DataFrame sin imputación",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if ruta_excel:
            merged_data.to_excel(ruta_excel, index=False)
            log += f"\nEl DataFrame sin imputación se ha exportado a:\n{ruta_excel}\n"
        else:
            log += "\nNo se exportó el DataFrame sin imputación.\n"
    else:
        log += "\nEl usuario decidió no exportar el DataFrame sin imputación.\n"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Diálogo para exportar las estadísticas descriptivas
    # ─────────────────────────────────────────────────────────────────────────
    if messagebox.askyesno("Exportar Estadísticas", "¿Desea exportar las estadísticas descriptivas del merge?"):
        ruta_excel_stats = asksaveasfilename(
            title="Guardar estadísticas descriptivas",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if ruta_excel_stats:
            merged_stats.to_excel(ruta_excel_stats, index=True)
            log += f"\nLas estadísticas descriptivas se han exportado a:\n{ruta_excel_stats}\n"
        else:
            log += "\nNo se exportaron las estadísticas descriptivas.\n"
    else:
        log += "\nEl usuario decidió no exportar las estadísticas descriptivas.\n"
    
    txt_log.delete(1.0, tk.END)
    txt_log.insert(tk.END, log)
    messagebox.showinfo("Merge finalizado", "Merge sin imputación completado.")
    
def merge_con_imputacion():
    """
    Fusiona los ficheros cargados y luego aplica imputación en heart_rate y cnfs:
    Para cada registro cuyo valor de cnfs < 50 se realiza lo siguiente:
      1) Método basado en regla: Se buscan candidatos en hr_df, dentro de 0 a 4 segundos posteriores
         (mismo día, hora y minuto) que tengan un valor de cnfs mayor que el actual.
         Si se encuentra el candidato con el mayor cnfs, se imputa su heart_rate y cnfs,
         se marca imputed=1, se guarda el tiempo del candidato en time_imputed y se indica el método "rule".
      2) Si no se puede imputar, se conservan los valores originales (imputed=0, imputed_method="none").

    Además, se guardan las columnas originales en "heart_rate_origin" y "cnfs_origin".
    Se muestra una barra de progreso solo para los registros con cnfs < 50.
    Al finalizar se le preguntará al usuario si desea exportar el DataFrame resultante a Excel.
    """
    global loaded_files, merged_data, merged_stats, ml_model

    if not loaded_files:
        if not auto_load_files():
            messagebox.showerror("Error", "No se pudieron cargar los ficheros automáticamente.")
            return

    # 1. Identificar la tabla original de heart_rate
    hr_key = None
    for key in loaded_files:
        if "heart_rate" in key.lower():
            hr_key = key
            break
    if hr_key is None:
        messagebox.showerror("Error", "No se encontró el fichero 'heart_rate' entre los cargados.")
        return

    hr_df = loaded_files[hr_key].copy()
    if "tiempo_dt" not in hr_df.columns:
        hr_df["tiempo_dt"] = pd.to_datetime(hr_df["tiempo"], format="%d/%m/%Y %H:%M:%S", errors='coerce')

    # 2. Realizar el merge (igual que sin imputación)
    main_key = None
    for key in loaded_files:
        if "interruptions" in key.lower():
            main_key = key
            break
    if not main_key:
        messagebox.showerror("Error", "No se encontró el fichero 'interruptions' entre los cargados.")
        return

    main_df = loaded_files[main_key]
    merged_df = main_df.copy()
    for key, df in loaded_files.items():
        if key == main_key:
            continue
        df_sec = df.drop_duplicates(subset=["tiempo"])
        merged_df = merged_df.merge(df_sec, on="tiempo", how="left")
    merged_df = merged_df.drop_duplicates()
    if "tiempo_dt" not in merged_df.columns:
        merged_df["tiempo_dt"] = pd.to_datetime(merged_df["tiempo"], format="%d/%m/%Y %H:%M:%S", errors='coerce')

    # 3. Guardar columnas originales
    merged_df["heart_rate_origin"] = merged_df["heart_rate"]
    merged_df["cnfs_origin"] = merged_df["cnfs"]

    # 4. Agregar columnas para imputación
    merged_df["imputed"] = 0
    merged_df["time_imputed"] = ""
    merged_df["imputed_method"] = "none"

    # 5. Filtrar registros con cnfs < 50 (los que se imputarán)
    mask_imput = merged_df["cnfs"].astype(float) < 50
    total_to_imput = mask_imput.sum()
    df_to_imput = merged_df[mask_imput].copy()
    
    log = f"Total de filas con cnfs < 50 (a imputar): {total_to_imput}\n\n"

    # 6. Guardar el índice original para volcar los cambios después
    df_to_imput["__idx__"] = df_to_imput.index
    imput_list = df_to_imput.to_dict("records")

    # 7. Crear ventana de progreso para imputación de heart_rate y cnfs
    progress_window = tk.Toplevel()
    progress_window.title("Progreso de Imputación")
    progress_label = tk.Label(progress_window, text="Iniciando imputación...")
    progress_label.pack(pady=5)
    progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=300, mode="determinate")
    progress_bar.pack(pady=5)
    n_rows_imput = len(imput_list)
    progress_bar["maximum"] = n_rows_imput

    # 8. Función de imputación para cada registro (método basado en regla)
    # >>> ESTE BLOQUE SE HA COMENTADO, POR LO QUE NO SE REALIZA LA IMPUTACIÓN POR REGLA <<<
    """
    def impute_hr(record):
        try:
            current_cnfs = float(record["cnfs"])
        except:
            return record["heart_rate"], record["cnfs"], "", 0, "none"
        if current_cnfs >= 50:
            return record["heart_rate"], record["cnfs"], "", 0, "none"
        
        try:
            current_time = datetime.datetime.strptime(record["tiempo"], "%d/%m/%Y %H:%M:%S")
        except Exception:
            return record["heart_rate"], record["cnfs"], "", 0, "none"
        
        time_window_mask = (hr_df["tiempo_dt"] - current_time).dt.total_seconds().between(0, 3)
        same_minute_mask = (
            (hr_df["tiempo_dt"].dt.date == current_time.date()) &
            (hr_df["tiempo_dt"].dt.hour == current_time.hour) &
            (hr_df["tiempo_dt"].dt.minute == current_time.minute) &
            (hr_df["tiempo_dt"].dt.second > current_time.second)
        )
        mask_rule = time_window_mask & same_minute_mask & (hr_df["cnfs"].astype(float) > current_cnfs)
        candidates = hr_df[mask_rule]
        if not candidates.empty:
            candidate = candidates.sort_values(by="cnfs", ascending=False).iloc[0]
            candidate_cnfs = float(candidate["cnfs"])
            if candidate_cnfs > current_cnfs:
                time_imp = candidate["tiempo_dt"].strftime("%d/%m/%Y %H:%M:%S")
                return candidate["heart_rate"], candidate["cnfs"], time_imp, 1, "rule"
        return record["heart_rate"], record["cnfs"], "", 0, "none"
    
    for i, record in enumerate(imput_list):
        new_hr, new_cnfs, time_imp, flag, method = impute_hr(record)
        record["heart_rate"] = new_hr
        record["cnfs"] = new_cnfs
        record["time_imputed"] = time_imp
        record["imputed"] = flag
        record["imputed_method"] = method

        if i % 50 == 0:
            progress_bar["value"] = i
            progress_label["text"] = f"Imputando registro {i} de {n_rows_imput}"
            progress_window.update_idletasks()
    """
    # Fin del bloque comentado

    progress_window.destroy()

    # 9. Volcar los cambios de vuelta a merged_df usando el índice original
    for record in imput_list:
        idx = record["__idx__"]
        merged_df.at[idx, "heart_rate"] = record["heart_rate"]
        merged_df.at[idx, "cnfs"] = record["cnfs"]
        merged_df.at[idx, "time_imputed"] = record["time_imputed"]
        merged_df.at[idx, "imputed"] = record["imputed"]
        merged_df.at[idx, "imputed_method"] = record["imputed_method"]

    # NUEVA SECCIÓN: Imputación para las variables "x" y "y" (forward fill estricto de la fila anterior)
    # 1) Guardar los valores originales de "x" e "y"
    merged_df["x_original"] = merged_df["x"]
    merged_df["y_original"] = merged_df["y"]

    # 2) Ordenar por tiempo_dt
    merged_df = merged_df.sort_values("tiempo_dt").copy()

    # 3) Crear ventana de progreso para la imputación de x e y
    progress_xy = tk.Toplevel()
    progress_xy.title("Progreso de imputación de x e y")
    progress_xy_label = tk.Label(progress_xy, text="Iniciando imputación en x e y (inmediato anterior)...")
    progress_xy_label.pack(pady=5)
    progress_xy_bar = ttk.Progressbar(progress_xy, orient="horizontal", length=300, mode="determinate")
    progress_xy_bar.pack(pady=5)
    total_rows_xy = merged_df.shape[0]
    progress_xy_bar["maximum"] = total_rows_xy

    # 4) Iterar sobre cada fila. Si ambos son 0/nulos, se imputan con el último valor final (last_x, last_y).
    last_x = None
    last_y = None

    for i, idx in enumerate(merged_df.index):
        current_x = merged_df.at[idx, "x"]
        current_y = merged_df.at[idx, "y"]

        # Verificar si ambos son 0 o NaN
        both_missing = (pd.isna(current_x) or current_x == 0) and (pd.isna(current_y) or current_y == 0)

        if both_missing and (last_x is not None and last_y is not None):
            # Imputar usando el valor final de la fila anterior
            final_x = last_x
            final_y = last_y
        else:
            # Conservar los valores actuales (sean parciales o completos)
            final_x = current_x
            final_y = current_y

        # Asignar el valor final a la fila
        merged_df.at[idx, "x"] = final_x
        merged_df.at[idx, "y"] = final_y

        # Actualizar referencias con los valores finales (imputados o reales)
        last_x = final_x
        last_y = final_y

        progress_xy_bar["value"] = i + 1
        progress_xy_label["text"] = f"Imputando x e y: registro {i+1} de {total_rows_xy}"
        progress_xy.update_idletasks()

    progress_xy.destroy()

    # 5) Crear la columna "x_y_imputed": 1 si se imputó (si originalmente ambas eran 0/nulas y se actualizaron), 0 si no.
    merged_df["x_y_imputed"] = (
        ((merged_df["x_original"] != merged_df["x"]) | (merged_df["y_original"] != merged_df["y"])) &
        ((merged_df["x_original"] == 0) | (merged_df["x_original"].isna())) &
        ((merged_df["y_original"] == 0) | (merged_df["y_original"].isna()))
    ).astype(int)

    # NUEVA SECCIÓN: Calcular y agregar al log la cantidad de imputaciones en "x" e "y"
    n_imputed_x = (
        ((merged_df["x_original"] == 0) | (merged_df["x_original"].isna())) &
        ~(merged_df["x"].isna() | (merged_df["x"] == 0))
    ).sum()
    n_imputed_y = (
        ((merged_df["y_original"] == 0) | (merged_df["y_original"].isna())) &
        ~(merged_df["y"].isna() | (merged_df["y"] == 0))
    ).sum()
    n_initial_x = ((merged_df["x_original"] == 0) | (merged_df["x_original"].isna())).sum()
    n_initial_y = ((merged_df["y_original"] == 0) | (merged_df["y_original"].isna())).sum()

    log += f"\nTotal de registros imputados en 'x': {n_imputed_x} (de {n_initial_x} con 0 o null en x)"
    log += f"\nTotal de registros imputados en 'y': {n_imputed_y} (de {n_initial_y} con 0 o null en y)\n"

    # 10. Convertir a numérico solo las columnas que deben ser numéricas,
    # excluyendo las de fecha y las originales
    cols_to_exclude = ["tiempo", "tiempo_dt", "time_imputed", "heart_rate_origin", "cnfs_origin", "imputed_method"]
    cols_to_convert = [col for col in merged_df.columns if col not in cols_to_exclude]
    for col in cols_to_convert:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

    tamaño_interruptions = main_df.shape[0]
    log += f"\nTamaño de la base 'interruptions' (sin duplicados): {tamaño_interruptions} registros\n\n"
    log += "Primeros 5 registros del DataFrame fusionado (después de imputación):\n"
    log += merged_df.head().to_string(index=False) + "\n\n"

    duplicados_merged = merged_df[merged_df.duplicated(keep=False)]
    log += "**Listado de duplicados en el merge:**\n"
    if not duplicados_merged.empty:
        for grupo, sub_df in merged_df.groupby(list(merged_df.columns)):
            indices = sub_df.index.tolist()
            if len(indices) > 1:
                log += f"Registro duplicado: {grupo} en filas: {indices}\n"
    else:
        log += "No se encontraron duplicados en el merge.\n"

    filas_iniciales = merged_df.shape[0]
    merged_df_sin_dup = merged_df.drop_duplicates()
    filas_finales = merged_df_sin_dup.shape[0]
    log += f"\nDuplicados removidos en el merge: {filas_iniciales - filas_finales}\n"
    log += f"Registros después del merge: {filas_finales}\n\n"

    n_imputed = merged_df_sin_dup["imputed"].astype(int).sum()
    log += f"Cantidad de registros imputados: {n_imputed}\n"

    stats = {}
    numeric_cols = merged_df_sin_dup.columns.drop(["tiempo", "time_imputed", "imputed_method"], errors='ignore')
    for col in numeric_cols:
        stats[col] = {
            "Mínimo": merged_df_sin_dup[col].min(),
            "Máximo": merged_df_sin_dup[col].max(),
            "Promedio": merged_df_sin_dup[col].mean(),
            "Q1": merged_df_sin_dup[col].quantile(0.25),
            "Q3": merged_df_sin_dup[col].quantile(0.75),
            "Nulos": merged_df_sin_dup[col].isnull().sum(),
            "Ceros": (merged_df_sin_dup[col] == 0).sum()
        }
    merged_stats_df = pd.DataFrame(stats).T
    log += "Estadísticas descriptivas del merge:\n" + merged_stats_df.to_string() + "\n\n"

    merged_data = merged_df_sin_dup.copy()
    merged_stats = merged_stats_df.copy()

    log += "El DataFrame fusionado con imputación se ha guardado en 'merged_data' y sus estadísticas en 'merged_stats'.\n"

    # 12. Preguntar al usuario si desea exportar el DataFrame con imputación
    if messagebox.askyesno("Exportar Base", "¿Desea exportar el DataFrame con imputación?"):
        ruta_excel = filedialog.asksaveasfilename(
            title="Guardar DataFrame con imputación",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if ruta_excel:
            merged_data.to_excel(ruta_excel, index=False)
            log += f"\nEl DataFrame con imputación se ha exportado a:\n{ruta_excel}\n"
        else:
            log += "\nNo se exportó el DataFrame con imputación (no se seleccionó ruta).\n"
    else:
        log += "\nEl usuario decidió no exportar el DataFrame con imputación.\n"

    txt_log.delete(1.0, tk.END)
    txt_log.insert(tk.END, log)
    messagebox.showinfo("Merge con imputación", "Merge con imputación completado.")



def merge_datos():
    """Al pulsar 'merge_datos' se muestra un pop up para seleccionar el método de merge."""
    popup_merge_options()


def assign_human_activity(row, hr_mean, hr_std):
    """
    Asigna de forma arbitraria una etiqueta específica para la actividad humana
    basada en la frecuencia cardíaca (heart_rate) en relación a la media y la desviación estándar.
    
    Regla propuesta:
      - Si heart_rate > (hr_mean + 1.0*hr_std): "Caminar"
      - Si heart_rate > (hr_mean + 0.75*hr_std): "Cocinar"
      - Si heart_rate > (hr_mean + 0.5*hr_std): "Andar"
      - Si heart_rate está entre (hr_mean - 0.5*hr_std) y (hr_mean + 0.5*hr_std): "Trabajar"
      - Si heart_rate < (hr_mean - 0.75*hr_std): "Ir al baño"
      - En otro caso: "Comer"
    """
    if row["heart_rate"] > (hr_mean + 1.0 * hr_std):
        return "Caminar"
    elif row["heart_rate"] > (hr_mean + 0.75 * hr_std):
        return "Cocinar"
    elif row["heart_rate"] > (hr_mean + 0.5 * hr_std):
        return "Andar"
    elif (row["heart_rate"] >= (hr_mean - 0.5 * hr_std)) and (row["heart_rate"] <= (hr_mean + 0.5 * hr_std)):
        return "Trabajar"
    elif row["heart_rate"] < (hr_mean - 0.75 * hr_std):
        return "Ir al baño"
    else:
        return "Comer"

def mode_1(series: pd.Series):
    """
    Calcula la moda de una serie, devolviendo la primera moda
    en caso de múltiples valores más frecuentes.
    """
    m = series.mode(dropna=True)
    if len(m) > 0:
        return m.iloc[0]
    return None

from tkinter.filedialog import asksaveasfilename

def deteccion_de_eventos():
    """Abre una ventana con las opciones de clasificación de eventos."""
    ventana_opciones = tk.Toplevel()
    ventana_opciones.title("Método de Clasificación de Eventos")
    ventana_opciones.geometry("350x200")
    
    tk.Label(ventana_opciones, text="Seleccione el método de clasificación de eventos:").pack(pady=10)
    
    btn_reglas = tk.Button(ventana_opciones, text="Clasificación basada en Reglas", width=30,
                            command=lambda: [ventana_opciones.destroy(), clasificacion_por_reglas()])
    btn_reglas.pack(pady=5)
    
    btn_cluster = tk.Button(ventana_opciones, text="Clasificación por Clustering", width=30,
                            command=lambda: [ventana_opciones.destroy(), clasificacion_por_clustering()])
    btn_cluster.pack(pady=5)

    btn_cluster = tk.Button(ventana_opciones, text="Clasificación por DBSCAN", width=30,
                            command=lambda: [ventana_opciones.destroy(), clasificacion_por_dbscan()])
    btn_cluster.pack(pady=5)

    
def clasificacion_por_reglas():
    """
    Mejora la clasificación de 'Reposo' subdividiéndolo en 'Reposo Diurno' y 'Reposo Nocturno',
    y luego reasigna a 'Dormir' únicamente si:
      - Estaba en 'Reposo Nocturno'
      - No hay movimiento (any_motion == 0)
      - Frecuencia cardíaca < (hr_mean - 1.0*hr_std)
    
    Finalmente, se añaden estadísticas descriptivas (media, mediana, moda, min, max)
    para cada evento en las variables heart_rate, cnfs, any_motion, high_g, onskin.
    Se da la opción de exportar dichas estadísticas a Excel, así como la base completa con
    la columna de eventos, mediante diálogos "Sí/No".
    """
    global merged_data
    if merged_data is None:
        messagebox.showwarning("Advertencia", "No se ha realizado el merge. Ejecuta 'merge_datos' o 'merge_con_imputacion' primero.")
        return

    df_events = merged_data.copy()
    df_events["event_label"] = "Sin evento"
    
    # Verificar que existan las columnas necesarias
    for col in ["heart_rate", "cnfs", "high_g", "any_motion", "onskin"]:
        if col not in df_events.columns:
            messagebox.showerror("Error", f"Columna requerida '{col}' no encontrada en merged_data.")
            return
    
    # Convertir la columna de tiempo a datetime para poder extraer la hora
    df_events["tiempo_dt"] = pd.to_datetime(df_events["tiempo"], format="%d/%m/%Y %H:%M:%S", errors='coerce')
    
    hr_mean = df_events["heart_rate"].mean()
    hr_std = df_events["heart_rate"].std()
    
    # 1) Caídas
    fall_threshold = 15
    mask_fall = df_events["high_g"] > fall_threshold
    df_events.loc[mask_fall, "event_label"] = "Caida leve"
    mask_fall_grave = mask_fall & (df_events["heart_rate"] < (hr_mean - hr_std))
    df_events.loc[mask_fall_grave, "event_label"] = "Caida grave"
    
    # 2) Eventos emocionales
    mask_emotional = (df_events["heart_rate"] > (hr_mean + 2 * hr_std)) & (df_events["cnfs"] >= 0.8)
    df_events.loc[mask_emotional, "event_label"] = "Emocional (estres/anxiedad)"
    
    # 3) Actividad humana (any_motion != 0) y sin etiqueta asignada
    mask_human = (df_events["any_motion"] != 0) & (df_events["event_label"] == "Sin evento")
    df_events.loc[mask_human, "event_label"] = df_events[mask_human].apply(
        lambda row: assign_human_activity(row, hr_mean, hr_std), axis=1
    )
    
    # 4) Asignar "Reposo" a lo que quede sin etiqueta
    df_events.loc[df_events["event_label"] == "Sin evento", "event_label"] = "Reposo"
    
    # 5) Subdividir "Reposo" en "Reposo Nocturno" y "Reposo Diurno" según la hora
    mask_reposo = df_events["event_label"] == "Reposo"
    df_events.loc[
        mask_reposo & ((df_events["tiempo_dt"].dt.hour >= 22) | (df_events["tiempo_dt"].dt.hour < 6)),
        "event_label"
    ] = "Reposo Nocturno"
    
    df_events.loc[
        mask_reposo & ((df_events["tiempo_dt"].dt.hour >= 6) & (df_events["tiempo_dt"].dt.hour < 22)),
        "event_label"
    ] = "Reposo Diurno"
    
    # 6) Reasignar a "Dormir" únicamente si cumple condiciones
    mask_nocturno = (df_events["event_label"] == "Reposo Nocturno")
    mask_sin_mov = (df_events["any_motion"] == 0)
    mask_hr_baja = (df_events["heart_rate"] < (hr_mean - 1.0 * hr_std))
    df_events.loc[mask_nocturno & mask_sin_mov & mask_hr_baja, "event_label"] = "Dormir"
    
    # 7) Clasificación general ("No wearable" si onskin != 3)
    df_events["general_label"] = df_events["event_label"]
    df_events.loc[df_events["onskin"] != 3, "general_label"] = "No wearable"
    
    # ─────────────────────────────────────────────────────────────────────────────
    # SECCIÓN DE LOG Y GRÁFICAS
    # ─────────────────────────────────────────────────────────────────────────────
    
    general_counts = df_events["general_label"].value_counts()
    log = "=== Clasificación general de eventos ===\n"
    log += general_counts.to_string() + "\n\n"
    log += "Primeros 5 registros (general):\n"
    log += df_events.head().to_string(index=False) + "\n\n"
    
    # Pequeño gráfico de torta
    plt.figure()
    plt.pie(general_counts, labels=general_counts.index, autopct='%1.1f%%')
    plt.title("Distribución de eventos (general)")
    plt.show()
    
    # Resumen de intervalos (hora inicio y fin) y conteos
    summary = df_events.groupby("general_label").agg(
        inicio=("tiempo_dt", "min"),
        fin=("tiempo_dt", "max"),
        count=("general_label", "count")
    ).reset_index()
    total = df_events.shape[0]
    summary["Porcentaje"] = summary["count"] / total * 100
    summary["inicio"] = summary["inicio"].dt.strftime("%H:%M")
    summary["fin"] = summary["fin"].dt.strftime("%H:%M")
    summary_table = summary.rename(columns={
        "general_label": "Evento",
        "inicio": "Hora inicio",
        "fin": "Hora fin",
        "count": "Cantidad",
        "Porcentaje": "Porcentaje (%)"
    })
    
    log += "Tabla resumen de eventos (general):\n"
    log += summary_table.to_string(index=False) + "\n\n"
    
    # DETALLE NO WEARABLE
    df_no_wearable = df_events[df_events["onskin"] != 3].copy()
    if not df_no_wearable.empty:
        df_no_wearable["detailed_label"] = "NW_" + df_no_wearable["event_label"]
        nw_counts = df_no_wearable["detailed_label"].value_counts()
        log += "=== Clasificación detallada de registros NO WEARABLE ===\n"
        log += nw_counts.to_string() + "\n\n"
        
        plt.figure()
        plt.pie(nw_counts, labels=nw_counts.index, autopct='%1.1f%%')
        plt.title("Distribución de eventos (detallado No wearable)")
        plt.show()
        
        summary_nw = df_no_wearable.groupby("detailed_label").agg(
            inicio=("tiempo_dt", "min"),
            fin=("tiempo_dt", "max"),
            count=("detailed_label", "count")
        ).reset_index()
        total_nw = df_no_wearable.shape[0]
        summary_nw["Porcentaje"] = summary_nw["count"] / total_nw * 100
        summary_nw["inicio"] = summary_nw["inicio"].dt.strftime("%H:%M")
        summary_nw["fin"] = summary_nw["fin"].dt.strftime("%H:%M")
        summary_nw_table = summary_nw.rename(columns={
            "detailed_label": "Evento",
            "inicio": "Hora inicio",
            "fin": "Hora fin",
            "count": "Cantidad",
            "Porcentaje": "Porcentaje (%)"
        })
        
        log += "Tabla resumen de eventos (detallado No wearable):\n"
        log += summary_nw_table.to_string(index=False) + "\n\n"
    else:
        log += "No se detectaron registros NO WEARABLE en detalle.\n\n"
    
    # ─────────────────────────────────────────────────────────────────────────────
    # PREGUNTAR AL USUARIO SI DESEA EXPORTAR ESTADÍSTICAS
    # ─────────────────────────────────────────────────────────────────────────────
    desc_cols = ["heart_rate", "cnfs", "any_motion", "high_g", "onskin"]
    agg_dict = {col: ["mean", "median", mode_1, "min", "max"] for col in desc_cols}
    df_stats = df_events.groupby("event_label").agg(agg_dict)
    df_stats.columns = ["_".join(col).strip() for col in df_stats.columns.values]
    
    export_stats = messagebox.askyesno("Exportar Estadísticas",
                                       "¿Deseas exportar las estadísticas descriptivas a Excel?")
    if export_stats:
        ruta_excel_stats = asksaveasfilename(
            title="Guardar estadísticas descriptivas (Excel)",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if ruta_excel_stats:
            df_stats.to_excel(ruta_excel_stats, index=True)
            log += f"Se han exportado las estadísticas descriptivas a:\n{ruta_excel_stats}\n"
        else:
            log += "Se eligió exportar estadísticas, pero no se seleccionó una ruta.\n"
    else:
        log += "El usuario eligió NO exportar las estadísticas descriptivas.\n"
    
    # ─────────────────────────────────────────────────────────────────────────────
    # PREGUNTAR AL USUARIO SI DESEA EXPORTAR LA BASE CON EVENTOS
    # ─────────────────────────────────────────────────────────────────────────────
    export_base = messagebox.askyesno("Exportar Base de Eventos",
                                      "¿Deseas exportar la base completa con la columna de eventos?")
    if export_base:
        ruta_excel_eventos = asksaveasfilename(
            title="Guardar base completa con EVENTOS (Excel)",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if ruta_excel_eventos:
            df_events.to_excel(ruta_excel_eventos, index=False)
            log += f"\nEl DataFrame completo con la columna 'event_label' se ha exportado a:\n{ruta_excel_eventos}\n"
        else:
            log += "Se eligió exportar la base de eventos, pero no se seleccionó una ruta.\n"
    else:
        log += "El usuario eligió NO exportar la base completa con eventos.\n"
    
    # Finalmente, mostrar el log en la interfaz
    txt_log.delete(1.0, tk.END)
    txt_log.insert(tk.END, log)



def clasificacion_por_clustering():
    """
    Aplica clustering (K-means) al DataFrame fusionado utilizando campos seleccionados del DataFrame merge.
    Los campos posibles a usar para el clustering (para el análisis) son: any_motion, high_g, heart_rate, cnfs y onskin.
    Además, se incluyen obligatoriamente dos variables derivadas:
      - time_bin_numeric: se calcula agrupando la marca de tiempo en ventanas de 5 minutos (minutos desde la medianoche).
      - dist_travelled: distancia recorrida entre registros consecutivos (calculada a partir de "x" e "y", pero no se utiliza en el clustering).
    
    Se ejecuta el clustering únicamente con las filas donde onskin == 3, y se excluyen "x" y "y" de las variables de clustering.
    
    La función determina automáticamente el número óptimo de clusters mediante el silhouette score y genera un resumen,
    junto con gráficos de la distribución de clusters, evolución temporal y (opcionalmente) la visualización espacial.
    
    Finalmente, se ofrecen opciones para exportar el DataFrame resultante, el resumen y la información de los centroides.
    """
    log = ""
    # Crear ventana de progreso
    total_steps = 11
    progress_window = tk.Toplevel()
    progress_window.title("Progreso de Clustering")
    progress_label = tk.Label(progress_window, text="Iniciando...")
    progress_label.pack(pady=5)
    progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=300, mode="determinate")
    progress_bar.pack(pady=5)
    progress_bar["maximum"] = total_steps

    def update_progress(step, message):
        progress_bar["value"] = step
        progress_label.config(text=message)
        progress_window.update_idletasks()

    global merged_data
    if merged_data is None:
        messagebox.showwarning("Advertencia", "No se ha realizado el merge. Ejecuta 'merge_datos' o 'merge_con_imputacion' primero.")
        progress_window.destroy()
        return

    # Paso 1: Verificar columnas requeridas
    update_progress(1, "Verificando columnas requeridas...")
    required_cols = ["any_motion", "high_g", "heart_rate", "cnfs", "x", "y", "onskin", "tiempo"]
    for col in required_cols:
        if col not in merged_data.columns:
            messagebox.showerror("Error", f"Columna requerida '{col}' no encontrada en merged_data.")
            progress_window.destroy()
            return

    # Paso 2: Convertir 'tiempo' a datetime y extraer hora, minuto y segundo
    update_progress(2, "Convirtiendo 'tiempo' a datetime...")
    try:
        merged_data["tiempo_dt"] = pd.to_datetime(merged_data["tiempo"], format="%d/%m/%Y %H:%M:%S", errors='coerce')
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo convertir 'tiempo' a datetime: {e}")
        progress_window.destroy()
        return

    merged_data["hora"] = merged_data["tiempo_dt"].dt.hour
    merged_data["minuto"] = merged_data["tiempo_dt"].dt.minute
    merged_data["segundo"] = merged_data["tiempo_dt"].dt.second

    # Paso 3: Calcular características temporales (ventanas de 5 minutos)
    update_progress(3, "Calculando características temporales...")
    merged_data["time_bin"] = merged_data["tiempo_dt"].dt.floor("5min")
    merged_data["time_bin_numeric"] = merged_data["time_bin"].dt.hour * 60 + merged_data["time_bin"].dt.minute

    # Paso 4: Calcular características espaciales (distancia recorrida)
    update_progress(4, "Calculando características espaciales...")
    merged_data = merged_data.sort_values("tiempo_dt").copy()
    merged_data["dist_travelled"] = merged_data[["x", "y"]].diff().apply(
        lambda row: np.sqrt(row.iloc[0]**2 + row.iloc[1]**2), axis=1
    )
    merged_data["dist_travelled"] = merged_data["dist_travelled"].fillna(0)

    # Paso 5: Extraer características para el clustering
    update_progress(5, "Extrayendo características para clustering...")
    # Definir la lista de variables para el clustering (excluyendo "x" y "y")
    features = ["any_motion", "high_g", "heart_rate", "cnfs", "onskin", "time_bin_numeric", "dist_travelled"]
    # Filtrar para que se usen únicamente las filas donde onskin == 3
    df_cluster = merged_data.loc[merged_data["onskin"] == 3, features].copy()
    for col in features:
        df_cluster[col] = pd.to_numeric(df_cluster[col], errors='coerce')
    df_cluster = df_cluster.dropna()

    from sklearn.neighbors import LocalOutlierFactor
    # Aplicar LOF para eliminar outliers:
    
    initial_count = len(df_cluster)

    n_neighbors = min(340, len(df_cluster))  # Ajusta según convenga
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.05)
    y_pred = lof.fit_predict(df_cluster)  # Inliers = 1, outliers = -1

    mask_inliers = (y_pred == 1)
    final_count = mask_inliers.sum()  # Cantidad de registros inliers

    eliminated_count = initial_count - final_count

    df_cluster = df_cluster[mask_inliers]

    log += f"Registros iniciales: {initial_count}\n"
    log += f"Registros eliminados por LOF: {eliminated_count}\n"
    log += f"Registros finales tras LOF: {final_count}\n"

    if df_cluster.empty:
        messagebox.showerror("Error", "No hay suficientes datos para realizar clustering después de eliminar datos faltantes.")
        progress_window.destroy()
        return

    # Paso 6: Escalar las características
    update_progress(6, "Escalando características...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    # Paso 7: Determinar el número óptimo de clusters usando el silhouette score
    update_progress(7, "Determinando número óptimo de clusters...")
    best_score = -1
    best_n_clusters = 2  # mínimo
    max_clusters = min(11, len(df_cluster))
    for n in range(2, max_clusters):
        update_progress(7, f"Evaluando clustering con {n} clusters...")
        kmeans_temp = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels_temp = kmeans_temp.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels_temp)
        if score > best_score:
            best_score = score
            best_n_clusters = n

    explanation = (
        "El número óptimo de clusters se determinó evaluando diferentes valores de k "
        "y seleccionando aquel que maximiza el silhouette score. Un mayor silhouette score "
        "indica que los clusters están bien separados y son coherentes internamente. "
        f"En este caso, se eligieron {best_n_clusters} clusters con un silhouette score de {best_score:.3f}."
    )

    # Paso 8: Ejecutar clustering final con k-means
    update_progress(8, f"Ejecutando clustering final con {best_n_clusters} clusters...")
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    merged_data["cluster_label"] = None
    merged_data.loc[df_cluster.index, "cluster_label"] = cluster_labels

    # Paso 9: Generar resumen de clustering
    update_progress(9, "Generando resumen de clustering...")
    summary = merged_data.loc[df_cluster.index].groupby("cluster_label").agg(
        count=("cluster_label", "size"),
        mean_any_motion=("any_motion", "mean"),
        mean_high_g=("high_g", "mean"),
        mean_heart_rate=("heart_rate", "mean"),
        mean_onskin=("onskin", "mean"),
        mean_time_bin=("time_bin_numeric", "mean"),
        mean_dist_travelled=("dist_travelled", "mean")
    ).reset_index()
    cluster_counts = merged_data.loc[df_cluster.index, "cluster_label"].value_counts().sort_index()
    log = "=== Resultados del Clustering ===\n"
    log += explanation + "\n\n"
    log += "Resumen de clusters:\n" + summary.to_string(index=False) + "\n\n"
    log += "Conteo de registros por cluster:\n" + cluster_counts.to_string() + "\n\n"

    log += f"Registros iniciales: {initial_count}\n"
    log += f"Registros eliminados por LOF: {eliminated_count}\n"
    log += f"Registros finales tras LOF: {final_count}\n"


    # Paso 10: Mostrar gráficos
    update_progress(10, "Mostrando gráficos...")
    plt.figure()
    plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%')
    plt.title("Distribución de clusters")
    plt.show()

    # Aunque "x" y "y" se excluyeron para clustering, se pueden usar para visualizar
    plt.figure()
    scatter = plt.scatter(merged_data.loc[df_cluster.index, "x"],
                          merged_data.loc[df_cluster.index, "y"],
                          c=merged_data.loc[df_cluster.index, "cluster_label"],
                          cmap='viridis')
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("Coordenada x")
    plt.ylabel("Coordenada y")
    plt.title("Distribución Espacial (x vs y) por Cluster")
    plt.show()

    df_time = merged_data.loc[df_cluster.index].copy()
    df_time = df_time.sort_values("tiempo_dt")
    plt.figure()
    scatter = plt.scatter(df_time["tiempo_dt"], df_time["heart_rate"],
                          c=df_time["cluster_label"], cmap='viridis')
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("Tiempo")
    plt.ylabel("Heart Rate")
    plt.title("Evolución Temporal de Heart Rate por Cluster")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Paso 10.1: Agregar las distancias antes de exportar
    def calcular_distancias_y_centroides(df, features, cluster_label_col="cluster_label"):
        import numpy as np
        import pandas as pd
        # Crear una Serie de pandas para las distancias con el mismo índice que df
        distancias = pd.Series(np.zeros(len(df)), index=df.index)
        centroides = {}
        # Agrupar por cluster y calcular centroide y distancias Euclidianas
        for label, grupo in df.groupby(cluster_label_col):
            centroide = grupo[features].mean().values
            centroides[label] = centroide
            dists = np.sqrt(((grupo[features].values - centroide)**2).sum(axis=1))
            distancias.loc[grupo.index] = dists
        return distancias, centroides

    # Aplicar sobre el subconjunto de registros usados en clustering (usando los índices de df_cluster)
    distancias, centroides_dict = calcular_distancias_y_centroides(merged_data.loc[df_cluster.index], features, "cluster_label")
    # Agregar la columna "dist_to_centroid" en merged_data para esos registros
    merged_data.loc[df_cluster.index, "dist_to_centroid"] = distancias

    # Para cada feature, agregar una columna con el valor del centroide asignado al cluster de cada registro
    for i, feat in enumerate(features):
        col_centroid = f"centroid_{feat}"
        merged_data.loc[df_cluster.index, col_centroid] = merged_data.loc[df_cluster.index, "cluster_label"].map(
            lambda lbl: centroides_dict[lbl][i] if lbl in centroides_dict else np.nan
        )


    # Paso 11: Preguntar por exportación de resultados
    update_progress(11, "Preguntando por exportación...")
    if messagebox.askyesno("Exportar Base", "¿Desea exportar el DataFrame con etiquetas de clustering?"):
        ruta_excel = filedialog.asksaveasfilename(
            title="Guardar DataFrame con clustering",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if ruta_excel:
            merged_data.to_excel(ruta_excel, index=False)
            log += f"\nEl DataFrame con clustering se ha exportado a:\n{ruta_excel}\n"
        else:
            log += "\nNo se exportó el DataFrame con clustering (no se seleccionó ruta).\n"
    else:
        log += "\nEl usuario decidió no exportar el DataFrame con clustering.\n"
    if messagebox.askyesno("Exportar Estadísticas", "¿Desea exportar el resumen de clustering a Excel?"):
        ruta_excel_stats = filedialog.asksaveasfilename(
            title="Guardar resumen de clustering (Excel)",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if ruta_excel_stats:
            summary.to_excel(ruta_excel_stats, index=False)
            log += f"\nEl resumen de clustering se ha exportado a:\n{ruta_excel_stats}\n"
        else:
            log += "\nNo se exportó el resumen de clustering (no se seleccionó ruta).\n"
    else:
        log += "\nEl usuario decidió no exportar el resumen de clustering.\n"

    # Exportación de los centroides
    if messagebox.askyesno("Exportar Centroides", "¿Desea exportar la información de los centroides?"):
        centroids = kmeans.cluster_centers_
        centroids_original = scaler.inverse_transform(centroids)
        # Usar los mismos campos de clustering (features) para los centroides
        centroid_df = pd.DataFrame(centroids_original, columns=features)
        centroid_df.insert(0, "cluster_label", range(best_n_clusters))
        ruta_centroides = filedialog.asksaveasfilename(
            title="Guardar Centroides (Excel)",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if ruta_centroides:
            centroid_df.to_excel(ruta_centroides, index=False)
            log += f"\nLa información de los centroides se ha exportado a:\n{ruta_centroides}\n"
        else:
            log += "\nNo se exportó la información de los centroides (no se seleccionó ruta).\n"
    else:
        log += "\nEl usuario decidió no exportar la información de los centroides.\n"

    txt_log.delete(1.0, tk.END)
    txt_log.insert(tk.END, log)

    update_progress(total_steps, "Clustering completado.")
    # Final del proceso inicial de clustering
    progress_window.destroy()
    messagebox.showinfo("Clustering completado", "La clasificación por clustering se ha completado.")

    # Preguntar si se desea reclusterizar
    # BLOQUE NUEVO: Diálogo para seleccionar filtros de reclusterización
    def show_recluster_options():
        option_window = tk.Toplevel()
        option_window.title("Opciones de Reclustering")
        tk.Label(option_window, text="Seleccione el/los filtro(s) a aplicar:").pack(padx=10, pady=5)
        
        var_lof = tk.BooleanVar(value=False)
        var_centroid = tk.BooleanVar(value=False)
        
        chk_lof = tk.Checkbutton(option_window, text="Filtrado global con LOF", variable=var_lof)
        chk_lof.pack(anchor="w", padx=10, pady=2)
        chk_centroid = tk.Checkbutton(option_window, text="Filtrado por distancia al centroide", variable=var_centroid)
        chk_centroid.pack(anchor="w", padx=10, pady=2)
        
        def accept():
            option_window.destroy()
            return
            
        btn_ok = tk.Button(option_window, text="Aceptar", command=accept)
        btn_ok.pack(pady=10)
        
        # Esperar hasta que se cierre la ventana
        option_window.grab_set()
        option_window.wait_window()
        return var_lof.get(), var_centroid.get()
    
    if messagebox.askyesno("Reclusterizar", "¿Desea reclusterizar eliminando registros atípicos?\nPuede aplicar LOF, por distancia al centroide, o ambos."):
        apply_lof, apply_centroid = show_recluster_options()
        
        # Aplicar filtro global con LOF si fue seleccionado
        if apply_lof:
            from sklearn.neighbors import LocalOutlierFactor
            n_neighbors = min(15, len(merged_data))
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.05)
            X_all = merged_data[features].dropna()
            y_pred = lof.fit_predict(X_all)  # Inliers=1, outliers=-1
            merged_data = merged_data.loc[X_all.index][y_pred == 1].copy()
            messagebox.showinfo("Reclusterización (LOF)", 
                                "Filtrado global con LOF completado. Se han eliminado los registros atípicos.")
        
        # Aplicar filtro por distancia al centroide si fue seleccionado
        if apply_centroid:
            cluster_to_refine = simpledialog.askinteger("Cluster a refinar", "Ingrese el número del cluster a refinar:")
            if cluster_to_refine is not None:
                # Extraer registros del cluster problemático
                df_cluster_problem = merged_data[merged_data["cluster_label"] == cluster_to_refine].copy()
                if df_cluster_problem.empty:
                    messagebox.showwarning("Advertencia", f"No se encontraron registros para el cluster {cluster_to_refine}.")
                else:
                    # Calcular el centroide usando las mismas features
                    centroide = df_cluster_problem[features].mean().values
                    # Calcular la distancia Euclidiana para cada registro
                    distancias = np.sqrt(((df_cluster_problem[features].values - centroide) ** 2).sum(axis=1))
                    # Definir un umbral (por ejemplo, media + 2*std)
                    umbral = distancias.mean() + 2 * distancias.std()
                    df_filtrado = df_cluster_problem[distancias <= umbral]
                    merged_data = merged_data[merged_data["cluster_label"] != cluster_to_refine]
                    merged_data = pd.concat([merged_data, df_filtrado], ignore_index=True)
                    messagebox.showinfo("Reclusterización (Centroide)",
                                        f"Filtrado por centroide completado para el cluster {cluster_to_refine}.\nUmbral: {umbral:.2f}")
        
        # Reiniciar el proceso completo de clustering
        clasificacion_por_clustering()




def clasificacion_por_dbscan():
    """
    Aplica clustering DBSCAN al DataFrame fusionado utilizando las siguientes características:
      - any_motion, high_g, heart_rate, x, y, onskin (características originales)
      - Características temporales: registros agrupados en ventanas de 5 minutos, 
        expresadas como minutos desde la medianoche (time_bin_numeric).
      - Características espaciales: distancia recorrida entre registros consecutivos (dist_travelled).

    Además, se extraen y agregan tres columnas separadas: "hora", "minuto" y "segundo",
    extraídas del campo "tiempo_dt" (convertido a datetime), para que aparezcan en la base final.

    La función determina automáticamente un valor de eps utilizando el percentil 90 de las distancias
    al k-ésimo vecino (min_samples=5 por defecto) y aplica DBSCAN sin intervención manual.
    
    Se añade la columna 'cluster_label' al DataFrame y se genera un resumen (conteo y medias por cluster).
    También se generan gráficos de:
      - Distribución de clusters (gráfico de torta).
      - Distribución espacial (x vs y) coloreado por cluster.
      - Evolución temporal de heart_rate, coloreado por cluster.

    Finalmente, se ofrecen opciones para exportar el DataFrame resultante y el resumen.
    """
    # Crear ventana de progreso
    total_steps = 11
    progress_window = tk.Toplevel()
    progress_window.title("Progreso de DBSCAN")
    progress_label = tk.Label(progress_window, text="Iniciando...")
    progress_label.pack(pady=5)
    progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=300, mode="determinate")
    progress_bar.pack(pady=5)
    progress_bar["maximum"] = total_steps

    def update_progress(step, message):
        progress_bar["value"] = step
        progress_label.config(text=message)
        progress_window.update_idletasks()

    global merged_data
    if merged_data is None:
        messagebox.showwarning("Advertencia", "No se ha realizado el merge. Ejecuta 'merge_datos' o 'merge_con_imputacion' primero.")
        progress_window.destroy()
        return

    # Paso 1: Verificar columnas requeridas
    update_progress(1, "Verificando columnas requeridas...")
    required_cols = ["any_motion", "high_g", "heart_rate", "x", "y", "onskin", "tiempo"]
    for col in required_cols:
        if col not in merged_data.columns:
            messagebox.showerror("Error", f"Columna requerida '{col}' no encontrada en merged_data.")
            progress_window.destroy()
            return

    # Paso 2: Convertir 'tiempo' a datetime y extraer hora, minuto y segundo
    update_progress(2, "Convirtiendo 'tiempo' a datetime...")
    try:
        merged_data["tiempo_dt"] = pd.to_datetime(merged_data["tiempo"], format="%d/%m/%Y %H:%M:%S", errors='coerce')
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo convertir 'tiempo' a datetime: {e}")
        progress_window.destroy()
        return

    # Agregar columnas separadas para hora, minuto y segundo
    merged_data["hora"] = merged_data["tiempo_dt"].dt.hour
    merged_data["minuto"] = merged_data["tiempo_dt"].dt.minute
    merged_data["segundo"] = merged_data["tiempo_dt"].dt.second

    # Paso 3: Calcular características temporales (ventanas de 5 minutos)
    update_progress(3, "Calculando características temporales...")
    merged_data["time_bin"] = merged_data["tiempo_dt"].dt.floor("5min")
    merged_data["time_bin_numeric"] = merged_data["time_bin"].dt.hour * 60 + merged_data["time_bin"].dt.minute

    # Paso 4: Calcular características espaciales (distancia recorrida)
    update_progress(4, "Calculando características espaciales...")
    merged_data = merged_data.sort_values("tiempo_dt").copy()
    merged_data["dist_travelled"] = merged_data[["x", "y"]].diff().apply(
        lambda row: np.sqrt(row.iloc[0]**2 + row.iloc[1]**2), axis=1
    )
    merged_data["dist_travelled"] = merged_data["dist_travelled"].fillna(0)

    # Paso 5: Extraer características para clustering (sin imputar valores faltantes)
    update_progress(5, "Extrayendo características para clustering...")
    features = ["any_motion", "high_g", "heart_rate", "cnfs", "onskin", "time_bin_numeric"]
    df_cluster = merged_data[features].copy()
    for col in features:
        df_cluster[col] = pd.to_numeric(df_cluster[col], errors='coerce')
    df_cluster = df_cluster.dropna()  # Eliminamos registros con valores faltantes
    if df_cluster.empty:
        messagebox.showerror("Error", "No hay suficientes datos para realizar clustering.")
        progress_window.destroy()
        return

    # Paso 6: Escalar las características
    update_progress(6, "Escalando características...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    # Paso 7: Determinar parámetros de DBSCAN automáticamente
    update_progress(7, "Determinando parámetros de DBSCAN...")
    min_samples = 5  # Valor por defecto
    neigh = NearestNeighbors(n_neighbors=min_samples)
    neigh.fit(X_scaled)
    distances, _ = neigh.kneighbors(X_scaled)
    kth_distances = distances[:, -1]
    # Elegir eps como el percentil 90 de las distancias al k-ésimo vecino
    eps = np.percentile(kth_distances, 90)
    update_progress(7, f"Parámetros: eps={eps:.3f}, min_samples={min_samples}")

    # Paso 8: Ejecutar DBSCAN
    update_progress(8, "Ejecutando DBSCAN...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    merged_data["cluster_label"] = None
    merged_data.loc[df_cluster.index, "cluster_label"] = dbscan_labels

    # Contar clusters y puntos de ruido
    unique_labels = np.unique(dbscan_labels)
    if -1 in unique_labels:
        num_clusters = len(unique_labels) - 1  # Excluyendo el ruido (-1)
        num_noise = (dbscan_labels == -1).sum()
    else:
        num_clusters = len(unique_labels)
        num_noise = 0

    explanation = (
        "DBSCAN agrupa datos basándose en la densidad:\n"
        "Al determinar automáticamente el parámetro eps utilizando el percentil 90 de las distancias al "
        "k-ésimo vecino (con min_samples=5), el algoritmo se adapta a la estructura de los datos.\n"
        "Los grupos (clusters) se forman en las áreas donde los puntos están suficientemente densos, "
        "mientras que aquellos puntos que no alcanzan esa densidad se etiquetan como ruido (-1).\n"
        f"Se formaron {num_clusters} clusters y se identificaron {num_noise} puntos como ruido.\n"
    )
    update_progress(8, f"DBSCAN completado: {num_clusters} clusters, {num_noise} puntos como ruido.")

    # Paso 9: Generar resumen de clustering
    update_progress(9, "Generando resumen de clustering...")
    summary = merged_data.loc[df_cluster.index].groupby("cluster_label").agg(
        count=("cluster_label", "size"),
        mean_any_motion=("any_motion", "mean"),
        mean_high_g=("high_g", "mean"),
        mean_heart_rate=("heart_rate", "mean"),
        #mean_x=("x", "mean"),
        #mean_y=("y", "mean"),
        mean_onskin=("onskin", "mean"),
        mean_time_bin=("time_bin_numeric", "mean")
        #mean_dist_travelled=("dist_travelled", "mean")
    ).reset_index()
    cluster_counts = merged_data.loc[df_cluster.index, "cluster_label"].value_counts().sort_index()
    log = "=== Resultados de DBSCAN ===\n"
    log += explanation + "\n"
    log += "Resumen de clusters:\n" + summary.to_string(index=False) + "\n\n"
    log += "Conteo de registros por cluster:\n" + cluster_counts.to_string() + "\n\n"

    # Paso 10: Mostrar gráficos
    update_progress(10, "Mostrando gráficos...")
    plt.figure()
    plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%')
    plt.title("Distribución de clusters (DBSCAN)")
    plt.show()

    plt.figure()
    scatter = plt.scatter(merged_data.loc[df_cluster.index, "x"],
                          merged_data.loc[df_cluster.index, "y"],
                          c=merged_data.loc[df_cluster.index, "cluster_label"],
                          cmap='viridis')
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("Coordenada x")
    plt.ylabel("Coordenada y")
    plt.title("Distribución Espacial (x vs y) por Cluster (DBSCAN)")
    plt.show()

    df_time = merged_data.loc[df_cluster.index].copy()
    df_time = df_time.sort_values("tiempo_dt")
    plt.figure()
    scatter = plt.scatter(df_time["tiempo_dt"], df_time["heart_rate"],
                          c=df_time["cluster_label"], cmap='viridis')
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("Tiempo")
    plt.ylabel("Heart Rate")
    plt.title("Evolución Temporal de Heart Rate por Cluster (DBSCAN)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Paso 11: Preguntar por exportación
    update_progress(11, "Preguntando por exportación...")
    if messagebox.askyesno("Exportar Base", "¿Desea exportar el DataFrame con etiquetas de clustering (DBSCAN)?"):
        ruta_excel = filedialog.asksaveasfilename(
            title="Guardar DataFrame con clustering (DBSCAN)",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if ruta_excel:
            merged_data.to_excel(ruta_excel, index=False)
            log += f"\nEl DataFrame con clustering se ha exportado a:\n{ruta_excel}\n"
        else:
            log += "\nNo se exportó el DataFrame con clustering (no se seleccionó ruta).\n"
    else:
        log += "\nEl usuario decidió no exportar el DataFrame con clustering.\n"
    if messagebox.askyesno("Exportar Estadísticas", "¿Desea exportar el resumen de clustering a Excel?"):
        ruta_excel_stats = filedialog.asksaveasfilename(
            title="Guardar resumen de clustering (Excel)",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if ruta_excel_stats:
            summary.to_excel(ruta_excel_stats, index=False)
            log += f"\nEl resumen de clustering se ha exportado a:\n{ruta_excel_stats}\n"
        else:
            log += "\nNo se exportó el resumen de clustering (no se seleccionó ruta).\n"
    else:
        log += "\nEl usuario decidió no exportar el resumen de clustering.\n"

    txt_log.delete(1.0, tk.END)
    txt_log.insert(tk.END, log)
    update_progress(total_steps, "DBSCAN completado.")
    progress_window.destroy()
    messagebox.showinfo("DBSCAN completado", "La clasificación por DBSCAN se ha completado.")

# Configuración de la ventana principal
ventana = tk.Tk()
ventana.title("Cargar, Fusionar y Detectar Eventos")

btn_cargar = tk.Button(ventana, text="Cargar ficheros", command=cargar_fichero)
btn_cargar.pack(pady=10)

btn_merge = tk.Button(ventana, text="merge_datos", command=merge_datos)
btn_merge.pack(pady=10)

btn_eventos = tk.Button(ventana, text="deteccion de enventos", command=deteccion_de_eventos)
btn_eventos.pack(pady=10)

txt_log = scrolledtext.ScrolledText(ventana, width=100, height=30)
txt_log.pack(padx=10, pady=10)

ventana.mainloop()
