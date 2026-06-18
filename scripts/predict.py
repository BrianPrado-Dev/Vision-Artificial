# -*- coding: utf-8 -*-
"""
===============================================================================
 predict.py  -  INFERENCIA / PREDICCIÓN con un modelo YOLOv8 ya entrenado
===============================================================================
 Proyecto : Detección de defectos en piezas de fundición CNC y desgaste en
            herramientas de fresado en una línea de producción.
 Autor    : <ESCRIBE TU NOMBRE AQUÍ>
 Curso    : <NOMBRE DE LA ASIGNATURA>
 Fecha    : <FECHA>
-------------------------------------------------------------------------------
 ¿QUÉ HACE ESTE SCRIPT?
 Usa el modelo YA ENTRENADO (archivo best.pt generado por train.py) para
 DETECTAR defectos en imágenes o videos NUEVOS que el modelo nunca ha visto.
 Por cada detección dibuja una "bounding box" (caja delimitadora) con:
   - El nombre de la clase (ej. "porosidad", "desgaste_flanco").
   - La confianza (probabilidad de 0 a 1 de que la detección sea correcta).
 Finalmente, GUARDA las imágenes/videos anotados en la carpeta 'evidencias/'.

 DIFERENCIA CLAVE CON train.py:
   - train.py  = el modelo APRENDE (modifica sus pesos).
   - predict.py= el modelo APLICA lo aprendido (los pesos NO cambian).

 USO BÁSICO (desde la raíz del proyecto):
   # Sobre una sola imagen:
   python scripts/predict.py --source datasets/defectos_cnc_fundicion/test/images/pieza1.jpg

   # Sobre una carpeta completa de imágenes:
   python scripts/predict.py --source datasets/defectos_cnc_fundicion/test/images

   # Sobre un video:
   python scripts/predict.py --source evidencias/videos/linea_produccion.mp4

   # Usando la webcam/cámara industrial en vivo (fuente = 0):
   python scripts/predict.py --source 0
===============================================================================
"""

# ----------------------------------------------------------------------------
# 1. IMPORTACIÓN DE LIBRERÍAS
# ----------------------------------------------------------------------------
import argparse           # Lee parámetros desde la terminal
from pathlib import Path  # Manejo de rutas multiplataforma

from ultralytics import YOLO  # Clase principal del modelo YOLOv8


# ----------------------------------------------------------------------------
# 2. LECTURA DE ARGUMENTOS DESDE LA LÍNEA DE COMANDOS
# ----------------------------------------------------------------------------
def parse_arguments():
    """Define los parámetros configurables de la inferencia."""
    parser = argparse.ArgumentParser(
        description="Inferencia con YOLOv8 para detección de defectos CNC."
    )

    parser.add_argument(
        "--model", type=str, default="models/best.pt",
        help="Ruta al modelo entrenado (best.pt). Cópialo a la carpeta 'models/'."
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Fuente a analizar: ruta de una imagen, de un video, de una "
             "carpeta de imágenes, o '0' para la cámara en vivo."
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Umbral de confianza (0-1). Solo se muestran detecciones con "
             "confianza mayor a este valor. Subir = menos falsos positivos."
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="Umbral IoU para NMS: elimina cajas duplicadas sobre el mismo objeto."
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Tamaño de inferencia (debe coincidir con el usado al entrenar)."
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Dispositivo: 'auto', 'cpu' o índice de GPU como '0'."
    )
    parser.add_argument(
        "--project", type=str, default="evidencias",
        help="Carpeta raíz donde se guardan las imágenes/videos con detecciones."
    )
    parser.add_argument(
        "--name", type=str, default="predicciones",
        help="Subcarpeta para guardar los resultados de esta corrida."
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Muestra en pantalla los resultados en tiempo real (requiere monitor)."
    )

    return parser.parse_args()


# ----------------------------------------------------------------------------
# 3. VERIFICACIÓN DE QUE EL MODELO ENTRENADO EXISTE
# ----------------------------------------------------------------------------
def verificar_modelo(model_path):
    """
    Comprueba que el archivo de pesos (best.pt) exista antes de continuar.
    Da un error claro si el usuario olvidó entrenar o copiar el modelo.
    """
    ruta = Path(model_path)
    if not ruta.exists():
        raise FileNotFoundError(
            f"\n[ERROR] No se encontró el modelo entrenado: '{ruta}'.\n"
            f"        1) Ejecuta primero 'python scripts/train.py'.\n"
            f"        2) Copia el archivo 'best.pt' generado a la carpeta 'models/'.\n"
        )
    print(f"[INFO] Modelo encontrado: {ruta.resolve()}")
    return ruta


def seleccionar_dispositivo(device_arg):
    """Convierte 'auto' en GPU(0)/CPU según disponibilidad; si no, respeta el valor dado."""
    if device_arg == "auto":
        try:
            import torch
            return 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_arg


# ----------------------------------------------------------------------------
# 4. FUNCIÓN PRINCIPAL DE INFERENCIA
# ----------------------------------------------------------------------------
def main():
    # 4.1 Leemos los parámetros de la terminal.
    args = parse_arguments()

    print("=" * 70)
    print(" INICIO DE LA INFERENCIA - YOLOv8 (Detección de defectos CNC)")
    print("=" * 70)

    # 4.2 Validamos el modelo y elegimos el hardware.
    verificar_modelo(args.model)
    device = seleccionar_dispositivo(args.device)

    # 4.3 Cargamos el modelo entrenado en memoria.
    print(f"[INFO] Cargando modelo entrenado: {args.model}")
    model = YOLO(args.model)

    print("\n----- CONFIGURACIÓN DE LA INFERENCIA -----")
    print(f"  Fuente       : {args.source}")
    print(f"  Confianza min: {args.conf}")
    print(f"  Dispositivo  : {device}")
    print(f"  Salida       : {Path(args.project) / args.name}")
    print("------------------------------------------\n")

    # 4.4 EJECUTAMOS LA PREDICCIÓN.
    #     Ultralytics detecta automáticamente si 'source' es imagen, carpeta,
    #     video o cámara. El parámetro save=True dibuja las cajas y guarda los
    #     archivos anotados en  <project>/<name>.
    resultados = model.predict(
        source=args.source,   # Qué analizar (imagen/video/carpeta/cámara)
        conf=args.conf,       # Umbral de confianza
        iou=args.iou,         # Umbral IoU para NMS
        imgsz=args.imgsz,     # Tamaño de inferencia
        device=device,        # GPU o CPU
        save=True,            # Guarda las imágenes/videos con las cajas dibujadas
        project=args.project, # Carpeta raíz de salida -> 'evidencias/'
        name=args.name,       # Subcarpeta -> 'predicciones'
        exist_ok=True,        # Permite sobrescribir la carpeta si ya existe
        show=args.show,       # Muestra en pantalla si se pidió con --show
        line_width=2,         # Grosor de las cajas dibujadas
    )

    # 4.5 RESUMEN DE DETECCIONES (muy útil como evidencia en el informe).
    #     Recorremos cada resultado (una imagen o un fotograma) y contamos
    #     cuántos objetos de cada clase se detectaron.
    print("\n----- RESUMEN DE DETECCIONES -----")
    nombres_clases = model.names  # Diccionario {0:'pieza_conforme', 1:'porosidad', ...}
    total_detecciones = 0

    for i, r in enumerate(resultados):
        cajas = r.boxes  # Conjunto de cajas detectadas en esta imagen/fotograma
        n = len(cajas) if cajas is not None else 0
        total_detecciones += n

        # Construimos un texto con el conteo por clase, ej: "porosidad x2, grieta x1"
        conteo = {}
        if n > 0:
            for clase_id in cajas.cls.tolist():
                nombre = nombres_clases[int(clase_id)]
                conteo[nombre] = conteo.get(nombre, 0) + 1
        detalle = ", ".join(f"{k} x{v}" for k, v in conteo.items()) if conteo else "sin defectos"
        print(f"  [{i + 1:>3}] {Path(r.path).name}: {n} detección(es) -> {detalle}")

    # 4.6 Mensaje final indicando dónde quedaron las evidencias.
    carpeta_salida = Path(args.project) / args.name
    print("\n" + "=" * 70)
    print(" INFERENCIA FINALIZADA")
    print("=" * 70)
    print(f"[INFO] Total de objetos detectados: {total_detecciones}")
    print(f"[INFO] Evidencias guardadas en   : {carpeta_salida.resolve()}")
    print("=" * 70)


# ----------------------------------------------------------------------------
# 5. PUNTO DE ENTRADA DEL PROGRAMA
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
