# -*- coding: utf-8 -*-
"""
===============================================================================
 train.py  -  ENTRENAMIENTO de un modelo YOLOv8 (Ultralytics)
===============================================================================
 Proyecto : Detección de defectos en piezas de fundición CNC y desgaste en
            herramientas de fresado en una línea de producción.
 Autor    : <ESCRIBE TU NOMBRE AQUÍ>
 Curso    : <NOMBRE DE LA ASIGNATURA>
 Fecha    : <FECHA>
-------------------------------------------------------------------------------
 ¿QUÉ HACE ESTE SCRIPT?
 Automatiza la fase de APRENDIZAJE de una red neuronal de detección de objetos
 de la familia YOLO ("You Only Look Once"). En palabras simples:
   - Le mostramos al modelo MILES de imágenes ya etiquetadas (con sus defectos
     marcados por cajas / "bounding boxes").
   - El modelo ajusta sus parámetros internos (pesos) para aprender a reconocer
     esos defectos por sí solo en imágenes nuevas.

 CONCEPTO CLAVE - "Transfer Learning" (aprendizaje por transferencia):
   No entrenamos desde cero. Partimos de un modelo (yolov8n.pt) que YA fue
   entrenado con millones de imágenes genéricas (dataset COCO). Ese modelo ya
   "sabe ver" formas, bordes y texturas. Nosotros solo lo RE-ESPECIALIZAMOS en
   nuestro problema industrial. Esto reduce drásticamente el tiempo de
   entrenamiento y la cantidad de imágenes necesarias.

 USO BÁSICO (desde la terminal, en la raíz del proyecto):
   python scripts/train.py
 USO PERSONALIZADO (ejemplos):
   python scripts/train.py --epochs 150 --batch 8 --model yolov8s.pt
   python scripts/train.py --data datasets/defectos_cnc_fundicion/data.yaml --device cpu
===============================================================================
"""

# ----------------------------------------------------------------------------
# 1. IMPORTACIÓN DE LIBRERÍAS
# ----------------------------------------------------------------------------
import argparse            # Lee parámetros (hiperparámetros) desde la terminal
from pathlib import Path   # Manejo de rutas que funciona igual en Windows/Linux/Mac

import torch               # Motor de deep learning; lo usamos para detectar la GPU
from ultralytics import YOLO
from yaml import parser  # Clase principal del modelo YOLOv8


# ----------------------------------------------------------------------------
# 2. LECTURA DE ARGUMENTOS (HIPERPARÁMETROS) DESDE LA LÍNEA DE COMANDOS
# ----------------------------------------------------------------------------
def parse_arguments():
    """
    Define y lee los parámetros configurables del entrenamiento.

    Usar argparse (en lugar de "quemar" los valores en el código) es una buena
    práctica: permite experimentar con distintos hiperparámetros SIN modificar
    el código fuente, simplemente cambiando lo que escribimos en la terminal.
    """
    parser = argparse.ArgumentParser(
        description="Entrenamiento de YOLOv8 para detección de defectos CNC."
    )

    # --- Datos y modelo base ---
    parser.add_argument(
        "--data", type=str,
        default="datasets/roboflow_fundicion/data.yaml", 
        help="Ruta al archivo data.yaml que describe el dataset."
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="Modelo base preentrenado. n=nano(rápido), s=small, m=medium, "
             "l=large, x=xlarge(preciso pero lento)."
    )

    # --- Hiperparámetros del entrenamiento ---
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Épocas: cuántas veces el modelo recorre TODO el dataset. "
             "Pocas = no aprende (underfitting); demasiadas = se memoriza (overfitting)."
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Tamaño (px) al que se redimensionan las imágenes. 640 es estándar."
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size: cuántas imágenes se procesan a la vez. "
             "Si te quedas sin memoria de GPU/RAM, baja este valor (8, 4...)."
    )
    parser.add_argument(
        "--patience", type=int, default=50,
        help="'Early stopping': detiene el entrenamiento si en N épocas no mejora."
    )

    # --- Hardware ---
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Dispositivo de cómputo: 'auto', 'cpu', o índice de GPU como '0'."
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Procesos en paralelo para cargar imágenes. En Windows, si da "
             "problemas, usa 0 o 2."
    )

    # --- Organización de los resultados ---
    parser.add_argument(
        "--project", type=str, default="runs/entrenamiento",
        help="Carpeta donde se guardan los resultados del entrenamiento."
    )
    parser.add_argument(
        "--name", type=str, default="defectos_yolov8",
        help="Nombre de la subcarpeta de este experimento."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Reanuda un entrenamiento interrumpido desde el último checkpoint."
    )

    return parser.parse_args()


# ----------------------------------------------------------------------------
# 3. SELECCIÓN AUTOMÁTICA DEL DISPOSITIVO (GPU o CPU)
# ----------------------------------------------------------------------------
def seleccionar_dispositivo(device_arg):
    """
    Decide si entrenar en GPU (rápido) o CPU (lento pero siempre disponible).

    - Una GPU NVIDIA con CUDA puede ser 10-50x más rápida que la CPU.
    - torch.cuda.is_available() devuelve True si PyTorch detecta una GPU usable.
    """
    if device_arg == "auto":
        if torch.cuda.is_available():
            nombre_gpu = torch.cuda.get_device_name(0)
            print(f"[INFO] GPU detectada: {nombre_gpu}. Se entrenará en GPU (device=0).")
            return 0  # 0 = primera GPU
        else:
            print("[INFO] No se detectó GPU. Se entrenará en CPU (será más lento).")
            return "cpu"
    # Si el usuario forzó un valor concreto ('cpu', '0', '0,1'...), lo respetamos.
    print(f"[INFO] Dispositivo seleccionado manualmente: {device_arg}")
    return device_arg


# ----------------------------------------------------------------------------
# 4. VERIFICACIÓN DE QUE EL DATASET EXISTE
# ----------------------------------------------------------------------------
def verificar_dataset(data_path):
    """
    Comprueba que el archivo data.yaml exista ANTES de empezar, para dar un
    mensaje de error claro en lugar de un error críptico a mitad del proceso.
    """
    ruta = Path(data_path)
    if not ruta.exists():
        raise FileNotFoundError(
            f"\n[ERROR] No se encontró el archivo de configuración: '{ruta}'.\n"
            f"        Verifica la ruta o créalo siguiendo el formato del README.\n"
        )
    print(f"[INFO] Archivo de dataset encontrado: {ruta.resolve()}")
    return ruta


# ----------------------------------------------------------------------------
# 5. FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ----------------------------------------------------------------------------
def main():
    # 5.1 Leemos los parámetros que el usuario pasó por la terminal.
    args = parse_arguments()

    print("=" * 70)
    print(" INICIO DEL ENTRENAMIENTO - YOLOv8 (Detección de defectos CNC)")
    print("=" * 70)

    # 5.2 Validamos que el dataset exista y elegimos el hardware.
    verificar_dataset(args.data)
    device = seleccionar_dispositivo(args.device)

    # 5.3 Cargamos el modelo base preentrenado (transfer learning).
    #     Si el archivo .pt no está localmente, Ultralytics lo descarga solo.
    print(f"[INFO] Cargando modelo base: {args.model}")
    model = YOLO(args.model)

    # 5.4 Mostramos un resumen de la configuración antes de arrancar.
    print("\n----- CONFIGURACIÓN DEL ENTRENAMIENTO -----")
    print(f"  Dataset      : {args.data}")
    print(f"  Modelo base  : {args.model}")
    print(f"  Épocas       : {args.epochs}")
    print(f"  Tamaño imagen: {args.imgsz} px")
    print(f"  Batch size   : {args.batch}")
    print(f"  Dispositivo  : {device}")
    print("-------------------------------------------\n")

    # 5.5 ¡ENTRENAMIENTO! Esta única llamada hace todo el trabajo pesado:
    #     - Carga las imágenes y sus etiquetas.
    #     - Aplica "data augmentation" (rotaciones, brillo, etc.) para generalizar.
    #     - Calcula el error (loss) y ajusta los pesos en cada época.
    #     - Valida el modelo y guarda automáticamente el MEJOR (best.pt).
    resultados = model.train(
        data=args.data,          # Configuración del dataset (data.yaml)
        epochs=args.epochs,      # Número de pasadas completas al dataset
        imgsz=args.imgsz,        # Resolución de entrada
        batch=args.batch,        # Imágenes por lote
        device=device,           # GPU o CPU
        workers=args.workers,    # Hilos de carga de datos
        patience=args.patience,  # Early stopping
        project=args.project,    # Carpeta raíz de resultados
        name=args.name,          # Subcarpeta del experimento
        resume=args.resume,      # Reanudar si se interrumpió
        plots=True,              # Genera gráficas de métricas (muy útil para el informe)
        verbose=True,            # Muestra detalle por época
    )

    # 5.6 Localizamos el modelo entrenado para indicárselo al usuario.
    #     'best.pt'  -> los pesos con la MEJOR métrica de validación (usar este).
    #     'last.pt'  -> los pesos de la última época.
    carpeta_pesos = Path(args.project) / args.name / "weights"
    mejor_modelo = carpeta_pesos / "best.pt"

    print("\n" + "=" * 70)
    print(" ENTRENAMIENTO FINALIZADO CORRECTAMENTE")
    print("=" * 70)
    print(f"[INFO] Resultados y gráficas en: {Path(args.project) / args.name}")
    print(f"[INFO] MEJOR modelo entrenado  : {mejor_modelo}")
    print("[SIGUIENTE PASO] Copia 'best.pt' a la carpeta 'models/' y ejecuta:")
    print(f"   python scripts/predict.py --model {mejor_modelo} --source <tu_imagen_o_video>")
    print("=" * 70)


# ----------------------------------------------------------------------------
# 6. PUNTO DE ENTRADA DEL PROGRAMA
# ----------------------------------------------------------------------------
# Esta condición asegura que main() solo se ejecute cuando corremos el archivo
# directamente (python scripts/train.py), y NO si alguien lo importa como módulo.
if __name__ == "__main__":
    main()
