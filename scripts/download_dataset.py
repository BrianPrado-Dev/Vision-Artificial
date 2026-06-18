# -*- coding: utf-8 -*-
"""
===============================================================================
 download_dataset.py  -  Descarga AUTOMÁTICA de un dataset público de defectos
                         de fundición desde Roboflow Universe.
===============================================================================
 Proyecto : Detección de defectos en piezas de fundición CNC y desgaste en
            herramientas de fresado en una línea de producción.
 Autor    : Brian Prado
 Curso    : Visión Artificial
-------------------------------------------------------------------------------
 ¿QUÉ HACE ESTE SCRIPT?
 Usa el cliente oficial de Roboflow para DESCARGAR un dataset ya etiquetado
 (imágenes + etiquetas en formato YOLOv8) y dejarlo listo dentro de la carpeta
 'datasets/' del proyecto, para poder entrenar de inmediato con train.py.

 ------------------------------------------------------------------------------
 REQUISITO IMPORTANTE: API KEY DE ROBOFLOW (GRATIS)
 ------------------------------------------------------------------------------
 Roboflow exige una clave personal (gratuita) para descargar datasets:
   1) Crea una cuenta en https://app.roboflow.com (es gratis).
   2) Entra a Settings -> "API Keys" y copia tu "Private API Key".
   3) Pásala al script de UNA de estas dos formas (NO la subas a GitHub):

      OPCIÓN A (recomendada) - variable de entorno:
        # Windows (PowerShell):
        $env:ROBOFLOW_API_KEY = "TU_API_KEY"
        python scripts/download_dataset.py

        # Linux / macOS:
        export ROBOFLOW_API_KEY="TU_API_KEY"
        python scripts/download_dataset.py

      OPCIÓN B - argumento directo:
        python scripts/download_dataset.py --api-key TU_API_KEY

      OPCIÓN C - archivo .env (YA CONFIGURADO en este proyecto):
        Coloca en un archivo '.env' en la raíz del proyecto la línea:
          ROBOFLOW_API_KEY=tu_clave
        El archivo .env está en .gitignore, por lo que NUNCA se sube a GitHub.

 ------------------------------------------------------------------------------
 ¿CÓMO ELEGIR OTRO DATASET? (el "snippet" de descarga)
 ------------------------------------------------------------------------------
 Cada dataset en Roboflow Universe tiene un botón "Download Dataset" que muestra
 un fragmento de código con TRES identificadores:
        rf.workspace("WORKSPACE").project("PROJECT").version(VERSION)
 Solo copia esos tres valores y pásalos con --workspace, --project y --version.

 DATASETS PÚBLICOS DE DETECCIÓN (bounding boxes) ENCONTRADOS COMO EJEMPLO:
   * Casting detection  -> workspace="new-workspace-kmz9b"  project="casting-detection-leboi"
   * Defect Detection   -> workspace="test-i6bsm"           project="defect-detection-r9fgh"
   (Los nombres de las clases vienen DENTRO del dataset descargado; pueden no
    coincidir con las 9 clases del data.yaml plantilla de este proyecto.)

 USO BÁSICO (desde la raíz del proyecto, con la API key ya configurada):
   python scripts/download_dataset.py
 USO CON OTRO DATASET:
   python scripts/download_dataset.py --workspace test-i6bsm --project defect-detection-r9fgh
===============================================================================
"""

# ----------------------------------------------------------------------------
# 1. IMPORTACIÓN DE LIBRERÍAS ESTÁNDAR
# ----------------------------------------------------------------------------
import argparse           # Lee parámetros desde la terminal
import os                 # Para leer variables de entorno (la API key)
import sys                # Para terminar el programa con un código de error
from pathlib import Path  # Manejo de rutas multiplataforma


# ----------------------------------------------------------------------------
# 2. LECTURA DE ARGUMENTOS DESDE LA LÍNEA DE COMANDOS
# ----------------------------------------------------------------------------
def parse_arguments():
    """Define los parámetros configurables de la descarga."""
    parser = argparse.ArgumentParser(
        description="Descarga un dataset público de defectos desde Roboflow Universe."
    )

    # --- Credencial ---
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key privada de Roboflow. Si se omite, se lee de la variable "
             "de entorno ROBOFLOW_API_KEY."
    )

    # --- Identificadores del dataset (el 'snippet' de Roboflow) ---
    parser.add_argument(
        "--workspace", type=str, default="new-workspace-kmz9b",
        help="ID del workspace de Roboflow (parte del snippet de descarga)."
    )
    parser.add_argument(
        "--project", type=str, default="casting-detection-leboi",
        help="ID del proyecto de Roboflow (parte del snippet de descarga)."
    )
    parser.add_argument(
        "--version", type=int, default=None,
        help="Número de versión del dataset. Si se omite, se intenta detectar "
             "automáticamente la versión más reciente."
    )

    # --- Formato y destino ---
    parser.add_argument(
        "--format", type=str, default="yolov8",
        help="Formato de exportación. Para este proyecto usa 'yolov8'."
    )
    parser.add_argument(
        "--output-dir", type=str, default="datasets/roboflow_fundicion",
        help="Carpeta destino donde se guardará el dataset descargado."
    )

    return parser.parse_args()


# ----------------------------------------------------------------------------
# 3. OBTENCIÓN SEGURA DE LA API KEY
# ----------------------------------------------------------------------------
def cargar_api_key_desde_dotenv():
    """
    Lee ROBOFLOW_API_KEY desde un archivo '.env' en la raíz del proyecto.

    Ventaja de seguridad: así la clave NO se escribe en el código (que sí se
    sube a GitHub) y, a la vez, no hay que exportarla a mano en cada terminal.
    El archivo .env está incluido en .gitignore, por lo que NUNCA se sube.
    """
    # La raíz del proyecto es la carpeta padre de 'scripts/'.
    raiz_proyecto = Path(__file__).resolve().parent.parent
    archivo_env = raiz_proyecto / ".env"
    if not archivo_env.exists():
        return None
    try:
        for linea in archivo_env.read_text(encoding="utf-8").splitlines():
            linea = linea.strip()
            # Ignoramos líneas vacías y comentarios (#).
            if not linea or linea.startswith("#") or "=" not in linea:
                continue
            clave, _, valor = linea.partition("=")
            if clave.strip() == "ROBOFLOW_API_KEY":
                # Quitamos espacios y comillas que pudieran rodear al valor.
                return valor.strip().strip('"').strip("'")
    except Exception:
        return None
    return None


def obtener_api_key(api_key_arg):
    """
    Devuelve la API key priorizando:
      1) El argumento --api-key (si se pasó).
      2) La variable de entorno ROBOFLOW_API_KEY.
      3) El archivo .env en la raíz del proyecto.
    Si no hay ninguna, detiene el programa con instrucciones claras.
    """
    api_key = (
        api_key_arg
        or os.environ.get("ROBOFLOW_API_KEY")
        or cargar_api_key_desde_dotenv()
    )
    if not api_key:
        print(
            "\n[ERROR] No se encontró la API key de Roboflow.\n"
            "        Configúrala de una de estas formas (ninguna se sube a GitHub):\n"
            "          - Archivo .env : ROBOFLOW_API_KEY=tu_clave\n"
            "          - PowerShell   : $env:ROBOFLOW_API_KEY = \"tu_clave\"\n"
            "          - Linux/macOS  : export ROBOFLOW_API_KEY=\"tu_clave\"\n"
            "          - Argumento    : --api-key tu_clave\n"
            "        (Obtén una gratis en https://app.roboflow.com -> Settings -> API Keys)\n"
        )
        sys.exit(1)
    return api_key


# ----------------------------------------------------------------------------
# 4. RESOLUCIÓN DE LA VERSIÓN DEL DATASET
# ----------------------------------------------------------------------------
def resolver_version(proyecto, version_arg):
    """
    Decide qué versión del dataset descargar.
    - Si el usuario indicó --version, se respeta ese número.
    - Si no, se consulta la lista de versiones disponibles y se usa la más alta.
    - Si no se puede consultar, se usa la versión 1 como último recurso.
    """
    if version_arg is not None:
        print(f"[INFO] Usando la versión indicada por el usuario: v{version_arg}")
        return version_arg

    # Intentamos autodetectar la versión más reciente.
    try:
        versiones = proyecto.versions()  # Lista de objetos 'Version'
        numeros = []
        for v in versiones:
            # El atributo .version puede ser un entero o una cadena tipo ".../3";
            # nos quedamos siempre con el número final.
            crudo = getattr(v, "version", None)
            if crudo is None:
                continue
            numeros.append(int(str(crudo).rstrip("/").split("/")[-1]))
        if numeros:
            ultima = max(numeros)
            print(f"[INFO] Versión más reciente detectada automáticamente: v{ultima}")
            return ultima
    except Exception as e:
        print(f"[AVISO] No se pudo autodetectar la versión ({e}). Se usará v1.")

    return 1


# ----------------------------------------------------------------------------
# 5. RESUMEN DEL DATASET DESCARGADO (clases y archivo de configuración)
# ----------------------------------------------------------------------------
def resumir_dataset(carpeta_destino):
    """
    Lee el data.yaml que Roboflow incluye dentro del dataset descargado y
    muestra las clases reales y la ruta del archivo, útil para entrenar luego.
    """
    yaml_path = Path(carpeta_destino) / "data.yaml"
    if not yaml_path.exists():
        print(f"[AVISO] No se encontró data.yaml en {carpeta_destino}.")
        return None

    try:
        import yaml  # PyYAML, ya incluido en requirements.txt
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        clases = config.get("names", [])
        print("\n----- CONTENIDO DEL DATASET DESCARGADO -----")
        print(f"  Archivo de configuración : {yaml_path}")
        print(f"  Número de clases (nc)    : {config.get('nc', len(clases))}")
        print(f"  Clases                   : {clases}")
        print("--------------------------------------------")
    except Exception as e:
        print(f"[AVISO] No se pudo leer el data.yaml descargado: {e}")
        return None

    return yaml_path


# ----------------------------------------------------------------------------
# 6. FUNCIÓN PRINCIPAL
# ----------------------------------------------------------------------------
def main():
    # 6.1 Leemos parámetros y la API key.
    args = parse_arguments()

    print("=" * 70)
    print(" DESCARGA DE DATASET PÚBLICO - Roboflow Universe")
    print("=" * 70)

    api_key = obtener_api_key(args.api_key)

    # 6.2 Importamos el cliente de Roboflow (con mensaje claro si falta).
    try:
        from roboflow import Roboflow
    except ImportError:
        print(
            "\n[ERROR] No está instalado el paquete 'roboflow'.\n"
            "        Instálalo con:  pip install roboflow\n"
            "        (o  pip install -r requirements.txt)\n"
        )
        sys.exit(1)

    # 6.3 Nos aseguramos de que exista la carpeta destino.
    destino = Path(args.output_dir)
    destino.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Workspace : {args.workspace}")
    print(f"[INFO] Proyecto  : {args.project}")
    print(f"[INFO] Formato   : {args.format}")
    print(f"[INFO] Destino   : {destino.resolve()}\n")

    # 6.4 Nos conectamos a Roboflow y seleccionamos el proyecto.
    try:
        rf = Roboflow(api_key=api_key)
        proyecto = rf.workspace(args.workspace).project(args.project)
    except Exception as e:
        print(
            f"\n[ERROR] No se pudo acceder al proyecto en Roboflow:\n  {e}\n"
            "        Verifica:\n"
            "          - Que tu API key sea correcta.\n"
            "          - Que --workspace y --project coincidan EXACTAMENTE con el\n"
            "            'snippet' que aparece en la página del dataset.\n"
        )
        sys.exit(1)

    # 6.5 Resolvemos la versión y descargamos.
    version_num = resolver_version(proyecto, args.version)
    print(f"\n[INFO] Descargando dataset (v{version_num}) en formato '{args.format}'...")
    try:
        version_obj = proyecto.version(version_num)
        # download() descarga y descomprime el dataset en 'location'.
        version_obj.download(args.format, location=str(destino))
    except Exception as e:
        print(
            f"\n[ERROR] Falló la descarga del dataset:\n  {e}\n"
            "        Sugerencia: abre la página del dataset en Roboflow Universe,\n"
            "        pulsa 'Download Dataset' -> 'show download code' y copia los\n"
            "        valores exactos de workspace, project y version.\n"
        )
        sys.exit(1)

    # 6.6 Mostramos un resumen y los siguientes pasos.
    yaml_path = resumir_dataset(destino)

    print("\n" + "=" * 70)
    print(" DESCARGA COMPLETADA CORRECTAMENTE")
    print("=" * 70)
    if yaml_path:
        print("[SIGUIENTE PASO] Entrena directamente con el dataset descargado:")
        print(f"   python scripts/train.py --data {yaml_path}")
    print(
        "\n[NOTA] Las clases del dataset descargado son las REALES de ese dataset\n"
        "       público y pueden diferir de las 9 clases del data.yaml plantilla\n"
        "       de este proyecto. Para el entrenamiento usa el data.yaml que vino\n"
        "       DENTRO de la carpeta descargada (el que se indica arriba)."
    )
    print("=" * 70)


# ----------------------------------------------------------------------------
# 7. PUNTO DE ENTRADA DEL PROGRAMA
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
