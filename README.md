# 🔧 Sistema de Visión Artificial para Detección de Defectos en Piezas de Fundición CNC y Desgaste de Herramientas de Fresado

> Proyecto de detección de objetos en tiempo real basado en **YOLOv8 (Ultralytics)**, orientado al control de calidad automatizado en una línea de producción industrial CNC.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green)
![Ultralytics](https://img.shields.io/badge/Framework-Ultralytics-orange)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

## 👤 Autor

    Nombre      | Carrera |
    Brian Prado | Ing Mecatronica |

> **Asignatura:** Vision Artificial  
> **Docente:** Jose Ramon Navarro   
> **Fecha de entrega:** 19-06-2026

---

## 📑 Tabla de Contenido
1. [Descripción del Proyecto](#-descripción-del-proyecto)
2. [Estructura del Repositorio](#-estructura-del-repositorio)
3. [Requisitos Previos](#-requisitos-previos)
4. [Instalación Paso a Paso](#-instalación-paso-a-paso)
5. [Preparación del Dataset](#-preparación-del-dataset)
6. [Uso: Entrenamiento (`train.py`)](#-uso-entrenamiento-trainpy)
7. [Uso: Inferencia (`predict.py`)](#-uso-inferencia-predictpy)
8. [Caso de Estudio (Aplicación Práctica en Planta)](#-caso-de-estudio-aplicación-práctica-en-planta)
9. [Tecnologías Utilizadas](#-tecnologías-utilizadas)

---

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema de **inspección visual automatizada** capaz de identificar, en tiempo real, dos tipos de problemas comunes en una línea de manufactura CNC:

1. **Defectos en piezas de fundición:** porosidad, grietas, inclusiones, rechupes y rebabas.
2. **Desgaste en herramientas de fresado:** desgaste de flanco y rotura/astillamiento del filo de corte.

El sistema utiliza una red neuronal convolucional de detección de objetos de la familia **YOLO (You Only Look Once)**, concretamente **YOLOv8**, por su excelente equilibrio entre **velocidad** (apto para tiempo real) y **precisión**, características indispensables en un entorno industrial.

El proyecto detecta **9 clases**:

| ID | Clase | Descripción |
|----|-------|-------------|
| 0 | `pieza_conforme` | Pieza correcta, sin defectos (referencia OK) |
| 1 | `porosidad` | Burbujas/poros de gas atrapado en la fundición |
| 2 | `grieta` | Fisura o grieta superficial |
| 3 | `inclusion` | Material extraño incrustado (arena, escoria) |
| 4 | `rechupe` | Cavidad por contracción al solidificar el metal |
| 5 | `rebaba` | Exceso de material en bordes (flash / burr) |
| 6 | `herramienta_ok` | Filo de fresa en buen estado |
| 7 | `desgaste_flanco` | Desgaste en el flanco de la herramienta |
| 8 | `filo_roto` | Astillamiento o rotura del filo de corte |

---

## 📂 Estructura del Repositorio

```
Vision-Artificial/
│
├── datasets/                          # Datos para entrenar y probar el modelo
│   └── defectos_cnc_fundicion/
│       ├── data.yaml                  # Configuración del dataset (rutas + clases)
│       ├── train/                     # ~70% de los datos (entrenamiento)
│       │   ├── images/                #   imagen001.jpg, imagen002.jpg, ...
│       │   └── labels/                #   imagen001.txt, imagen002.txt, ...
│       ├── valid/                     # ~20% de los datos (validación)
│       │   ├── images/
│       │   └── labels/
│       └── test/                      # ~10% de los datos (prueba final)
│           ├── images/
│           └── labels/
│
├── scripts/                          # Código fuente del proyecto
│   ├── train.py                      # Script de ENTRENAMIENTO del modelo
│   └── predict.py                    # Script de INFERENCIA / predicción
│
├── models/                           # Modelos entrenados (pesos .pt)
│   └── best.pt                       # Mejor modelo (copiar aquí tras entrenar)
│
├── evidencias/                       # Resultados con bounding boxes dibujadas
│   ├── imagenes/                     # Capturas de detecciones en imágenes
│   ├── videos/                       # Videos de prueba y resultados
│   └── predicciones/                 # Salida automática de predict.py
│
├── runs/                             # (Auto-generada) logs y gráficas de entrenamiento
│
├── requirements.txt                  # Dependencias de Python del proyecto
├── .gitignore                        # Archivos que no se suben a GitHub
└── README.md                         # Este archivo (documentación)
```

---

## ⚙️ Requisitos Previos

- **Python 3.10 o superior** ([descargar](https://www.python.org/downloads/)).
- **pip** (gestor de paquetes de Python, incluido con Python).
- **Git** ([descargar](https://git-scm.com/downloads)).
- *(Opcional pero recomendado)* Una **GPU NVIDIA con CUDA** para acelerar el entrenamiento. Sin GPU el proyecto funciona igual, solo que más lento.

---

## 🚀 Instalación Paso a Paso

### 1️⃣ Clonar el repositorio
```bash
git clone https://github.com/<TU_USUARIO>/Vision-Artificial.git
cd Vision-Artificial
```

### 2️⃣ Crear y activar un entorno virtual (recomendado)
Aísla las dependencias del proyecto del resto del sistema.

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Instalar las dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Verificar la instalación
```bash
yolo version
```
Si muestra la versión de Ultralytics, ¡todo está listo! ✅

---

## 🗂️ Preparación del Dataset

El modelo necesita imágenes **etiquetadas** (con los defectos marcados mediante cajas). Recomendaciones:

1. **Recolecta imágenes** de las piezas y herramientas (cámara industrial o celular).
2. **Etiqueta** los defectos con [Roboflow](https://roboflow.com/) o [LabelImg](https://github.com/heartexlabs/labelImg), exportando en formato **YOLOv8**.
3. **Coloca** las imágenes y etiquetas en `datasets/defectos_cnc_fundicion/` siguiendo la estructura `train/valid/test`.
4. **Ajusta** la ruta `path:` dentro de `data.yaml` si fuese necesario.

**Formato de cada etiqueta (`.txt`)** — una línea por objeto, valores normalizados de 0 a 1:
```
<class_id> <x_centro> <y_centro> <ancho> <alto>
# Ejemplo: una porosidad (clase 1) centrada en la imagen
1 0.500000 0.500000 0.120000 0.090000
```

> 💡 Cada imagen `imagenXYZ.jpg` debe tener su archivo de etiqueta `imagenXYZ.txt` con el mismo nombre.

---

## 🎓 Uso: Entrenamiento (`train.py`)

Entrena el modelo con tu dataset. Desde la raíz del proyecto:

```bash
# Entrenamiento con valores por defecto (100 épocas, modelo nano)
python scripts/train.py

# Entrenamiento personalizado (ejemplos)
python scripts/train.py --epochs 150 --batch 8 --model yolov8s.pt
python scripts/train.py --device cpu          # forzar CPU
python scripts/train.py --device 0            # forzar GPU 0
```

**Parámetros principales:**

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `--data` | `datasets/.../data.yaml` | Archivo de configuración del dataset |
| `--model` | `yolov8n.pt` | Modelo base (n/s/m/l/x) |
| `--epochs` | `100` | Número de épocas |
| `--imgsz` | `640` | Tamaño de imagen (px) |
| `--batch` | `16` | Imágenes por lote |
| `--device` | `auto` | `auto`, `cpu` o índice de GPU |

Al terminar, el mejor modelo queda en `runs/entrenamiento/defectos_yolov8/weights/best.pt`.

➡️ **Copia ese `best.pt` a la carpeta `models/`** para usarlo en la inferencia:
```bash
# Windows
copy runs\entrenamiento\defectos_yolov8\weights\best.pt models\best.pt
# Linux / macOS
cp runs/entrenamiento/defectos_yolov8/weights/best.pt models/best.pt
```

---

## 🔍 Uso: Inferencia (`predict.py`)

Aplica el modelo entrenado sobre imágenes, carpetas, videos o cámara en vivo. Los resultados con las cajas dibujadas se guardan en `evidencias/predicciones/`.

```bash
# Sobre una sola imagen
python scripts/predict.py --model models/best.pt --source datasets/defectos_cnc_fundicion/test/images/pieza1.jpg

# Sobre una carpeta completa de imágenes
python scripts/predict.py --model models/best.pt --source datasets/defectos_cnc_fundicion/test/images

# Sobre un video
python scripts/predict.py --model models/best.pt --source evidencias/videos/linea_produccion.mp4

# Cámara industrial / webcam en vivo (fuente = 0)
python scripts/predict.py --model models/best.pt --source 0 --show
```

**Parámetros principales:**

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `--model` | `models/best.pt` | Modelo entrenado a usar |
| `--source` | *(obligatorio)* | Imagen, carpeta, video o `0` (cámara) |
| `--conf` | `0.25` | Umbral mínimo de confianza |
| `--iou` | `0.45` | Umbral IoU para eliminar cajas duplicadas |
| `--show` | desactivado | Muestra resultados en pantalla en tiempo real |

---

## 🏭 Caso de Estudio (Aplicación Práctica en Planta)

> Esta sección describe cómo este modelo de IA pasaría de un script en una laptop a un **sistema de inspección 100% integrado** en una línea de producción CNC real.

### 🎯 El Problema a Resolver

En muchas plantas de manufactura, el control de calidad de las piezas fundidas/mecanizadas todavía se realiza de forma **manual y por muestreo**, lo que genera problemas críticos:

- **Lentitud:** un operario no puede inspeccionar el 100% de las piezas a la velocidad de la banda transportadora (que puede mover decenas de piezas por minuto).
- **Imprecisión y subjetividad:** la detección de microporos, microgrietas o el desgaste incipiente de una fresa depende de la vista, la experiencia y el cansancio del inspector. Un mismo defecto puede aceptarse en un turno y rechazarse en otro.
- **Detección tardía del desgaste de herramienta:** cuando una fresa se desgasta, comienza a producir piezas fuera de tolerancia. Si no se detecta a tiempo, se fabrican **lotes completos de piezas defectuosas** antes de que alguien lo note, generando desperdicio (scrap) y costos elevados.
- **Costo y trazabilidad:** el rechazo manual no deja registro digital, dificultando el análisis estadístico de causas raíz (SPC) y la mejora continua.

**Objetivo:** sustituir la inspección manual por un sistema de visión artificial que inspeccione el **100%** de las piezas en tiempo real, de forma **objetiva, trazable y automática**, y que además **anticipe el cambio de herramienta** antes de que produzca piezas malas.

---

### 🛠️ Hardware Propuesto

| Componente | Especificación sugerida | Función en el sistema |
|------------|-------------------------|------------------------|
| **Cámara industrial** | Cámara **GigE Vision** (Basler / FLIR) de 5–12 MP, obturador global (*global shutter*) | Captura imágenes nítidas de la pieza en movimiento sin distorsión (*motion blur*). El cable GigE permite distancias largas y sincronización por trigger. |
| **Óptica (lente)** | Lente de montura C con distancia focal acorde a la distancia de trabajo | Enfoca la región de inspección con la resolución espacial necesaria para ver microdefectos. |
| **Iluminación** | Iluminación LED industrial controlada: **domo difuso** (para superficies brillantes/metálicas) o **luz rasante (low-angle)** para resaltar grietas y rebabas | La iluminación es el factor #1 de éxito: estabiliza el contraste de los defectos y elimina reflejos/sombras variables. |
| **Sensor de presencia** | Sensor fotoeléctrico / encoder en la banda | Genera el *trigger* exacto: avisa cuándo una pieza está en posición frente a la cámara. |
| **Unidad de cómputo (Edge)** | **NVIDIA Jetson Orin / AGX** (GPU industrial embebida) o PC industrial con GPU; alternativamente **Google Coral Edge TPU** para el modelo cuantizado | Ejecuta el modelo YOLOv8 localmente (*edge computing*), con baja latencia y sin depender de la nube. |
| **PLC** | PLC industrial (Siemens S7-1200/1500, Allen-Bradley) | "Cerebro" de la automatización: recibe el veredicto de la IA y controla los actuadores y la banda. |
| **Actuador de rechazo** | **Pistón/actuador neumático** (con válvula y compresor) o **brazo robótico** (cobot tipo UR) | Físicamente retira/desvía la pieza defectuosa de la línea. |
| **HMI / Pantalla** | Panel HMI o monitor industrial | Muestra estadísticas en vivo, alertas y permite al operario supervisar el sistema. |
| **Red / Gabinete** | Switch industrial, gabinete IP65, fuente regulada | Protege la electrónica del ambiente de planta (polvo, vibración, viruta). |

---

### 🔄 Flujo de Funcionamiento (Integración Completa)

```
   ┌──────────────┐   trigger    ┌──────────────┐   imagen      ┌────────────────────┐
   │ Sensor de    │─────────────▶│   Cámara     │──────────────▶│  Unidad Edge (GPU) │
   │ presencia    │              │   GigE +     │   (GigE)      │   Modelo YOLOv8     │
   │ en la banda  │              │ iluminación  │               │  (predict / infer.) │
   └──────────────┘              └──────────────┘               └─────────┬──────────┘
                                                                          │ veredicto
                                                                          │ (OK / DEFECTO + clase)
                                                                          ▼
   ┌──────────────┐   señal      ┌──────────────┐   señal/IO    ┌────────────────────┐
   │  Actuador    │◀─────────────│     PLC      │◀──────────────│  Resultado + clase  │
   │ neumático /  │   descarte   │ (S7/AB)      │  (Ethernet/IP │  (vía OPC-UA/MQTT/   │
   │ brazo robot  │              │              │   o Modbus)   │   E/S digitales)    │
   └──────────────┘              └──────┬───────┘               └────────────────────┘
                                        │
                                        ▼  alertas y estadísticas
                                  ┌──────────────┐
                                  │   HMI /      │  "Cambiar fresa: desgaste detectado"
                                  │  Dashboard   │  "Tasa de defectos: 2.3%"
                                  └──────────────┘
```

**Paso a paso:**

1. **Detección de la pieza:** la pieza avanza sobre la banda transportadora. Al pasar frente a la estación, el **sensor fotoeléctrico** dispara un *trigger*.

2. **Captura sincronizada:** el trigger activa simultáneamente la **iluminación LED** y el **obturador de la cámara GigE**, que captura una imagen nítida de la pieza (gracias al *global shutter*, sin emborronamiento por movimiento).

3. **Procesamiento con YOLOv8 (inferencia en el Edge):** la imagen viaja por el cable GigE a la **unidad de cómputo embebida** (Jetson/PC industrial). Allí se ejecuta la lógica de `predict.py`: el modelo YOLOv8 analiza la imagen en **milisegundos** y devuelve:
   - Las **bounding boxes** de cada defecto encontrado.
   - La **clase** (porosidad, grieta, desgaste_flanco, etc.).
   - La **confianza** de cada detección.
   
   El software aplica entonces una **regla de decisión**: por ejemplo, *"si se detecta cualquier clase de defecto con confianza ≥ 0.5 → pieza NO CONFORME"*.

4. **Comunicación con la maquinaria (IA → PLC):** la unidad Edge envía el veredicto al **PLC** mediante un protocolo industrial (**OPC-UA**, **MQTT**, **Modbus TCP** o simples **señales digitales de E/S**). El mensaje incluye OK/DEFECTO y, opcionalmente, la clase de defecto.

5. **Acción física de descarte:** el **PLC** interpreta la señal y, en el instante exacto en que la pieza llega a la estación de rechazo:
   - Si la pieza es **conforme** → continúa hacia el área de empaque.
   - Si es **defectuosa** → el PLC activa el **actuador neumático** (que empuja la pieza a un contenedor de scrap) o comanda el **brazo robótico** para retirarla y clasificarla por tipo de defecto.

6. **Mantenimiento predictivo de herramienta:** cuando el modelo detecta de forma recurrente las clases `desgaste_flanco` o `filo_roto` en la **herramienta de fresado**, el sistema **alerta en el HMI** que se acerca el fin de vida útil de la fresa, permitiendo programar su cambio **antes** de fabricar piezas fuera de tolerancia. Esto convierte el mantenimiento correctivo en **mantenimiento predictivo**.

7. **Trazabilidad y mejora continua:** cada veredicto se registra (imagen + clase + timestamp) en una base de datos. El **dashboard/HMI** muestra KPIs en vivo (tasa de defectos, defecto más frecuente, OEE), habilitando el **Control Estadístico de Procesos (SPC)** y la mejora continua.

---

### ✅ Beneficios Esperados

- **Inspección del 100%** de la producción (vs. muestreo manual).
- **Objetividad y repetibilidad:** mismo criterio las 24 horas, sin fatiga.
- **Reducción de scrap** gracias a la detección temprana del desgaste de herramienta.
- **Trazabilidad digital** para auditorías y análisis de causa raíz.
- **Retorno de inversión (ROI)** por reducción de reclamos de cliente y reprocesos.

---

## 🧰 Tecnologías Utilizadas

- **[Ultralytics YOLOv8](https://docs.ultralytics.com/)** — modelo de detección de objetos.
- **[PyTorch](https://pytorch.org/)** — motor de deep learning.
- **[OpenCV](https://opencv.org/)** — procesamiento de imágenes y video.
- **Python 3.10+** — lenguaje de programación.

---

> 📌 *Proyecto académico desarrollado con fines educativos para la asignatura de Visión Artificial / Ingeniería Mecatrónica.*
