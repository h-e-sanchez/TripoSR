# triposr — pipeline local imagen→3D en CPU

> *De una foto a un modelo `.glb` en ~34 segundos, corriendo 100 % en CPU: generación 3D local sin GPU NVIDIA, sin CUDA y sin servicios cloud.*

Adaptamos [TripoSR](https://github.com/VAST-AI-Research/TripoSR) para ejecutarse en hardware de escritorio estándar (Windows 10, CPU Intel, GPU AMD RX 6600), con interfaz web Gradio y CLI. Este repositorio es un fork de trabajo del [upstream oficial](https://github.com/VAST-AI-Research/TripoSR) con los ajustes necesarios para Windows + CPU y una UI ampliada.

## Características de ingeniería

- **Inferencia 100 % CPU:** parche en `tsr/models/isosurface.py` que reemplaza `torchmcubes` (requiere compilar C++/CUDA) por `PyMCubes` con wheel precompilado — cero compilación, cero toolchain.
- **Dependencias pineadas y reproducibles:** `requirements-windows-cpu.txt` congela el combo `gradio 4.44.1 + fastapi 0.115.6 + starlette 0.41.3 + pydantic 2.10.6`; versiones modernas de starlette/pydantic rompen la UI de Gradio 4.x.
- **Entorno desacoplado de la nube:** el venv vive en `%USERPROFILE%\venvs\triposr`, fuera de carpetas sincronizadas (OneDrive/Drive), evitando la sincronización de miles de archivos de paquetes.
- **Doble vía de ejecución:** interfaz web para uso interactivo y CLI para automatización por lotes.

## Guía de uso rápido

### Interfaz web (recomendada)

Doble clic en `launch-web.bat`, o desde la terminal:

```batch
launch-web.bat
```

El script valida el entorno y levanta la UI en **http://127.0.0.1:7860**. Sube una imagen, el fondo se remueve automáticamente y descargas el modelo como `.glb` (color por vértice).

Controles de calidad disponibles en la UI:

- **Resolución Marching Cubes** (32–512, default 256): detalle geométrico de la malla; más alto = más polígonos y más tiempo (sobre ~320 el retorno disminuye).
- **Foreground Ratio** (0.5–1.0): encuadre del objeto antes de inferir; probar 0.75–0.90 puede mejorar la reconstrucción.
- **Bake de textura UV** (512–4096 px): en vez de color por vértice, hornea un atlas de textura y entrega un ZIP con `OBJ + MTL + PNG` listo para arrastrar a Blender (la textura carga sola). El rasterizado corre por OpenGL sobre la GPU AMD.
- **Estado**: reporta tiempo de generación y conteo de caras/vértices de cada corrida.

Equivalente manual:

```powershell
& "$env:USERPROFILE\venvs\triposr\Scripts\python.exe" gradio_app.py
```

### CLI (automatización)

```powershell
& "$env:USERPROFILE\venvs\triposr\Scripts\python.exe" run.py <imagen> --device cpu --output-dir output --model-save-format glb
```

Salida en `output/0/mesh.glb`. Rendimiento verificado: ~34 s de inferencia por imagen.

### Reinstalación del entorno

```powershell
python -m venv "$env:USERPROFILE\venvs\triposr"
& "$env:USERPROFILE\venvs\triposr\Scripts\pip.exe" install torch --index-url https://download.pytorch.org/whl/cpu
& "$env:USERPROFILE\venvs\triposr\Scripts\pip.exe" install -r requirements-windows-cpu.txt
& "$env:USERPROFILE\venvs\triposr\Scripts\pip.exe" install onnxruntime
```

> **Importante:** usa siempre el Python del venv y no actualices los paquetes pineados del stack web.

## Cambios respecto al upstream

Diff completo contra `VAST-AI-Research/TripoSR@main` — cinco archivos:

| Archivo | Cambio |
|:---|:---|
| `tsr/models/isosurface.py` | `torchmcubes` → `PyMCubes` (extracción de malla sin compilar C++/CUDA) |
| `requirements-windows-cpu.txt` | **Nuevo** — pins verificados para Windows + CPU |
| `gradio_app.py` | UI en español con tooltips, slider MC hasta 512, bake de textura UV (ZIP `OBJ+MTL+PNG`) y caja de estado con métricas |
| `launch-web.bat` | **Nuevo** — lanzador con validación de entorno |
| `README.md` | Este documento (el original se conserva en el upstream) |

## Créditos y referencias

- **[TripoSR](https://github.com/VAST-AI-Research/TripoSR)** — Tripo AI × Stability AI. Reconstrucción 3D feedforward desde una sola imagen. Licencia MIT. Paper: [arXiv:2403.02151](https://arxiv.org/abs/2403.02151).
- **[TRELLIS.2](https://github.com/microsoft/TRELLIS.2)** — Microsoft Research. Lo evaluamos como motor principal por su calidad PBR, pero requiere GPU NVIDIA con 24 GB de VRAM, CUDA y Linux. Para esa calidad sin hardware dedicado, la ruta es su [Space en Hugging Face](https://huggingface.co/spaces/microsoft/TRELLIS.2) o GPU cloud.

Los pesos del modelo se descargan automáticamente desde Hugging Face en la primera ejecución.

---

*Optimizamos para lo que el hardware disponible puede entregar hoy: iteración local rápida primero, calidad cloud cuando el proyecto lo exija.*
