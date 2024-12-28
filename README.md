# Video Subtitle Tool

Video Subtitle Tool es una utilidad en Python para generar subtítulos utilizando el modelo **WhisperX**. Permite extraer audios, transcribir y alinear textos, y quemar subtítulos en los videos. Además, soporta configuraciones personalizadas de fuentes para los subtítulos.

## Características

- **Extracción de audio** de videos para procesamiento posterior.
- **Generación de subtítulos** en formato SRT.
- **Quemado de subtítulos** directamente en los videos.
- Soporte para **subtítulos palabra por palabra**.
- Opciones de configuración para fuentes personalizadas.
- Basado en **WhisperX** para alineaciones de tiempo precisas.

## Instalación

### Requisitos previos

Asegúrate de tener instalado:

- Python 3.9 o superior
- ffmpeg (puedes instalarlo con tu gestor de paquetes, como `apt`, `brew` o descargándolo desde el [sitio oficial](https://ffmpeg.org/)).

### Instalación con Poetry

Clona el repositorio y navega a la carpeta del proyecto:

```bash
git clone https://github.com/SebasCrucer/subwhiz
cd video_subtitle_tool
```

Instala las dependencias usando Poetry:

```bash
poetry install
```

Activa el entorno virtual:

```bash
poetry shell
```

## Uso

### Ejemplo básico

```python
from video_subtitle_tool import VideoSubtitleTool

# Crear instancia de la herramienta
tool = VideoSubtitleTool(output_dir="output", language="es", verbose=True)

# Lista de videos a procesar
video_paths = ["video1.mp4", "video2.mp4"]

# Procesar videos
subtitles = tool.process_videos(video_paths, output_srt=True, srt_only=False, task="transcribe")

print("Subtítulos generados:", subtitles)
```

### Opciones avanzadas

- **Generar solo subtítulos SRT:**

  ```python
  subtitles = tool.process_videos(video_paths, output_srt=True, srt_only=True)
  ```

- **Quemar subtítulos con fuente personalizada:**

  ```python
  tool.process_videos(
      video_paths,
      custom_font_dir="/path/to/fonts",
      custom_font_name="MyCustomFont"
  )
  ```

- **Generar subtítulos palabra por palabra:**

  ```python
  tool.process_videos(video_paths, word_by_word=True)
  ```

## Dependencias

- `ffmpeg-python`
- `torch`
- `numpy`
- `whisperx` (instalado desde GitHub)

Estas dependencias se instalan automáticamente con Poetry.

## Contribuir

1. Haz un fork del repositorio.
2. Crea una rama para tu función o arreglo:

   ```bash
   git checkout -b mi-nueva-funcionalidad
   ```

3. Realiza tus cambios y haz commit:

   ```bash
   git commit -m "Agrega mi nueva funcionalidad"
   ```

4. Envía un pull request.

## Licencia

Este proyecto está licenciado bajo la [MIT License](LICENSE).

## Autores

- [SebasCrucer](https://github.com/SebasCrucer)

## Notas

Este proyecto requiere hardware con capacidad de procesamiento CUDA para obtener el mejor rendimiento, aunque también puede ejecutarse en CPU.

