import os
import ffmpeg
import tempfile
from typing import Iterator, TextIO
import torch
import whisperx
import gc
import shutil

class SubWhiz:
    def __init__(
            self,
            model,
            output_dir: str = ".",
            language: str = "es",
            verbose: bool = False,
    ):
        """
        Inicializa la herramienta con el modelo whisperx.

        :param model: Modelo whisperx pre-entrenado
        :param output_dir: Directorio de salida para los archivos generados
        :param language: Lenguaje para la transcripción
        :param verbose: Si True, imprime información adicional durante la ejecución
        """
        print("version: 0.0.1")
        self.output_dir = output_dir
        self.language = language
        self.verbose = verbose

        os.makedirs(self.output_dir, exist_ok=True)
        self.model = model


    def process_videos(self, video_paths: list, output_srt: bool = False,
                       srt_only: bool = False, task: str = "transcribe",
                       custom_font_dir: str = None,
                       custom_font_name: str = None,
                       word_by_word: bool = False):
        """
        Procesa la lista de videos, genera o quema los subtítulos.

        :param video_paths: lista de rutas a los videos
        :param output_srt: si True, los .srt se guardarán en el directorio de salida
        :param srt_only: si True, solo se generarán los .srt y NO se quema subtítulo
        :param task: 'transcribe' o 'translate' para el modelo
        :param custom_font_dir: ruta a la carpeta donde está tu fuente TTF
        :param custom_font_name: nombre interno de la fuente TTF
        :param word_by_word: si True, genera subtítulos palabra por palabra
        """
        audio_paths = self.extract_audio(video_paths)
        subtitles = self.generate_subtitles(audio_paths, output_srt, srt_only, task, word_by_word)

        if srt_only:
            return subtitles

        self.add_subtitles_to_videos(
            subtitles,
            custom_font_dir=custom_font_dir,
            custom_font_name=custom_font_name
        )
        return subtitles

    def extract_audio(self, video_paths: list):
        temp_dir = tempfile.gettempdir()
        audio_paths = {}

        for path in video_paths:
            if self.verbose:
                print(f"Extracting audio from {self.get_filename(path)}...")
            output_path = os.path.join(temp_dir, f"{self.get_filename(path)}.wav")

            ffmpeg.input(path).output(
                output_path,
                acodec="pcm_s16le", ac=1, ar="16k"
            ).run(quiet=not self.verbose, overwrite_output=True)

            audio_paths[path] = output_path

        return audio_paths

    def generate_subtitles(self, audio_paths: dict, output_srt: bool,
                           srt_only: bool, task: str, word_by_word: bool):
        """
        Genera los subtítulos utilizando whisperx para una alineación precisa.

        :param audio_paths: Diccionario con rutas de audio
        :param output_srt: Si True, guarda los archivos .srt en el directorio de salida
        :param srt_only: Si True, solo genera los .srt sin quemar los subtítulos
        :param task: 'transcribe' o 'translate' para el modelo
        :param word_by_word: Si True, genera subtítulos palabra por palabra
        :return: Diccionario con rutas de los archivos .srt generados
        """
        subtitles_path = {}

        for path, audio_path in audio_paths.items():
            srt_path = self.output_dir if output_srt else tempfile.gettempdir()
            srt_path = os.path.join(srt_path, f"{self.get_filename(path)}.srt")

            if self.verbose:
                print(f"Generando subtítulos para {self.get_filename(path)}...")

            # Transcribir el audio
            result = self.model.transcribe(audio_path, task=task, language=self.language)

            # Alinear los timestamps
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.model.device)
            result = whisperx.align(result["segments"], model_a, metadata, audio_path, self.model.device,
                                    return_char_alignments=False)

            # Guardar el archivo SRT con los timestamps precisos
            with open(srt_path, "w", encoding="utf-8") as srt:
                if word_by_word:
                    self.write_srt_word_by_word(result["segments"], srt)
                else:
                    self.write_srt(result["segments"], srt)

            subtitles_path[path] = srt_path

            # Liberar memoria si es necesario
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

        return subtitles_path

    def add_subtitles_to_videos(self, subtitles: dict,
                                custom_font_dir: str = None,
                                custom_font_name: str = None):
        """
        Quita el audio del video original, agrega el filtro 'subtitles'
        y exporta el video con subtítulos quemados.
        Si se especifican custom_font_dir y custom_font_name, se usarán
        para buscar una fuente personalizada.
        """
        for path, srt_path in subtitles.items():
            out_path = os.path.join(self.output_dir, f"{self.get_filename(path)}.mp4")

            if self.verbose:
                print(f"Adding subtitles to {self.get_filename(path)}...")

            video = ffmpeg.input(path)
            audio = video.audio

            # Preparamos opciones para el filtro 'subtitles'
            # force_style te permite definir fuente, tamaño, color, etc.
            # Si tenemos fuente personalizada:
            if custom_font_dir and custom_font_name:
                filter_sub = video.filter(
                    "subtitles",
                    srt_path,
                    fontsdir=custom_font_dir,
                    force_style=(
                        f"FontName={custom_font_name},"
                        "FontSize=16,"
                        "BackColour= & H80000000,"
                        "Spacing = 0.2," 
                        "Outline = 0,"
                        "Shadow = 0.75,"
                        "MarginV=70"
                    )
                )
            else:
                # Si no se especifica, usar Arial por defecto.
                filter_sub = video.filter(
                    "subtitles",
                    srt_path,
                    force_style=(
                        "FontName=Arial,"
                        "FontSize=16,"
                        "BackColour= & H80000000,"
                        "Spacing = 0.2," 
                        "Outline = 0,"
                        "Shadow = 0.75,"
                        "MarginV=70"
                    )
                )

            try:
                temp_out_path = os.path.join(tempfile.gettempdir(), f"{self.get_filename(path)}_temp.mp4")
                final_out_path = os.path.join(self.output_dir, f"{self.get_filename(path)}.mp4")
                # Generar video subtitulado en ruta temporal
                ffmpeg.concat(filter_sub, audio, v=1, a=1).output(
                    temp_out_path
                ).run(
                    quiet=not self.verbose,
                    overwrite_output=True
                )

                # Mover el archivo temporal al directorio final
                shutil.move(temp_out_path, final_out_path)

                if self.verbose:
                    print(f"Saved subtitled video to {os.path.abspath(final_out_path)}.")
            except ffmpeg.Error as e:
                print("ffmpeg error:", e.stderr.decode('utf8'))
                raise

    @staticmethod
    def write_srt(transcript: Iterator[dict], file: TextIO):
        """
        Escribe un archivo SRT utilizando los timestamps precisos de whisperx.

        :param transcript: Iterador de segmentos de transcripción alineados
        :param file: Objeto de archivo donde se escribirá el SRT
        """
        line_counter = 1
        for segment in transcript:
            text = segment["text"].strip().replace('-->', '->')
            start_time = segment['start']
            end_time = segment['end']

            print(f"{line_counter}", file=file)
            print(
                f"{SubWhiz.format_timestamp(start_time, always_include_hours=True)} --> "
                f"{SubWhiz.format_timestamp(end_time, always_include_hours=True)}",
                file=file,
            )
            print(text, file=file)
            print("", file=file)
            line_counter += 1

    @staticmethod
    def write_srt_word_by_word(transcript: Iterator[dict], file: TextIO):
        """
        Escribe un archivo SRT palabra por palabra utilizando los timestamps precisos de whisperx.

        :param transcript: Iterador de segmentos de transcripción alineados
        :param file: Objeto de archivo donde se escribirá el SRT
        """
        line_counter = 1
        for segment in transcript:
            words = segment.get("words", [])
            for word_info in words:
                word = word_info["word"].strip().replace('-->', '->')
                start_time = word_info['start']
                end_time = word_info['end']

                print(f"{line_counter}", file=file)
                print(
                    f"{SubWhiz.format_timestamp(start_time, always_include_hours=True)} --> "
                    f"{SubWhiz.format_timestamp(end_time, always_include_hours=True)}",
                    file=file,
                )
                print(word, file=file)
                print("", file=file)
                line_counter += 1

    @staticmethod
    def format_timestamp(seconds: float, always_include_hours: bool = False):
        assert seconds >= 0, "non-negative timestamp expected"
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        secs = milliseconds // 1_000
        milliseconds -= secs * 1_000

        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
        return f"{hours_marker}{minutes:02d}:{secs:02d},{milliseconds:03d}"

    @staticmethod
    def get_filename(path):
        return os.path.splitext(os.path.basename(path))[0]