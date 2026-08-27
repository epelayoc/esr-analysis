%%writefile esr_pipeline.py


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
esr_pipeline.py
================

Pipeline unificado end-to-end para el análisis de ESRs (Evaluation Summary
Reports) de Horizon Europe. Une en un único script tres etapas que antes
vivían en ficheros independientes:

    1. EXTRACCIÓN   (antes ``esr_zips.py``)
       ZIP(s) -> PDFs -> datos estructurados (Excel/CSV/Parquet), con
       anonimización y cifrado AES-GCM opcionales de las columnas sensibles.

    2. CLASIFICACIÓN (antes ``esr_class.py``)
       Excel de ESRs -> extracción de frases negativas de los criterios de
       evaluación -> clasificación por similitud de embeddings contra un
       conjunto de topics predefinidos -> Excel con hojas por bloque
       (Excellence / Impact / Implementation) y resúmenes agrupados.

    3. RESUMEN LLM  (antes ``esr_summary.py``)
       Excel de resúmenes por bloque -> resumen en lenguaje natural por
       "Activity"/"Call" mediante un LLM pequeño (p.ej. Gemma), en lotes.

Las tres etapas se orquestan mediante ``ESRMasterPipeline``, que permite
ejecutar el proceso completo o solo una parte de él (por ejemplo, partir de
un Excel de extracción ya existente y solo clasificar + resumir, o solo
clasificar sin resumir). Cada etapa mantiene las clases originales (con
pequeños renombrados para evitar colisiones de nombres, ver más abajo) y
sus métodos se comportan exactamente igual que en los scripts originales.

Renombrados realizados para unificar el módulo
-----------------------------------------------
- ``ESRPipeline`` (etapa de extracción, en ``esr_zips.py``)
      -> ``ESRExtractionPipeline``
- ``ESRPipeline`` (etapa de clasificación, en ``esr_class.py``)
      -> ``ESRClassificationPipeline``
- ``ESRIO`` (etapa de clasificación, en ``esr_class.py``)
      -> ``ESRClassificationIO``
- ``ExcelSummaryProcessor`` (etapa de resumen, en ``esr_summary.py``)
      -> se mantiene, pero también se expone como ``ESRSummaryProcessor``
         para mayor coherencia de nomenclatura (alias, no rompe nada).

Todas las dependencias "pesadas" u opcionales (Google Colab, Kaggle
Secrets, ``cryptography``, ``nltk``, ``sentence-transformers``, ``torch`` /
``transformers`` / ``bitsandbytes``) se importan de forma perezosa/opcional:
si no están instaladas, solo falla la etapa que las necesita, no el resto
del pipeline. Esto permite, por ejemplo, ejecutar solo la extracción de
PDFs en una máquina sin GPU ni modelos de embeddings instalados.

Uso por línea de comandos
--------------------------
    # Pipeline completo: ZIPs -> extracción -> clasificación -> resumen LLM
    python esr_pipeline.py full --zips a.zip b.zip --output-dir ./out

    # Solo extracción
    python esr_pipeline.py extract --zips a.zip --output-dir ./out --export out/esr_data.xlsx

    # Solo clasificación, partiendo de un Excel ya extraído
    python esr_pipeline.py classify --input out/esr_data.xlsx --output out/classified.xlsx

    # Solo resumen LLM, partiendo del Excel clasificado
    python esr_pipeline.py summarize --input out/classified.xlsx --output out/classified_with_summary.xlsx

También puede usarse de forma programática importando ``ESRMasterPipeline``
o cualquiera de las clases individuales de cada etapa.
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import re
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from PyPDF2 import PdfReader

# ---------------------------------------------------------------------------
# Logging (único logger compartido por las 3 etapas)
# ---------------------------------------------------------------------------
logger = logging.getLogger("esr_pipeline")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# tqdm (barra de progreso). Se intenta la variante "auto" (funciona igual en
# notebook y en terminal); si no está instalada, se usa un passthrough.
# ---------------------------------------------------------------------------
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

# ---------------------------------------------------------------------------
# Integración opcional con Google Colab (Drive, subida de ficheros, Secrets).
# El pipeline funciona igual sin Colab; simplemente estas utilidades quedan
# deshabilitadas y hay que pasar rutas locales explícitas.
# ---------------------------------------------------------------------------
try:
    from google.colab import userdata as colab_userdata  # type: ignore
    from google.colab import drive as colab_drive  # type: ignore
    from google.colab import files as colab_files  # type: ignore
    IN_COLAB = True
except ImportError:  # pragma: no cover
    colab_userdata = None
    colab_drive = None
    colab_files = None
    IN_COLAB = False

# Cargar el token de Hugging Face si estamos en Colab y existe en los Secrets
if IN_COLAB and colab_userdata is not None:
    try:
        os.environ["HF_TOKEN"] = colab_userdata.get('HF_TOKEN')
    except Exception:
        pass
        
# ---------------------------------------------------------------------------
# cryptography (AES-GCM), usado solo por la etapa de extracción para cifrar
# columnas sensibles. Opcional: si no está instalado, simplemente no se puede
# cifrar/descifrar, pero el resto del pipeline funciona con normalidad.
# ---------------------------------------------------------------------------
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:  # pragma: no cover
    AESGCM = None  # type: ignore

# ---------------------------------------------------------------------------
# nltk (tokenización de frases), usado solo por la etapa de clasificación.
# ---------------------------------------------------------------------------
try:
    import nltk
except ImportError:  # pragma: no cover
    nltk = None  # type: ignore


def _ensure_nltk_punkt() -> None:
    """Descarga (si hace falta) el tokenizador de frases de NLTK."""
    if nltk is None:
        raise ImportError(
            "El paquete 'nltk' es necesario para la etapa de clasificación. "
            "Instálalo con: pip install nltk"
        )
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:  # noqa: BLE001 - la descarga es best-effort
        pass


# ---------------------------------------------------------------------------
# sentence-transformers (embeddings de topics), usado solo por la etapa de
# clasificación.
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore


def _ensure_sentence_transformers() -> None:
    if SentenceTransformer is None or util is None:
        raise ImportError(
            "El paquete 'sentence-transformers' es necesario para la etapa "
            "de clasificación. Instálalo con: pip install sentence-transformers"
        )


# ---------------------------------------------------------------------------
# torch / transformers / bitsandbytes, usados solo por la etapa de resumen
# LLM. Se importan de forma perezosa dentro de LLMSummarizerEngine para no
# obligar a instalarlos si solo se quieren usar las etapas de extracción o
# clasificación.
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    _TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    BitsAndBytesConfig = None  # type: ignore
    _TRANSFORMERS_AVAILABLE = False


def _ensure_transformers() -> None:
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "Los paquetes 'torch' y 'transformers' son necesarios para la "
            "etapa de resumen LLM. Instálalos con: "
            "pip install torch transformers accelerate bitsandbytes"
        )


def _maybe_login_huggingface(env_var: str = "HF_TOKEN") -> None:
    """Inicia sesión en el Hugging Face Hub si hay un token disponible.

    Busca el token, por orden, en: variable de entorno, Kaggle Secrets.
    Es un "best effort": si no encuentra token o falla el login, no lanza
    ninguna excepción (algunos modelos no requieren autenticación).
    """
    hf_token = os.environ.get(env_var)

    if not hf_token:
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore

            hf_token = UserSecretsClient().get_secret(env_var)
        except Exception:  # noqa: BLE001
            hf_token = None

    if not hf_token:
        return

    try:
        from huggingface_hub import login  # type: ignore

        login(token=hf_token)
        logger.info("Sesión iniciada en Hugging Face Hub.")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"No se pudo iniciar sesión en Hugging Face Hub: {e}")



# =============================================================================
# =============================================================================
# ETAPA 1 · EXTRACCIÓN  (ZIP -> PDF -> datos estructurados)
# Clases: ESR, ESRExtractor, ESRZipProcessor, ESRAnonymizer, ESRExporter,
#         ESRExtractionPipeline
# =============================================================================
# =============================================================================
# =============================================================================
# ESR  ------------------------------------------------------------- datos + estado
# =============================================================================
@dataclass
class ESR:
    """Representa un único ESR: los datos extraídos de un PDF y su estado.

    Es un contenedor puro de datos (sin lógica de extracción, anonimización
    ni exportación); esas responsabilidades viven en las otras clases.
    """

    #: Código de error, igual que en la implementación original:
    #: 0 OK, 1 PDF, 2 BasicInfo, 3 Scores, 4 Criteria
    ERROR_MEANINGS = {
        0: "OK - No errors detected.",
        1: "PDF - Error during PDF reading or content extraction.",
        2: (
            "BasicInfo - Error extracting basic information (e.g., Call, "
            "Type, Number, Acronym, Title, Activity)."
        ),
        3: "Scores - Error extracting scores.",
        4: "Criteria - Error extracting criteria sections.",
    }

    source_path: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    error_code: int = 0

    @property
    def is_valid(self) -> bool:
        """True si no se detectó ningún error durante la extracción."""
        return self.error_code == 0

    @property
    def error_message(self) -> str:
        """Descripción legible del ``error_code`` actual."""
        return self.ERROR_MEANINGS.get(self.error_code, "Unknown Error Code")

    def to_dict(self) -> Dict[str, Any]:
        """Copia de los datos extraídos como diccionario plano."""
        return dict(self.data)

    def to_series(self) -> pd.Series:
        """Los datos extraídos como ``pandas.Series`` (una fila)."""
        return pd.Series(self.data)

    @staticmethod
    def to_dataframe(records: Sequence["ESR"]) -> pd.DataFrame:
        """Convierte una lista de ``ESR`` en un único ``DataFrame``.

        Se mantiene el comportamiento original: todos los registros se
        incluyen (también los que tuvieron algún error de extracción), ya
        que el error simplemente se registra/loguea aparte.
        """
        if not records:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in records])

    def __repr__(self) -> str:  # pragma: no cover - solo cosmético
        acronym = self.data.get("Acronym", "?")
        return f"ESR(acronym={acronym!r}, error_code={self.error_code})"


# =============================================================================
# ESRExtractor  ------------------------------------------------------- PDF -> datos
# =============================================================================
class ESRExtractor:
    """Extrae los datos de un ESR a partir de un PDF (equivalente a
    ``ESR_generate`` / ``ESR_Extraction_error`` en el notebook original)."""

    #: Patrón para eliminar el pie de página repetido en cada hoja del PDF.
    FOOTER_PATTERN = r"\d{8,9}/.*?-\d{2}/\d{2}/\d{4}-\d{2}:\d{2}:\d{2}\s+\d+\s+/\s*\d+"

    #: Patrones de información básica.
    BASIC_INFO_PATTERNS = {
        "Call": r"Call: (\S+)",
        "Type": r"Type of action: (\S+)",
        "Number": r"Proposal number: (\d+)",
        "Acronym": r"Proposal acronym: ([a-zA-Z0-9 ._-]+)",
        "Title": r"(?:Proposal )?[Tt]itle:\s*([\s\S]+?)(?:\n[A-Z][^:\n]+:|$)",
        "Activity": r"Activity: (\S+)",
        "Abstract": r"Abstract:\s*(.*?)\s*Evaluation Summary Report",
    }

    #: Patrón de puntuaciones ("Score: 4.5000") y de umbral a eliminar antes.
    SCORE_PATTERN = r"Score:\s*(\d+\.\d{1,4})"
    THRESHOLD_PATTERN = r"\(Threshold:.*?\)"

    #: Delimitadores de los 3 criterios de evaluación.
    CRITERION_DELIMITERS = (
        r"Criterion 1 - Excellence|Criterion 2 - Impact|Criterion 3 - Quality and"
        r" efficiency of the implementation|Scope of the"
        r" (?:application|proposal)"
    )

    #: Delimitador final de cada criterio, según el tipo de acción.
    CRITERION_END_BY_TYPE = {
        "HORIZON-CSA": [r"methodology\.", r"activities\.", r"expertise\."],
        "HORIZON-FPA": [
            "objectives",
            r"and impacts specified in the work programme\.",
            "participants",
        ],
    }
    CRITERION_END_DEFAULT = [r"appropriate\.", r"activities\.", r"expertise\."]

    def _read_pdf_text(self, file_path: str) -> Tuple[str, bool]:
        """Lee todas las páginas de un PDF y concatena su texto.

        Devuelve ``(texto, ok)`` donde ``ok`` es ``False`` si hubo un error
        de lectura (equivalente a ``isERR == 1``).
        """
        txt = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    txt += content
            return txt, True
        except Exception as e:  # noqa: BLE001 - se registra y se propaga el estado
            logger.info(f"Error detected: {e}")
            return txt, False

    def _strip_footer(self, txt: str) -> str:
        return re.sub(self.FOOTER_PATTERN, "", txt, flags=re.DOTALL)

    def _extract_basic_info(self, txt: str) -> Tuple[Dict[str, Any], bool]:
        row_data: Dict[str, Any] = {}
        ok = True
        for key, pattern in self.BASIC_INFO_PATTERNS.items():
            if key == "Abstract":
                match = re.search(pattern, txt, re.DOTALL)
            else:
                match = re.search(pattern, txt)

            if match:
                row_data[key] = match.group(1)
            else:
                row_data[key] = "No encontrado"
                ok = False
        return row_data, ok

    def _extract_scores(self, txt: str) -> Tuple[Dict[str, Any], str, bool]:
        txt_no_threshold = re.sub(self.THRESHOLD_PATTERN, "", txt)
        scores = [float(n) for n in re.findall(self.SCORE_PATTERN, txt_no_threshold)]

        ok = True
        if not scores:
            logger.info("No valid patterns found for the text")
            ok = False
            total_sum = 0
        else:
            total_sum = sum(scores)

        row_data: Dict[str, Any] = {"Total Score": total_sum}
        row_data["S_Excellence"] = scores[0] if len(scores) > 0 else None
        row_data["S_Impact"] = scores[1] if len(scores) > 1 else None
        row_data["S_Implement"] = scores[2] if len(scores) > 2 else None
        return row_data, txt_no_threshold, ok

    def _extract_criteria(self, txt: str, action_type: str) -> Tuple[Dict[str, Any], bool]:
        delimiter_end = self.CRITERION_END_BY_TYPE.get(action_type, self.CRITERION_END_DEFAULT)

        criterion = re.split(self.CRITERION_DELIMITERS, txt, flags=re.DOTALL)
        ok = True
        if len(criterion) == 5:
            criterion = criterion[1:4]
            for i in range(len(criterion)):
                text = re.sub(r"[ \t]+", " ", criterion[i])
                text = re.sub(r"\n- ", "\n", text)
                pattern = r"Score:.*?" + delimiter_end[i]
                criterion[i] = re.sub(pattern, "", text, count=1, flags=re.DOTALL).strip()
        else:
            ok = False
            criterion = ["", "", ""]

        row_data = {
            "C_Excellence": criterion[0] if len(criterion) > 0 else "",
            "C_Impact": criterion[1] if len(criterion) > 1 else "",
            "C_Implement": criterion[2] if len(criterion) > 2 else "",
        }
        return row_data, ok

    def extract(self, file_path: str) -> ESR:
        """Extrae los datos de un PDF de ESR y devuelve un objeto ``ESR``.

        Equivalente a ``ESR_generate`` del notebook original, pero devolviendo
        un único registro (``ESR``) en lugar de concatenar directamente en un
        DataFrame; la concatenación la realiza ``ESRZipProcessor`` / ``ESR.to_dataframe``.
        """
        error_code = 0  # 0:OK, 1: PDF, 2: BasicInfo, 3: Scores, 4: Criteria

        txt, pdf_ok = self._read_pdf_text(file_path)
        if not pdf_ok:
            error_code = 1

        txt = self._strip_footer(txt)

        row_data, basic_ok = self._extract_basic_info(txt)
        if not basic_ok:
            error_code = 2

        score_data, txt, scores_ok = self._extract_scores(txt)
        row_data.update(score_data)
        if not scores_ok:
            error_code = 3

        criteria_data, criteria_ok = self._extract_criteria(txt, row_data.get("Type"))
        row_data.update(criteria_data)
        if not criteria_ok:
            error_code = 4

        return ESR(source_path=file_path, data=row_data, error_code=error_code)


# =============================================================================
# ESRZipProcessor  ------------------------------------------------------ ZIP -> PDFs
# =============================================================================
class ESRZipProcessor:
    """Descomprime ZIP(s) (incluyendo ZIPs anidados), localiza los PDFs de ESR
    (``*esr.pdf``) y usa un ``ESRExtractor`` para construir la lista/DataFrame
    de resultados. Permite subir ZIPs interactivamente en Google Colab y consolidar
    el resultado con un DataFrame provisto desde un archivo Excel.
    """

    def __init__(
        self,
        extractor: Optional[ESRExtractor] = None,
        temp_folder: str = "/content/temp_unzip_content",
        colab_output_folder: str = "/content/PDF",
        upload_temp_dir: str = "/content/uploaded_zips_for_processing",
        error_log_file: str = "extraction_errors.log",
    ) -> None:
        self.extractor = extractor or ESRExtractor()
        self.temp_folder = temp_folder
        self.output_folder = colab_output_folder
        self.upload_temp_dir = upload_temp_dir
        self.error_log_file = error_log_file

        #: Registros (`ESR`) acumulados tras la última llamada a `process`.
        self.records: List[ESR] = []
        #: Log de fichero -> zip de origen, tras la última llamada a `process`.
        self.file_log_data: List[Dict[str, str]] = []

        # Asegurar la creación del directorio de salida
        os.makedirs(self.output_folder, exist_ok=True)

    # -- utilidades internas -------------------------------------------------
    def _resolve_output_path(self, output_dir: Optional[str]) -> str:
        return output_dir or self.output_folder

    @staticmethod
    def _extract_nested_zips(root_folder: str, log_f, errors_occurred: bool) -> bool:
        """Descomprime recursivamente cualquier ZIP anidado dentro de root_folder."""
        zips_extracted = True
        processed_zips = set()
        while zips_extracted:
            zips_extracted = False
            for root, _, filenames in os.walk(root_folder):
                for f in filenames:
                    if f.lower().endswith(".zip"):
                        zp = os.path.join(root, f)
                        if zp not in processed_zips:
                            try:
                                with zipfile.ZipFile(zp, "r") as nz:
                                    nz.extractall(os.path.dirname(zp))
                                processed_zips.add(zp)
                                zips_extracted = True
                                os.remove(zp)
                            except Exception as e:  # noqa: BLE001
                                warning_msg = f"  ⚠️ Warning: Could not extract nested zip {f}: {e}"
                                logger.info(warning_msg)
                                log_f.write(f"{datetime.now()} - {warning_msg}\n")
                                errors_occurred = True
        return errors_occurred

    # -- API pública -----------------------------------------------------
    def unzip(
        self,
        zip_source_paths: Optional[Sequence[str]] = None,
        output_dir: Optional[str] = None,
        input_excel_path: Optional[str] = None,
        upload_zips: bool = False,
    ) -> Tuple[pd.DataFrame, str, bool]:
        """Procesa uno o varios ZIPs y devuelve ``(df, error_log_path, errors_occurred)``.

        Parámetros
        ----------
        zip_source_paths: rutas explícitas a ZIPs a procesar.
        output_dir: carpeta de salida donde copiar los PDFs procesados y el
            log de errores (por defecto /content/PDF).
        input_excel_path: ruta opcional a un archivo Excel (.xlsx, .xls) para
            leer y combinar/anexar su Dataframe al Dataframe extraído de los PDFs.
        upload_zips: forzar la subida interactiva mediante `files.upload()`.
            Si `zip_source_paths` es `None`, se activa por defecto en Google Colab.
        """
        full_output_path = self._resolve_output_path(output_dir)
        full_error_log_path = os.path.join(full_output_path, self.error_log_file)

        if not os.path.exists(full_output_path):
            os.makedirs(full_output_path)
            
        pdf_output_dir = os.path.join(full_output_path, "PDFs")

        if not os.path.exists(pdf_output_dir):
            os.makedirs(pdf_output_dir, exist_ok=True)
        # Limpieza previa de carpeta temporal
        if os.path.exists(self.temp_folder):
            shutil.rmtree(self.temp_folder)

        zip_files_to_process: List[str] = []
        is_uploaded_temp = False

        # 1. Determinar el origen de los archivos ZIP
        if zip_source_paths is not None:
            zip_files_to_process = list(zip_source_paths)
        elif upload_zips or IN_COLAB:
            if colab_files is not None:
                if not os.path.exists(self.upload_temp_dir):
                    os.makedirs(self.upload_temp_dir)

                print("Please upload your ZIP files.")
                uploaded = colab_files.upload()

                if uploaded:
                    for fn in uploaded.keys():
                        local_path = os.path.join(self.upload_temp_dir, fn)
                        with open(local_path, "wb") as f:
                            f.write(uploaded[fn])
                        zip_files_to_process.append(local_path)
                    is_uploaded_temp = True
                else:
                    print("No files uploaded.")
            else:
                logger.warning("⚠️ google.colab.files no está disponible en este entorno.")

        records: List[ESR] = []
        total_found_files = 0
        file_log_data: List[Dict[str, str]] = []
        errors_occurred = False

        try:
            if zip_files_to_process:
                with open(full_error_log_path, "a", encoding="utf-8") as log_f:
                    log_f.write(f"\n--- Log started: {datetime.now()} ---\n")

                    for current_zip_full_path in tqdm(zip_files_to_process, desc="Processing ZIP files"):
                        zip_file_name = os.path.basename(current_zip_full_path)
                        logger.info(f"📂 Processing ZIP: {zip_file_name}")
                        if os.path.exists(self.temp_folder):
                            shutil.rmtree(self.temp_folder)
                        os.makedirs(self.temp_folder)

                        try:
                            with zipfile.ZipFile(current_zip_full_path, "r") as zip_ref:
                                zip_ref.extractall(self.temp_folder)
                        except Exception as e:  # noqa: BLE001
                            error_msg = f"❌ Error unzipping {zip_file_name}: {e}"
                            logger.info(error_msg)
                            log_f.write(f"{datetime.now()} - {error_msg}\n")
                            errors_occurred = True
                            continue

                        errors_occurred = self._extract_nested_zips(self.temp_folder, log_f, errors_occurred)

                        esr_pdfs_in_zip = []
                        for root, _, filenames in os.walk(self.temp_folder):
                            for fname in filenames:
                                if fname.lower().endswith("esr.pdf"):
                                    esr_pdfs_in_zip.append(os.path.join(root, fname))

                        for source_path in tqdm(
                            esr_pdfs_in_zip,
                            desc=f"  Extracting PDFs from {zip_file_name}",
                            leave=False,
                        ):
                            fname = os.path.basename(source_path)

                            # 1. Extraer datos con ESRExtractor
                            try:
                                esr = self.extractor.extract(source_path)
                                records.append(esr)
                                if not esr.is_valid:
                                    error_msg = f"  ⚠️ Extraction error {esr.error_code} in {fname}"
                                    logger.info(error_msg)
                                    log_f.write(f"{datetime.now()} - {error_msg}\n")
                                    errors_occurred = True
                            except Exception as e:  # noqa: BLE001
                                error_msg = f"  ❌ Failed to extract data from {fname}: {e}"
                                logger.info(error_msg)
                                log_f.write(f"{datetime.now()} - {error_msg}\n")
                                errors_occurred = True
                                continue

                           # 2. Guardar copia del PDF en la subcarpeta PDFs
                            dest_file_name = fname
                            dest_path = os.path.join(pdf_output_dir, dest_file_name)

                            counter = 0
                            while os.path.exists(dest_path):
                                counter += 1
                                name, ext = os.path.splitext(fname)
                                dest_file_name = f"{name}_{counter}{ext}"
                                dest_path = os.path.join(pdf_output_dir, dest_file_name)

                            shutil.copy2(source_path, dest_path)

                            file_log_data.append({"ZIP": zip_file_name, "File": dest_file_name})
                            total_found_files += 1

                    log_f.write(f"--- Log finished: {datetime.now()} ---\n")
            else:
                logger.info("No ZIP files found or uploaded to process.")
        finally:
            # Limpieza de carpetas temporales tras el procesamiento
            if os.path.exists(self.temp_folder):
                shutil.rmtree(self.temp_folder)
            if is_uploaded_temp and os.path.exists(self.upload_temp_dir):
                shutil.rmtree(self.upload_temp_dir)

        self.records = records
        self.file_log_data = file_log_data

        # Generar DataFrame con los registros extraídos
        df = ESR.to_dataframe(records) if records else pd.DataFrame()

        # Incorporar DataFrame desde archivo Excel si se proporciona
        if input_excel_path:
            if os.path.exists(input_excel_path):
                try:
                    logger.info(f"📄 Loading existing Excel data from: {input_excel_path}")
                    excel_df = pd.read_excel(input_excel_path)
                    df = pd.concat([df, excel_df], ignore_index=True)
                except Exception as e:
                    logger.error(f"❌ Error reading Excel file at {input_excel_path}: {e}")
            else:
                logger.warning(f"⚠️ Warning: input_excel_path specified ({input_excel_path}) but file does not exist.")

        return df, full_error_log_path, errors_occurred


# =============================================================================
# ESRAnonymizer  ---------------------------------------------- anonimización/cifrado
# =============================================================================
class ESRAnonymizer:
    """Anonimiza texto libre (criterios) y cifra/descifra columnas sensibles
    con AES-GCM. Equivalente a ``ESR_encode`` / ``ESR_decode`` +
    ``load_aesgcm_cipher`` / ``encode_aesgcm`` / ``decode_aesgcm`` del
    notebook original.
    """

    #: Palabras "seguras" que, aunque empiecen por mayúscula, no deben
    #: anonimizarse (vocabulario técnico habitual en propuestas Horizon Europe).
    WHITELIST: List[str] = [
        'Abscissa', 'Absorption', 'Abstract', 'Academia', 'Academic', 'Academics', 'Academy', 'Acceleration', 'Accelerator', 'Acceptability', 'Acceptance', 'Acceptors', 'Access', 'Accessibility', 'Accessible', 'Acclaim', 'Accord', 'According', 'Accountability', 'Accreditation', 'Accuracy', 'Achieve', 'Achievement', 'Achievements', 'Acid', 'Acoustic', 'Acre', 'Acta', 'Action', 'Actionbot', 'Actions', 'Activated', 'Activation', 'Active', 'Activities', 'Activity', 'Actor', 'Acts', 'Actual', 'Actuators', 'Adaptability', 'Adaptation', 'Adaptations', 'Adapters', 'Adaptive', 'Additionally', 'Additive', 'Address', 'Addresses', 'Addressing', 'Adequate', 'Adjacent', 'Administration', 'Administrations', 'Administrative', 'Adopt', 'Adopter', 'Adopters', 'Adoption', 'Advance', 'Advanced', 'Advancements', 'Advances', 'Advancing', 'Adversarial', 'Advice', 'Advisor', 'Advisory', 'Advocating', 'Aerial', 'Aeronautic', 'Aeronautics', 'Aeronis', 'Aerospace', 'Affiliated', 'African', 'Afro', 'Afrofeminist', 'Ageement', 'Ageing', 'Agencies', 'Agency', 'Agenda', 'Agendas', 'Agent', 'Agentar', 'Agentic', 'Agents', 'Aggregator', 'Agile', 'Aging', 'Agora', 'Agreement', 'Agreements', 'Agri', 'Agricultural', 'Agriculture', 'Agrifood', 'Agro', 'Aiaas', 'Aided', 'Airborne', 'Aire', 'Airport', 'Albanian', 'Alert', 'Algorithm', 'Algorithmic', 'Algorithms', 'Align', 'Alkaline', 'Alliance', 'Alliances', 'Allocation', 'Allocations', 'Alloying', 'Alloys', 'Also', 'Alternative', 'Although', 'Aluminum', 'Ambassador', 'Ambassadors', 'Ambient', 'Ambition', 'Ambitions', 'American', 'Amharic', 'Amounts', 'Amplifiers', 'Analog', 'Analyse', 'Analyses', 'Analysis', 'Analytical', 'Analytics', 'Analyze', 'Analyzer', 'Angels', 'Angewandten', 'Angle', 'Animal', 'Animation', 'Anion', 'Annealing', 'Annex', 'Annual', 'Anode', 'Anodic', 'Anomaly', 'Anonymization', 'Another', 'Answer', 'Answering', 'Antimony', 'Anytime', 'Aplite', 'Appearance', 'Appliance', 'Appliances', 'Applicants', 'Application', 'Applications', 'Applied', 'Apply', 'Approach', 'Approaches', 'Appropriate', 'Approximate', 'Approximately', 'Apps', 'Aquaculture', 'Architect', 'Architectural', 'Architecture', 'Architectures', 'Archiv', 'Archive', 'Arctic', 'Area', 'Areas', 'Argentinian', 'Argument', 'Arise', 'Armed', 'Arms', 'Around', 'Array', 'Arrays', 'Arrow', 'Arsenic', 'Artefact', 'Artery', 'Article', 'Articles', 'Articulated', 'Artificial', 'Arts', 'Ashesi', 'Asian', 'Aspect', 'Aspects', 'Assembly', 'Assess', 'Assessment', 'Assessments', 'Asset', 'Assets', 'Assignment', 'Assistance', 'Assistant', 'Assistants', 'Assisted', 'Assistive', 'Associated', 'Association', 'Associations', 'Assurance', 'Asymmetric', 'Atomic', 'Atomisation', 'Attention', 'Attractiveness', 'Attribution', 'Audience', 'Audiences', 'Audit', 'Augmentation', 'Augmented', 'August', 'Authentication', 'Authoring', 'Authorities', 'Authority', 'Autism', 'Auto', 'Automate', 'Automated', 'Automatic', 'Automation', 'Automotive', 'Autonomic', 'Autonomisation', 'Autonomous', 'Autonomy', 'Available', 'Avalanche', 'Avalanches', 'Avatar', 'Average', 'Aviation', 'Avoidance', 'Awards', 'Aware', 'Awareness', 'Axis', 'Backbone', 'Background', 'Balance', 'Band', 'Banking', 'Banks', 'Barium', 'Barrier', 'Barriers', 'Base', 'Based', 'Baseline', 'Bases', 'Basic', 'Batteries', 'Battery', 'Bauxite', 'Bauxitic', 'Beacon', 'Because', 'Beds', 'Behavioral', 'Behaviour', 'Behavioural', 'Being', 'Bejing', 'Belief', 'Belo', 'Belong', 'Bench', 'Benchmarking', 'Beneficiaries', 'Beneficiary', 'Benefit', 'Benefits', 'Beryllium', 'Best', 'Beta', 'Better', 'Beverage', 'Beyond', 'Bias', 'Bidirectional', 'Bigin', 'Bill', 'Billion', 'Bini', 'Biobased', 'Bioeconomy', 'Bioenergy', 'Biology', 'Biomedical', 'Bisphenol', 'Blades', 'Block', 'Blockchain', 'Blocks', 'Bloomberg', 'Blooming', 'Blue', 'Blueprint', 'Bluetooth', 'Board', 'Boards', 'Body', 'Bonn', 'Book', 'Boost', 'Booster', 'Boron', 'Bottleneck', 'Boundaries', 'Boundary', 'Braille', 'Brain', 'Branding', 'Brave', 'Brazilian', 'Breakdown', 'Breaking', 'Breast', 'Bridge', 'Bridges', 'Bridging', 'Brief', 'Briefs', 'Brightway', 'Bristol', 'Broad', 'Broadband', 'Broadcasting', 'Broader', 'Budget', 'Bugs', 'Build', 'Building', 'Buildings', 'Built', 'Burden', 'Business', 'Businesses', 'Busitema', 'Butadiene', 'Buyers', 'Cabling', 'Calculator', 'Calculus', 'Call', 'Calls', 'Camera', 'Campaign', 'Campus', 'Canadian', 'Cancer', 'Canopies', 'Canvas', 'Capabilities', 'Capability', 'Capacity', 'Capital', 'Caps', 'Capture', 'Carbide', 'Carbine', 'Carbon', 'Carbonate', 'Card', 'Cardiologists', 'Cards', 'Care', 'Career', 'Case', 'Cases', 'Casting', 'Catalan', 'Catalogue', 'Catalyst', 'Categorisation', 'Category', 'Causal', 'Cavity', 'Cell', 'Cells', 'Cellulose', 'Center', 'Centered', 'Centers', 'Central', 'Centre', 'Centred', 'Centres', 'Centric', 'Ceramics', 'Ceremony', 'Certain', 'Certificate', 'Certification', 'Certified', 'Chain', 'Challenge', 'Challenges', 'Champions', 'Change', 'Changes', 'Channels', 'Chapter', 'Characterisation', 'Characterization', 'Characterizer', 'Charge', 'Charging', 'Chart', 'Charter', 'Charts', 'Chat', 'Chatbot', 'Check', 'Checklist', 'Chemical', 'Chemicals', 'Chemistry', 'Chichewa', 'Chilean', 'Chilout', 'Chinese', 'Chip', 'Chips', 'Chiral', 'Chloride', 'Chromebooks', 'Chronic', 'Circle', 'Circuit', 'Circuits', 'Circular', 'Circularise', 'Circularity', 'Cities', 'Citiverse', 'Citizen', 'Citizens', 'City', 'Civil', 'Claim', 'Claims', 'Clarity', 'Class', 'Classic', 'Classical', 'Classification', 'Clean', 'Client', 'Clients', 'Climate', 'Clinical', 'Clocks', 'Close', 'Closing', 'Clothing', 'Cloud', 'Clubs', 'Cluster', 'Clustering', 'Clusters', 'Coach', 'Coaches', 'Coaching', 'Coalition', 'Coast', 'Coastal', 'Cobalt', 'Cobra', 'Cockpit', 'Coco', 'Code', 'Codes', 'Cognition', 'Cognitive', 'Coils', 'Colab', 'Collaboration', 'Collaborative', 'Collection', 'Collective', 'Collectively', 'College', 'Collision', 'Colony', 'Comb', 'Combinatorial', 'Combined', 'Combining', 'Commercial', 'Commercialization', 'Commission', 'Commissioning', 'Commitment', 'Committee', 'Committees', 'Common', 'Commons', 'Communication', 'Communications', 'Communities', 'Community', 'Companies', 'Companion', 'Company', 'Comparative', 'Compared', 'Comparison', 'Competence', 'Competency', 'Competition', 'Competitive', 'Competitiveness', 'Compiler', 'Completion', 'Complex', 'Complexity', 'Compliance', 'Compliant', 'Component', 'Components', 'Composites', 'Composition', 'Compound', 'Comprehensive', 'Compression', 'Compressive', 'Computadores', 'Computation', 'Computational', 'Compute', 'Computer', 'Computers', 'Computing', 'Comunica', 'Concentrates', 'Concept', 'Concepts', 'Concern', 'Concerning', 'Concerns', 'Conditions', 'Conduct', 'Conference', 'Confiance', 'Confidence', 'Confidential', 'Conflict', 'Conformity', 'Congress', 'Conjugate', 'Connected', 'Connectivity', 'Connectors', 'Conscious', 'Consensus', 'Considering', 'Consolidation', 'Consortium', 'Constitutional', 'Construction', 'Consultant', 'Consultants', 'Consulting', 'Consumables', 'Consumer', 'Consumption', 'Contact', 'Contactile', 'Container', 'Content', 'Context', 'Contextual', 'Continual', 'Continue', 'Continuity', 'Continuous', 'Continuum', 'Contract', 'Contrastive', 'Contribute', 'Contributing', 'Contribution', 'Contributions', 'Contributor', 'Control', 'Controller', 'Convention', 'Convergence', 'Conversational', 'Conversion', 'Convert', 'Cooperation', 'Cooperative', 'Coordinated', 'Coordination', 'Coordinator', 'Copper', 'Copying', 'Core', 'Coronary', 'Corporate', 'Corporation', 'Corralling', 'Correcting', 'Correction', 'Correctional', 'Correlation', 'Cortex', 'Cosmetics', 'Cosserat', 'Cost', 'Costing', 'Costs', 'Council', 'Counter', 'Countries', 'Country', 'Coupled', 'Course', 'Coursera', 'Courses', 'Covid', 'Creating', 'Creation', 'Creative', 'Creativity', 'Creator', 'Credential', 'Credentials', 'Credibility', 'Crisis', 'Criteria', 'Criterion', 'Critical', 'Criticality', 'Cross', 'Crucially', 'Crystal', 'Cultural', 'Cultured', 'Cultures', 'Culturing', 'Cura', 'Curation', 'Current', 'Curriculum', 'Custodian', 'Custom', 'Customer', 'Cutting', 'Cyber', 'Cybersec', 'Cybersecurity', 'Cycle', 'Daily', 'Dangerous', 'Danish', 'Dashboard', 'Data', 'Database', 'Databases', 'Datacenter', 'Datahub', 'Datalift', 'Dataset', 'Datasets', 'Dataspace', 'Dataspaces', 'Days', 'Deaf', 'Deal', 'Deals', 'Death', 'Decade', 'Decarbonisation', 'Decentralised', 'Decentralized', 'Decide', 'Decision', 'Declaration', 'Declarations', 'Decrease', 'Decreasing', 'Deep', 'Deepfake', 'Defect', 'Defense', 'Definition', 'Degree', 'Delamination', 'Delay', 'Delayed', 'Delays', 'Demand', 'Demining', 'Demo', 'Democracy', 'Democratic', 'Demonstrate', 'Demonstrated', 'Demonstrating', 'Demonstration', 'Demonstrator', 'Demonstrators', 'Denim', 'Department', 'Dependability', 'Dependencies', 'Dependency', 'Deployment', 'Deposition', 'Depreciation', 'Describe', 'Description', 'Descriptions', 'Design', 'Designer', 'Designing', 'Destination', 'Destinations', 'Destructive', 'Details', 'Detection', 'Detector', 'Deutsches', 'Develop', 'Developer', 'Developers', 'Developing', 'Development', 'Developments', 'Device', 'Devices', 'Diabetes', 'Diabetic', 'Diagnostic', 'Diagnostics', 'Dialogue', 'Diamond', 'Diamonds', 'Dichalcogenides', 'Didactic', 'Difficulties', 'Difficulty', 'Diffractive', 'Diffuse', 'Diffusion', 'Digital', 'Digitalisation', 'Digitally', 'Digitisation', 'Digitization', 'Diligence', 'Dimension', 'Dimensional', 'Diode', 'Dioxide', 'Direct', 'Directed', 'Directive', 'Directorate', 'Directory', 'Dirty', 'Disassembly', 'Disaster', 'Disciplined', 'Discord', 'Discover', 'Discovery', 'Discrete', 'Discrimination', 'Discriminatory', 'Disease', 'Diseases', 'Disinformation', 'Disorder', 'Dispatch', 'Dispenser', 'Displays', 'Dispute', 'Disseminate', 'Dissemination', 'Distributed', 'Distribution', 'Distributions', 'District', 'Diversity', 'Does', 'Dots', 'Double', 'Doughnut', 'Down', 'Downloads', 'Draft', 'Drilling', 'Drinking', 'Driven', 'Driver', 'Driverless', 'Driving', 'Drone', 'Drones', 'Droplets', 'Drug', 'Dual', 'Duality', 'Dull', 'Dust', 'Dynamic', 'Dynamics', 'Each', 'Early', 'Earth', 'Eastern', 'Ecodesign', 'Ecoinvent', 'Ecolabel', 'Ecole', 'Ecological', 'Ecology', 'Economic', 'Economically', 'Economics', 'Economy', 'Ecosystem', 'Ecosystems', 'Ecotourism', 'Edge', 'Edges', 'Edtech', 'Education', 'Educational', 'Effect', 'Efficiency', 'Efficient', 'Effort', 'Egyptian', 'Elaboration', 'Elastic', 'Electric', 'Electrical', 'Electricity', 'Electro', 'Electrochemical', 'Electrodialysis', 'Electroencephalography', 'Electrolysers', 'Electrolysis', 'Electrolyte', 'Electrolytic', 'Electromyography', 'Electron', 'Electronic', 'Electronica', 'Electronics', 'Electrotechnical', 'Element', 'Elements', 'Ellipsoid', 'Embedded', 'Embeddings', 'Embodied', 'Emergency', 'Emerging', 'Emission', 'Emitting', 'Emotion', 'Emotional', 'Emphatic', 'Employ', 'Employee', 'Employees', 'Empower', 'Empowerment', 'Enable', 'Enabled', 'Enablers', 'Enabling', 'Encryption', 'Endorsement', 'Energy', 'Enforcement', 'Engage', 'Engagement', 'Engine', 'Engineered', 'Engineering', 'English', 'Enhance', 'Enhanced', 'Enhancement', 'Enhancing', 'Ensure', 'Ensuring', 'Enterprise', 'Enterprises', 'Entertainment', 'Entity', 'Entrepreneurs', 'Entrepreneurship', 'Entropy', 'Entry', 'Envirometrics', 'Environ', 'Environment', 'Environmental', 'Environments', 'Epichlorohydrin', 'Epitaxy', 'Equal', 'Equality', 'Equation', 'Equi', 'Equipment', 'Equity', 'Equivalents', 'Error', 'Errors', 'Esan', 'Establish', 'Establishing', 'Establishments', 'Estimates', 'Estimation', 'Ethical', 'Ethics', 'Ethiopian', 'Ethnicity', 'Etsako', 'Euros', 'Eutectic', 'Evaluation', 'Evaluator', 'Even', 'Event', 'Events', 'Evidence', 'Evolutionary', 'Evolving', 'Evry', 'Example', 'Examples', 'Excel', 'Excellence', 'Exchange', 'Execute', 'Execution', 'Executive', 'Executives', 'Exercise', 'Existing', 'Exitosa', 'Expected', 'Experience', 'Experiences', 'Experimental', 'Experimentation', 'Experiments', 'Expert', 'Expertise', 'Experts', 'Explain', 'Explainability', 'Explainable', 'Explainer', 'Explainers', 'Explanation', 'Exploit', 'Exploitable', 'Exploitation', 'Exploration', 'Explore', 'Explorers', 'Explotation', 'Expo', 'Exposure', 'Express', 'Extended', 'Extending', 'Extensible', 'Extension', 'Extensions', 'External', 'Extract', 'Extraction', 'Extreme', 'Eyeflow', 'Eyewear', 'Fabless', 'Fabric', 'Fabrication', 'Fabrix', 'Fabry', 'Face', 'Facilitate', 'Facilitators', 'Facilities', 'Facility', 'Factories', 'Factors', 'Factory', 'Fail', 'Failure', 'Fair', 'Fairness', 'Family', 'Fark', 'Farm', 'Farmer', 'Farmerfirst', 'Farmers', 'Farming', 'Fashion', 'Fast', 'Fault', 'Feasibility', 'Federal', 'Federated', 'Federation', 'Feedback', 'Feeding', 'Feel', 'Fellowships', 'Festival', 'Fetal', 'Fiber', 'Fibre', 'Fidelity', 'Field', 'Figure', 'Figures', 'Filing', 'Final', 'Finally', 'Finance', 'Financial', 'Financing', 'Findability', 'Findable', 'Fine', 'Finite', 'Finland', 'Finnish', 'Fintech', 'Fire', 'Firmware', 'First', 'Fisher', 'Fisheries', 'Flagship', 'Flaire', 'Flemish', 'Flexibility', 'Flexible', 'Flight', 'Flow', 'Flower', 'Fluid', 'Fluorescence', 'Fluorimetry', 'Fluoroalkyl', 'Focus', 'Focused', 'Following', 'Food', 'Foot', 'Footprint', 'Force', 'Forces', 'Forecasting', 'Foreground', 'Foresight', 'Forest', 'Forestry', 'Forgotten', 'Fork', 'Forks', 'Form', 'Formal', 'Format', 'Formia', 'Forming', 'Formnext', 'Forschung', 'Forum', 'Forward', 'Foster', 'Fostering', 'Foundation', 'Foundational', 'Foundations', 'Foundry', 'Four', 'Fragmented', 'Frame', 'Framework', 'Frameworks', 'France', 'Fraud', 'Free', 'Freedom', 'Freelance', 'Frequency', 'Fresh', 'From', 'Frontier', 'Frontline', 'Frugal', 'Fsas', 'Fuel', 'Fuer', 'Full', 'Fully', 'Function', 'Functional', 'Functionality', 'Fund', 'Fundamental', 'Funding', 'Fundings', 'Fungible', 'Funka', 'Furthermore', 'Fused', 'Fusion', 'Future', 'Fuzzy', 'Gain', 'Gallium', 'Game', 'Games', 'Gamma', 'Gannt', 'Gant', 'Gantt', 'Gaps', 'Garment', 'Gases', 'Gate', 'Gateway', 'Gbaud', 'Gbps', 'Gender', 'Gendered', 'General', 'Generalisability', 'Generalisation', 'Generalist', 'Generalization', 'Generally', 'Generation', 'Generative', 'Generator', 'Generic', 'Genome', 'Geological', 'Geologists', 'Geospatial', 'Geotechnical', 'German', 'Germanium', 'Gesellschaft', 'Ghost', 'Giga', 'Given', 'Glasses', 'Glaucoma', 'Global', 'Glove', 'Glucose', 'Glue', 'Glycine', 'Goal', 'Goals', 'Godspeed', 'Gold', 'Golden', 'Good', 'Goods', 'Governance', 'Governing', 'Government', 'Gradient', 'Grant', 'Granular', 'Graph', 'Graphene', 'Graphenea', 'Graphical', 'Graphs', 'Grating', 'Gratings', 'Gravimetric', 'Green', 'Greener', 'Greenhouse', 'Grid', 'Grids', 'Ground', 'Group', 'Grouping', 'Groups', 'Growth', 'Guarantees', 'Guarantying', 'Guidance', 'Guide', 'Guided', 'Guidelines', 'Guiding', 'Guinea', 'Hackathon', 'Hackathons', 'Hall', 'Handbook', 'Handover', 'Haptic', 'Haptics', 'Harbour', 'Hard', 'Hardware', 'Harm', 'Harvester', 'Harvesting', 'Hausa', 'Hazards', 'Head', 'Healing', 'Health', 'Healthcare', 'Hear', 'Hearing', 'Heart', 'Heat', 'Heater', 'Heath', 'Heating', 'Heavy', 'Hebbian', 'Helium', 'Helix', 'Helpdesk', 'Heritage', 'Heter', 'Heterogeneity', 'High', 'Higher', 'Highlight', 'Highly', 'History', 'Hochschule', 'Holistic', 'Hollow', 'Holonic', 'Holoportation', 'Home', 'Homebase', 'Horizon', 'Horn', 'Hospital', 'Hospitals', 'Host', 'Hotlines', 'Household', 'However', 'Hubs', 'Human', 'Humanism', 'Humanitarian', 'Humanities', 'Humanity', 'Humanizing', 'Humanoid', 'Humans', 'Hybrid', 'Hydraulics', 'Hydrogen', 'Hydrothermal', 'Hydroxide', 'Hyper', 'Hyperconnected', 'Hyperloop', 'Hyperpersonalisation', 'Hyperpesonalisation', 'Hyperspectral', 'Hyperstore', 'Hypoxia', 'Icelandic', 'Identification', 'Identifier', 'Identifiers', 'Identify', 'Identifying', 'Identities', 'Identity', 'Igbo', 'Illumination', 'Image', 'Imagination', 'Imaging', 'Immersive', 'Impact', 'Impacts', 'Impaired', 'Impairment', 'Implement', 'Implementation', 'Implementations', 'Implementing', 'Implication', 'Important', 'Improve', 'Improved', 'Improvement', 'Improvements', 'Improving', 'Inability', 'Inadequate', 'Inaugural', 'Incident', 'Including', 'Inclusion', 'Inclusive', 'Inclusivity', 'Incomes', 'Incomplete', 'Incorporating', 'Increase', 'Increased', 'Increasing', 'Incremental', 'Incubation', 'Incubator', 'Indaba', 'Independent', 'Index', 'Indexing', 'Indian', 'Indicate', 'Indicator', 'Indicators', 'Indigenous', 'Indium', 'Individual', 'Individuals', 'Indo', 'Induced', 'Inducted', 'Industrial', 'Industrialization', 'Industries', 'Industry', 'Infections', 'Inference', 'Infinite', 'Informatics', 'Information', 'Informed', 'Informer', 'Infotainment', 'Infra', 'Infrared', 'Infrastructure', 'Infrastructures', 'Infused', 'Inherently', 'Initial', 'Initiate', 'Initiation', 'Initiative', 'Initiatives', 'Injection', 'Injury', 'Inner', 'Insight', 'Inspection', 'Inspire', 'Inspired', 'Instances', 'Institut', 'Institute', 'Institutes', 'Institutional', 'Institutions', 'Instituto', 'Instruction', 'Instrument', 'Instruments', 'Insufficient', 'Insulator', 'Insurance', 'Integrate', 'Integrated', 'Integration', 'Integrations', 'Integrators', 'Integrity', 'Intellectual', 'Intelligence', 'Intelligent', 'Intense', 'Intensive', 'Intent', 'Intents', 'Inter', 'Interaction', 'Interactions', 'Interactive', 'Interband', 'Intercommunication', 'Interconnected', 'Interdependencies', 'Interdisciplinarity', 'Interdisciplinary', 'Interest', 'Interface', 'Interfaces', 'Interference', 'Interferometers', 'Interferometry', 'Interim', 'Intermediaries', 'Intermediate', 'Internal', 'Internally', 'International', 'Internet', 'Interoperability', 'Interoperable', 'Interpretability', 'Interpretation', 'Interreg', 'Interrogator', 'Intersector', 'Intertwined', 'Interventional', 'Interventions', 'Interviews', 'Intra', 'Introducing', 'Introduction', 'Introspection', 'Intrusion', 'Invasive', 'Inventorise', 'Investigate', 'Investigation', 'Investigations', 'Investigative', 'Investment', 'Investors', 'Invoiced', 'Invoices', 'Involvement', 'Ionic', 'Iridium', 'Irish', 'Islands', 'Isostatic', 'Isotopologue', 'Issue', 'Issues', 'Item', 'Iteration', 'Iterative', 'Japanese', 'Jobs', 'Joint', 'Journal', 'Journalism', 'Journals', 'Journey', 'Jumpstarter', 'June', 'Junior', 'Just', 'Justice', 'Justification', 'Kaolinite', 'Kazakh', 'Kerf', 'Kimbundu', 'Kinematic', 'Kinyarwanda', 'Kiswahili', 'Kits', 'Klaveness', 'Know', 'Knowledge', 'Korean', 'Label', 'Labelled', 'Laboratories', 'Laboratory', 'Labs', 'Lack', 'Lake', 'Lakehouse', 'Landing', 'Landscape', 'Language', 'Large', 'Larger', 'Larus', 'Laser', 'Lasers', 'Latent', 'Latin', 'Lattice', 'Launch', 'Layer', 'Layers', 'Leach', 'Lead', 'Leader', 'Leaders', 'Leadership', 'Leading', 'Leaf', 'Lean', 'Leaning', 'Learn', 'Learning', 'Least', 'Lece', 'Ledger', 'Legal', 'Legislation', 'Letters', 'Leuven', 'Level', 'Levelized', 'Levels', 'Leveraging', 'Levers', 'Lexicology', 'Liaison', 'Liberate', 'Liberties', 'Librarian', 'Libraries', 'Library', 'Libre', 'Licence', 'Licences', 'License', 'Licor', 'Lidar', 'Life', 'Lifecycle', 'Lifelong', 'Lifetime', 'Light', 'Lighthouse', 'Lighthouses', 'Lightwave', 'Lightweight', 'Like', 'Likelihood', 'Limited', 'Limiting', 'Limits', 'Line', 'Lineage', 'Linguistic', 'Link', 'Linked', 'Linkedin', 'Links', 'Liquid', 'Liquids', 'Lisesi', 'List', 'Literature', 'Lithium', 'Little', 'Live', 'Living', 'Load', 'Local', 'Localisation', 'Localization', 'Locations', 'Lock', 'Logic', 'Logics', 'Logistics', 'Long', 'Loop', 'Loss', 'Lower', 'Lowering', 'Luganda', 'Lugbarati', 'Lump', 'Lunar', 'Lyondellbasel', 'Mach', 'Machine', 'Machinery', 'Machines', 'Machining', 'Made', 'Magnesium', 'Magnetic', 'Magnetisation', 'Magnetite', 'Mailbox', 'Main', 'Mainstreaming', 'Maintenance', 'Major', 'Make', 'Makers', 'Making', 'Malnutrition', 'Management', 'Manager', 'Managers', 'Mandate', 'Manifesto', 'Manipulator', 'Manpower', 'Manual', 'Manufacturers', 'Manufacturing', 'Many', 'Mapping', 'Maps', 'Marine', 'Maritime', 'Mark', 'Market', 'Marketable', 'Marketing', 'Marketplace', 'Marketplaces', 'Markets', 'Markup', 'Masakhane', 'Massive', 'Master', 'Masters', 'Matadero', 'Matadi', 'Matching', 'Matchmaking', 'Material', 'Materialise', 'Materials', 'Maternal', 'Maternity', 'Mathematics', 'Matrix', 'Mature', 'Maturity', 'Means', 'Measurable', 'Measure', 'Measurement', 'Measurements', 'Measures', 'Meat', 'Mechanical', 'Mechanics', 'Mechanism', 'Mechanisms', 'Mechano', 'Mechatronic', 'Mechatronics', 'Media', 'Medical', 'Medicine', 'Mediterranean', 'Medium', 'Meet', 'Meetings', 'Megawatt', 'Melt', 'Melting', 'Meltwater', 'Member', 'Membrane', 'Memo', 'Memorandum', 'Memory', 'Mergeable', 'Merits', 'Mesh', 'Meta', 'Metadata', 'Metal', 'Metallurgy', 'Metals', 'Metaverse', 'Metaverses', 'Methane', 'Method', 'Methodical', 'Methodological', 'Methodologies', 'Methodology', 'Methods', 'Metric', 'Metrics', 'Metrology', 'Mexican', 'Mexico', 'Micro', 'Microdisplay', 'Microdisplays', 'Microfactory', 'Microfibre', 'Microfluidics', 'Microgrid', 'Micromachining', 'Microscope', 'Microscopy', 'Microwave', 'Middleware', 'Mild', 'Milestone', 'Milestones', 'Mill', 'Millenion', 'Mind', 'Minds', 'Mines', 'Minimally', 'Minimize', 'Minimum', 'Mining', 'Ministries', 'Ministry', 'Minor', 'Minorities', 'Missing', 'Mission', 'Mitigation', 'Mixed', 'Mixture', 'Mobile', 'Mobility', 'Mockups', 'Modal', 'Model', 'Modeling', 'Modelling', 'Models', 'Modern', 'Modular', 'Modulated', 'Modulator', 'Modulators', 'Module', 'Modules', 'Molecular', 'Molten', 'Monetisation', 'Monitor', 'Monitoring', 'Monolith', 'Monolithic', 'Monte', 'Month', 'Months', 'More', 'Moreover', 'Morphemic', 'Most', 'Motion', 'Motors', 'Move', 'Moving', 'Mton', 'Multi', 'Multidisciplinary', 'Multilingual', 'Multimedia', 'Multimodal', 'Multiparty', 'Multiplexed', 'Multiplier', 'Multiplying', 'Multiscale', 'Multispectral', 'Muscles', 'Museum', 'Mutual', 'Named', 'National', 'Nations', 'Natively', 'Natural', 'Navigation', 'Near', 'Nebulous', 'Neck', 'Needed', 'Needs', 'Negative', 'Neglected', 'Negotiation', 'Nervous', 'Nets', 'Network', 'Networking', 'Networks', 'Neural', 'Neuro', 'Neurodegenerative', 'Neuromorphic', 'Neurosymbolic', 'Neurotechnology', 'Nevertheless', 'News', 'Newsrooms', 'Newtech', 'Next', 'Nexus', 'Nickel', 'Night', 'Niobate', 'Niobium', 'Nitride', 'Nitrogen', 'Node', 'Nodes', 'Noisy', 'Nomination', 'Normung', 'North', 'Norway', 'Norwegian', 'Nose', 'Novel', 'Nuclear', 'Number', 'Numurus', 'Nyex', 'Object', 'Objective', 'Objectives', 'Objects', 'Observation', 'Observatory', 'Observer', 'Obstructive', 'Obtainable', 'Ocean', 'Oceanic', 'Offerings', 'Office', 'Officer', 'Offices', 'Offshore', 'Olivine', 'Olympic', 'Omnidirectional', 'Once', 'Online', 'Only', 'Ontologies', 'Ontology', 'Open', 'Openness', 'Opera', 'Operate', 'Operating', 'Operation', 'Operational', 'Operationalization', 'Operations', 'Operator', 'Operators', 'Opportunities', 'Opportunity', 'Optic', 'Optical', 'Optically', 'Optics', 'Optimal', 'Optimisation', 'Optimised', 'Optimising', 'Optimization', 'Optimized', 'Options', 'Optoacoustic', 'Optoelectronic', 'Oral', 'Orange', 'Orbit', 'Orchestration', 'Orchestrator', 'Organic', 'Organisation', 'Organisations', 'Organization', 'Organizational', 'Organizations', 'Organize', 'Orientation', 'Orientations', 'Origami', 'Origin', 'Original', 'Other', 'Otherwise', 'Outcome', 'Outcomes', 'Outline', 'Output', 'Outputs', 'Outreach', 'Over', 'Overall', 'Overflow', 'Overhaul', 'Oversight', 'Overview', 'Owners', 'Ownership', 'Oxidation', 'Oxide', 'Ozone', 'Pack', 'Package', 'Packages', 'Packaging', 'Pact', 'Page', 'Pancake', 'Panel', 'Panelfit', 'Panels', 'Paper', 'Papers', 'Paradigm', 'Paradigms', 'Paralympic', 'Parametric', 'Paris', 'Parity', 'Park', 'Parking', 'Parliament', 'Part', 'Parter', 'Partial', 'Participant', 'Participants', 'Participation', 'Participatory', 'Particle', 'Parties', 'Partitioning', 'Partner', 'Partners', 'Partnership', 'Partnerships', 'Parts', 'Party', 'Pashto', 'Passive', 'Passport', 'Passports', 'Patent', 'Patents', 'Path', 'Paths', 'Pathway', 'Pathways', 'Patient', 'Patients', 'Patterning', 'Pedagogical', 'Peer', 'People', 'Percentage', 'Perception', 'Percutaneous', 'Perfect', 'Performance', 'Performant', 'Periodic', 'Periodically', 'Perovskite', 'Person', 'Personal', 'Personalisation', 'Personalised', 'Personalized', 'Personas', 'Personnel', 'Perspective', 'Pert', 'Peru', 'Pervasive', 'Phabulous', 'Phactory', 'Pharia', 'Pharma', 'Pharmaceutical', 'Phase', 'Phased', 'Phaseform', 'Phases', 'Phenomena', 'Phonon', 'Phosphorus', 'Photo', 'Photon', 'Photonic', 'Photonics', 'Photonicsens', 'Photons', 'Photovoltaic', 'Photovoltaics', 'Physic', 'Physical', 'Physician', 'Physics', 'Physiological', 'Pidgin', 'Pilar', 'Pillar', 'Pillars', 'Pilot', 'Piloted', 'Piloting', 'Pilots', 'Pioneer', 'Pipeline', 'Pipelines', 'Piston', 'Place', 'Plan', 'Plane', 'Planet', 'Planetary', 'Planned', 'Planning', 'Plans', 'Plant', 'Planting', 'Plants', 'Plasma', 'Plasmon', 'Plastic', 'Plastronics', 'Platform', 'Platforms', 'Plausibility', 'Play', 'Playbook', 'Player', 'Playful', 'Please', 'Pledger', 'Plesh', 'Plethora', 'Plots', 'Plug', 'Pluggable', 'Plus', 'Pneumatic', 'Podcasts', 'Point', 'Pointer', 'Points', 'Poland', 'Poled', 'Policies', 'Policy', 'Policymakers', 'Polish', 'Political', 'Politics', 'Pollutants', 'Pollution', 'Poly', 'Polyethylene', 'Polylactic', 'Polymer', 'Polymers', 'Polytechnic', 'Poor', 'Port', 'Portable', 'Portal', 'Portals', 'Portfolio', 'Portuguese', 'Position', 'Positive', 'Possible', 'Post', 'Postdocs', 'Potassium', 'Potential', 'Power', 'Powered', 'Powerful', 'Practice', 'Practices', 'Precision', 'Prediction', 'Predictive', 'Preliminary', 'Preparation', 'Preservation', 'Preserving', 'Pressure', 'Pretraining', 'Prevention', 'Pricing', 'Primarily', 'Primary', 'Principle', 'Principles', 'Print', 'Printed', 'Printing', 'Priorities', 'Priority', 'Privacy', 'Private', 'Prize', 'Procedure', 'Procedures', 'Process', 'Processes', 'Processing', 'Processor', 'Processors', 'Procurement', 'Produce', 'Produced', 'Producer', 'Product', 'Production', 'Productivity', 'Products', 'Professionals', 'Profil', 'Profiles', 'Profiling', 'Profit', 'Program', 'Programmable', 'Programme', 'Programmers', 'Programmes', 'Programming', 'Programs', 'Progress', 'Progressivist', 'Project', 'Projection', 'Projections', 'Projective', 'Projects', 'Promote', 'Promoting', 'Prompting', 'Proof', 'Proofs', 'Propagation', 'Property', 'Proposal', 'Proposed', 'Proposition', 'Prospective', 'Prospects', 'Prosper', 'Protect', 'Protected', 'Protection', 'Protective', 'Protein', 'Protocol', 'Protocols', 'Proton', 'Prototype', 'Prototypes', 'Prototyping', 'Provenance', 'Provide', 'Provider', 'Providers', 'Providing', 'Proximal', 'Pseudonymization', 'Psychology', 'Public', 'Publication', 'Publications', 'Pulaar', 'Pulmonary', 'Pulpe', 'Pulse', 'Pump', 'Pumps', 'Purchase', 'Purchasing', 'Purification', 'Purity', 'Purpan', 'Purpose', 'Pushing', 'Putting', 'Quadrants', 'Quadruple', 'Qualidade', 'Qualification', 'Qualitative', 'Quality', 'Quantified', 'Quantitative', 'Quantum', 'Quartz', 'Quasi', 'Qubit', 'Query', 'Question', 'Qunatum', 'Radar', 'Radiant', 'Radio', 'Radiology', 'Rails', 'Railway', 'Railways', 'Raise', 'Raised', 'Raman', 'Random', 'Ranging', 'Rapid', 'Raspberry', 'Rate', 'Rating', 'Reach', 'Reactor', 'Readiness', 'Ready', 'Real', 'Realistic', 'Reality', 'Reallocate', 'Reasonable', 'Reasoning', 'Recall', 'Receiver', 'Recognition', 'Recommendation', 'Recommendations', 'Recommenders', 'Records', 'Recovery', 'Recruitment', 'Recyclability', 'Recycling', 'Redesign', 'Redshift', 'Reduce', 'Reduced', 'Reducing', 'Reduction', 'Reference', 'Refinery', 'Reflective', 'Reformer', 'Regarding', 'Region', 'Regional', 'Register', 'Registry', 'Regulation', 'Regulations', 'Regulatory', 'Reinforced', 'Reinforcement', 'Related', 'Release', 'Relevance', 'Relevant', 'Reliability', 'Reluctance', 'Remaining', 'Reman', 'Remanufacture', 'Remanufacturing', 'Remote', 'Remotely', 'Renewable', 'Rental', 'Renting', 'Repair', 'Repeat', 'Replacement', 'Replication', 'Report', 'Reporting', 'Reports', 'Repositories', 'Repository', 'Representation', 'Representative', 'Representatives', 'Reproducible', 'Requirements', 'Resale', 'Rescue', 'Research', 'Researcher', 'Researchers', 'Reservoir', 'Residence', 'Residency', 'Residential', 'Residual', 'Residue', 'Resilience', 'Resilient', 'Resonance', 'Resonant', 'Resonator', 'Resource', 'Resources', 'Respective', 'Respiratory', 'Response', 'Responsibilities', 'Responsibility', 'Responsible', 'Result', 'Results', 'Resultss', 'Retail', 'Retailers', 'Retrieval', 'Return', 'Reusability', 'Reusable', 'Reuse', 'Revenue', 'Reverse', 'Review', 'Reviews', 'Rhythmic', 'Rich', 'Ridge', 'Right', 'Rights', 'Rigid', 'Ring', 'Rising', 'Risk', 'Risks', 'River', 'Road', 'Roadmap', 'Roadmapping', 'Roadmaps', 'Robominers', 'Robot', 'Robotic', 'Robotics', 'Robotized', 'Robotnik', 'Robots', 'Robust', 'Robustness', 'Roles', 'Roll', 'Romani', 'Romanian', 'Romanic', 'Rome', 'Room', 'Rooms', 'Root', 'Route', 'Routing', 'Rubber', 'Rulebook', 'Rules', 'Runtime', 'Rural', 'Ruralities', 'Russian', 'Rust', 'Ruthenium', 'Saclay', 'Safe', 'Safer', 'Safety', 'Saharan', 'Salaries', 'Sales', 'Sand', 'Sandbox', 'Sans', 'Sars', 'Satellite', 'Satellites', 'Satisfaction', 'Scaffots', 'Scalability', 'Scalable', 'Scale', 'Scaleup', 'Scaling', 'Scan', 'Scanning', 'Scattering', 'Scenario', 'Scenarios', 'Scheduling', 'Scheme', 'School', 'Science', 'Sciences', 'Scientific', 'Scientists', 'Scope', 'Scoping', 'Score', 'Scorecard', 'Scouting', 'Scrap', 'Screen', 'Screening', 'Scribe', 'Seal', 'Search', 'Second', 'Secondary', 'Secret', 'Secretariat', 'Sect', 'Section', 'Sector', 'Sectors', 'Secure', 'Security', 'Seed', 'Select', 'Selection', 'Self', 'Selling', 'Semantic', 'Semi', 'Semiconductor', 'Semiconductors', 'Senior', 'Sense', 'Senses', 'Sensing', 'Sensitive', 'Sensitivity', 'Sensor', 'Sensors', 'Sensory', 'Sentient', 'Sentiment', 'Sequential', 'Servant', 'Server', 'Servers', 'Service', 'Serviceable', 'Services', 'Servoing', 'Setting', 'Several', 'Severity', 'Sewer', 'Sextuple', 'Sexual', 'Shadow', 'Shaped', 'Share', 'Shared', 'Sharing', 'Sheet', 'Shelf', 'Shell', 'Shells', 'Shield', 'Ship', 'Shipping', 'Shop', 'Shopfloor', 'Short', 'Shortcoming', 'Shortcomings', 'Shot', 'Side', 'Sign', 'Signal', 'Significance', 'Significant', 'Silica', 'Silicon', 'Silos', 'Silver', 'Simpl', 'Simula', 'Simulation', 'Simulations', 'Simulator', 'Simulink', 'Simultaneous', 'Since', 'Single', 'Sintef', 'Sintering', 'Siren', 'Sistemas', 'Site', 'Sites', 'Situ', 'Situation', 'Skill', 'Skills', 'Skin', 'Slag', 'Slice', 'Sliding', 'Slot', 'Slovenian', 'Small', 'Smaller', 'Smart', 'Smarter', 'Smelting', 'Snap', 'Social', 'Societal', 'Society', 'Socio', 'Sociology', 'Sociotechnical', 'Sock', 'Soft', 'Software', 'Softwarisation', 'Solar', 'Soldadura', 'Solid', 'Soluboard', 'Solution', 'Solutions', 'Solvent', 'Solvents', 'Some', 'Source', 'Sources', 'Sovereign', 'Sovereignty', 'Space', 'Spaces', 'Spanish', 'Spark', 'Spatial', 'Special', 'Specialisation', 'Specialist', 'Specialized', 'Specialty', 'Species', 'Specific', 'Specification', 'Specifications', 'Spectral', 'Spectrometers', 'Spectroscopic', 'Spectroscopy', 'Spectrum', 'Speech', 'Spheres', 'Spiking', 'Spin', 'Spine', 'Spinning', 'Spintronic', 'Spoke', 'Spoken', 'Sports', 'Spot', 'Sprint', 'Sprints', 'Squares', 'Squeezed', 'Stability', 'Stack', 'Staff', 'Stage', 'Stakeholder', 'Stakeholders', 'Standard', 'Standardisation', 'Standardization', 'Standards', 'Stars', 'Start', 'Starting', 'Startup', 'Startups', 'State', 'States', 'Station', 'Stations', 'Statistical', 'Steam', 'Steel', 'Steering', 'Step', 'Steps', 'Stereopsia', 'Stewardship', 'Stimulated', 'Stimulation', 'Stockholms', 'Stop', 'Storage', 'Storages', 'Store', 'Stories', 'Storytelling', 'Strategic', 'Strategies', 'Strategy', 'Streaming', 'Streams', 'Strengthening', 'Strengths', 'Stress', 'Strong', 'Structural', 'Structuration', 'Structure', 'Structured', 'Structures', 'Student', 'Students', 'Studies', 'Studio', 'Studios', 'Study', 'Styrene', 'Subarachnoid', 'Subcontracting', 'Subcontractor', 'Subcritical', 'Substance', 'Substances', 'Success', 'Successful', 'Such', 'Sufficient', 'Suitability', 'Suitable', 'Suite', 'Sulfur', 'Summaries', 'Summarisation', 'Summary', 'Summer', 'Summit', 'Summits', 'Super', 'Supercomputers', 'Supervise', 'Supervised', 'Supervision', 'Supply', 'Support', 'Supporting', 'Surface', 'Surgery', 'Surgical', 'Surrounding', 'Surveillance', 'Survey', 'Surveys', 'Sustainability', 'Sustainable', 'Swahili', 'Swan', 'Swappable', 'Swarm', 'Swarmchestrate', 'Swedish', 'Swept', 'Swiss', 'Switching', 'Syllabus', 'Symbiosis', 'Symbiotic', 'Symbolic', 'Symphony', 'Synchronisation', 'Synchrony', 'Syndicated', 'Syndrome', 'Synergies', 'Synergy', 'Synopsis', 'Synthesis', 'Synthesize', 'Synthetic', 'Sypin', 'System', 'Systematic', 'Systemic', 'Systems', 'Table', 'Tables', 'Tablet', 'Tactile', 'Tailings', 'Tailor', 'Taiwanese', 'Take', 'Taken', 'Talent', 'Talents', 'Tangible', 'Tank', 'Tanzanian', 'Tape', 'Target', 'Targeted', 'Targets', 'Task', 'Tasks', 'Taxonomy', 'Teach', 'Teaching', 'Team', 'Teaming', 'Tech', 'Technical', 'Technically', 'Technique', 'Techniques', 'Technische', 'Techno', 'Technological', 'Technologies', 'Technology', 'Techs', 'Tecnalia', 'Tecnologia', 'Tecnologias', 'Teknorot', 'Telco', 'Tele', 'Telecom', 'Telecommunication', 'Telecommunications', 'Telecoms', 'Telemetry', 'Teleworking', 'Temperature', 'Template', 'Tenant', 'Tenders', 'Tensor', 'Terephthalate', 'Term', 'Terminal', 'Terminals', 'Terms', 'Terrestrial', 'Territories', 'Test', 'Testbed', 'Testbeds', 'Testing', 'Text', 'Textile', 'Textiles', 'Theatre', 'Theme', 'Themes', 'Theoretical', 'Theory', 'There', 'Thermal', 'Thermo', 'Thermodynamics', 'Thermohuman', 'These', 'They', 'Things', 'Think', 'Thinking', 'Third', 'This', 'Thixotropic', 'Thought', 'Thoughts', 'Thread', 'Threads', 'Threat', 'Three', 'Thrombectomy', 'Through', 'Throughput', 'Tier', 'Tiered', 'Tilted', 'Time', 'Tiny', 'Tirana', 'Tissue', 'Titanate', 'Titanium', 'Together', 'Tokenisation', 'Tokens', 'Tolerant', 'Tomography', 'Tons', 'Tool', 'Toolbox', 'Toolkit', 'Toolkits', 'Toolpath', 'Tools', 'Toolset', 'Topic', 'Topics', 'Topological', 'Torque', 'Total', 'Tourism', 'Toxicity', 'Traceability', 'Track', 'Tract', 'Trade', 'Trades', 'Traditional', 'Traffic', 'Train', 'Trainer', 'Trainers', 'Training', 'Transactions', 'Transcontinuum', 'Transcriptomics', 'Transdisciplinary', 'Transducer', 'Transducers', 'Transfer', 'Transform', 'Transformation', 'Transformer', 'Transformers', 'Transient', 'Transilience', 'Transistor', 'Transit', 'Transition', 'Translate', 'Translation', 'Transmission', 'Transparency', 'Transparent', 'Transport', 'Transportation', 'Transversal', 'Travel', 'Traveling', 'Treatment', 'Trial', 'Triboelectric', 'Trinity', 'Tropical', 'Truck', 'True', 'Trully', 'Truly', 'Trust', 'Trusted', 'Trustworthiness', 'Trustworthy', 'Tuberculosis', 'Turbine', 'Turkish', 'Turnover', 'Tutor', 'Tutors', 'Tweezers', 'Twin', 'Twining', 'Twinning', 'Twins', 'Type', 'Ukrainian', 'Umbundu', 'Unavailability', 'Unconscious', 'Under', 'Undersea', 'Understanding', 'Understandings', 'Undertaking', 'Undertakings', 'Unexpected', 'Unforeseen', 'Unified', 'Unify', 'Union', 'Unions', 'Unique', 'Unit', 'Units', 'Unity', 'Universal', 'Universidad', 'Universite', 'Universiteit', 'Universitet', 'Universities', 'University', 'Unmanned', 'Unreal', 'Upcycling', 'Upskilling', 'Urban', 'Urbanism', 'Usability', 'Usable', 'Usage', 'Useful', 'User', 'Users', 'Using', 'Utility', 'Utilization', 'Vacuum', 'Valance', 'Validate', 'Validation', 'Valley', 'Valleytronic', 'Valleytronics', 'Valorisation', 'Valorization', 'Valuation', 'Value', 'Values', 'Vapor', 'Vapour', 'Variability', 'Vector', 'Vehicle', 'Vehicles', 'Vendor', 'Vendors', 'Verifiable', 'Verification', 'Vertical', 'Very', 'Vessel', 'Viable', 'Vibe', 'Video', 'Videoconferencing', 'Violet', 'Virtual', 'Virtualization', 'Vision', 'Visual', 'Visualisation', 'Visualization', 'Vitro', 'Vocabulary', 'Vocational', 'Voice', 'Volatile', 'Volcanic', 'Volcano', 'Volcanos', 'Voltaic', 'Volume', 'Volumetric', 'Voluntary', 'Voucher', 'Vulnerabilities', 'Vulnerability', 'Vulnerable', 'Waded', 'Wafer', 'Wallet', 'Ware', 'Warehouse', 'Warming', 'Waste', 'Wastewater', 'Water', 'Waterfall', 'Wave', 'Waveguide', 'Waveguides', 'Wavelength', 'Weakness', 'Weaknesses', 'Wearable', 'Weather', 'Weaving', 'Webinar', 'Webinars', 'Website', 'Weeder', 'Week', 'Weight', 'Welding', 'Well', 'Wellbeing', 'Wells', 'What', 'Wheel', 'Wheely', 'Where', 'Whereas', 'Which', 'While', 'White', 'Whole', 'Wide', 'Widening', 'Wider', 'Wiki', 'Wikidata', 'Wikipedia', 'Wildlife', 'Will', 'Wind', 'Wire', 'Wireless', 'Wiring', 'Wisely', 'With', 'Wolof', 'Women', 'Wood', 'Word', 'Wordalisation', 'Work', 'Workbench', 'Worker', 'Workers', 'Workflow', 'Workflows', 'Workforce', 'Working', 'Workload', 'Workpackage', 'Workpackages', 'Workplace', 'Workplan', 'Workprogram', 'Works', 'Workshop', 'Workshops', 'Workspace', 'Workspaces', 'Workstations', 'World', 'Worlds', 'Writing', 'Year', 'Yoruba', 'Young', 'Zero', 'Zinc', 'Zirconium', 'Zone', 'Zoonotic', 'Zurcher',
    ]

    #: Palabras comunes y nombres de iniciativas digitales de la UE a ignorar.
    BLACKLIST: List[str] = [
        'NAME', 'NUMBER', 'ACRONYM', 'TITLE', 'TRL', 'GANTT', 'PERT', 'FAIR', 'AI4EUROPE', 'CHIPS', 'CHIPS-JU', 'ECSEL', 'EDIH', 'DAIRO', 'DESTIN-E', 'EDMO', 'EFIS', 'ENISA', 'ERGA', 'EUROHPC', 'EUROHPC-JU', 'EUROCORES', 'ELRC',
        'EU-ROBOTICS', 'GAIA-X', 'GEANT', 'IPCEI', 'FLAGSHIP', 'TESTA', 'EFFRA', 'AI4EU', 'EOSC',
        'HORIZON', 'DIGITAL', 'EUROPE', 'P4PLANET', 'IAM4EU', 'PHOTONICS21',
    ]

    #: Entidades (siglas) con 3 o más participaciones a reemplazar por "[NAME]".
    ENTITIES: List[str] = [
        'AAU', 'ABI', 'AIT', 'ALS', 'AMU', 'ARC', 'ART', 'ASC', 'ATB', 'ATC', 'AUS', 'AVL', 'BBC',
        'BED', 'BIU', 'BME', 'BSC', 'CAP', 'CBS', 'CCT', 'CEA', 'CNR', 'CRF', 'DBC', 'DCU', 'DLR',
        'DTI', 'DTU', 'EAB', 'ECL', 'ECO', 'EDF', 'EGM', 'ENG', 'EUR', 'EUT', 'EWF', 'F6S', 'FAU',
        'FBA', 'FBK', 'FHG', 'FPG', 'FZJ', 'GFT', 'HHI', 'HIT', 'HPE', 'HUB', 'IBM', 'IIT', 'IKL',
        'IMC', 'IMM', 'IMT', 'ING', 'ITA', 'ITI', 'JSI', 'K3Y', 'KIT', 'KTH', 'KUL', 'LIC', 'LIU',
        'LMS', 'LTU', 'LUH', 'MAG', 'MDH', 'MED', 'MOD', 'MPG', 'NEC', 'NOK', 'NXP', 'NXW', 'ORU',
        'OW2', 'PAL', 'PPC', 'PRE', 'RAI', 'ROB', 'RUB', 'SAL', 'SFU', 'SIE', 'STM', 'SYN', 'TAU',
        'TEC', 'TEI', 'TEK', 'THL', 'TID', 'TNI', 'TNO', 'TSI', 'TUB', 'TUC', 'TUD', 'TUE', 'TUK',
        'TUM', 'TUW', 'UAB', 'UAM', 'UBI', 'UBX', 'UCC', 'UCD', 'UCL', 'UCM', 'UCY', 'UIA', 'UIO',
        'ULB', 'UMA', 'UMU', 'UNE', 'UNI', 'UOB', 'UOM', 'UPB', 'UPC', 'UPF', 'UPM', 'UPV', 'UR1',
        'USE', 'UTH', 'UVA', 'VIF', 'VIV', 'VLC', 'VRT', 'VTT', 'VUB', 'WLT', 'WSE', 'WWU',
    ]

    #: Columnas que se cifran por defecto con AES-GCM.
    DEFAULT_ENCRYPT_COLUMNS = ["Number", "Acronym", "Title", "Abstract"]

    #: Regex reutilizados por la lógica de anonimización contextual.
    CONTEXT_PATTERN = r"(?<!^[\s\n])(?<![.!?]\s)\b[A-Z][a-z]{3,}\b"
    DOUBLE_WORD_PATTERN = r"\b[A-Z][A-Z0-9-]{2,}[A-Z0-9]\s+[A-Z][A-Z0-9-]{1,}[A-Z0-9]\b"
    ACRONYM_PATTERN = r"\b[A-Z][A-Z0-9-]{1,}[A-Z]\b"

    def __init__(
        self,
        cipher: Optional["AESGCM"] = None,
        blacklist: Optional[Iterable[str]] = None,
        entities: Optional[Iterable[str]] = None,
        whitelist: Optional[Iterable[str]] = None,
    ) -> None:
        self.cipher = cipher
        self.blacklist = set(blacklist) if blacklist is not None else set(self.BLACKLIST)
        self.entities = set(entities) if entities is not None else set(self.ENTITIES)
        self.whitelist = set(whitelist) if whitelist is not None else set(self.WHITELIST)

    # -- gestión de la clave AES-GCM ------------------------------------
    @classmethod
    def load_cipher_from_colab_secret(cls, secret_name: str = "AESGCM_KEY") -> Optional["AESGCM"]:
        """Carga la clave AES-GCM desde Colab Secrets (equivalente a
        ``load_aesgcm_cipher``). Devuelve ``None`` si no está disponible."""
        if AESGCM is None:
            logger.info("⚠️ El paquete 'cryptography' no está instalado.")
            return None
        if not IN_COLAB:
            logger.info("⚠️ Colab Secrets no está disponible fuera de Google Colab.")
            return None
        try:
            base64_key_str = colab_userdata.get(secret_name)
            if not base64_key_str:
                raise ValueError(f"Secret '{secret_name}' not found or is empty.")
            aesgcm_key = base64.urlsafe_b64decode(base64_key_str)
            logger.info(f"✅ AES-GCM key loaded from Colab Secrets: '{secret_name}'")
            return AESGCM(aesgcm_key)
        except Exception as e:  # noqa: BLE001
            logger.info(f"⚠️ Error loading AES-GCM key from secrets: {e}")
            logger.info("Please generate an AES-GCM key and save it to Colab Secrets.")
            return None

    @classmethod
    def load_cipher_from_env(cls, env_var: str = "AESGCM_KEY") -> Optional["AESGCM"]:
        """Carga la clave AES-GCM (base64 urlsafe) desde una variable de entorno."""
        if AESGCM is None:
            logger.info("⚠️ El paquete 'cryptography' no está instalado.")
            return None
        base64_key_str = os.environ.get(env_var)
        if not base64_key_str:
            logger.info(f"⚠️ Variable de entorno '{env_var}' no definida.")
            return None
        aesgcm_key = base64.urlsafe_b64decode(base64_key_str)
        return AESGCM(aesgcm_key)

    @classmethod
    def generate_key(cls, bit_length: int = 128) -> "AESGCM":
        """Genera una clave AES-GCM temporal (SOLO para pruebas, no producción)."""
        if AESGCM is None:
            raise RuntimeError("El paquete 'cryptography' no está instalado.")
        return AESGCM(AESGCM.generate_key(bit_length=bit_length))

    # -- cifrado / descifrado de un valor individual ----------------------
    def encode_value(self, message: str) -> str:
        """Cifra un string con AES-GCM y devuelve nonce+ciphertext en hexadecimal."""
        if not isinstance(self.cipher, AESGCM):
            raise RuntimeError("AES-GCM cipher not initialized. Check key loading.")
        message_bytes = message.encode("utf-8")
        nonce = os.urandom(12)
        ciphertext = self.cipher.encrypt(nonce, message_bytes, None)
        return (nonce + ciphertext).hex()

    def decode_value(self, encrypted_data_hex: str) -> str:
        """Descifra un string previamente cifrado con :meth:`encode_value`."""
        if not isinstance(self.cipher, AESGCM):
            raise RuntimeError("AES-GCM cipher not initialized. Check key loading.")
        encrypted_data_bytes = bytes.fromhex(encrypted_data_hex)
        nonce = encrypted_data_bytes[:12]
        ciphertext = encrypted_data_bytes[12:]
        decrypted_bytes = self.cipher.decrypt(nonce, ciphertext, None)
        return decrypted_bytes.decode("utf-8")

    # -- anonimización de texto libre --------------------------------------
    def _replace_caps(self, match: "re.Match") -> str:
        word = match.group(0)
        if word in self.entities:
            return "[NAME]"
        return "[NAME]" if word not in self.blacklist else word

    def _context_anonymize(self, match: "re.Match") -> str:
        word = match.group(0)
        if word in self.whitelist:
            return word
        return "[NAME]"

    def _clean_text(self, text: Any, number: Any, acronym: Any, title: Any) -> str:
        if not isinstance(text, str) or not text:
            return ""

        if number and str(number) != "No encontrado":
            text = text.replace(str(number), "[NUMBER]")
        if acronym and str(acronym) != "No encontrado":
            text = text.replace(str(acronym), "[ACRONYM]")
        if title and str(title) != "No encontrado":
            text = text.replace(str(title), "[TITLE]")

        text = re.sub(self.CONTEXT_PATTERN, self._context_anonymize, text)
        text = re.sub(self.DOUBLE_WORD_PATTERN, "[NAME]", text)
        text = re.sub(self.ACRONYM_PATTERN, self._replace_caps, text)
        return text

    def anonymize(
        self,
        df: pd.DataFrame,
        encrypt_cols: Optional[List[str]] = None,
        encrypt: bool = True,
    ) -> pd.DataFrame:
        """Anonimiza los criterios de texto libre y, opcionalmente, cifra
        columnas sensibles. Equivalente a ``ESR_encode``.

        Parámetros
        ----------
        df: DataFrame con los datos de ESRs.
        encrypt_cols: columnas a cifrar (por defecto ``DEFAULT_ENCRYPT_COLUMNS``).
        encrypt: si ``False``, no se realiza cifrado aunque haya un ``cipher``
            configurado (solo se anonimiza el texto libre).
        """
        df_anon = df.copy()

        numbers = df_anon.get("Number", [""] * len(df_anon))
        acronyms = df_anon.get("Acronym", [""] * len(df_anon))
        titles = df_anon.get("Title", [""] * len(df_anon))

        crit_cols = ["C_Excellence", "C_Impact", "C_Implement"]
        has_expanded_crit = all(col in df_anon.columns for col in crit_cols)

        if has_expanded_crit:
            for col in crit_cols:
                df_anon[col] = [
                    self._clean_text(text, num, acr, tit)
                    for text, num, acr, tit in zip(df_anon[col], numbers, acronyms, titles)
                ]
        elif "Criterion" in df_anon.columns:
            df_anon["Criterion"] = [
                [self._clean_text(c, num, acr, tit) for c in crits]
                if isinstance(crits, list)
                else crits
                for crits, num, acr, tit in zip(df_anon["Criterion"], numbers, acronyms, titles)
            ]

        if encrypt and self.cipher is not None:
            if encrypt_cols is None:
                encrypt_cols = self.DEFAULT_ENCRYPT_COLUMNS
            for col in encrypt_cols:
                if col in df_anon.columns:
                    df_anon[col] = [self.encode_value(str(val)) for val in df_anon[col]]

        return df_anon

    def deanonymize(self, df: pd.DataFrame, target_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Descifra las columnas cifradas con AES-GCM. Equivalente a ``ESR_decode``.

        Nota: la anonimización de texto libre (sustitución por ``[NAME]``,
        ``[NUMBER]``...) no es reversible por diseño; sólo se revierte el
        cifrado AES-GCM.
        """
        if self.cipher is None:
            logger.info(
                "⚠️ No se ha proporcionado la clave AESGCM. Se devuelve el "
                "DataFrame sin modificaciones."
            )
            return df

        df_deanon = df.copy()
        target_cols = target_cols or self.DEFAULT_ENCRYPT_COLUMNS

        for col in target_cols:
            if col in df_deanon.columns:
                df_deanon[col] = [
                    self.decode_value(str(val)) if pd.notna(val) else val
                    for val in df_deanon[col]
                ]
        return df_deanon


# =============================================================================
# ESRExporter  --------------------------------------------------- Excel/CSV/Parquet
# =============================================================================
class ESRExporter:
    """Lee/escribe DataFrames de ESRs en Excel, CSV y Parquet, incluyendo los
    flujos de cifrado/descifrado de Excel (equivalente a ``EXCEL_encode`` /
    ``EXCEL_decode``)."""

    def __init__(self, anonymizer: Optional[ESRAnonymizer] = None) -> None:
        self.anonymizer = anonymizer

    # -- exportación genérica ---------------------------------------------
    @staticmethod
    def to_excel(df: pd.DataFrame, output_path: str, sheet_name: Union[str, int] = 0) -> str:
        out_sheet = sheet_name if isinstance(sheet_name, str) else "Sheet1"
        df.to_excel(output_path, index=False, sheet_name=out_sheet)
        logger.info(f"📄 Excel guardado correctamente en: {output_path}")
        return output_path

    @staticmethod
    def to_csv(df: pd.DataFrame, output_path: str, **kwargs: Any) -> str:
        df.to_csv(output_path, index=False, **kwargs)
        logger.info(f"📄 CSV guardado correctamente en: {output_path}")
        return output_path

    @staticmethod
    def to_parquet(df: pd.DataFrame, output_path: str, **kwargs: Any) -> str:
        df.to_parquet(output_path, index=False, **kwargs)
        logger.info(f"📄 Parquet guardado correctamente en: {output_path}")
        return output_path

    @staticmethod
    def read_excel(input_path: str, sheet_name: Union[str, int] = 0) -> pd.DataFrame:
        return pd.read_excel(input_path, sheet_name=sheet_name)

    # -- flujos de cifrado/descifrado sobre Excel --------------------------
    def excel_encode(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        columns: Optional[List[str]] = None,
        sheet_name: Union[str, int] = 0,
    ) -> pd.DataFrame:
        """Lee un Excel, anonimiza/cifra con ``self.anonymizer`` y opcionalmente
        guarda el resultado. Equivalente a ``EXCEL_encode``."""
        if self.anonymizer is None:
            raise RuntimeError("Se necesita un ESRAnonymizer para excel_encode().")

        df = self.read_excel(input_path, sheet_name=sheet_name)
        df_encoded = self.anonymizer.anonymize(df, encrypt_cols=columns)

        if output_path:
            self.to_excel(df_encoded, output_path, sheet_name=sheet_name)

        return df_encoded

    def excel_decode(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        columns: Optional[List[str]] = None,
        sheet_name: Union[str, int] = 0,
    ) -> pd.DataFrame:
        """Lee un Excel cifrado, descifra con ``self.anonymizer`` y opcionalmente
        guarda el resultado. Equivalente a ``EXCEL_decode``."""
        if self.anonymizer is None:
            raise RuntimeError("Se necesita un ESRAnonymizer para excel_decode().")

        df = self.read_excel(input_path, sheet_name=sheet_name)
        df_decoded = self.anonymizer.deanonymize(df, target_cols=columns)

        if output_path:
            self.to_excel(df_decoded, output_path, sheet_name=sheet_name)
            logger.info(f"📄 Excel desencriptado guardado correctamente en: {output_path}")

        return df_decoded

    # -- exportación con formato inferido por extensión --------------------
    def export(self, df: pd.DataFrame, output_path: str, **kwargs: Any) -> str:
        ext = os.path.splitext(output_path)[1].lower()
        if ext in (".xlsx", ".xls"):
            return self.to_excel(df, output_path, **kwargs)
        if ext == ".csv":
            return self.to_csv(df, output_path, **kwargs)
        if ext == ".parquet":
            return self.to_parquet(df, output_path, **kwargs)
        raise ValueError(f"Formato de exportación no soportado: {ext}")


# =============================================================================
# ESRPipeline  --------------------------------------------------- orquestación end-to-end
# =============================================================================
class ESRExtractionPipeline:
    """Orquesta ``ESRExtractor`` -> ``ESRZipProcessor`` -> ``ESRAnonymizer`` ->
    ``ESRExporter`` como un único pipeline ejecutable."""

    def __init__(
        self,
        extractor: Optional[ESRExtractor] = None,
        zip_processor: Optional[ESRZipProcessor] = None,
        anonymizer: Optional[ESRAnonymizer] = None,
        exporter: Optional[ESRExporter] = None,
    ) -> None:
        self.extractor = extractor or ESRExtractor()
        self.zip_processor = zip_processor or ESRZipProcessor(extractor=self.extractor)
        self.anonymizer = anonymizer or ESRAnonymizer()
        self.exporter = exporter or ESRExporter(anonymizer=self.anonymizer)

    def run(
        self,
        zip_source_paths: Optional[Sequence[str]] = None,
        output_dir: str = "esr_output",
        anonymize: bool = False,
        encrypt: bool = False,
        export_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ejecuta el pipeline completo: descomprime ZIPs, extrae datos de los
        PDFs, opcionalmente anonimiza/cifra, y opcionalmente exporta.

        Devuelve un diccionario con: ``dataframe``, ``error_log_path``,
        ``errors_occurred`` y (si se exportó) ``export_path``.
        """
        df, error_log_path, errors_occurred = self.zip_processor.unzip(
            zip_source_paths=zip_source_paths, output_dir=output_dir
        )

        if anonymize and not df.empty:
            df = self.anonymizer.anonymize(df, encrypt=encrypt)

        result: Dict[str, Any] = {
            "dataframe": df,
            "error_log_path": error_log_path,
            "errors_occurred": errors_occurred,
        }

        if export_path and not df.empty:
            result["export_path"] = self.exporter.export(df, export_path)

        return result


# =============================================================================
# =============================================================================
# ETAPA 2 · CLASIFICACIÓN  (frases negativas -> topics por similitud semántica)
# Clases: TextPreprocessor, TopicIndex, ESRAnalyzer, ESRReportGenerator,
#         ESRSummaryVisualizer, ESRClassificationPipeline, ESRClassificationIO
# =============================================================================
# =============================================================================
VERDICT_KEYWORDS = frozenset({"shortcoming", "weakness"})
MIN_VERDICT_SENTENCE_LEN = 60

NEGATIVE_PATTERNS: list[str] = [
    r"\bshortcoming(?:s)?\b",
    r"\bweakness(?:es)?\b",
    r"\blimit(?:ed|ations?)?\b",
    r"\bincremental(?:ly)?\b",
    r"\black(?:s|ing)?\b",
    r"\binsufficient(?:ly)?\b",
    r"\bunclear(?:ly)?\b",
    r"\bvague(?:ly|ness)?\b",
    r"\bnot\s+(?:sufficiently|fully|clearly|completely|adequate(?:ly)?|convincing|described|addressed)\b",
]

NOISE_PATTERNS: list[str] = [
    r"\bnot\s+(?:adequately|sufficiently)\b",
    r"\binsufficiently\b",
    r"\badequately\b",
    r"\bsufficiently\b",
    r"\bconvincing\b",
    r"\bthis\s+is\s+a\s+(?:weakness|shortcoming)\b",
    r"\bas\s+a\s+result\b",
    r"\bhowever\b",
    r"\bmoreover\b",
    r"\binadequately\b",
]

TOPICS_EXCELLENCE: dict[str, str] = {
    "objectives_and_ambition": """
        Project objectives, scientific objectives, research goals, project vision,
        ambition, innovation ambition, expected achievements, scope, alignment with
        the call topic, relevance to the work programme, scientific excellence,
        novelty of the concept, strategic objectives, project rationale,
        clarity of objectives, measurable goals.
    """,
    "state_of_the_art": """
        State of the art, existing knowledge, literature review, prior work,
        competing approaches, scientific baseline, technological baseline,
        research gap, innovation beyond the state of the art, originality,
        novelty compared to existing solutions, scientific advances,
        current limitations of existing methods.
    """,
    "methodology": """
        Scientific methodology, research methods, experimental methodology,
        project approach, work methodology, scientific approach,
        use cases, datasets, data collection, data preparation,
        training data, validation methodology, experimental protocol,
        scientific justification, assumptions, reproducibility,
        technical feasibility, AI models, model training,
        workflow, scientific evidence.
    """,
    "design": """
        System architecture, technical architecture, software architecture,
        cloud-edge architecture, distributed architecture,
        component integration, interoperability,
        technical design, engineering design,
        algorithms, AI algorithms, optimisation methods,
        communication protocols, APIs,
        cybersecurity, resilience, robustness,
        software components, hardware components,
        sensing configuration, orchestration,
        digital infrastructure, technical implementation.
    """,
    "validation": """
        Validation, verification, evaluation methodology,
        performance assessment, benchmarks,
        KPIs, metrics, quantitative targets,
        testing procedures, pilot demonstrations,
        demonstrators, experimental validation,
        baseline comparison, performance evidence,
        scalability validation, robustness testing,
        cost analysis, energy consumption evaluation,
        technology readiness, acceptance criteria.
    """,
    "interdisciplinarity_and_gender": """
        Interdisciplinary research, multidisciplinary collaboration,
        cross-disciplinary integration, social sciences,
        humanities integration, gender dimension,
        sex and gender analysis, diversity considerations,
        inclusiveness, responsible research and innovation.
    """,
    "open_science": """
        Open science, FAIR principles,
        data management plan, DMP,
        open access publications,
        open-source software,
        open-source licences,
        repositories, research data sharing,
        reproducibility, scientific transparency,
        metadata standards, data stewardship.
    """,
    "engagement": """
        Stakeholder engagement,
        end-user involvement,
        citizen engagement,
        industrial participation,
        co-creation,
        user requirements,
        participatory design,
        consultation activities,
        living labs,
        stakeholder feedback.
    """,
    "concepts_and_models": """
        Conceptual framework,
        theoretical framework,
        scientific concepts,
        reference model,
        mathematical model,
        simulation model,
        digital twin,
        ontology,
        modelling assumptions,
        analytical framework,
        conceptual architecture.
    """,
    "resources": """
        Budget, requested funding,
        budget justification,
        financial resources,
        financial feasibility,
        resource allocation,
        person-month allocation,
        staffing,
        human resources,
        availability of personnel,
        equipment costs,
        travel costs,
        procurement,
        infrastructure costs,
        operational costs,
        cost efficiency,
        value for money,
        financial planning.
    """,
}

TOPICS_IMPACT: dict[str, str] = {
    "pathways_to_impact": (
        "Credibility of pathways, impact pathway, causal links, expected outcomes, "
        "theory of change, and strategic plan alignment."
    ),
    "scale_and_significance": (
        "Scale of contribution, significance of impact, potential reach, European "
        "dimensions, global impact, and quantified performance indicators."
    ),
    "scientific_impact": (
        "New scientific opportunities, reinforcing European research, academic "
        "impact, advancing scientific fields, and world-class knowledge creation."
    ),
    "economic_and_technological_impact": (
        "Economic growth, industrial competitiveness, market creation, job "
        "creation, European technological sovereignty, ROI, and business growth."
    ),
    "societal_and_environmental_impact": (
        "Societal benefits, public health, environmental sustainability, climate "
        "change mitigation, Green Deal alignment, and Sustainable Development "
        "Goals (SDG)."
    ),
    "dissemination_plan": (
        "Dissemination strategy, scientific publications, targeted dissemination, "
        "conference presentations, workshops, and sharing results with "
        "stakeholders."
    ),
    "exploitation_strategy": (
        "Exploitation plan, market uptake, commercialisation plan, business "
        "models, go-to-market strategy, and product or service deployment."
    ),
    "intellectual_property_management": (
        "IPR strategy, intellectual property rights, patenting, licensing, "
        "freedom to operate (FTO), and consortium asset protection."
    ),
    "communication_activities": (
        "Communication plan, public outreach, citizen awareness, media coverage, "
        "social media strategy, project website, and public engagement."
    ),
    "policy_and_standardization": (
        "Policy contribution, policy briefs, regulatory frameworks, "
        "standardisation activities, contribution to standards, and "
        "policy-makers engagement."
    ),
    "barriers_to_impact": (
        "Potential barriers, market hurdles, regulatory obstacles, user "
        "acceptance, socio-economic barriers, and impact risks."
    ),
}

TOPICS_IMPLEMENTATION: dict[str, str] = {
    "work_plan_structure": (
        "Work plan, work package structure, WP breakdown, task dependencies, "
        "project timeline, Gantt chart, and workflow organisation."
    ),
    "resource_allocation": (
        "Person-months, effort allocation, PM distribution, budget "
        "justification, equipment and travel costs, and cost-effectiveness."
    ),
    "milestones_and_deliverables": (
        "Milestones, deliverables, deliverable schedule, verification means, "
        "acceptance criteria, reporting schedule, and project outputs."
    ),
    "consortium_composition_and_synergy": (
        "Consortium composition, partner complementarity, complementary "
        "expertise, partner roles, industrial/academic participation, and "
        "partnership strength."
    ),
    "participant_capacity_and_expertise": (
        "Operational capacity, institutional expertise, technical competence, "
        "track record, key personnel, and available research infrastructure."
    ),
    "project_governance_and_management": (
        "Project management, governance structure, decision-making process, "
        "coordinator role, PMO, internal communication, and conflict "
        "resolution."
    ),
    "risk_management_and_contingency": (
        "Risk assessment, critical risks, risk probability and impact, risk "
        "mitigation measures, contingency plans, and fallback strategies."
    ),
    "quality_assurance_and_monitoring": (
        "Quality assurance, continuous progress monitoring, internal reviews, "
        "advisory board, reporting mechanisms, and performance KPIs."
    ),
    "data_and_infrastructure_management": (
        "Computing and storage infrastructure, hosting environment, cloud "
        "infrastructure, operational support, access rights, and data storage."
    ),
    "innovation_management": (
        "Innovation management, technology transfer, knowledge management, IP "
        "management, exploitation planning, and innovation roadmaps."
    ),
    "communication": (
        "Communication activities, outreach, project website, social media, "
        "branding, public awareness, media relations, and promotional "
        "material."
    ),
    "dissemination": (
        "Dissemination strategy, scientific dissemination, knowledge transfer, "
        "publications, stakeholder communication, and sharing results."
    ),
    "exploitation": (
        "Exploitation strategy, market uptake, commercialisation, technology "
        "transfer, business development, spin-offs, and business models."
    ),
    "subcontracting_and_third_parties": (
        "Subcontracting, third-party involvement, linked third parties, "
        "affiliated entities, external expertise, procurement, and external "
        "services."
    ),
}

ALL_TOPIC_CATEGORIES: dict[str, dict[str, str]] = {
    "C_Excellence": TOPICS_EXCELLENCE,
    "C_Impact": TOPICS_IMPACT,
    "C_Implement": TOPICS_IMPLEMENTATION,
}

SCORE_COLUMN_MAP: dict[str, str] = {
    "C_Excellence": "S_Excellence",
    "C_Impact": "S_Impact",
    "C_Implement": "S_Implement",
}

BLOCK_SHEET_NAMES = {
    "C_Excellence": "Excellence",
    "C_Impact": "Impact",
    "C_Implement": "Implementation",
}

BLOCK_SUMMARY_SHEET_NAMES = {
    "C_Excellence": "Excellence_Summary",
    "C_Impact": "Impact_Summary",
    "C_Implement": "Implementation_Summary",
}

REQUIRED_METADATA_COLUMNS = ["Activity", "Call", "Number"]

SIMILARITY_THRESHOLD = 0.30
MODEL_NAME = "BAAI/bge-m3"
ENCODE_BATCH_SIZE = 64

IQR_MULTIPLIER = 1.5
AMBIGUITY_RATIO_THRESHOLD = 0.98
MIN_GROUP_SIZE_FOR_IQR = 4

DEFAULT_TOP_N_SENTENCES = 10

RESULT_COLUMNS = [
    "sentence",
    "keyword",
    "dist",
    "Call",
    "Activity",
    "Number",
    "Score",
    "Block",
    "classification",
]

_negative_re = re.compile("|".join(NEGATIVE_PATTERNS), re.IGNORECASE)
_noise_re = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)


# ============================================================
# 1. Text Preprocessing
# ============================================================
class TextPreprocessor:
    @classmethod
    def extract_negative_sentences(cls, text: str) -> list[str]:
        sentences = nltk.sent_tokenize(text)
        result = []
        for sentence in sentences:
            clean = sentence.replace("\n", " ").strip()
            if clean and _negative_re.search(clean):
                result.append(clean)
        return result

    @classmethod
    def is_verdict_only(cls, sentence: str) -> bool:
        lowered = sentence.lower()
        if not any(kw in lowered for kw in VERDICT_KEYWORDS):
            return False
        return len(sentence.strip()) < MIN_VERDICT_SENTENCE_LEN

    @classmethod
    def clean_for_embedding(cls, sentence: str) -> str:
        cleaned = _noise_re.sub(" ", sentence.lower())
        return re.sub(r"\s+", " ", cleaned).strip()


# ============================================================
# 2. Topic Index
# ============================================================
@dataclass
class TopicIndex:
    model: SentenceTransformer
    data: dict[str, dict] = field(default_factory=dict)
    encode_batch_size: int = ENCODE_BATCH_SIZE
    similarity_threshold: float = SIMILARITY_THRESHOLD

    @classmethod
    def build(
        cls,
        model_name: str = MODEL_NAME,
        topic_categories: dict[str, dict[str, str]] | None = None,
        device: str | None = None,
        batch_size: int = ENCODE_BATCH_SIZE,
    ) -> TopicIndex:
        _ensure_sentence_transformers()
        model = SentenceTransformer(model_name, device=device)
        index = cls(model=model, encode_batch_size=batch_size)
        categories = topic_categories or ALL_TOPIC_CATEGORIES
        for category_name, topic_dict in categories.items():
            names = list(topic_dict.keys())
            embeddings = model.encode(
                list(topic_dict.values()),
                convert_to_tensor=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )
            index.data[category_name] = {"names": names, "embeddings": embeddings}
        return index

    def _score_sentences(self, sentences: list[str], topic_category: str):
        topic_data = self.data.get(topic_category)
        if not topic_data:
            raise ValueError(f"Unknown topic category: {topic_category}")
        cleaned = [TextPreprocessor.clean_for_embedding(s) for s in sentences]
        embeddings = self.model.encode(
            cleaned,
            convert_to_tensor=True,
            batch_size=self.encode_batch_size,
            show_progress_bar=False,
        )
        similarity_matrix = util.cos_sim(embeddings, topic_data["embeddings"])
        return topic_data["names"], similarity_matrix

    def classify_detailed(
        self, sentences: list[str], topic_category: str = "C_Excellence", top_k: int = 1
    ) -> list[dict]:
        topic_names, similarity_matrix = self._score_sentences(sentences, topic_category)
        results = []
        for i in range(len(sentences)):
            scores = similarity_matrix[i]
            k = min(top_k, len(topic_names))
            top_scores, top_indices = scores.topk(k)
            matches = [
                (topic_names[idx.item()], score.item())
                for score, idx in zip(top_scores, top_indices)
                if score.item() >= self.similarity_threshold
            ]
            if not matches:
                matches = [("unclassified", scores.max().item())]
            k2 = min(2, len(topic_names))
            top2_scores, top2_indices = scores.topk(k2)
            primary_topic = topic_names[top2_indices[0].item()]
            primary_val = top2_scores[0].item()
            secondary_topic = topic_names[top2_indices[1].item()] if k2 >= 2 else None
            secondary_val = top2_scores[1].item() if k2 >= 2 else float("nan")
            results.append(
                {
                    "matches": matches,
                    "primary_topic": primary_topic,
                    "primary_val": primary_val,
                    "secondary_topic": secondary_topic,
                    "secondary_val": secondary_val,
                }
            )
        return results


# ============================================================
# 3. Analyzer
# ============================================================
class ESRAnalyzer:
    def __init__(self, topic_index: TopicIndex):
        self.topic_index = topic_index

    def analyse_text(
        self, text: str, topic_category: str = "C_Excellence", top_k: int = 1
    ) -> pd.DataFrame:
        cols = [
            "sentence",
            "topic",
            "val",
            "primary_topic",
            "primary_val",
            "secondary_topic",
            "secondary_val",
        ]
        if not isinstance(text, str):
            return pd.DataFrame(columns=cols)

        negative_sentences = TextPreprocessor.extract_negative_sentences(text)
        valid_sentences = [
            s for s in negative_sentences if not TextPreprocessor.is_verdict_only(s)
        ]
        if not valid_sentences:
            return pd.DataFrame(columns=cols)

        detailed_results = self.topic_index.classify_detailed(
            valid_sentences, topic_category=topic_category, top_k=top_k
        )

        rows = []
        for sentence, info in zip(valid_sentences, detailed_results):
            for topic, score in info["matches"]:
                rows.append(
                    {
                        "sentence": sentence,
                        "topic": topic,
                        "val": score,
                        "primary_topic": info["primary_topic"],
                        "primary_val": info["primary_val"],
                        "secondary_topic": info["secondary_topic"],
                        "secondary_val": info["secondary_val"],
                    }
                )
        return pd.DataFrame(rows)


# ============================================================
# 4. Report Generator
# ============================================================
class ESRReportGenerator:
    """Outlier detection, sheet building, and pivot table formatting."""

    IQR_MULTIPLIER = IQR_MULTIPLIER
    AMBIGUITY_RATIO_THRESHOLD = AMBIGUITY_RATIO_THRESHOLD
    RESULT_COLUMNS = RESULT_COLUMNS

    @classmethod
    def flag_outliers_and_ambiguous(
        cls,
        df: pd.DataFrame,
        topic_col: str = "primary_topic",
        val_primary_col: str = "primary_val",
        val_secondary_col: str = "secondary_val",
        iqr_multiplier: float = IQR_MULTIPLIER,
        ratio_threshold: float = AMBIGUITY_RATIO_THRESHOLD,
        min_group_size: int = MIN_GROUP_SIZE_FOR_IQR,
        keep_intermediate_columns: bool = False,
    ) -> pd.DataFrame:
        df_processed = df.copy()
        if df_processed.empty:
            df_processed["classification_status"] = pd.Series(dtype="object")
            return df_processed

        df_processed["dist_primary"] = 1 - df_processed[val_primary_col]
        df_processed["dist_secondary"] = 1 - df_processed[val_secondary_col]

        df_processed["is_outlier"] = False
        for topic in df_processed[topic_col].dropna().unique():
            topic_mask = df_processed[topic_col] == topic
            distances = df_processed.loc[topic_mask, "dist_primary"]
            if len(distances) < min_group_size:
                continue
            q1 = np.percentile(distances, 25)
            q3 = np.percentile(distances, 75)
            iqr = q3 - q1
            upper_fence = q3 + (iqr_multiplier * iqr)
            outlier_mask = topic_mask & (df_processed["dist_primary"] > upper_fence)
            df_processed.loc[outlier_mask, "is_outlier"] = True

        with np.errstate(divide="ignore", invalid="ignore"):
            df_processed["ambiguity_ratio"] = (
                df_processed["dist_primary"] / df_processed["dist_secondary"]
            )
        df_processed["is_ambiguous"] = (
            df_processed["ambiguity_ratio"] > ratio_threshold
        ) & df_processed[val_secondary_col].notna()

        df_processed["classification_status"] = "Valid"
        df_processed.loc[df_processed["is_ambiguous"], "classification_status"] = "Ambiguous"
        df_processed.loc[df_processed["is_outlier"], "classification_status"] = "Outlier"

        if not keep_intermediate_columns:
            df_processed = df_processed.drop(
                columns=[
                    "dist_primary",
                    "dist_secondary",
                    "is_outlier",
                    "is_ambiguous",
                    "ambiguity_ratio",
                ],
                errors="ignore",
            )
        return df_processed

    @classmethod
    def build_all_raw_data_sheet(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Build All_raw_data sheet.

        Columns: [sentence, keyword, dist, Call, Activity, Number, Score, Block, classification]
        - Valid rows: keyword=primary_topic, dist=primary_val
        - Ambiguous rows: duplicated (1st=primary, 2nd=secondary)
        """
        if df.empty or "classification_status" not in df.columns:
            return pd.DataFrame(columns=cls.RESULT_COLUMNS)

        df_valid_amb = df[df["classification_status"].isin(["Valid", "Ambiguous"])].copy()
        if df_valid_amb.empty:
            return pd.DataFrame(columns=cls.RESULT_COLUMNS)

        primary_rows = pd.DataFrame(
            {
                "sentence": df_valid_amb["sentence"],
                "keyword": df_valid_amb["primary_topic"],
                "dist": df_valid_amb["primary_val"],
                "Call": df_valid_amb["Call"],
                "Activity": df_valid_amb["Activity"],
                "Number": df_valid_amb["Number"],
                "Score": df_valid_amb["Score"],
                "Block": df_valid_amb["Block"],
                "classification": df_valid_amb["classification_status"],
            }
        )

        ambiguous_mask = df_valid_amb["classification_status"] == "Ambiguous"
        if ambiguous_mask.any():
            amb = df_valid_amb[ambiguous_mask].copy()
            secondary_rows = pd.DataFrame(
                {
                    "sentence": amb["sentence"],
                    "keyword": amb["secondary_topic"],
                    "dist": amb["secondary_val"],
                    "Call": amb["Call"],
                    "Activity": amb["Activity"],
                    "Number": amb["Number"],
                    "Score": amb["Score"],
                    "Block": amb["Block"],
                    "classification": amb["classification_status"],
                }
            )
            combined = pd.concat([primary_rows, secondary_rows], ignore_index=True)
        else:
            combined = primary_rows

        return combined[cls.RESULT_COLUMNS]

    @classmethod
    def build_unpivoted_block_sheet(cls, df: pd.DataFrame, block_name: str) -> pd.DataFrame:
        """Build unpivoted block DataFrame selecting non-outlier rows.

        dist = 1.0 for Valid, 0.5 for Ambiguous (for both rows).
        """
        if df.empty or "classification_status" not in df.columns:
            return pd.DataFrame(columns=cls.RESULT_COLUMNS)

        df_block = df[
            (df["Block"] == block_name)
            & (df["classification_status"].isin(["Valid", "Ambiguous"]))
        ].copy()
        if df_block.empty:
            return pd.DataFrame(columns=cls.RESULT_COLUMNS)

        primary_rows = pd.DataFrame(
            {
                "sentence": df_block["sentence"],
                "keyword": df_block["primary_topic"],
                "Call": df_block["Call"],
                "Activity": df_block["Activity"],
                "Number": df_block["Number"],
                "Score": df_block["Score"],
                "Block": df_block["Block"],
                "classification": df_block["classification_status"],
            }
        )
        primary_rows["dist"] = np.where(
            df_block["classification_status"].eq("Valid"), 1.0, 0.5
        )

        ambiguous_mask = df_block["classification_status"] == "Ambiguous"
        if ambiguous_mask.any():
            amb = df_block[ambiguous_mask].copy()
            secondary_rows = pd.DataFrame(
                {
                    "sentence": amb["sentence"],
                    "keyword": amb["secondary_topic"],
                    "Call": amb["Call"],
                    "Activity": amb["Activity"],
                    "Number": amb["Number"],
                    "Score": amb["Score"],
                    "Block": amb["Block"],
                    "classification": amb["classification_status"],
                }
            )
            secondary_rows["dist"] = 0.5
            combined = pd.concat([primary_rows, secondary_rows], ignore_index=True)
        else:
            combined = primary_rows

        return combined[cls.RESULT_COLUMNS]

    @classmethod
    def pivot_block_sheet(cls, unpivoted_df: pd.DataFrame) -> pd.DataFrame:
        """Pivot block sheet so topic keywords become columns."""
        if unpivoted_df.empty:
            return pd.DataFrame()

        index_cols = [
            "Activity",
            "Call",
            "Number",
            "Score",
            "Block",
            "classification",
            "sentence",
        ]
        existing_index = [c for c in index_cols if c in unpivoted_df.columns]

        pivoted = pd.pivot_table(
            unpivoted_df,
            index=existing_index,
            columns="keyword",
            values="dist",
            aggfunc="mean",
            fill_value=0.0,
        ).reset_index()

        pivoted.columns.name = None
        return pivoted

    @classmethod
    def build_block_sheets(cls, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Returns pivoted DataFrames for Excellence, Impact, Implementation."""
        sheets = {}
        for block_name in ALL_TOPIC_CATEGORIES:
            unpivoted = cls.build_unpivoted_block_sheet(df, block_name)
            sheets[block_name] = cls.pivot_block_sheet(unpivoted)
        return sheets

    @classmethod
    def extract_top_sentences_and_group(
        cls,
        dfE_pivoted: pd.DataFrame,
        dfI_pivoted: pd.DataFrame,
        dfQ_pivoted: pd.DataFrame,
        top_n: int = DEFAULT_TOP_N_SENTENCES,
        output_excel_path: str | Path | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Groups pivoted block sheets by Activity and extracts top_n closest sentences."""
        blocks = {
            "C_Excellence": dfE_pivoted,
            "C_Impact": dfI_pivoted,
            "C_Implement": dfQ_pivoted,
        }
        summaries: dict[str, pd.DataFrame] = {}

        for block_name, df_pivoted in blocks.items():
            if df_pivoted is None or df_pivoted.empty:
                summaries[block_name] = pd.DataFrame()
                continue

            metadata_cols = [
                "Activity",
                "Call",
                "Number",
                "Score",
                "Block",
                "classification",
                "sentence",
            ]
            topic_cols = [c for c in df_pivoted.columns if c not in metadata_cols]

            agg_dict = {t: "mean" for t in topic_cols}
            if "Call" in df_pivoted.columns:
                agg_dict["Call"] = "first"
            if "Score" in df_pivoted.columns:
                agg_dict["Score"] = "first"
            if "Block" in df_pivoted.columns:
                agg_dict["Block"] = "first"

            grouped = df_pivoted.groupby("Activity").agg(agg_dict).reset_index()

            def _get_top_sentences(g):
                scores = g[topic_cols].max(axis=1) if topic_cols else pd.Series(1, index=g.index)
                top_rows = g.assign(_score=scores).sort_values(by="_score", ascending=False).head(top_n)
                return "\n".join([f"- {s}" for s in top_rows["sentence"]])

            top_sentences_series = df_pivoted.groupby("Activity", group_keys=False).apply(_get_top_sentences)
            top_sentences_df = top_sentences_series.reset_index(name=f"Top_{top_n}_Sentences")

            merged_summary = pd.merge(grouped, top_sentences_df, on="Activity")
            base_cols = [c for c in ["Activity", "Call", "Score", "Block"] if c in merged_summary.columns]
            final_cols = base_cols + topic_cols + [f"Top_{top_n}_Sentences"]
            summaries[block_name] = merged_summary[final_cols].sort_values(by="Activity").reset_index(drop=True)

        return summaries


# ============================================================
# 5. Summary Visualizer
# ============================================================
class ESRSummaryVisualizer:
    """Visualizes summary results grouped by Activity."""

    @classmethod
    def format_activity_summary(cls, summary_df: pd.DataFrame) -> pd.DataFrame:
        if summary_df.empty:
            return summary_df
        return summary_df.sort_values(by="Activity").reset_index(drop=True)


# ============================================================
# 6. Pipeline
# ============================================================
class ESRClassificationPipeline:
    def __init__(self, topic_index: TopicIndex):
        _ensure_nltk_punkt()
        self.topic_index = topic_index
        self.analyzer = ESRAnalyzer(topic_index)

    @staticmethod
    def _validate_columns(df: pd.DataFrame, column_name: str, score_column: str) -> None:
        missing = [
            c for c in [column_name, score_column, *REQUIRED_METADATA_COLUMNS]
            if c not in df.columns
        ]
        if missing:
            raise KeyError(
                f"Input DataFrame missing required column(s): {missing}. Available: {list(df.columns)}"
            )

    def run_analysis_column(
        self,
        df: pd.DataFrame,
        column_name: str = "C_Excellence",
        num_rows_to_process: int | None = None,
        top_k: int = 1,
    ) -> pd.DataFrame:
        if column_name not in SCORE_COLUMN_MAP:
            raise ValueError(f"Unknown column '{column_name}'. Expected: {list(SCORE_COLUMN_MAP)}")
        score_column = SCORE_COLUMN_MAP[column_name]
        self._validate_columns(df, column_name, score_column)

        df_subset = df.head(num_rows_to_process) if num_rows_to_process else df
        all_results = []

        for row in tqdm(
            df_subset.itertuples(index=False),
            total=len(df_subset),
            desc=f"Processing {column_name} ({len(df_subset)} rows)",
        ):
            raw_text = getattr(row, column_name)
            text_str = str(raw_text) if not pd.isna(raw_text) else ""

            results_for_text = self.analyzer.analyse_text(
                text_str, topic_category=column_name, top_k=top_k
            )
            if results_for_text.empty:
                continue

            results_for_text["Call"] = row.Call
            results_for_text["Activity"] = row.Activity
            results_for_text["Number"] = row.Number
            results_for_text["Score"] = getattr(row, score_column)
            results_for_text["Block"] = column_name
            all_results.append(results_for_text)

        if not all_results:
            return pd.DataFrame(
                columns=["sentence", "topic", "val", "Call", "Activity", "Number", "Score", "Block"]
            )

        return pd.concat(all_results, ignore_index=True)

    def analysis(
        self,
        df: pd.DataFrame,
        columns_to_process: list[str] | None = None,
        num_rows_to_process: int | None = None,
        top_k: int = 1,
        output_path: str | Path | None = None,
        output_format: str = "excel",
        keep_embeddings: bool = False,
        flag_outliers: bool = True,
        iqr_multiplier: float = IQR_MULTIPLIER,
        ratio_threshold: float = AMBIGUITY_RATIO_THRESHOLD,
        top_n_sentences: int = DEFAULT_TOP_N_SENTENCES,
    ) -> pd.DataFrame:
        start_time_total = time.time()
        columns_to_process = columns_to_process or list(SCORE_COLUMN_MAP)
        num_rows_to_process = num_rows_to_process or len(df)

        all_combined_results = []
        for col_name in columns_to_process:
            logger.info("Running analysis for column: %s", col_name)
            results_df = self.run_analysis_column(
                df,
                column_name=col_name,
                num_rows_to_process=num_rows_to_process,
                top_k=top_k,
            )
            if not results_df.empty:
                all_combined_results.append(results_df)

        if not all_combined_results:
            logger.warning("No results generated.")
            return pd.DataFrame()

        final_combined_df = pd.concat(all_combined_results, ignore_index=True)
        final_combined_df = final_combined_df.drop(columns=["topic", "val"], errors="ignore")

        elapsed = time.time() - start_time_total
        logger.info("Total processing time: %.2f seconds", elapsed)

        if not keep_embeddings and "embedding" in final_combined_df.columns:
            final_combined_df = final_combined_df.drop(columns=["embedding"])

        if flag_outliers and "primary_val" in final_combined_df.columns:
            final_combined_df = ESRReportGenerator.flag_outliers_and_ambiguous(
                final_combined_df,
                iqr_multiplier=iqr_multiplier,
                ratio_threshold=ratio_threshold,
            )

        output_path = ESRClassificationIO.resolve_output_path(output_path, output_format, num_rows_to_process)
        ESRClassificationIO.save_results(
            final_combined_df,
            output_path,
            output_format,
            top_n_sentences=top_n_sentences,
        )

        return final_combined_df


# ============================================================
# I/O Helpers
# ============================================================
OUTPUT_FORMATS = ("parquet", "excel", "both")


class ESRClassificationIO:
    @staticmethod
    def resolve_output_path(
        output_path: str | Path | None,
        output_format: str,
        num_rows_to_process: int,
    ) -> Path:
        if output_path is not None:
            return Path(output_path)
        ext = "xlsx" if output_format == "excel" else "parquet"
        return Path(f"classified_all_results.{ext}")

    @staticmethod
    def save_results(
        df: pd.DataFrame,
        output_path: str | Path,
        output_format: str,
        top_n_sentences: int = DEFAULT_TOP_N_SENTENCES,
    ) -> None:
        if output_format not in OUTPUT_FORMATS:
            raise ValueError(f"Unknown output_format '{output_format}'.")

        output_path = Path(output_path)

        if output_format in ("parquet", "both"):
            parquet_path = output_path.with_suffix(".parquet")
            df.to_parquet(parquet_path, index=False)
            logger.info("Results saved to %s", parquet_path)

        if output_format in ("excel", "both"):
            excel_path = output_path.with_suffix(".xlsx")

            all_raw_df = ESRReportGenerator.build_all_raw_data_sheet(df)
            block_sheets = ESRReportGenerator.build_block_sheets(df)

            summaries = ESRReportGenerator.extract_top_sentences_and_group(
                block_sheets.get("C_Excellence", pd.DataFrame()),
                block_sheets.get("C_Impact", pd.DataFrame()),
                block_sheets.get("C_Implement", pd.DataFrame()),
                top_n=top_n_sentences,
            )

            try:
                with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                    if not all_raw_df.empty:
                        all_raw_df.to_excel(writer, sheet_name="All_raw_data", index=False)

                    for block_name, sheet_df in block_sheets.items():
                        if not sheet_df.empty:
                            sheet_name = BLOCK_SHEET_NAMES.get(block_name, block_name)
                            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

                    for block_name, summary_df in summaries.items():
                        if not summary_df.empty:
                            sheet_name = BLOCK_SUMMARY_SHEET_NAMES.get(block_name, block_name)
                            summary_df.to_excel(writer, sheet_name=sheet_name, index=False)

                logger.info("Combined Excel workbook saved to %s", excel_path)
            except Exception as e:
                logger.error("Failed to write Excel file: %s", e)
                raise

    @staticmethod
    def load_input_dataframe(input_path: str | Path) -> pd.DataFrame:
        input_path = Path(input_path)
        if input_path.exists():
            return pd.read_excel(input_path)
        if IN_COLAB:
            uploaded = colab_files.upload()
            filename = next(iter(uploaded.keys()))
            return pd.read_excel(filename)
        raise FileNotFoundError(f"Input file '{input_path}' not found.")



# =============================================================================
# =============================================================================
# ETAPA 3 · RESUMEN LLM  (resumen en lenguaje natural por Activity/Call)
# Clases: LLMSummarizerEngine, ExcelSummaryProcessor (alias: ESRSummaryProcessor)
# =============================================================================
# =============================================================================
class LLMSummarizerEngine:
    """
    Envoltorio sobre Transformers para Gemma 4 (E2B por defecto).

    - Cuantización 4-bit NF4 vía bitsandbytes (recomendado en T4 de 16GB).
    - Generación por lotes con model.generate() (control total de truncado
      y de manejo de errores, a diferencia de pipeline()).
    - Si ocurre un CUDA OOM, el lote se divide recursivamente en dos mitades
      hasta que cada sublote quepa en memoria. NINGUNA fila se descarta:
      todas las mitades se procesan y se recombinan en orden.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-4-E2B-it",
        device: str | None = None,
        max_new_tokens: int = 150,
        max_input_tokens: int = 2048,
        use_4bit: bool = True,
        hf_login: bool = True,
    ):
        _ensure_transformers()
        if hf_login:
            _maybe_login_huggingface()

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Cargando '{model_name}' en '{self.device}'...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Requisito clave para batching con modelos CausalLM decoder-only
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = None
        if self.device == "cuda" and use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Cuantización 4-bit NF4 activada.")

        # sdpa es la opción estable por defecto; flash-attention2 no siempre
        # está disponible/instalado en el entorno de Kaggle.
        attn_impl = "sdpa" if self.device == "cuda" else "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if self.device == "cuda" else None,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            quantization_config=quantization_config,
            attn_implementation=attn_impl,
        )
        if self.device == "cpu":
            self.model.to("cpu")
        self.model.eval()

        logger.info("Modelo cargado correctamente.")

    # --------------------------------------------------------
    def _build_prompt(self, comments: str) -> str:
        """Construye el prompt instruccional (Gemma 4 soporta 'system' nativamente)."""
        if not isinstance(comments, str) or not comments.strip():
            return ""



        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert evaluator of research/grant proposals. For each call you are given a set of sentences of DIFERENT proposals in the call "
                    "Synthesize ONLY the main core weaknesses raised across the comments into a high-level summary of ALL the proposals in that call.\n\n"
                    "Rules:\n"
                    "- one single paragraph\n"
                    "- objective, neutral language\n"
                    "- GROUP RELATED TOPICS TOGETHER: cluster all references to the same thematic area "
                    "(e.g., all issues regarding industrial applications/use cases, metrics/KPIs, or hardware/resources) "
                    "into a single unified mention to avoid scattering related points\n"
                    "- HIGH-LEVEL ABSTRACTION: omit hyper-specific domain jargon, specific algorithms, hardware models, or niche technical terms "
                    "(synthesize the TYPE of flaw instead, e.g., 'technical objectives', 'domain-specific applications', 'target hardware/infrastructure')\n"
                    "- focus on evaluation categories (e.g., lack of measurable metrics, poor roadmap alignment, insufficient resource estimation, vague scope)\n"
                    "- no bullet points, no repetition\n"
                    "- maximum 120 words\n"
                    "- do not invent information\n"
                    "- do not mention strengths"
                ),
            },
            {"role": "user", "content": f"Evaluation comments:\n\n{comments.strip()}"},
        ]

        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback por si el chat template de este checkpoint concreto no
            # admite el rol 'system' (p.ej. una revisión distinta del modelo).
            merged = [
                {"role": "user", "content": f"{messages[0]['content']}\n\n{messages[1]['content']}"}
            ]
            return self.tokenizer.apply_chat_template(
                merged, tokenize=False, add_generation_prompt=True
            )

    # --------------------------------------------------------
    def _tokenize(self, prompts: list[str]):
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors="pt",
        )
        return {k: v.to(self.model.device) for k, v in encoded.items()}

    def _generate_raw(self, prompts: list[str]) -> list[str]:
        inputs = self._tokenize(prompts)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[:, input_len:]
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [t.strip() for t in texts]

    # --------------------------------------------------------
    def generate_summaries_batch(self, batch_sentences: list[str]) -> list[str]:
        """
        Genera resúmenes para un lote de N filas.

        Si hay CUDA OOM, el lote se bisecta recursivamente y ambas mitades
        se procesan por separado; los resultados se devuelven en el mismo
        orden y con la misma longitud que la entrada (ninguna fila se pierde).
        """
        n = len(batch_sentences)
        if n == 0:
            return []

        prompts = [self._build_prompt(s) for s in batch_sentences]

        # Filas vacías no necesitan pasar por el modelo
        non_empty_idx = [i for i, p in enumerate(prompts) if p]
        if not non_empty_idx:
            return [""] * n

        try:
            start = time.perf_counter()
            raw = self._generate_raw([prompts[i] for i in non_empty_idx])
            elapsed = time.perf_counter() - start
            logger.info(f"Batch={len(non_empty_idx)} tiempo={elapsed:.2f}s ({elapsed/len(non_empty_idx):.2f}s/fila)")

            results = [""] * n
            for idx, summary in zip(non_empty_idx, raw):
                results[idx] = summary
            return results

        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()

            if n == 1:
                logger.error("OOM incluso con una sola fila; se marca como error.")
                return ["ERROR: CUDA OUT OF MEMORY"]

            logger.warning(f"CUDA OOM con batch={n}; dividiendo en dos mitades.")
            mid = n // 2
            left = self.generate_summaries_batch(batch_sentences[:mid])
            right = self.generate_summaries_batch(batch_sentences[mid:])
            return left + right


# ============================================================
# 2. Procesador de Excel (una fila = una call)
# ============================================================
class ExcelSummaryProcessor:
    """
    Lee un Excel donde cada fila es una 'call' y una columna contiene las
    frases negativas concatenadas de esa call. Genera un resumen por fila
    procesando en lotes de `batch_size` filas a la vez.
    """

    def __init__(self, llm_engine: LLMSummarizerEngine, batch_size: int = 4):
        self.llm_engine = llm_engine
        self.batch_size = batch_size

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "sentences",
        batch_size: int | None = None,
        summary_column: str = "LLM_Summary",
    ) -> pd.DataFrame:
        if df.empty or text_column not in df.columns:
            return df

        n_batch = batch_size or self.batch_size
        df = df.copy()
        texts = df[text_column].fillna("").astype(str).tolist()

        summaries: list[str] = []
        for i in tqdm(range(0, len(texts), n_batch), desc=f"Resumiendo (batch={n_batch})"):
            chunk = texts[i : i + n_batch]
            summaries.extend(self.llm_engine.generate_summaries_batch(chunk))

        df[summary_column] = summaries
        return df

    def process_excel(
        self,
        input_excel_path: str | Path,
        output_excel_path: str | Path,
        text_column: str = "sentences",
        sheet_name: str | list[str] | None = None,
        batch_size: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Procesa todas las hojas (o las indicadas en `sheet_name`) que contengan
        la columna `text_column`. Las demás hojas se copian sin modificar.
        """
        input_excel_path = Path(input_excel_path)
        output_excel_path = Path(output_excel_path)

        if not input_excel_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {input_excel_path}")

        logger.info(f"Leyendo Excel: {input_excel_path}")
        all_sheets = pd.read_excel(input_excel_path, sheet_name=None)

        target_sheets = None
        if sheet_name is not None:
            target_sheets = [sheet_name] if isinstance(sheet_name, str) else list(sheet_name)

        processed = {}
        for name, df in all_sheets.items():
            should_process = (target_sheets is None or name in target_sheets) and (text_column in df.columns)
            if should_process:
                logger.info(f"Procesando hoja '{name}' ({len(df)} filas/calls)")
                processed[name] = self.process_dataframe(df, text_column=text_column, batch_size=batch_size)
            else:
                processed[name] = df.copy()

        with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
            for name, frame in processed.items():
                frame.to_excel(writer, sheet_name=name, index=False)

        logger.info(f"Archivo final guardado en: {output_excel_path}")
        return processed


#: Alias por coherencia de nomenclatura con el resto de etapas (mismo
#: comportamiento que ``ExcelSummaryProcessor``; se puede usar cualquiera
#: de los dos nombres indistintamente).
ESRSummaryProcessor = ExcelSummaryProcessor


# =============================================================================
# =============================================================================
# ORQUESTACIÓN END-TO-END · ESRMasterPipeline
#
# Une las 3 etapas (extracción, clasificación, resumen LLM) en un único
# pipeline ejecutable completo o parcialmente. Cada etapa puede:
#   - ejecutarse encadenada a la anterior (usa el DataFrame en memoria), o
#   - arrancar de forma independiente desde un fichero Excel ya generado
#     por una ejecución previa de la etapa anterior.
# =============================================================================
# =============================================================================
class ESRMasterPipeline:
    """Orquesta ``ESRExtractionPipeline`` -> ``ESRClassificationPipeline`` ->
    (``LLMSummarizerEngine`` + ``ExcelSummaryProcessor``) como un único
    pipeline, permitiendo ejecutar el proceso completo o solo una parte.

    Los componentes "pesados" (el modelo de embeddings de la etapa de
    clasificación y el LLM de la etapa de resumen) se construyen de forma
    perezosa la primera vez que se necesitan, y se cachean para
    reutilizarse en llamadas posteriores a ``run`` sobre la misma
    instancia.
    """

    def __init__(
        self,
        extraction_pipeline: Optional["ESRExtractionPipeline"] = None,
        topic_index: Optional["TopicIndex"] = None,
        llm_engine: Optional["LLMSummarizerEngine"] = None,
    ) -> None:
        self.extraction_pipeline = extraction_pipeline
        self.topic_index = topic_index
        self.llm_engine = llm_engine

    # ------------------------------------------------------------------
    # Etapa 1: extracción
    # ------------------------------------------------------------------
    def run_extraction(
        self,
        zip_source_paths: Optional[Sequence[str]] = None,
        output_dir: str = "esr_output",
        anonymize: bool = False,
        encrypt: bool = False,
        export_path: Optional[str] = None,
        input_excel_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Descomprime ZIP(s), extrae los datos de los PDFs de ESR y,
        opcionalmente, anonimiza/cifra y exporta el resultado.

        Devuelve un diccionario con ``dataframe``, ``error_log_path``,
        ``errors_occurred`` y, si se exportó, ``export_path``.
        """
        pipeline = self.extraction_pipeline or ESRExtractionPipeline()
        self.extraction_pipeline = pipeline

        result = pipeline.run(
            zip_source_paths=zip_source_paths,
            output_dir=output_dir,
            anonymize=anonymize,
            encrypt=encrypt,
            export_path=export_path,
        )

        # Si además se indicó un Excel de entrada preexistente, se combina
        # con lo extraído (equivalente a `ESRZipProcessor.unzip(..., input_excel_path=...)`).
        if input_excel_path:
            df = result["dataframe"]
            if os.path.exists(input_excel_path):
                logger.info(f"📄 Combinando con datos existentes de: {input_excel_path}")
                existing_df = pd.read_excel(input_excel_path)
                result["dataframe"] = pd.concat([df, existing_df], ignore_index=True)
            else:
                logger.warning(
                    f"⚠️ input_excel_path especificado ({input_excel_path}) pero no existe."
                )
                
        pre_classified_file = os.path.join(output_dir, "extracted_raw_data.xlsx")
        result["dataframe"].to_excel(pre_classified_file, index=False)
        logger.info(f"📊 Excel tras extracción guardado en: {pre_classified_file}")
        
        return result

    # ------------------------------------------------------------------
    # Etapa 2: clasificación
    # ------------------------------------------------------------------
    def _get_topic_index(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
    ) -> "TopicIndex":
        if self.topic_index is None:
            logger.info(f"Cargando modelo de embeddings '{model_name}'...")
            self.topic_index = TopicIndex.build(model_name=model_name, device=device)
        return self.topic_index

    def run_classification(
        self,
        df: Optional[pd.DataFrame] = None,
        input_path: Optional[str] = None,
        columns_to_process: Optional[List[str]] = None,
        num_rows_to_process: Optional[int] = None,
        top_k: int = 1,
        output_path: Optional[Union[str, Path]] = None,
        output_format: str = "excel",
        keep_embeddings: bool = False,
        flag_outliers: bool = True,
        iqr_multiplier: float = IQR_MULTIPLIER,
        ratio_threshold: float = AMBIGUITY_RATIO_THRESHOLD,
        top_n_sentences: int = DEFAULT_TOP_N_SENTENCES,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
    ) -> pd.DataFrame:
        """Clasifica las frases negativas de los criterios de evaluación por
        similitud de embeddings frente a un conjunto de topics predefinidos.

        Se puede pasar directamente un DataFrame (por ejemplo, encadenado
        desde ``run_extraction``) o una ruta ``input_path`` a un Excel ya
        extraído (para ejecutar esta etapa de forma independiente).
        """
        if df is None:
            if not input_path:
                raise ValueError("Debes indicar 'df' o 'input_path' para la etapa de clasificación.")
            df = ESRClassificationIO.load_input_dataframe(input_path)

        topic_index = self._get_topic_index(model_name=model_name, device=device)
        pipeline = ESRClassificationPipeline(topic_index)

        return pipeline.analysis(
            df,
            columns_to_process=columns_to_process,
            num_rows_to_process=num_rows_to_process,
            top_k=top_k,
            output_path=output_path,
            output_format=output_format,
            keep_embeddings=keep_embeddings,
            flag_outliers=flag_outliers,
            iqr_multiplier=iqr_multiplier,
            ratio_threshold=ratio_threshold,
            top_n_sentences=top_n_sentences,
        )

    # ------------------------------------------------------------------
    # Etapa 3: resumen LLM
    # ------------------------------------------------------------------
    def _get_llm_engine(
        self,
        model_name: str = "google/gemma-4-E2B-it",
        device: Optional[str] = None,
        max_new_tokens: int = 150,
        use_4bit: bool = True,
    ) -> "LLMSummarizerEngine":
        if self.llm_engine is None:
            self.llm_engine = LLMSummarizerEngine(
                model_name=model_name,
                device=device,
                max_new_tokens=max_new_tokens,
                use_4bit=use_4bit,
            )
        return self.llm_engine

    def run_summary(
        self,
        input_excel_path: Union[str, Path],
        output_excel_path: Union[str, Path],
        text_column: str = "Top_10_Sentences",
        sheet_name: Optional[Union[str, List[str]]] = None,
        batch_size: int = 4,
        llm_model_name: str = "google/gemma-4-E2B-it",
        device: Optional[str] = None,
        max_new_tokens: int = 150,
        use_4bit: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Genera, por cada hoja/'call', un resumen en lenguaje natural de
        las frases negativas más representativas, usando un LLM pequeño.

        Requiere que ``input_excel_path`` sea un Excel con hojas que
        contengan la columna ``text_column`` (típicamente el fichero
        generado por ``run_classification``, con hojas
        ``Excellence_Summary`` / ``Impact_Summary`` / ``Implementation_Summary``).
        """
        engine = self._get_llm_engine(
            model_name=llm_model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            use_4bit=use_4bit,
        )
        processor = ExcelSummaryProcessor(llm_engine=engine, batch_size=batch_size)
        return processor.process_excel(
            input_excel_path=input_excel_path,
            output_excel_path=output_excel_path,
            text_column=text_column,
            sheet_name=sheet_name,
            batch_size=batch_size,
        )

    # ------------------------------------------------------------------
    # Pipeline completo (o parcial)
    # ------------------------------------------------------------------
    def run(
        self,
        # --- control de qué etapas ejecutar ---
        do_extract: bool = True,
        do_classify: bool = True,
        do_summarize: bool = True,
        # --- etapa 1: extracción ---
        zip_source_paths: Optional[Sequence[str]] = None,
        extraction_output_dir: str = "esr_output",
        anonymize: bool = False,
        encrypt: bool = False,
        extraction_export_path: Optional[str] = None,
        extraction_input_excel_path: Optional[str] = None,
        # --- etapa 2: clasificación ---
        classification_input_path: Optional[str] = None,
        classification_columns: Optional[List[str]] = None,
        classification_rows: Optional[int] = None,
        classification_top_k: int = 1,
        classification_output_path: Optional[Union[str, Path]] = None,
        classification_output_format: str = "excel",
        keep_embeddings: bool = False,
        flag_outliers: bool = True,
        iqr_multiplier: float = IQR_MULTIPLIER,
        ratio_threshold: float = AMBIGUITY_RATIO_THRESHOLD,
        top_n_sentences: int = DEFAULT_TOP_N_SENTENCES,
        embedding_model_name: str = MODEL_NAME,
        embedding_device: Optional[str] = None,
        # --- etapa 3: resumen LLM ---
        summary_input_path: Optional[Union[str, Path]] = None,
        summary_output_path: Optional[Union[str, Path]] = None,
        summary_text_column: Optional[str] = None,
        summary_sheet_names: Optional[List[str]] = None,
        summary_batch_size: int = 4,
        llm_model_name: str = "google/gemma-4-E2B-it",
        llm_device: Optional[str] = None,
        llm_max_new_tokens: int = 150,
        llm_use_4bit: bool = True,
    ) -> Dict[str, Any]:
        """Ejecuta el pipeline completo (extracción + clasificación +
        resumen LLM) o solo el subconjunto de etapas solicitado.

        Si una etapa se desactiva (``do_extract=False`` / ``do_classify=False``)
        pero la siguiente etapa sí está activa, esta lee su entrada del
        fichero Excel indicado (``classification_input_path`` /
        ``summary_input_path``) en lugar de encadenar el DataFrame en
        memoria. Esto permite, por ejemplo, re-lanzar solo la clasificación
        o solo el resumen sobre resultados ya generados.

        Devuelve un diccionario con las claves relevantes para las etapas
        ejecutadas: ``extraction`` (dict de ``run_extraction``),
        ``classification_df`` (DataFrame) y ``summary`` (dict de hojas).
        """
        results: Dict[str, Any] = {}
        classification_df: Optional[pd.DataFrame] = None

        # ---------------- Etapa 1: extracción ----------------
        if do_extract:
            extraction_result = self.run_extraction(
                zip_source_paths=zip_source_paths,
                output_dir=extraction_output_dir,
                anonymize=anonymize,
                encrypt=encrypt,
                export_path=extraction_export_path,
                input_excel_path=extraction_input_excel_path,
            )
            results["extraction"] = extraction_result
            classification_df = extraction_result["dataframe"]

        # ---------------- Etapa 2: clasificación ----------------
        if do_classify:
            df_for_classification = classification_df
            path_for_classification = classification_input_path
            if df_for_classification is None and path_for_classification is None:
                raise ValueError(
                    "La etapa de clasificación necesita datos: activa 'do_extract' o "
                    "indica 'classification_input_path' con un Excel ya extraído."
                )

            classification_df = self.run_classification(
                df=df_for_classification,
                input_path=path_for_classification,
                columns_to_process=classification_columns,
                num_rows_to_process=classification_rows,
                top_k=classification_top_k,
                output_path=classification_output_path,
                output_format=classification_output_format,
                keep_embeddings=keep_embeddings,
                flag_outliers=flag_outliers,
                iqr_multiplier=iqr_multiplier,
                ratio_threshold=ratio_threshold,
                top_n_sentences=top_n_sentences,
                model_name=embedding_model_name,
                device=embedding_device,
            )
            results["classification_df"] = classification_df

        # ---------------- Etapa 3: resumen LLM ----------------
        if do_summarize:
            # El Excel de entrada de esta etapa es siempre un fichero en
            # disco (el LLM se ejecuta hoja a hoja sobre un Excel ya
            # pivotado/agrupado por Activity), así que si se acaba de
            # generar en la etapa 2, se usa esa ruta de salida.
            resolved_summary_input = summary_input_path
            if resolved_summary_input is None:
                if do_classify and classification_output_path:
                    resolved_summary_input = Path(classification_output_path).with_suffix(".xlsx")
                elif do_classify:
                    resolved_summary_input = ESRClassificationIO.resolve_output_path(
                        None, classification_output_format, classification_rows or 0
                    ).with_suffix(".xlsx")

            if resolved_summary_input is None:
                raise ValueError(
                    "La etapa de resumen necesita un Excel de entrada: activa 'do_classify' "
                    "(con salida en Excel) o indica 'summary_input_path'."
                )

            resolved_summary_output = summary_output_path
            if resolved_summary_output is None:
                resolved_summary_input_p = Path(resolved_summary_input)
                resolved_summary_output = resolved_summary_input_p.with_name(
                    resolved_summary_input_p.stem + "_with_summary.xlsx"
                )

            resolved_text_column = summary_text_column or f"Top_{top_n_sentences}_Sentences"
            resolved_sheet_names = summary_sheet_names or [
                "Excellence_Summary",
                "Impact_Summary",
                "Implementation_Summary",
            ]

            summary_result = self.run_summary(
                input_excel_path=resolved_summary_input,
                output_excel_path=resolved_summary_output,
                text_column=resolved_text_column,
                sheet_name=resolved_sheet_names,
                batch_size=summary_batch_size,
                llm_model_name=llm_model_name,
                device=llm_device,
                max_new_tokens=llm_max_new_tokens,
                use_4bit=llm_use_4bit,
            )
            results["summary"] = summary_result
            results["summary_output_path"] = str(resolved_summary_output)

        return results


# =============================================================================
# =============================================================================
# CLI unificado
# =============================================================================
# =============================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    # Parser "padre" con opciones comunes a todos los subcomandos (p.ej.
    # --log-level), para que puedan indicarse tanto antes como después del
    # nombre del subcomando:
    #   python esr_pipeline.py --log-level DEBUG extract ...
    #   python esr_pipeline.py extract --log-level DEBUG ...
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    parser = argparse.ArgumentParser(
        description=(
            "Pipeline unificado de extracción, clasificación y resumen LLM de ESRs. "
            "Usa un subcomando para ejecutar el proceso completo o solo una parte."
        ),
        parents=[common],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---------------- extract ----------------
    p_extract = subparsers.add_parser(
        "extract", parents=[common], help="Solo la etapa 1: ZIP(s) -> Excel/CSV/Parquet de ESRs."
    )
    p_extract.add_argument("--zips", nargs="*", default=None, help="Rutas a ficheros ZIP a procesar.")
    p_extract.add_argument("--input-folder", default=None, help="Carpeta con ZIPs a procesar (si no se indica --zips).")
    p_extract.add_argument("--output-dir", default="esr_output", help="Carpeta de salida para PDFs copiados y el log de errores.")
    p_extract.add_argument("--export", dest="export_path", default=None, help="Ruta del fichero de resultados a exportar (.xlsx, .csv o .parquet).")
    p_extract.add_argument("--input-excel", dest="input_excel_path", default=None, help="Excel existente a combinar con lo extraído.")
    p_extract.add_argument("--anonymize", action="store_true", help="Anonimiza el texto libre de los criterios.")
    p_extract.add_argument("--encrypt", action="store_true", help="Cifra con AES-GCM las columnas sensibles (requiere --anonymize y una clave).")
    p_extract.add_argument("--aesgcm-key-env", default="AESGCM_KEY", help="Variable de entorno con la clave AES-GCM en base64 urlsafe.")

    # ---------------- classify ----------------
    p_classify = subparsers.add_parser("classify", parents=[common], help="Solo la etapa 2: Excel de ESRs -> clasificación por topics.")
    p_classify.add_argument("--input", default="all_esr_data.xlsx", help="Excel de entrada (salida de la etapa de extracción).")
    p_classify.add_argument("--rows", type=int, default=None)
    p_classify.add_argument("--columns", nargs="+", default=None, choices=list(SCORE_COLUMN_MAP))
    p_classify.add_argument("--top-k", type=int, default=1)
    p_classify.add_argument("--output", default=None)
    p_classify.add_argument("--output-format", default="excel", choices=list(OUTPUT_FORMATS))
    p_classify.add_argument("--keep-embeddings", action="store_true")
    p_classify.add_argument("--no-flag-outliers", action="store_true")
    p_classify.add_argument("--iqr-multiplier", type=float, default=IQR_MULTIPLIER)
    p_classify.add_argument("--ratio-threshold", type=float, default=AMBIGUITY_RATIO_THRESHOLD)
    p_classify.add_argument("--top-n-sentences", type=int, default=DEFAULT_TOP_N_SENTENCES)
    p_classify.add_argument("--model-name", default=MODEL_NAME)
    p_classify.add_argument("--device", default=None)

    # ---------------- summarize ----------------
    p_summarize = subparsers.add_parser("summarize", parents=[common], help="Solo la etapa 3: Excel clasificado -> resúmenes LLM.")
    p_summarize.add_argument("--input", default="classified_all_results.xlsx")
    p_summarize.add_argument("--output", default=None)
    p_summarize.add_argument("--llm-model", dest="llm_model_name", default="google/gemma-4-E2B-it")
    p_summarize.add_argument("--device", default=None)
    p_summarize.add_argument("--batch-size", type=int, default=32)
    p_summarize.add_argument("--max-new-tokens", type=int, default=150)
    p_summarize.add_argument("--no-4bit", action="store_true")
    p_summarize.add_argument("--text-column", default="Top_10_Sentences")
    p_summarize.add_argument(
        "--sheet-name", nargs="+",
        default=["Excellence_Summary", "Impact_Summary", "Implementation_Summary"],
        help="Una o varias hojas separadas por espacio. Usa 'all' para todas.",
    )

    # ---------------- full ----------------
    p_full = subparsers.add_parser("full", parents=[common], help="Pipeline completo: extracción + clasificación + resumen LLM.")
    p_full.add_argument("--zips", nargs="*", default=None)
    p_full.add_argument("--input-folder", default=None)
    p_full.add_argument("--output-dir", default="esr_output")
    p_full.add_argument("--anonymize", action="store_true")
    p_full.add_argument("--encrypt", action="store_true")
    p_full.add_argument("--aesgcm-key-env", default="AESGCM_KEY")
    p_full.add_argument("--classification-output", dest="classification_output_path", default=None, help="Excel de la etapa de clasificación (por defecto: <output-dir>/classified_all_results.xlsx).")
    p_full.add_argument("--top-k", type=int, default=1)
    p_full.add_argument("--top-n-sentences", type=int, default=DEFAULT_TOP_N_SENTENCES)
    p_full.add_argument("--embedding-model", dest="embedding_model_name", default=MODEL_NAME)
    p_full.add_argument("--embedding-device", default=None)
    p_full.add_argument("--summary-output", dest="summary_output_path", default=None, help="Excel final con los resúmenes LLM.")
    p_full.add_argument("--llm-model", dest="llm_model_name", default="google/gemma-4-E2B-it")
    p_full.add_argument("--llm-device", default=None)
    p_full.add_argument("--summary-batch-size", type=int, default=4)
    p_full.add_argument("--llm-max-new-tokens", type=int, default=150)
    p_full.add_argument("--no-4bit", action="store_true")
    p_full.add_argument("--skip-classify", action="store_true", help="Ejecuta solo la extracción (equivalente a 'extract').")
    p_full.add_argument("--skip-summarize", action="store_true", help="No ejecuta la etapa de resumen LLM.")

    return parser


def _resolve_zip_sources(zips: Optional[List[str]], input_folder: Optional[str]) -> Optional[List[str]]:
    if zips is not None:
        return zips
    if input_folder:
        return [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.lower().endswith(".zip")
        ]
    return None


def main(argv: Optional[Sequence[str]] = None) -> Any:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # Nota: no se llama a logging.basicConfig() para evitar duplicar líneas
    # de log (nuestro logger ya tiene su propio StreamHandler configurado
    # en la cabecera del módulo); basta con ajustar su nivel.
    logger.setLevel(getattr(logging, args.log_level))

    master = ESRMasterPipeline()

    if args.command == "extract":
        zip_source_paths = _resolve_zip_sources(args.zips, args.input_folder)
        cipher = None
        if args.encrypt:
            cipher = ESRAnonymizer.load_cipher_from_env(args.aesgcm_key_env)
            if cipher is None and IN_COLAB:
                cipher = ESRAnonymizer.load_cipher_from_colab_secret(args.aesgcm_key_env)
            if cipher is None:
                logger.info("⚠️ No se pudo cargar la clave AES-GCM; se omitirá el cifrado.")

        anonymizer = ESRAnonymizer(cipher=cipher)
        exporter = ESRExporter(anonymizer=anonymizer)
        master.extraction_pipeline = ESRExtractionPipeline(
            anonymizer=anonymizer,
            exporter=exporter,
            zip_processor=ESRZipProcessor(colab_output_folder=args.output_dir),
        )

        result = master.run_extraction(
            zip_source_paths=zip_source_paths,
            output_dir=args.output_dir,
            anonymize=args.anonymize,
            encrypt=args.encrypt,
            export_path=args.export_path,
            input_excel_path=args.input_excel_path,
        )
        df = result["dataframe"]
        logger.info(f"\n✨ Extracción completada. Total de PDFs procesados: {len(df)}")
        if "export_path" in result:
            logger.info(f"📊 Resultado guardado en {result['export_path']}")
        if result["errors_occurred"]:
            logger.info(f"⚠️ Se registraron errores en {result['error_log_path']}.")
        return result

    if args.command == "classify":
        result_df = master.run_classification(
            input_path=args.input,
            columns_to_process=args.columns,
            num_rows_to_process=args.rows,
            top_k=args.top_k,
            output_path=args.output,
            output_format=args.output_format,
            keep_embeddings=args.keep_embeddings,
            flag_outliers=not args.no_flag_outliers,
            iqr_multiplier=args.iqr_multiplier,
            ratio_threshold=args.ratio_threshold,
            top_n_sentences=args.top_n_sentences,
            model_name=args.model_name,
            device=args.device,
        )
        if not result_df.empty:
            logger.info("Vista previa de resultados:\n%s", result_df.head())
        return result_df

    if args.command == "summarize":
        sheet = None if args.sheet_name and "all" in [s.lower() for s in args.sheet_name] else args.sheet_name
        output_path = args.output or str(Path(args.input).with_name(Path(args.input).stem + "_with_summary.xlsx"))

        t0 = time.perf_counter()
        result = master.run_summary(
            input_excel_path=args.input,
            output_excel_path=output_path,
            text_column=args.text_column,
            sheet_name=sheet,
            batch_size=args.batch_size,
            llm_model_name=args.llm_model_name,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            use_4bit=not args.no_4bit,
        )
        elapsed = time.perf_counter() - t0
        logger.info(f"⏱️ Resumen completado en {elapsed:.2f}s. Archivo guardado en: {output_path}")
        return result

    if args.command == "full":
        zip_source_paths = _resolve_zip_sources(args.zips, args.input_folder)
        cipher = None
        if args.encrypt:
            cipher = ESRAnonymizer.load_cipher_from_env(args.aesgcm_key_env)
            if cipher is None and IN_COLAB:
                cipher = ESRAnonymizer.load_cipher_from_colab_secret(args.aesgcm_key_env)
            if cipher is None:
                logger.info("⚠️ No se pudo cargar la clave AES-GCM; se omitirá el cifrado.")

        anonymizer = ESRAnonymizer(cipher=cipher)
        exporter = ESRExporter(anonymizer=anonymizer)
        master.extraction_pipeline = ESRExtractionPipeline(
            anonymizer=anonymizer,
            exporter=exporter,
            zip_processor=ESRZipProcessor(colab_output_folder=args.output_dir),
        )

        classification_output_path = args.classification_output_path or os.path.join(
            args.output_dir, "classified_all_results.xlsx"
        )

        results = master.run(
            do_extract=True,
            do_classify=not args.skip_classify,
            do_summarize=(not args.skip_classify) and (not args.skip_summarize),
            zip_source_paths=zip_source_paths,
            extraction_output_dir=args.output_dir,
            anonymize=args.anonymize,
            encrypt=args.encrypt,
            classification_top_k=args.top_k,
            classification_output_path=classification_output_path,
            top_n_sentences=args.top_n_sentences,
            embedding_model_name=args.embedding_model_name,
            embedding_device=args.embedding_device,
            summary_output_path=args.summary_output_path,
            llm_model_name=args.llm_model_name,
            llm_device=args.llm_device,
            summary_batch_size=args.summary_batch_size,
            llm_max_new_tokens=args.llm_max_new_tokens,
            llm_use_4bit=not args.no_4bit,
        )

        extraction_result = results.get("extraction", {})
        df = extraction_result.get("dataframe", pd.DataFrame())
        logger.info(f"\n✨ Extracción completada. Total de PDFs procesados: {len(df)}")
        if "classification_df" in results:
            logger.info(f"📊 Clasificación guardada en {classification_output_path}")
        if "summary_output_path" in results:
            logger.info(f"🧠 Resumen LLM guardado en {results['summary_output_path']}")
        return results

    raise ValueError(f"Comando desconocido: {args.command}")


if __name__ == "__main__":
    main()
