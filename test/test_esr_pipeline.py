import os
import zipfile
import pytest
import pandas as pd
from dataclasses import is_dataclass
from unittest.mock import patch

# Importamos las clases de tu script
try:
    from esr_zips_v3 import (
        ESR,
        ESRExtractor,
        ESRZipProcessor,
        ESRAnonymizer,
        ESRExporter,
        ESRPipeline,
    )
except ImportError:
    from esr_pipeline import (
        ESR,
        ESRExtractor,
        ESRZipProcessor,
        ESRAnonymizer,
        ESRExporter,
        ESRPipeline,
    )


# =============================================================================
# FIXTURES (Configuración de pruebas)
# =============================================================================

@pytest.fixture(autouse=True)
def disable_google_drive_mount(monkeypatch):
    """Evita que ESRZipProcessor intente montar Google Drive durante los tests."""
    try:
        import google.colab.drive
        monkeypatch.setattr("google.colab.drive.mount", lambda *args, **kwargs: None)
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def sample_esr():
    """Retorna una instancia básica válida de ESR."""
    return ESR(
        source_path="/fake/path/sample_esr.pdf",
        data={
            "Call": "HORIZON-CL4-2024",
            "Type": "HORIZON-IA",
            "Number": "101123456",
            "Acronym": "TESTPROJ",
            "Title": "Test Project Title",
            "Total Score": 14.5,
            "C_Excellence": "The proposal addresses John Doe from AAU.",
            "C_Impact": "High impact for Europe.",
            "C_Implement": "Work plan is solid.",
        },
        error_code=0,
    )


@pytest.fixture
def colab_pdf_path():
    """Detecta si el usuario subió un ESR.pdf real a /content/."""
    real_pdf = "/content/ESR.pdf"
    if os.path.exists(real_pdf):
        return real_pdf
    return None


@pytest.fixture
def mock_zip_with_pdf(tmp_path, colab_pdf_path):
    """Crea un ZIP de prueba en un directorio temporal."""
    zip_dir = tmp_path / "zips"
    zip_dir.mkdir()
    zip_file_path = zip_dir / "test_esrs.zip"

    if colab_pdf_path:
        pdf_content_path = colab_pdf_path
    else:
        dummy_pdf = tmp_path / "sample_esr.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 %EOF")
        pdf_content_path = str(dummy_pdf)

    with zipfile.ZipFile(zip_file_path, "w") as zf:
        zf.write(pdf_content_path, arcname="proposal_esr.pdf")

    return str(zip_file_path)


# =============================================================================
# PRUEBAS UNITARIAS
# =============================================================================

class TestESRDataClass:
    """Pruebas para el contenedor de datos ESR."""

    def test_esr_initialization(self, sample_esr):
        assert is_dataclass(sample_esr)
        assert sample_esr.is_valid is True
        assert sample_esr.error_message == "OK - No errors detected."

    def test_esr_error_state(self):
        esr_error = ESR(error_code=2)
        assert esr_error.is_valid is False
        assert "BasicInfo" in esr_error.error_message

    def test_to_dataframe(self, sample_esr):
        df = ESR.to_dataframe([sample_esr])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["Acronym"] == "TESTPROJ"


class TestESRAnonymizer:
    """Pruebas para la anonimización de texto y cifrado AES-GCM."""

    def test_aes_encryption_decryption(self):
        cipher_key = ESRAnonymizer.generate_key()
        anonymizer = ESRAnonymizer(cipher=cipher_key)

        original_text = "101123456"
        encoded = anonymizer.encode_value(original_text)
        decoded = anonymizer.decode_value(encoded)

        assert encoded != original_text
        assert decoded == original_text

    def test_text_anonymization(self, sample_esr):
        anonymizer = ESRAnonymizer()
        df = ESR.to_dataframe([sample_esr])  # Uso correcto de to_dataframe
        
        df_anon = anonymizer.anonymize(df, encrypt=False)
        
        cleaned_text = df_anon.iloc[0]["C_Excellence"]
        assert "TESTPROJ" not in cleaned_text


class TestESRExporter:
    """Pruebas para la exportación de archivos."""

    def test_export_formats(self, tmp_path, sample_esr):
        exporter = ESRExporter()
        df = ESR.to_dataframe([sample_esr])  # Uso correcto de to_dataframe

        # CSV
        csv_path = str(tmp_path / "out.csv")
        exporter.export(df, csv_path)
        assert os.path.exists(csv_path)

        # Excel
        excel_path = str(tmp_path / "out.xlsx")
        exporter.export(df, excel_path)
        assert os.path.exists(excel_path)

        # Parquet
        parquet_path = str(tmp_path / "out.parquet")
        exporter.export(df, parquet_path)
        assert os.path.exists(parquet_path)


# =============================================================================
# PRUEBAS DE INTEGRACIÓN
# =============================================================================

class TestESRExtractor:
    """Pruebas para la extracción desde un PDF real."""

    def test_real_pdf_extraction(self, colab_pdf_path):
        if not colab_pdf_path:
            pytest.skip("No se encontró '/content/ESR.pdf'. Omitiendo prueba de extracción real.")

        extractor = ESRExtractor()
        esr = extractor.extract(colab_pdf_path)

        assert isinstance(esr, ESR)
        assert esr.source_path == colab_pdf_path
        for key in ["Call", "Type", "Number", "Acronym"]:
            assert key in esr.data


class TestESRPipelineIntegration:
    """Prueba End-to-End del pipeline completo."""

    def test_full_pipeline_run(self, mock_zip_with_pdf, tmp_path):
        output_dir = str(tmp_path / "pipeline_out")
        export_file = str(tmp_path / "final_result.xlsx")

        test_cipher = ESRAnonymizer.generate_key()
        anonymizer = ESRAnonymizer(cipher=test_cipher)

        pipeline = ESRPipeline(anonymizer=anonymizer)

        result = pipeline.run(
            zip_source_paths=[mock_zip_with_pdf],
            output_dir=output_dir,
            anonymize=True,
            encrypt=True,
            export_path=export_file,
        )

        assert "dataframe" in result
        assert result["error_log_path"] is not None
        assert os.path.exists(export_file)
