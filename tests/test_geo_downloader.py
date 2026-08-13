"""Unit tests for src/data_processing/geo_downloader.py.

Tests cover the SOFT-parsing, verification, sample selection, and download
logic. All network calls are mocked via ``GEODownloader._http_get``.
"""

import gzip
import os
import tempfile
from unittest.mock import patch

import pytest

from src.data_processing.geo_downloader import (
    DatasetVerificationError,
    GEODownloader,
    GEOSample,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CONFIG = """\
geo_datasets:
  verified_datasets:
    - accession: "GSE100738"
      title: "Test ImmGen ATAC-seq"
      cell_type: "CD4_T_cell"
      assay_type: "ATAC-seq"
      organism: "Homo sapiens"
      url: "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100738"
      cell_type_default: "CD4_T_cell"
      sample_title_regex: ".*"
      supplementary_regex: ".*\\\\.gz$"
      peak_format: "narrowPeak"
      expected:
        organism: "Homo sapiens"
        library_strategy: "ATAC-seq"
    - accession: "GSE177046"
      title: "Test B cell ATAC-seq"
      cell_type: "B_cell"
      assay_type: "ATAC-seq"
      organism: "Homo sapiens"
      url: "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE177046"
      cell_type_default: "B_cell"
      sample_title_regex: ".*"
      supplementary_regex: ".*\\\\.gz$"
      peak_format: "narrowPeak"
      expected:
        organism: "Homo sapiens"
        library_strategy: "ATAC-seq"

reference_genome:
  version: "hg38"
  equivalent_builds: ["hg38", "GRCh38"]

paths:
  raw_dir: "{raw_dir}"
"""

SOFT_TEXT = """\
^SAMPLE = GSM2707001
!Sample_title = CD4+ T cell ATAC rep1
!Sample_organism_ch1 = Homo sapiens
!Sample_library_strategy = ATAC-seq
!Sample_characteristics_ch1 = cell type: CD4+ T cell
!Sample_supplementary_file = ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2707nnn/GSM2707001/suppl/GSM2707001_peaks.narrowPeak.gz
!Sample_data_processing = Assembly: GRCh38
^SAMPLE = GSM2707002
!Sample_title = CD4+ T cell ATAC rep2
!Sample_organism_ch1 = Homo sapiens
!Sample_library_strategy = ATAC-seq
!Sample_characteristics_ch1 = cell type: CD4+ T cell
!Sample_supplementary_file = ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2707nnn/GSM2707002/suppl/GSM2707002_peaks.narrowPeak.gz
"""


@pytest.fixture
def temp_env():
    """Create a temporary directory with a valid data_config.yaml."""
    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "data_config.yaml")
        raw_dir = os.path.join(tmp, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as fh:
            fh.write(SAMPLE_CONFIG.format(raw_dir=raw_dir))
        yield config_path, raw_dir


# ---------------------------------------------------------------------------
# GEOSample dataclass
# ---------------------------------------------------------------------------


class TestGEOSample:
    """Tests for the GEOSample dataclass."""

    def test_characteristic_found(self) -> None:
        """Return value when key matches."""
        sample = GEOSample(gsm="GSM1", characteristics=["cell type: CD4+ T cell"])
        assert sample.characteristic("cell type") == "CD4+ T cell"

    def test_characteristic_not_found(self) -> None:
        """Return None when key is absent."""
        sample = GEOSample(gsm="GSM1", characteristics=["tissue: brain"])
        assert sample.characteristic("cell type") is None

    def test_characteristic_case_insensitive(self) -> None:
        """Keys should be matched case-insensitively."""
        sample = GEOSample(gsm="GSM1", characteristics=["Cell Type: B cell"])
        assert sample.characteristic("cell type") == "B cell"


# ---------------------------------------------------------------------------
# Init and config
# ---------------------------------------------------------------------------


def test_geo_downloader_init_missing_config() -> None:
    """Missing config raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        GEODownloader(config_path="non_existent.yaml")


def test_get_verified_accessions(temp_env) -> None:
    """Verified accessions come from the config."""
    config_path, _ = temp_env
    dl = GEODownloader(config_path=config_path)
    assert dl.get_verified_accessions() == ["GSE100738", "GSE177046"]


def test_get_dataset_config_unknown(temp_env) -> None:
    """Unknown accession raises KeyError."""
    config_path, _ = temp_env
    dl = GEODownloader(config_path=config_path)
    with pytest.raises(KeyError, match="GSE999999"):
        dl.get_dataset_config("GSE999999")


# ---------------------------------------------------------------------------
# SOFT parsing
# ---------------------------------------------------------------------------


def test_parse_soft() -> None:
    """parse_soft extracts all sample records from SOFT text."""
    samples = GEODownloader.parse_soft(SOFT_TEXT)
    assert len(samples) == 2
    assert samples[0].gsm == "GSM2707001"
    assert samples[0].organism == "Homo sapiens"
    assert samples[0].library_strategy == "ATAC-seq"
    assert len(samples[0].supplementary_urls) == 1
    assert samples[1].gsm == "GSM2707002"


def test_extract_assembly() -> None:
    """extract_assembly finds the Assembly: line."""
    assert GEODownloader.extract_assembly(SOFT_TEXT) == "GRCh38"
    assert GEODownloader.extract_assembly("No assembly here.") is None


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@patch.object(GEODownloader, "_http_get")
def test_verify_dataset_success(mock_http, temp_env) -> None:
    """Successful verification when metadata matches config expectations."""
    config_path, _ = temp_env
    mock_http.return_value = SOFT_TEXT.encode("utf-8")
    dl = GEODownloader(config_path=config_path)
    result = dl.verify_dataset("GSE100738", use_cache=False)
    assert result["accession"] == "GSE100738"
    assert result["n_samples"] == 2


@patch.object(GEODownloader, "_http_get")
def test_verify_dataset_organism_mismatch(mock_http, temp_env) -> None:
    """Verification fails when organism doesn't match."""
    config_path, _ = temp_env
    bad_soft = SOFT_TEXT.replace("Homo sapiens", "Mus musculus")
    mock_http.return_value = bad_soft.encode("utf-8")
    dl = GEODownloader(config_path=config_path)
    with pytest.raises(DatasetVerificationError, match="organism"):
        dl.verify_dataset("GSE100738", use_cache=False)


# ---------------------------------------------------------------------------
# gzip validation
# ---------------------------------------------------------------------------


def test_is_valid_gzip_true(tmp_path) -> None:
    """A real gzip file returns True."""
    gz_path = str(tmp_path / "test.gz")
    with gzip.open(gz_path, "wb") as fh:
        fh.write(b"test content for gzip validation")
    assert GEODownloader._is_valid_gzip(gz_path) is True


def test_is_valid_gzip_false(tmp_path) -> None:
    """A plain text file returns False."""
    plain = str(tmp_path / "test.txt")
    with open(plain, "wb") as fh:
        fh.write(b"not a gzip file")
    assert GEODownloader._is_valid_gzip(plain) is False


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


@patch.object(GEODownloader, "_http_get")
def test_download_peak_file(mock_http, temp_env) -> None:
    """download_peak_file saves a valid gzip and returns the path."""
    config_path, raw_dir = temp_env
    dl = GEODownloader(config_path=config_path)

    # Produce real gzip bytes
    import io
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(b"chr1\t100\t200\tpeak_1\t500\t.\t10.0\t5.0\t3.0\t50\n")
    mock_http.return_value = buf.getvalue()

    record = {
        "accession": "GSE100738",
        "gsm": "GSM2707001",
        "url": "https://example.com/GSM2707001_peaks.narrowPeak.gz",
        "cell_type": "CD4_T_cell",
        "peak_format": "narrowPeak",
    }
    local = dl.download_peak_file(record, use_cache=False)
    assert os.path.exists(local)
    assert local.endswith("GSM2707001_peaks.narrowPeak.gz")


def test_download_peak_file_cached(temp_env) -> None:
    """Cached valid gzip is reused without network call."""
    config_path, raw_dir = temp_env
    dl = GEODownloader(config_path=config_path)

    # Pre-create a valid gzip cache
    peaks_dir = os.path.join(raw_dir, "peaks", "GSE100738")
    os.makedirs(peaks_dir, exist_ok=True)
    cached = os.path.join(peaks_dir, "GSM2707001_cached.gz")
    with gzip.open(cached, "wb") as fh:
        fh.write(b"cached peak data")

    record = {
        "accession": "GSE100738",
        "gsm": "GSM2707001",
        "url": "https://example.com/GSM2707001_cached.gz",
        "cell_type": "CD4_T_cell",
        "peak_format": "narrowPeak",
    }
    # No _http_get mock needed — cache should be used
    local = dl.download_peak_file(record, use_cache=True)
    assert local == cached
