"""Unit tests for src/data_processing/bed_processor.py."""

import os
import tempfile
import pandas as pd
import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from src.data_processing.bed_processor import BEDProcessor


@pytest.fixture
def temp_config_file():
    """Create a temporary data config file for testing BEDProcessor."""
    config_content = """
reference_genome:
  version: "hg38"
  primary_chromosomes:
    - "chr1"
    - "chr2"

sequence_encoding:
  sequence_length: 1000

paths:
  bed_dir: "{temp_dir}/bed"
  fasta_dir: "{temp_dir}/fasta"
"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "data_config.yaml")
        with open(config_path, "w") as f:
            f.write(config_content.format(temp_dir=temp_dir))
        yield config_path, temp_dir


@pytest.fixture
def sample_genome_fasta():
    """Create a mock reference genome FASTA file with chr1 (length 5000)."""
    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as f:
        seq_chr1 = Seq("A" * 2500 + "C" * 2500)
        rec1 = SeqRecord(seq_chr1, id="chr1", description="mock chromosome 1")
        SeqIO.write([rec1], f, "fasta")
        fasta_path = f.name
    yield fasta_path
    if os.path.exists(fasta_path):
        os.remove(fasta_path)


def test_bed_processor_init(temp_config_file):
    """Test initialization of BEDProcessor."""
    config_path, temp_dir = temp_config_file
    processor = BEDProcessor(config_path=config_path)

    assert processor.genome_version == "hg38"
    assert processor.target_window_len == 1000
    assert processor.primary_chromosomes == ["chr1", "chr2"]


def test_filter_primary_chromosomes(temp_config_file):
    """Test filtering out non-primary chromosomes."""
    config_path, _ = temp_config_file
    processor = BEDProcessor(config_path=config_path)

    df = pd.DataFrame({
        "chrom": ["chr1", "chr2", "chr1_gl000191_random", "chrUn_gl000211"],
        "start": [100, 200, 300, 400],
        "end": [500, 600, 700, 800]
    })

    filtered = processor.filter_primary_chromosomes(df)
    assert list(filtered["chrom"]) == ["chr1", "chr2"]


def test_resize_peak_to_window_summit_vs_midpoint(temp_config_file):
    """Test peak resizing using summit offset vs midpoint fallback."""
    config_path, _ = temp_config_file
    processor = BEDProcessor(config_path=config_path)

    # Peak: start=1000, end=1400 (length 400)
    # 1. With summit offset = 100 (relative to start 1000 -> summit at 1100)
    st, en, method = processor.resize_peak_to_window(1000, 1400, summit_offset=100, target_len=1000)
    assert method == "summit"
    # Center = 1100 -> start = 1100 - 500 = 600, end = 600 + 1000 = 1600
    assert st == 600
    assert en == 1600
    assert (en - st) == 1000

    # 2. Without summit offset -> midpoint fallback (center at 1200)
    st, en, method = processor.resize_peak_to_window(1000, 1400, summit_offset=None, target_len=1000)
    assert method == "midpoint"
    # Center = 1200 -> start = 1200 - 500 = 700, end = 700 + 1000 = 1700
    assert st == 700
    assert en == 1700


def test_load_peak_file(temp_config_file):
    """Test loading a narrowPeak file into a normalised DataFrame."""
    config_path, temp_dir = temp_config_file
    processor = BEDProcessor(config_path=config_path)

    # Standard 10-column narrowPeak line
    bed_content = "chr1\t1000\t1400\tpeak1\t100\t.\t2.5\t5.0\t3.0\t100\n"
    input_bed = os.path.join(temp_dir, "input.narrowPeak")
    with open(input_bed, "w") as f:
        f.write(bed_content)

    frame = processor.load_peak_file(input_bed, peak_format="narrowPeak")
    assert len(frame) == 1
    assert frame.iloc[0]["chrom"] == "chr1"
    assert frame.iloc[0]["summit_offset"] == 100
    assert frame.iloc[0]["fold_enrichment"] == 2.5


def test_extract_fasta(temp_config_file, sample_genome_fasta):
    """Test extracting FASTA sequences from a windows DataFrame."""
    config_path, temp_dir = temp_config_file
    processor = BEDProcessor(config_path=config_path)

    # Build a windows DataFrame matching the format produced by build_windows()
    windows = pd.DataFrame({
        "chrom": ["chr1"],
        "start": [1000],
        "end": [2000],
        "name": ["CD4_T_cell|chr1:1000-2000"],
        "cell_type": ["CD4_T_cell"],
        "peak_score": [100.0],
        "fold_enrichment": [2.5],
        "gsm": ["GSM0001"],
        "center_method": ["summit"],
    })

    fasta_out, meta_out = processor.extract_fasta(
        windows=windows,
        reference_fasta_path=sample_genome_fasta,
        fasta_filename="extracted.fasta",
        metadata_filename="extracted_meta.csv",
    )

    assert os.path.exists(fasta_out)
    records = list(SeqIO.parse(fasta_out, "fasta"))
    assert len(records) == 1
    assert len(records[0].seq) == 1000
    # In mock genome fasta, indices 1000-2000 are all 'A' (0-2500 are 'A')
    assert str(records[0].seq) == "A" * 1000

    # Metadata CSV should have one row
    assert os.path.exists(meta_out)
    meta = pd.read_csv(meta_out)
    assert len(meta) == 1
    assert meta.iloc[0]["cell_type"] == "CD4_T_cell"


def test_collapse_redundant_windows_keeps_strongest_per_element():
    """Near-identical windows from different donors collapse to the best one."""
    windows = pd.DataFrame(
        {
            "cell_type": ["B_cell"] * 3 + ["CD4_T_cell"],
            "chrom": ["chr1"] * 4,
            "win_start": [1000, 1030, 1055, 1010],
            "peak_score": [50.0, 120.0, 80.0, 10.0],
        }
    )

    collapsed = BEDProcessor.collapse_redundant_windows(windows, min_separation=500)

    # The three B_cell windows describe one element; the strongest survives.
    b_cell = collapsed[collapsed["cell_type"] == "B_cell"]
    assert len(b_cell) == 1
    assert b_cell.iloc[0]["peak_score"] == 120.0
    # A different cell type at the same locus is a distinct training example.
    assert len(collapsed[collapsed["cell_type"] == "CD4_T_cell"]) == 1


def test_collapse_redundant_windows_keeps_separated_windows():
    """Windows farther apart than min_separation are all retained."""
    windows = pd.DataFrame(
        {
            "cell_type": ["B_cell"] * 3,
            "chrom": ["chr1", "chr1", "chr2"],
            "win_start": [1000, 5000, 1000],
            "peak_score": [10.0, 20.0, 30.0],
        }
    )

    collapsed = BEDProcessor.collapse_redundant_windows(windows, min_separation=500)

    assert len(collapsed) == 3


def test_collapse_redundant_windows_disabled():
    """A non-positive separation leaves the frame untouched."""
    windows = pd.DataFrame(
        {
            "cell_type": ["B_cell", "B_cell"],
            "chrom": ["chr1", "chr1"],
            "win_start": [1000, 1001],
            "peak_score": [10.0, 20.0],
        }
    )

    assert len(BEDProcessor.collapse_redundant_windows(windows, min_separation=0)) == 2


def test_drop_shared_windows_removes_loci_called_in_several_cell_types():
    """A locus called in two cell types is ambiguous and goes; a unique one stays."""
    windows = pd.DataFrame(
        {
            "cell_type": ["B_cell", "CD4_T_cell", "microglia", "B_cell"],
            "chrom": ["chr1", "chr1", "chr2", "chr3"],
            # chr1 1000/1200 are the same element under two labels; chr2 and chr3
            # are each seen in only one cell type.
            "win_start": [1000, 1200, 5000, 7000],
            "peak_score": [10.0, 20.0, 30.0, 40.0],
        }
    )

    exclusive = BEDProcessor.drop_shared_windows(windows, min_separation=500)

    assert sorted(exclusive["chrom"]) == ["chr2", "chr3"]


def test_drop_shared_windows_keeps_same_locus_on_different_chromosomes():
    """Equal starts on different chromosomes are different loci."""
    windows = pd.DataFrame(
        {
            "cell_type": ["B_cell", "CD4_T_cell"],
            "chrom": ["chr1", "chr2"],
            "win_start": [1000, 1000],
            "peak_score": [10.0, 20.0],
        }
    )

    assert len(BEDProcessor.drop_shared_windows(windows, min_separation=500)) == 2


def test_drop_shared_windows_disabled():
    """A non-positive separation leaves the frame untouched."""
    windows = pd.DataFrame(
        {
            "cell_type": ["B_cell", "CD4_T_cell"],
            "chrom": ["chr1", "chr1"],
            "win_start": [1000, 1001],
            "peak_score": [10.0, 20.0],
        }
    )

    assert len(BEDProcessor.drop_shared_windows(windows, min_separation=0)) == 2
