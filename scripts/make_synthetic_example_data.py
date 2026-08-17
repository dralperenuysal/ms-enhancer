"""Generate a small synthetic ATAC-seq peak dataset + peak_manifest.json for
offline pipeline testing, so a stranger can run `build_dataset.py` (and the
rest of main.nf) without touching live GEO. NOT real biological data: peaks
are placed at real MS GWAS risk locus coordinates (data/bed/ms_gwas_loci_hg38.bed)
but scores/signal are fabricated. See docs/REPRODUCING.md section 10.

Usage:
    python scripts/make_synthetic_example_data.py
    # then:
    python scripts/build_dataset.py --manifest data/example_synthetic/peak_manifest.json
"""
import csv
import json
import os
import random

SEED = 42
OUT_DIR = "data/example_synthetic"
GWAS_BED = "data/bed/ms_gwas_loci_hg38.bed"
CELL_TYPES = ["CD4_T_cell", "B_cell", "microglia"]
PEAKS_PER_SAMPLE = 40
NARROWPEAK_COLUMNS = [
    "chrom", "start", "end", "name", "score", "strand",
    "signal_value", "p_value", "q_value", "summit_offset",
]


def load_loci(path):
    loci = []
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            loci.append((row[0], int(row[1]), int(row[2])))
    return loci


def make_peaks(rng, loci, n):
    peaks = []
    for i in range(n):
        chrom, lstart, lend = rng.choice(loci)
        width = rng.randint(200, 500)
        start = rng.randint(lstart, max(lstart, lend - width))
        end = start + width
        peaks.append({
            "chrom": chrom, "start": start, "end": end,
            "name": f"synthetic_peak_{i}", "score": rng.randint(100, 1000),
            "strand": ".", "signal_value": round(rng.uniform(2, 20), 3),
            "p_value": round(rng.uniform(2, 6), 4),
            "q_value": round(rng.uniform(1, 5), 4),
            "summit_offset": width // 2,
        })
    return peaks


def main():
    rng = random.Random(SEED)
    loci = load_loci(GWAS_BED)
    peaks_dir = os.path.join(OUT_DIR, "peaks")
    os.makedirs(peaks_dir, exist_ok=True)

    manifest = []
    for cell_type in CELL_TYPES:
        for rep in (1, 2):
            gsm = f"SYNTH_{cell_type}_{rep}"
            local_path = os.path.join(peaks_dir, f"{gsm}.narrowPeak")
            peaks = make_peaks(rng, loci, PEAKS_PER_SAMPLE)
            with open(local_path, "w", encoding="utf-8") as f:
                for p in peaks:
                    f.write("\t".join(str(p[c]) for c in NARROWPEAK_COLUMNS) + "\n")
            manifest.append({
                "accession": "SYNTHETIC_EXAMPLE",
                "gsm": gsm,
                "title": f"Synthetic example {cell_type} replicate {rep}",
                "cell_type": cell_type,
                "url": "",
                "peak_format": "narrowPeak",
                "local_path": local_path,
            })

    manifest_path = os.path.join(OUT_DIR, "peak_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} synthetic samples to {manifest_path}")


if __name__ == "__main__":
    main()
