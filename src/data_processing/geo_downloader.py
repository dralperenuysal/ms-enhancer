"""GEO metadata verification and peak-file downloader for MS-ENHANCER-GEN.

This module deliberately does two things that the original draft did not:

1. It *verifies* every accession listed in ``configs/data_config.yaml`` against the
   live GEO SOFT record (organism, library strategy, genome assembly, sample count)
   and raises :class:`DatasetVerificationError` on any mismatch. An accession whose
   claimed description does not survive this check must never reach the pipeline.
2. It downloads per-sample *peak call* files (MACS narrowPeak or HOMER findPeaks
   tables) rather than raw reads, so the pipeline can be reproduced without an
   alignment step.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

GEO_ACC_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"


class DatasetVerificationError(RuntimeError):
    """Raised when a configured GEO accession does not match its recorded description."""


@dataclass
class GEOSample:
    """A single GEO sample (GSM) with the fields this pipeline depends on.

    Attributes:
        gsm: GEO sample accession, e.g. ``GSM6094271``.
        title: Sample title as reported by GEO.
        characteristics: Raw ``!Sample_characteristics_ch1`` strings.
        supplementary_urls: Supplementary file URLs advertised by GEO.
        organism: Value of ``!Sample_organism_ch1``.
        library_strategy: Value of ``!Sample_library_strategy``.
    """

    gsm: str
    title: str = ""
    characteristics: List[str] = field(default_factory=list)
    supplementary_urls: List[str] = field(default_factory=list)
    organism: str = ""
    library_strategy: str = ""

    def characteristic(self, key: str) -> Optional[str]:
        """Return the value of a ``key: value`` characteristic, or None.

        Args:
            key: Characteristic key, e.g. ``"cell type"``.

        Returns:
            The value string, or None when the key is absent.
        """
        prefix = f"{key.lower()}:"
        for entry in self.characteristics:
            if entry.lower().startswith(prefix):
                return entry.split(":", 1)[1].strip()
        return None


class GEODownloader:
    """Verifies and downloads peak-call files for the configured GEO datasets."""

    def __init__(self, config_path: str = "configs/data_config.yaml") -> None:
        """Initialize from the data configuration file.

        Args:
            config_path: Path to ``data_config.yaml``.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If no verified datasets are configured.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as handle:
            self.config: Dict[str, Any] = yaml.safe_load(handle)

        paths = self.config.get("paths", {})
        self.raw_dir: str = paths.get("raw_dir", "data/raw")
        os.makedirs(self.raw_dir, exist_ok=True)

        geo_config = self.config.get("geo_datasets", {})
        self.verified_datasets: List[Dict[str, Any]] = geo_config.get("verified_datasets", [])
        if not self.verified_datasets:
            raise ValueError(
                f"No verified GEO datasets configured in {config_path}. "
                "Verify accessions against GEO and record them under geo_datasets.verified_datasets."
            )

        ref = self.config.get("reference_genome", {})
        self.accepted_builds: List[str] = [
            b.lower() for b in ref.get("equivalent_builds", [ref.get("version", "hg38")])
        ]

        logger.info(
            "GEODownloader initialised (raw_dir=%s, %d verified datasets, accepted builds=%s).",
            self.raw_dir,
            len(self.verified_datasets),
            self.accepted_builds,
        )

    # ------------------------------------------------------------------ config

    def get_verified_accessions(self) -> List[str]:
        """Return the configured, human-verified GEO accession numbers.

        Returns:
            List of GSE accession strings.
        """
        return [d["accession"] for d in self.verified_datasets if "accession" in d]

    def get_dataset_config(self, accession: str) -> Dict[str, Any]:
        """Look up the configuration block for one accession.

        Args:
            accession: GSE accession.

        Returns:
            The dataset configuration mapping.

        Raises:
            KeyError: If the accession is not configured.
        """
        for dataset in self.verified_datasets:
            if dataset.get("accession") == accession:
                return dataset
        raise KeyError(
            f"Accession '{accession}' is not in geo_datasets.verified_datasets. "
            "Unverified accessions are refused by design."
        )

    # ----------------------------------------------------------------- fetching

    @staticmethod
    def _http_get(url: str, timeout: int = 120) -> bytes:
        """Fetch a URL and return its body.

        Args:
            url: URL to fetch.
            timeout: Socket timeout in seconds.

        Returns:
            Response body bytes.
        """
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()

    def fetch_soft_metadata(self, accession: str, use_cache: bool = True) -> str:
        """Download (or reuse) the GEO SOFT sample metadata for a series.

        Args:
            accession: GSE accession.
            use_cache: Reuse ``data/raw/<accession>.txt`` when it already exists.

        Returns:
            SOFT text.

        Raises:
            DatasetVerificationError: If GEO returns no sample records.
        """
        cache_path = os.path.join(self.raw_dir, f"{accession}.txt")
        if use_cache and os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            logger.info("Using cached SOFT metadata for %s (%s).", accession, cache_path)
            with open(cache_path, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        else:
            url = f"{GEO_ACC_URL}?acc={accession}&targ=gsm&form=text&view=brief"
            logger.info("Fetching GEO SOFT metadata for %s ...", accession)
            text = self._http_get(url).decode("utf-8", errors="replace")
            with open(cache_path, "w", encoding="utf-8") as handle:
                handle.write(text)

        if "^SAMPLE" not in text:
            raise DatasetVerificationError(
                f"GEO returned no sample records for '{accession}'. The accession may not exist. "
                f"Checked {GEO_ACC_URL}?acc={accession}"
            )
        return text

    @staticmethod
    def parse_soft(text: str) -> List[GEOSample]:
        """Parse GEO SOFT sample text into :class:`GEOSample` records.

        Args:
            text: SOFT-formatted metadata.

        Returns:
            List of parsed samples, in file order.
        """
        samples: List[GEOSample] = []
        current: Optional[GEOSample] = None

        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if line.startswith("^SAMPLE"):
                current = GEOSample(gsm=line.split("=", 1)[1].strip())
                samples.append(current)
            elif current is None:
                continue
            elif line.startswith("!Sample_title"):
                current.title = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_characteristics_ch1"):
                current.characteristics.append(line.split("=", 1)[1].strip())
            elif line.startswith("!Sample_supplementary_file"):
                current.supplementary_urls.append(line.split("=", 1)[1].strip())
            elif line.startswith("!Sample_organism_ch1"):
                current.organism = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_library_strategy"):
                current.library_strategy = line.split("=", 1)[1].strip()

        return samples

    @staticmethod
    def extract_assembly(text: str) -> Optional[str]:
        """Extract the genome assembly declared in ``!Sample_data_processing``.

        Args:
            text: SOFT-formatted metadata.

        Returns:
            Assembly string (e.g. ``"GRCh38"``) or None when not declared.
        """
        match = re.search(r"Assembly:\s*(\S+)", text)
        return match.group(1).strip() if match else None

    # -------------------------------------------------------------- verification

    def verify_dataset(self, accession: str, use_cache: bool = True) -> Dict[str, Any]:
        """Check a configured accession against live GEO metadata.

        Args:
            accession: GSE accession to verify.
            use_cache: Reuse cached SOFT metadata when present.

        Returns:
            Dictionary with the observed organism, library strategy, assembly and
            sample count.

        Raises:
            DatasetVerificationError: On any disagreement with the ``expected``
                block in the configuration.
        """
        dataset = self.get_dataset_config(accession)
        expected = dataset.get("expected", {})

        text = self.fetch_soft_metadata(accession, use_cache=use_cache)
        samples = self.parse_soft(text)
        assembly = self.extract_assembly(text)

        organisms = sorted({s.organism for s in samples if s.organism})
        strategies = sorted({s.library_strategy for s in samples if s.library_strategy})
        observed = {
            "accession": accession,
            "organism": organisms,
            "library_strategy": strategies,
            "genome_build": assembly,
            "n_samples": len(samples),
        }

        problems: List[str] = []
        if "organism" in expected and organisms != [expected["organism"]]:
            problems.append(f"organism: expected ['{expected['organism']}'], GEO reports {organisms}")
        if "library_strategy" in expected and strategies != [expected["library_strategy"]]:
            problems.append(
                f"library_strategy: expected ['{expected['library_strategy']}'], GEO reports {strategies}"
            )
        if "n_samples" in expected and len(samples) != expected["n_samples"]:
            problems.append(f"n_samples: expected {expected['n_samples']}, GEO reports {len(samples)}")
        if assembly is not None and assembly.lower() not in self.accepted_builds:
            problems.append(
                f"genome_build: GEO reports '{assembly}', which is not one of the accepted "
                f"builds {self.accepted_builds}. Liftover is not implemented; refusing to mix builds."
            )

        if problems:
            raise DatasetVerificationError(
                f"GEO metadata for {accession} contradicts configs/data_config.yaml:\n  - "
                + "\n  - ".join(problems)
                + "\nFix the config (or drop the accession) before running the pipeline."
            )

        logger.info(
            "Verified %s: %s / %s / %s / %d samples.",
            accession,
            organisms,
            strategies,
            assembly,
            len(samples),
        )
        return observed

    def verify_all(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Verify every configured accession.

        Args:
            use_cache: Reuse cached SOFT metadata when present.

        Returns:
            List of observation dictionaries, one per accession.
        """
        return [self.verify_dataset(acc, use_cache=use_cache) for acc in self.get_verified_accessions()]

    # ------------------------------------------------------------------ samples

    def select_samples(self, accession: str, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Resolve the samples of a series to (cell_type, peak file URL) records.

        Samples that carry no recognised cell type, or advertise no supplementary
        file matching the dataset's ``supplementary_regex``, are skipped with a log
        line rather than silently dropped.

        Args:
            accession: GSE accession.
            use_cache: Reuse cached SOFT metadata when present.

        Returns:
            List of dicts with keys ``gsm``, ``title``, ``cell_type``, ``url``,
            ``peak_format`` and ``accession``.

        Raises:
            DatasetVerificationError: If no sample survives selection.
        """
        dataset = self.get_dataset_config(accession)
        text = self.fetch_soft_metadata(accession, use_cache=use_cache)
        samples = self.parse_soft(text)

        cell_type_map: Dict[str, str] = dataset.get("cell_type_map", {})
        title_regex = dataset.get("sample_title_regex")
        default_cell_type = dataset.get("cell_type_default")
        supp_regex = re.compile(dataset.get("supplementary_regex", ".*"))
        peak_format = dataset.get("peak_format", "narrowPeak")

        selected: List[Dict[str, Any]] = []
        for sample in samples:
            cell_type: Optional[str] = None
            if cell_type_map:
                raw = sample.characteristic("cell type")
                if raw is not None:
                    cell_type = cell_type_map.get(raw)
            if cell_type is None and title_regex and re.search(title_regex, sample.title):
                cell_type = default_cell_type
            if cell_type is None:
                continue

            urls = [u for u in sample.supplementary_urls if supp_regex.search(u)]
            if not urls:
                logger.warning(
                    "Sample %s (%s) matched cell type %s but has no supplementary file matching %r.",
                    sample.gsm,
                    sample.title,
                    cell_type,
                    dataset.get("supplementary_regex"),
                )
                continue

            selected.append(
                {
                    "accession": accession,
                    "gsm": sample.gsm,
                    "title": sample.title,
                    "cell_type": cell_type,
                    "url": urls[0].replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov"),
                    "peak_format": peak_format,
                }
            )

        if not selected:
            raise DatasetVerificationError(
                f"No usable samples selected from {accession}. Check cell_type_map / "
                f"sample_title_regex / supplementary_regex in configs/data_config.yaml."
            )

        by_type: Dict[str, int] = {}
        for record in selected:
            by_type[record["cell_type"]] = by_type.get(record["cell_type"], 0) + 1
        logger.info("Selected %d samples from %s: %s", len(selected), accession, by_type)
        return selected

    # ---------------------------------------------------------------- downloads

    def download_peak_file(
        self,
        record: Dict[str, Any],
        max_retries: int = 3,
        use_cache: bool = True,
    ) -> str:
        """Download one sample's peak file into ``raw_dir``, with retries and caching.

        The downloaded gzip is validated by decompressing its first block; a corrupt
        cached file is re-downloaded rather than being trusted.

        Args:
            record: Sample record from :meth:`select_samples`.
            max_retries: Number of download attempts before giving up.
            use_cache: Reuse an existing, valid local copy.

        Returns:
            Local path to the downloaded file.

        Raises:
            RuntimeError: If the file cannot be downloaded after ``max_retries``.
        """
        peaks_dir = os.path.join(self.raw_dir, "peaks", record["accession"])
        os.makedirs(peaks_dir, exist_ok=True)
        local_path = os.path.join(peaks_dir, os.path.basename(record["url"]))

        if use_cache and os.path.exists(local_path) and self._is_valid_gzip(local_path):
            logger.debug("Cache hit for %s (%s).", record["gsm"], local_path)
            return local_path

        last_error: Optional[BaseException] = None
        for attempt in range(1, max_retries + 1):
            try:
                payload = self._http_get(record["url"])
                with open(local_path, "wb") as handle:
                    handle.write(payload)
                if not self._is_valid_gzip(local_path):
                    raise OSError("downloaded file is not a readable gzip archive")
                logger.info(
                    "Downloaded %s -> %s (%d bytes, md5=%s).",
                    record["gsm"],
                    local_path,
                    len(payload),
                    hashlib.md5(payload).hexdigest(),
                )
                return local_path
            except Exception as error:  # noqa: BLE001 - retried and re-raised below
                last_error = error
                logger.warning(
                    "Attempt %d/%d failed for %s: %s", attempt, max_retries, record["url"], error
                )

        raise RuntimeError(
            f"Failed to download {record['url']} after {max_retries} attempts: {last_error}"
        )

    @staticmethod
    def _is_valid_gzip(path: str) -> bool:
        """Return True when ``path`` is a gzip file whose first block decompresses.

        Args:
            path: Local file path.

        Returns:
            True if the file is readable gzip, False otherwise.
        """
        try:
            with gzip.open(path, "rb") as handle:
                handle.read(1024)
            return True
        except Exception:  # noqa: BLE001 - any failure means "re-download"
            return False

    def download_all(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Verify every configured dataset, then download all selected peak files.

        Args:
            use_cache: Reuse cached metadata and peak files.

        Returns:
            Sample records extended with a ``local_path`` key.
        """
        manifest: List[Dict[str, Any]] = []
        for accession in self.get_verified_accessions():
            self.verify_dataset(accession, use_cache=use_cache)
            for record in self.select_samples(accession, use_cache=use_cache):
                record["local_path"] = self.download_peak_file(record, use_cache=use_cache)
                manifest.append(record)

        logger.info("Downloaded %d peak files across %d series.", len(manifest), len(self.verified_datasets))
        return manifest
