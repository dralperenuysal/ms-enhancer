"""Sequence and metadata encoder module for MS-ENHANCER-GEN.

Converts FASTA sequences and cell-type metadata into model-ready PyTorch tensors
and saves processed arrays for training.
"""

import os
import logging
from typing import List, Tuple, Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import torch
import yaml
from Bio import SeqIO

logger = logging.getLogger(__name__)


class SequenceEncoder:
    """Encoder for DNA FASTA sequences and cell-type metadata into PyTorch tensors."""

    BASE_TO_INDEX: Dict[str, int] = {
        'A': 0, 'a': 0,
        'C': 1, 'c': 1,
        'G': 2, 'g': 2,
        'T': 3, 't': 3,
    }

    # Byte value -> one-hot row. 4 marks "ambiguous", handled by ``n_handling``.
    _BYTE_TO_ROW: np.ndarray = np.full(256, 4, dtype=np.uint8)
    for _base, _row in BASE_TO_INDEX.items():
        _BYTE_TO_ROW[ord(_base)] = _row
    del _base, _row

    def __init__(self, config_path: str = "configs/data_config.yaml") -> None:
        """Initialize SequenceEncoder with configuration settings.

        Args:
            config_path: Path to the data configuration YAML file.

        Raises:
            FileNotFoundError: If the specified config file does not exist.
            KeyError: If required configuration keys are missing.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Data configuration file not found at: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        seq_config = self.config.get("sequence_encoding", {})
        self.sequence_length: int = seq_config.get("sequence_length", 1000)
        self.n_handling: str = seq_config.get("n_handling", "uniform")

        if self.n_handling not in ["uniform", "zero"]:
            raise ValueError(f"Invalid n_handling configuration '{self.n_handling}'. Must be 'uniform' or 'zero'.")

        cond_config = self.config.get("condition_encoding", {})
        self.cell_types: List[str] = cond_config.get("cell_types", ["CD4_T_cell", "B_cell", "microglia"])
        self.normalization_method: str = cond_config.get("normalization_method", "zscore")
        self.continuous_features: List[str] = cond_config.get("continuous_features", ["peak_score", "fold_enrichment"])
        # Populated by build_condition_vectors; persisted with the dataset so that
        # generation can reuse the exact training-time normalization constants.
        self.normalization_stats: Optional[Dict[str, Any]] = None

        logger.info(
            f"SequenceEncoder initialized (length={self.sequence_length}, n_handling='{self.n_handling}', "
            f"normalization='{self.normalization_method}')."
        )

    def one_hot_encode_sequence(self, sequence: str) -> np.ndarray:
        """One-hot encode a single DNA sequence of shape (4, L).

        Args:
            sequence: Input nucleotide sequence string.

        Returns:
            Numpy array of shape (4, len(sequence)) with float32 data type.

        Raises:
            ValueError: If sequence length does not match expected length.
        """
        if len(sequence) != self.sequence_length:
            raise ValueError(
                f"Sequence length {len(sequence)} does not match required length {self.sequence_length}."
            )

        # Vectorised over the whole sequence: a 256-entry lookup table maps every
        # possible byte to a row index, so no per-base Python loop is needed.
        codes = np.frombuffer(sequence.encode("ascii", errors="replace"), dtype=np.uint8)
        rows = self._BYTE_TO_ROW[codes]

        encoded = np.zeros((4, len(sequence)), dtype=np.float32)
        known = rows < 4
        encoded[rows[known], np.nonzero(known)[0]] = 1.0
        # Ambiguous/unknown nucleotides (N and other IUPAC codes).
        if self.n_handling == "uniform":
            encoded[:, ~known] = 0.25

        return encoded

    def decode_one_hot(self, tensor: Union[np.ndarray, torch.Tensor]) -> str:
        """Decode a (4, L) one-hot matrix back to a nucleotide string.

        Args:
            tensor: Matrix of shape (4, L) as numpy array or PyTorch tensor.

        Returns:
            Reconstructed DNA sequence string.

        Raises:
            ValueError: If input shape is invalid.
        """
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = tensor

        if arr.ndim != 2 or arr.shape[0] != 4:
            raise ValueError(f"Input tensor must have shape (4, L), got shape {arr.shape}.")

        bases = ["A", "C", "G", "T"]
        seq_chars: List[str] = []

        for i in range(arr.shape[1]):
            col = arr[:, i]
            # Check if uniform [0.25, 0.25, 0.25, 0.25] or zero [0, 0, 0, 0]
            if np.allclose(col, 0.25) or np.allclose(col, 0.0):
                seq_chars.append("N")
            else:
                max_idx = int(np.argmax(col))
                seq_chars.append(bases[max_idx])

        return "".join(seq_chars)

    def encode_fasta(self, fasta_path: str) -> Tuple[torch.Tensor, List[str]]:
        """Encode sequences from a FASTA file into a PyTorch tensor.

        Args:
            fasta_path: Path to the FASTA input file.

        Returns:
            Tuple of (sequences_tensor, header_list):
            - sequences_tensor: PyTorch FloatTensor of shape (N, 4, L)
            - header_list: List of FASTA header IDs

        Raises:
            FileNotFoundError: If fasta_path does not exist.
            ValueError: If fasta file contains no valid sequences or sequences with wrong length.
        """
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

        records = list(SeqIO.parse(fasta_path, "fasta"))
        if not records:
            raise ValueError(f"No FASTA records found in file: {fasta_path}")

        encoded_list: List[np.ndarray] = []
        headers: List[str] = []

        for record in records:
            seq_str = str(record.seq).upper()
            if len(seq_str) != self.sequence_length:
                logger.warning(
                    f"Skipping record {record.id}: length {len(seq_str)} != required {self.sequence_length}"
                )
                continue
            encoded_list.append(self.one_hot_encode_sequence(seq_str))
            headers.append(record.id)

        if not encoded_list:
            raise ValueError(f"No valid sequences of length {self.sequence_length} found in {fasta_path}.")

        encoded_arr = np.stack(encoded_list, axis=0)  # Shape: (N, 4, L)
        tensor = torch.from_numpy(encoded_arr).float()

        logger.info(f"Successfully encoded {len(headers)} sequences from {fasta_path} into tensor of shape {tensor.shape}.")
        return tensor, headers

    def build_condition_vectors(
        self,
        metadata_df: pd.DataFrame,
        stats: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Build condition matrix C of shape (N, k) from sample metadata.

        Continuous features are normalized according to self.normalization_method (z-score or minmax).
        Categorical cell type feature is one-hot encoded according to self.cell_types.

        The normalization constants are stored on ``self.normalization_stats`` so
        they can be persisted with the dataset. Generation must reuse the training
        constants — normalizing generation-time inputs against their own statistics
        would place them on a different scale than the model was trained on.

        Args:
            metadata_df: DataFrame containing metadata columns (e.g. 'cell_type', 'peak_score', 'fold_enrichment').
            stats: Previously fitted constants from :attr:`normalization_stats`. If
                given they are applied as-is instead of being re-estimated.

        Returns:
            PyTorch FloatTensor of shape (N, k).

        Raises:
            KeyError: If required metadata columns are missing.
            ValueError: If metadata DataFrame is empty.
        """
        if metadata_df.empty:
            raise ValueError("Metadata DataFrame cannot be empty.")

        if "cell_type" not in metadata_df.columns:
            raise KeyError("Metadata DataFrame must contain a 'cell_type' column.")

        for feat in self.continuous_features:
            if feat not in metadata_df.columns:
                raise KeyError(f"Metadata DataFrame missing continuous feature column: '{feat}'")

        # 1. One-hot encode categorical cell_type
        cell_type_arr = np.zeros((len(metadata_df), len(self.cell_types)), dtype=np.float32)
        for i, ct in enumerate(metadata_df["cell_type"]):
            if ct in self.cell_types:
                cell_type_arr[i, self.cell_types.index(ct)] = 1.0
            else:
                logger.warning(f"Unrecognized cell type '{ct}' at index {i}. Left as zero vector.")

        # 2. Normalize continuous features
        cont_vals = metadata_df[self.continuous_features].values.astype(np.float32)
        if self.normalization_method == "zscore":
            if stats is None:
                means = np.mean(cont_vals, axis=0, keepdims=True)
                stds = np.std(cont_vals, axis=0, keepdims=True)
                stds[stds == 0] = 1.0  # Prevent division by zero
            else:
                means = np.asarray(stats["center"], dtype=np.float32).reshape(1, -1)
                stds = np.asarray(stats["scale"], dtype=np.float32).reshape(1, -1)
            norm_cont_vals = (cont_vals - means) / stds
            center, scale = means, stds
        elif self.normalization_method == "minmax":
            if stats is None:
                mins = np.min(cont_vals, axis=0, keepdims=True)
                maxs = np.max(cont_vals, axis=0, keepdims=True)
                ranges = maxs - mins
                ranges[ranges == 0] = 1.0  # Prevent division by zero
            else:
                mins = np.asarray(stats["center"], dtype=np.float32).reshape(1, -1)
                ranges = np.asarray(stats["scale"], dtype=np.float32).reshape(1, -1)
            norm_cont_vals = (cont_vals - mins) / ranges
            center, scale = mins, ranges
        else:
            raise ValueError(f"Unsupported normalization method: '{self.normalization_method}'")

        self.normalization_stats: Dict[str, Any] = {
            "method": self.normalization_method,
            "features": list(self.continuous_features),
            "center": center.ravel().tolist(),
            "scale": scale.ravel().tolist(),
        }

        # Concatenate categorical and continuous features
        condition_matrix = np.hstack([cell_type_arr, norm_cont_vals])
        tensor = torch.from_numpy(condition_matrix).float()

        logger.info(f"Built condition tensor of shape {tensor.shape} for {len(metadata_df)} samples.")
        return tensor

    def process_and_save_dataset(
        self,
        fasta_path: str,
        metadata_path: str,
        output_filename: str = "processed_dataset.pt"
    ) -> str:
        """Process FASTA and metadata, then save tensors to processed_dir.

        Args:
            fasta_path: Path to FASTA sequence file.
            metadata_path: Path to CSV/TSV metadata file.
            output_filename: Filename for saving PyTorch dictionary dataset.

        Returns:
            Path to saved PyTorch file.
        """
        output_dir = self.config.get("paths", {}).get("processed_dir", "data/processed")
        os.makedirs(output_dir, exist_ok=True)

        seq_tensors, headers = self.encode_fasta(fasta_path)

        if metadata_path.endswith(".csv"):
            meta_df = pd.read_csv(metadata_path)
        else:
            meta_df = pd.read_csv(metadata_path, sep="\t")

        # Align metadata to the sequences by identifier rather than by position:
        # encode_fasta drops malformed records, which would silently shift every
        # subsequent condition vector onto the wrong sequence.
        if "peak_id" in meta_df.columns:
            indexed = meta_df.set_index("peak_id")
            missing = [h for h in headers if h not in indexed.index]
            if missing:
                raise ValueError(
                    f"{len(missing)} FASTA records have no metadata row (first: {missing[0]}). "
                    f"Regenerate both files with BEDProcessor.extract_fasta()."
                )
            meta_df = indexed.loc[headers].reset_index()
        elif len(meta_df) != len(headers):
            raise ValueError(
                f"Metadata has no 'peak_id' column and its length ({len(meta_df)}) does not match "
                f"the number of encoded sequences ({len(headers)}); alignment cannot be verified."
            )

        cond_tensors = self.build_condition_vectors(meta_df)

        if len(seq_tensors) != len(cond_tensors):
            raise ValueError(
                f"Mismatch between number of sequences ({len(seq_tensors)}) and metadata entries ({len(cond_tensors)})."
            )

        data_dict = {
            "sequences": seq_tensors,
            "conditions": cond_tensors,
            "headers": headers,
            "cell_types": self.cell_types,
            "continuous_features": self.continuous_features,
            "normalization_stats": self.normalization_stats,
        }

        save_path = os.path.join(output_dir, output_filename)
        torch.save(data_dict, save_path)
        logger.info(f"Successfully saved processed dataset to {save_path}")
        return save_path
