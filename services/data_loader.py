"""
Data Loader — VigilantMLOps

Central I/O layer for all dataset access. Two categories:

Pre-production (file-based)
    Raw  — UNSW-NB15 (4 part CSVs) and CICIoT2023 (169 part CSVs)
    Processed — balanced final stage only (train / test / val hybrid parquets)
    Blacklist — single IP blacklist parquet

Production (request-based)
    from_records() — converts incoming API request body records to a DataFrame;
                     no disk access, data lives only in memory for the request.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Config schemas
# ---------------------------------------------------------------------------


class RawDataPaths(BaseModel):
    unsw_nb15_dir: Path    # raw/UNSW-NB15/
    ciciot2023_dir: Path   # raw/CICIoT2023/


class DataPaths(BaseModel):
    raw: RawDataPaths
    balanced_dir: Path     # processed/balanced/  ← final pre-training stage
    blacklist: Path        # ip_blacklists/blacklist.parquet


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class DataLoader:
    """
    Unified I/O interface for all VigilantMLOps data sources.

    Parameters
    ----------
    paths:
        Configured paths for each data source.
    cache:
        When True (default), DataFrames are kept in memory after the first
        read.  Call ``invalidate()`` to evict entries and force a reload.
    """

    # Cache keys for named splits / sources
    _RAW_UNSW = "raw_unsw"
    _RAW_CICIOT = "raw_ciciot"
    _RAW_ALL = "raw_all"
    _BLACKLIST = "blacklist"

    def __init__(self, paths: DataPaths, *, cache: bool = True) -> None:
        self._paths = paths
        self._cache = cache
        self._store: dict[str, pl.DataFrame] = {}

    # ------------------------------------------------------------------
    # Pre-production — Raw data
    # ------------------------------------------------------------------

    def load_unsw_nb15(self) -> pl.DataFrame:
        """Load the four UNSW-NB15 main part files and concatenate."""
        if self._cache and self._RAW_UNSW in self._store:
            return self._store[self._RAW_UNSW]

        _UNSW_COLS = [
            "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
            "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
            "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb",
            "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
            "sjit", "djit", "stime", "ltime", "sintpkt", "dintpkt", "tcprtt",
            "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
            "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
            "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
            "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label",
        ]
        # sport, dsport, stcpb, dtcpb can contain hex literals (e.g. "0x000c").
        # Read them as Utf8 then normalise to Int64 after loading.
        _HEX_COLS = {"sport", "dsport", "stcpb", "dtcpb"}
        _schema_overrides = {col: pl.Utf8 for col in _HEX_COLS}

        part_glob = str(self._paths.raw.unsw_nb15_dir / "UNSW-NB15_[0-9].csv")
        df = pl.read_csv(
            part_glob,
            has_header=False,
            new_columns=_UNSW_COLS,
            infer_schema_length=10_000,
            schema_overrides=_schema_overrides,
        )

        # Normalise hex/decimal strings → Int64 (nulls stay null)
        df = df.with_columns(
            pl.when(pl.col(c).str.starts_with("0x"))
            .then(pl.col(c).str.slice(2).str.to_integer(base=16, strict=False))
            .otherwise(pl.col(c).cast(pl.Int64, strict=False))
            .alias(c)
            for c in _HEX_COLS
        )

        if self._cache:
            self._store[self._RAW_UNSW] = df
        return df

    def load_ciciot2023(self) -> pl.DataFrame:
        """Scan all 169 CICIoT2023 part CSVs and concatenate."""
        if self._cache and self._RAW_CICIOT in self._store:
            return self._store[self._RAW_CICIOT]

        part_glob = str(self._paths.raw.ciciot2023_dir / "part-*.csv")
        df = pl.read_csv(part_glob, infer_schema_length=10_000)

        if self._cache:
            self._store[self._RAW_CICIOT] = df
        return df

    def load_raw(self) -> pl.DataFrame:
        """Load and concatenate both raw sources (UNSW-NB15 + CICIoT2023)."""
        if self._cache and self._RAW_ALL in self._store:
            return self._store[self._RAW_ALL]

        # Normalise `label` to Utf8 in both sources before stacking —
        # UNSW-NB15 uses Int64 (0/1) while CICIoT2023 uses String category names.
        unsw = self.load_unsw_nb15().with_columns(pl.col("label").cast(pl.Utf8))
        ciciot = self.load_ciciot2023().with_columns(pl.col("label").cast(pl.Utf8))

        df = pl.concat(
            [unsw, ciciot],
            how="diagonal",  # fills missing columns with null
        )

        if self._cache:
            self._store[self._RAW_ALL] = df
        return df

    # ------------------------------------------------------------------
    # Pre-production — Processed (balanced final stage)
    # ------------------------------------------------------------------

    def load_train(self) -> pl.DataFrame:
        """Training split — also the drift reference distribution."""
        return self._load_balanced("train")

    def load_test(self) -> pl.DataFrame:
        return self._load_balanced("test")

    def load_val(self) -> pl.DataFrame:
        return self._load_balanced("val")

    def load_reference(self) -> pl.DataFrame:
        """Drift reference — training distribution (balanced hybrid split)."""
        return self.load_train()

    def load_split(self, split: str) -> pl.DataFrame:
        """
        Generic accessor for processed splits.
        ``split`` must be one of: train, test, val, reference.
        """
        dispatch = {
            "train": self.load_train,
            "test": self.load_test,
            "val": self.load_val,
            "reference": self.load_reference,
        }
        if split not in dispatch:
            raise ValueError(
                f"Unknown split '{split}'. Valid options: {list(dispatch)}"
            )
        return dispatch[split]()

    # ------------------------------------------------------------------
    # Blacklist
    # ------------------------------------------------------------------

    def load_blacklist(self) -> pl.DataFrame:
        if self._cache and self._BLACKLIST in self._store:
            return self._store[self._BLACKLIST]

        df = pl.read_parquet(self._paths.blacklist)

        if self._cache:
            self._store[self._BLACKLIST] = df
        return df

    # ------------------------------------------------------------------
    # Production — request body → DataFrame (no disk access)
    # ------------------------------------------------------------------

    @staticmethod
    def from_records(records: list[dict[str, Any]]) -> pl.DataFrame:
        """
        Convert incoming API request body records to a Polars DataFrame.
        Used for production data evaluation — data lives in memory only.
        """
        return pl.DataFrame(records)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate(self, key: str | None = None) -> None:
        """Evict one entry (or all entries) from the in-memory cache."""
        if key is None:
            self._store.clear()
        else:
            self._store.pop(key, None)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_balanced(self, split: str) -> pl.DataFrame:
        if self._cache and split in self._store:
            return self._store[split]

        path = self._paths.balanced_dir / f"{split}_nodup_hybrid.parquet"
        df = pl.read_parquet(path)

        if self._cache:
            self._store[split] = df
        return df
