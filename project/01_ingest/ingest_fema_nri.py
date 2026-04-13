"""
Ingest FEMA NRI DBF (946 MB, 468 fields) -> Parquet.

Loads ONLY the ~25 fields we need via a custom struct-based selective parser
to avoid the ~12 GB RAM requirement of loading all 468 columns.

Estimated runtime: 8–15 minutes (run once, then use the parquet).
"""

import sys
import struct
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from tqdm import tqdm
from config import FEMA_NRI_PATH, NRI_PARQUET, PROCESSED_RAW

# ── Fields to extract ─────────────────────────────────────────────────────────
NRI_KEEP = [
    'TRACTFIPS', 'STCOFIPS', 'POPULATION', 'BUILDVALUE', 'AGRIVALUE',
    'RISK_SCORE', 'RISK_RATNG', 'RESL_SCORE', 'RESL_RATNG',
    'SOVI_SCORE', 'EAL_SCORE',
    # Heat Wave
    'HWAV_RISKS', 'HWAV_RISKR', 'HWAV_EALT',
    # Inland Flood
    'IFLD_RISKS', 'IFLD_RISKR', 'IFLD_EALT',
    # Hurricane (Gulf Coast)
    'HRCN_RISKS', 'HRCN_RISKR',
    # Tornado
    'TRND_RISKS', 'TRND_RISKR',
    # Coastal Flood
    'CFLD_RISKS', 'CFLD_RISKR',
    # Drought
    'DRGT_RISKS', 'DRGT_RISKR',
    # Wildfire
    'WFIR_RISKS', 'WFIR_RISKR',
]


def parse_dbf_selective(filepath: Path, keep_fields: list[str]) -> pd.DataFrame:
    """
    Parse a DBF file, extracting only the specified field names.
    Returns a DataFrame with those columns.
    """
    keep_set = set(keep_fields)

    with open(filepath, 'rb') as f:
        # ── DBF header (32 bytes) ─────────────────────────────────────────────
        header = f.read(32)
        num_records  = struct.unpack('<I', header[4:8])[0]
        header_size  = struct.unpack('<H', header[8:10])[0]
        record_size  = struct.unpack('<H', header[10:12])[0]

        # ── Field descriptors (32 bytes each) ────────────────────────────────
        num_fields = (header_size - 32 - 1) // 32
        fields  = []
        offsets = [1]   # byte 0 in each record is the deletion flag
        for _ in range(num_fields):
            fd    = f.read(32)
            name  = fd[0:11].rstrip(b'\x00').decode('ascii', errors='ignore')
            ftype = chr(fd[11])
            flen  = fd[16]
            fields.append((name, ftype, flen))
            offsets.append(offsets[-1] + flen)

        # ── Build lookup: field_name -> (byte_offset, field_len, field_type) ──
        field_map = {
            name: (offsets[i], flen, ftype)
            for i, (name, ftype, flen) in enumerate(fields)
            if name in keep_set
        }
        missing = keep_set - set(field_map.keys())
        if missing:
            print(f"  WARNING: fields not found in DBF: {sorted(missing)}")

        # ── Read records ──────────────────────────────────────────────────────
        f.seek(header_size)
        records = []
        for _ in tqdm(range(num_records), desc="Parsing NRI DBF", unit="rec"):
            raw = f.read(record_size)
            if not raw:
                break
            if raw[0] == 0x2A:   # '*' = deleted record
                continue
            row = {}
            for fname, (off, fl, ft) in field_map.items():
                val = raw[off: off + fl].decode('latin-1').strip()
                if ft in ('N', 'F'):
                    row[fname] = float(val) if val else None
                else:
                    row[fname] = val
            records.append(row)

    return pd.DataFrame(records)


def ingest_fema_nri():
    print(f"Parsing FEMA NRI DBF: {FEMA_NRI_PATH}")
    print(f"  Records will be written to: {NRI_PARQUET}")
    PROCESSED_RAW.mkdir(parents=True, exist_ok=True)

    df = parse_dbf_selective(FEMA_NRI_PATH, NRI_KEEP)
    print(f"  Parsed {len(df):,} records, {len(df.columns)} columns")

    # ── Clean GEOID ───────────────────────────────────────────────────────────
    df['GEOID'] = df['TRACTFIPS'].astype(str).str.strip().str.zfill(11)
    df.drop(columns=['TRACTFIPS'], inplace=True)

    # ── Normalise rating strings ──────────────────────────────────────────────
    rating_cols = [c for c in df.columns if c.endswith('_RISKR') or c.endswith('_RATNG')]
    for col in rating_cols:
        df[col] = df[col].str.strip()

    # ── Coerce numeric columns ────────────────────────────────────────────────
    numeric_cols = [c for c in df.columns if c.endswith('_RISKS') or c.endswith('_EALT')
                    or c in ('RISK_SCORE','RESL_SCORE','SOVI_SCORE','EAL_SCORE','POPULATION','BUILDVALUE','AGRIVALUE')]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.to_parquet(NRI_PARQUET, index=False)
    print(f"  Saved -> {NRI_PARQUET}")
    print(f"  Shape: {df.shape}")
    print(f"  RISK_RATNG distribution:\n{df['RISK_RATNG'].value_counts()}")
    return df


if __name__ == "__main__":
    ingest_fema_nri()
