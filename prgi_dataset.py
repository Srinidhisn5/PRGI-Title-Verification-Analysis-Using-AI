import pandas as pd
import os


def _read_titles_from_file(path):
    """Read titles from a CSV or Excel file. Returns list of title strings preserving original text."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"PRGI file not found: {path}")

    _, ext = os.path.splitext(path)
    try:
        if ext.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, encoding='utf-8')
    except Exception as e:
        raise RuntimeError(f"Failed to read PRGI file {path}: {e}")

    # Determine title column (case-insensitive match for 'title')
    title_col = None
    for col in df.columns:
        if 'title' == str(col).strip().lower():
            title_col = col
            break
    if title_col is None:
        # Fallback: if single column, use it
        if len(df.columns) == 1:
            title_col = df.columns[0]
        else:
            raise RuntimeError(f"PRGI file {path} missing a clear title column")

    titles = []
    for val in df[title_col].tolist():
        if pd.isna(val):
            continue
        t = str(val)
        # Preserve row text as-is (only strip leading/trailing whitespace)
        t = t.strip()
        if t:
            titles.append(t)

    return titles


def load_prgi_titles():
    """Load only the authoritative PRGI_MASTER_DATASET file and return its titles.

    This function fails loudly if the master file is missing or malformed.
    """
    base_dir = os.path.dirname(__file__)
    db_dir = os.path.join(base_dir, 'database')

    master_base = 'PRGI_MASTER_DATASET'
    found = False
    titles = []
    for ext in ['.xlsx', '.xls', '.csv']:
        path = os.path.join(db_dir, master_base + ext)
        if os.path.exists(path):
            titles = _read_titles_from_file(path)
            found = True
            break

    if not found:
        raise RuntimeError(f"Missing authoritative PRGI master file '{master_base}' in {db_dir}")

    total = len(titles)
    # Strict startup log per requirements
    print(f"PRGI MASTER DATASET loaded: {total}")

    return titles
