import shutil
from pathlib import Path

import pandas as pd


BASE_DIR = Path(r"C:\Python\Personal\kfm-ms\codes")
PEST_DIR = BASE_DIR / "pest_level_1"
ORG_DIR = PEST_DIR / "org"

GRID_FILE = ORG_DIR / "grid_candidates.csv"


def backup_file(path: Path):
    if path.exists():
        backup_path = path.with_suffix(path.suffix + ".bak")
        shutil.copy(path, backup_path)
        print(f"[OK] Backup saved: {backup_path}")
    else:
        raise FileNotFoundError(f"File not found: {path}")


def edit_org_grid(level: str):
    level = str(level).strip()

    if not GRID_FILE.exists():
        raise FileNotFoundError(f"Missing org file: {GRID_FILE}")

    backup_file(GRID_FILE)

    df = pd.read_csv(GRID_FILE)

    print(f"[INFO] Loaded {GRID_FILE}")
    print(f"[INFO] Columns: {list(df.columns)}")
    print("[INFO] Preview before edit:")
    print(df.head(10))

    if "a" not in df.columns:
        raise KeyError("grid_candidates.csv must contain an 'a' column")

    if "qref" not in df.columns:
        raise KeyError("grid_candidates.csv must contain a 'qref' column")

    if level == "1":
        # Include zeros in the mean
        mean_qref = df["qref"].mean()

        # Preserve original pumping activity before overwriting qref
        active = df["qref"] > 0

        df["qref"] = mean_qref
        df["a"] = 0.0
        df.loc[active, "a"] = 1.0

        print("[INFO] Applied Level 1 template:")
        print("       all rows -> qref = mean including zeros")
        print("       active rows -> a = 1.0, zero-pumping rows -> a = 0.0")

    elif level == "2":
        # Exclude zeros from the mean
        active = df["qref"] > 0

        if active.any():
            mean_active_qref = df.loc[active, "qref"].mean()
        else:
            mean_active_qref = 0.0

        df["qref"] = mean_active_qref
        df["a"] = 1.0

        print("[INFO] Applied Level 2 template:")
        print("       all rows -> qref = mean of active values only, a = 1.0")

    elif level == "3":
        # Same as Level 2 for now
        active = df["qref"] > 0

        if active.any():
            mean_active_qref = df.loc[active, "qref"].mean()
        else:
            mean_active_qref = 0.0

        df["qref"] = mean_active_qref
        df["a"] = 1.0

        print("[INFO] Applied Level 3 template:")
        print("       all rows -> qref = mean of active values only, a = 1.0")

    else:
        raise ValueError("level must be '1', '2', or '3'")

    df.to_csv(GRID_FILE, index=False)

    print("[OK] Saved updated org grid_candidates.csv")
    print("[INFO] Preview after edit:")
    print(df.head(10))


if __name__ == "__main__":
    level = input("Enter inversion level to apply to org template (1, 2, or 3): ").strip()
    edit_org_grid(level)