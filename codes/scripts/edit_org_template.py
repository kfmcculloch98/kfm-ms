import shutil
from pathlib import Path

import pandas as pd


BASE_DIR = Path(r"C:\Python\Personal\kfm-ms\codes")
PEST_DIR = BASE_DIR / "pest_level_2"
ORG_DIR = PEST_DIR / "org"

GRID_FILE = ORG_DIR / "grid_candidates.csv"


def backup_file(path: Path):
    if path.exists():
        backup_path = path.with_suffix(path.suffix + ".bak")
        shutil.copy(path, backup_path)
        print(f"[OK] Backup saved: {backup_path}")
    else:
        raise FileNotFoundError(f"File not found: {path}")


def preview_file(path: Path, label: str, n: int = 10):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    print(f"[INFO] Preview of {label}: {path}")
    print(f"[INFO] Columns: {list(df.columns)}")
    print(df.head(n))
    return df


def edit_org_grid(level: str):
    level = str(level).strip()

    if not GRID_FILE.exists():
        raise FileNotFoundError(f"Missing org file: {GRID_FILE}")

    # Preview before edit
    df = preview_file(GRID_FILE, "org grid before edit")

    # Backup
    backup_file(GRID_FILE)

    required_cols = {"r", "c", "sp", "qref", "a"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"grid_candidates.csv must contain columns: {sorted(required_cols)}")

    active = df["qref"] > 0

    if level == "1":
        # Mean including zeros
        mean_qref = df["qref"].mean()

        # Active rows get the mean, inactive rows get 0
        df["qref"] = 0.0
        df.loc[active, "qref"] = mean_qref

        df["a"] = 0.0
        df.loc[active, "a"] = 1.0

        print("[INFO] Applied Level 1 template:")
        print("       active rows -> qref = mean including zeros, a = 1.0")
        print("       inactive rows -> qref = 0.0, a = 0.0")

    elif level == "2":
        # Mean excluding zeros
        if active.any():
            mean_active_qref = df.loc[active, "qref"].mean()
        else:
            mean_active_qref = 0.0

        df["qref"] = mean_active_qref
        df["a"] = 1.0

        print("[INFO] Applied Level 2 template:")
        print("       all rows -> qref = mean of active values only")
        print("       all rows -> a = 1.0")

    elif level == "3":
        # Same as Level 2 for now
        if active.any():
            mean_active_qref = df.loc[active, "qref"].mean()
        else:
            mean_active_qref = 0.0

        df["qref"] = mean_active_qref
        df["a"] = 1.0

        print("[INFO] Applied Level 3 template:")
        print("       all rows -> qref = mean of active values only")
        print("       all rows -> a = 1.0")

    else:
        raise ValueError("level must be '1', '2', or '3'")

    df.to_csv(GRID_FILE, index=False)

    print("[OK] Saved updated org grid_candidates.csv")
    print("[INFO] Preview after edit:")
    print(df.head(10))


if __name__ == "__main__":
    level = input("Enter inversion level to apply to org template (1, 2, or 3): ").strip()
    edit_org_grid(level)