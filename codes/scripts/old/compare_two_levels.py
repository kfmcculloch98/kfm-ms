import sys
from pathlib import Path

import pandas as pd
import pyemu


def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def load_pst(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing PST file: {path}")
    return pyemu.Pst(str(path))


def summarize_grid(df, label):
    print(f"\n=== {label} ===")
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    cols = [c for c in ["r", "c", "sp", "qref", "a"] if c in df.columns]
    if cols:
        print(df[cols].head(10))


def compare_dataframes(df1, df2, merge_cols, compare_cols, label1, label2):
    merged = df1.merge(df2, on=merge_cols, how="inner", suffixes=(f"_{label1}", f"_{label2}"))
    print(f"\n=== {label1} vs {label2}: {len(merged)} merged rows ===")

    for col in compare_cols:
        c1 = f"{col}_{label1}"
        c2 = f"{col}_{label2}"
        if c1 in merged.columns and c2 in merged.columns:
            neq = merged[c1].ne(merged[c2]).sum()
            print(f"{col}: differing rows = {neq}")

    return merged


def compare_pst(pst1, pst2, label1, label2):
    par1 = pst1.parameter_data.copy()
    par2 = pst2.parameter_data.copy()

    print(f"\n=== PST COMPARISON: {label1} vs {label2} ===")
    print(f"npar_adj {label1}: {(par1['partrans'].astype(str).str.lower() != 'fixed').sum()}")
    print(f"npar_adj {label2}: {(par2['partrans'].astype(str).str.lower() != 'fixed').sum()}")

    # Compare on parameter names/index
    idx1 = par1.index.astype(str)
    idx2 = par2.index.astype(str)

    common = idx1.intersection(idx2)
    print(f"Common parameter names: {len(common)}")

    if len(common) == 0:
        print("No common parameter names to compare.")
        return

    cols = [c for c in ["parval1", "parlbnd", "parubnd", "partrans"] if c in par1.columns and c in par2.columns]
    for col in cols:
        neq = (par1.loc[common, col].astype(str).values != par2.loc[common, col].astype(str).values).sum()
        print(f"{col}: differing common rows = {neq}")

    print("\nSample parameter rows from level 1:")
    print(par1.loc[common, ["parval1", "parlbnd", "parubnd", "partrans"]].head(10))

    print(f"\nSample parameter rows from level 2:")
    print(par2.loc[common, ["parval1", "parlbnd", "parubnd", "partrans"]].head(10))


def compare_two_levels(level1_dir: Path, level2_dir: Path):
    if not level1_dir.exists():
        raise FileNotFoundError(f"Missing level 1 directory: {level1_dir}")
    if not level2_dir.exists():
        raise FileNotFoundError(f"Missing level 2 directory: {level2_dir}")

    print(f"[INFO] Level 1 dir: {level1_dir}")
    print(f"[INFO] Level 2 dir: {level2_dir}")

    # ---------------------------------------------------------------------
    # Load core files
    # ---------------------------------------------------------------------
    grid1 = load_csv(level1_dir / "grid_candidates.csv")
    grid2 = load_csv(level2_dir / "grid_candidates.csv")

    org1 = load_csv(level1_dir / "org" / "grid_candidates.csv")
    org2 = load_csv(level2_dir / "org" / "grid_candidates.csv")

    pst1 = load_pst(level1_dir / "inversion_level_1.pst")
    pst2 = load_pst(level2_dir / "inversion_level_2.pst")

    # ---------------------------------------------------------------------
    # Summaries
    # ---------------------------------------------------------------------
    summarize_grid(grid1, "Level 1 grid_candidates.csv")
    summarize_grid(grid2, "Level 2 grid_candidates.csv")
    summarize_grid(org1, "Level 1 org/grid_candidates.csv")
    summarize_grid(org2, "Level 2 org/grid_candidates.csv")

    # ---------------------------------------------------------------------
    # Grid comparisons
    # ---------------------------------------------------------------------
    print("\n==============================")
    print("GRID_CANDIDATES.csv COMPARISON")
    print("==============================")

    merged_grid = compare_dataframes(
        grid1,
        grid2,
        merge_cols=["r", "c", "sp"],
        compare_cols=["qref", "a"],
        label1="l1",
        label2="l2"
    )

    print("\n==============================")
    print("ORG GRID_CANDIDATES.csv COMPARISON")
    print("==============================")

    merged_org = compare_dataframes(
        org1,
        org2,
        merge_cols=["r", "c", "sp"],
        compare_cols=["qref", "a"],
        label1="l1",
        label2="l2"
    )

    # ---------------------------------------------------------------------
    # PST comparison
    # ---------------------------------------------------------------------
    compare_pst(pst1, pst2, "l1", "l2")

    # ---------------------------------------------------------------------
    # Effective pumping summaries
    # ---------------------------------------------------------------------
    if {"qref_l1", "a_l1", "qref_l2", "a_l2"}.issubset(merged_org.columns):
        merged_org["eff_l1"] = merged_org["qref_l1"] * merged_org["a_l1"]
        merged_org["eff_l2"] = merged_org["qref_l2"] * merged_org["a_l2"]

        print("\n==============================")
        print("EFFECTIVE PUMPING COMPARISON")
        print("==============================")
        print("Level 1 effective pumping summary:")
        print(merged_org["eff_l1"].describe())

        print("\nLevel 2 effective pumping summary:")
        print(merged_org["eff_l2"].describe())

        print("\nRows where effective pumping differs:")
        print((merged_org["eff_l1"] != merged_org["eff_l2"]).sum())

        diff_rows = merged_org[merged_org["eff_l1"] != merged_org["eff_l2"]]
        print("\nSample differing rows:")
        print(diff_rows[["r", "c", "sp", "qref_l1", "a_l1", "eff_l1", "qref_l2", "a_l2", "eff_l2"]].head(20))

    # ---------------------------------------------------------------------
    # Optional: forward_run.py comparison
    # ---------------------------------------------------------------------
    fwd1 = level1_dir / "forward_run.py"
    fwd2 = level2_dir / "forward_run.py"

    if fwd1.exists() and fwd2.exists():
        txt1 = fwd1.read_text(encoding="utf-8", errors="ignore")
        txt2 = fwd2.read_text(encoding="utf-8", errors="ignore")
        print("\n==============================")
        print("FORWARD_RUN.PY COMPARISON")
        print("==============================")
        print(f"Identical: {txt1 == txt2}")
        if txt1 != txt2:
            print("First 500 chars of Level 1 forward_run.py:")
            print(txt1[:500])
            print("\nFirst 500 chars of Level 2 forward_run.py:")
            print(txt2[:500])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  uv run compare_two_levels.py <level1_dir> <level2_dir>")
        print("\nExample:")
        print(r"  uv run compare_two_levels.py C:\path\to\level1_pest C:\path\to\level2_pest")
        sys.exit(1)

    level1_dir = Path(sys.argv[1])
    level2_dir = Path(sys.argv[2])
    compare_two_levels(level1_dir, level2_dir)