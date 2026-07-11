import os
from pathlib import Path
import numpy as np
import pandas as pd
import re

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
BASE_DIR = Path(r"C:\Python\Personal\kfm-ms\codes")
PEST_DIR = BASE_DIR / "pest"
SIMS_DIR = BASE_DIR / "sims"
RESULTS_DIR = BASE_DIR / "results"

MASTER_TRUTH_FILE = "master_truth.csv"
GRID_CANDIDATES_FILE = "grid_candidates.csv"
PARAM_ORDER_FILE = "param_order.csv"


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def load_required_file(path, desc):
    if not path.exists():
        raise FileNotFoundError(f"Missing {desc}: {path}")
    return path


def parse_parnme(name):
    """
    Parse parameter name of form:
      q_r:12_c:34_sp:5
    Returns (r, c, sp) or (None, None, None) if parsing fails.
    """
    name = str(name).lower()
    m = re.search(r"r:(\d+).*c:(\d+).*sp:(\d+)", name)
    if not m:
        return None, None, None
    r, c, sp = map(int, m.groups())
    return r, c, sp


# -----------------------------------------------------------------------------
# CHECK 1: FILE AVAILABILITY
# -----------------------------------------------------------------------------
def check_workspace_files(root_name):
    real_ws = SIMS_DIR / root_name / "real"

    print("\n" + "=" * 80)
    print(f"[CHECK 1] Workspace file availability for root='{root_name}'")
    print("=" * 80)

    required = [
        (real_ws / MASTER_TRUTH_FILE, "master truth"),
        (real_ws / GRID_CANDIDATES_FILE, "grid candidates"),
        (real_ws / PARAM_ORDER_FILE, "parameter order"),
        (PEST_DIR / "G_real_basis.npy", "G matrix"),
        (PEST_DIR / "b_baseline_flat.npy", "baseline vector"),
    ]

    ok = True
    for path, label in required:
        if path.exists():
            print(f"[OK] {label}: {path}")
        else:
            print(f"[MISSING] {label}: {path}")
            ok = False

    return ok, real_ws


# -----------------------------------------------------------------------------
# CHECK 2: ORDER CONSISTENCY
# -----------------------------------------------------------------------------
def check_parameter_order(real_ws):
    print("\n" + "=" * 80)
    print("[CHECK 2] Parameter ordering consistency")
    print("=" * 80)

    order_file = real_ws / PARAM_ORDER_FILE
    grid_file = real_ws / GRID_CANDIDATES_FILE
    truth_file = real_ws / MASTER_TRUTH_FILE
    g_file = PEST_DIR / "G_real_basis.npy"

    df_order = pd.read_csv(order_file)
    df_grid = pd.read_csv(grid_file)
    df_truth = pd.read_csv(truth_file)
    G = np.load(g_file)

    print(f"[INFO] param_order rows = {len(df_order)}")
    print(f"[INFO] grid_candidates rows = {len(df_grid)}")
    print(f"[INFO] master_truth rows = {len(df_truth)}")
    print(f"[INFO] G shape = {G.shape}")

    if len(df_order) != G.shape[1]:
        print("[FAIL] param_order row count does not match G column count.")
        print(f"       len(param_order) = {len(df_order)}")
        print(f"       G.shape[1]       = {G.shape[1]}")
    else:
        print("[OK] param_order row count matches G column count.")

    if len(df_grid) != len(df_order):
        print("[WARN] grid_candidates row count differs from param_order row count.")
        print("       This is okay only if param_order is a filtered subset.")
    else:
        print("[OK] grid_candidates row count matches param_order row count.")

    if "parnme" not in df_order.columns:
        print("[FAIL] param_order missing 'parnme' column.")
    else:
        dup = df_order["parnme"].duplicated().sum()
        if dup > 0:
            print(f"[FAIL] Duplicate parameter names found in param_order: {dup}")
        else:
            print("[OK] No duplicate parameter names in param_order.")

    print("\n[INFO] First 10 parameter-order rows:")
    print(df_order.head(10).to_string(index=False))

    print("\n[INFO] Last 10 parameter-order rows:")
    print(df_order.tail(10).to_string(index=False))

    return df_order, df_truth, G


# -----------------------------------------------------------------------------
# CHECK 3: COLLOCATED BASIS BEHAVIOR
# -----------------------------------------------------------------------------
def summarize_basis_behavior(df_order, df_truth):
    print("\n" + "=" * 80)
    print("[CHECK 3] Colocated basis summary")
    print("=" * 80)

    true_locs = set(zip(df_truth["r"].astype(int), df_truth["c"].astype(int)))
    basis_locs = set(zip(df_order["r"].astype(int), df_order["c"].astype(int)))
    sp_values = sorted(df_truth["sp"].dropna().unique().tolist())

    print(f"[INFO] Unique real well locations: {len(true_locs)}")
    print(f"[INFO] Unique basis locations: {len(basis_locs)}")
    print(f"[INFO] Stress periods: {sp_values}")

    if basis_locs != true_locs:
        print("[WARN] Basis locations do not exactly match real well locations.")
        missing = true_locs - basis_locs
        extra = basis_locs - true_locs
        if missing:
            print(f"[WARN] Locations in truth but not basis: {sorted(list(missing))[:10]}")
        if extra:
            print(f"[WARN] Locations in basis but not truth: {sorted(list(extra))[:10]}")
    else:
        print("[OK] Basis locations exactly match the real well locations.")

    # In colocated design, all three levels should share the same parameter space.
    print("\n[Level 1]")
    print(f"  Active parameters : {len(df_order)}")
    print(f"  Fixed parameters  : 0")
    print("  Note: Level 1 differences should come from priors, not parameter reduction.")

    print("\n[Level 2]")
    print(f"  Active parameters : {len(df_order)}")
    print(f"  Fixed parameters  : 0")

    print("\n[Level 3]")
    print(f"  Active parameters : {len(df_order)}")
    print(f"  Fixed parameters  : 0")

    return {
        "true_locations": len(true_locs),
        "basis_locations": len(basis_locs),
        "sp_count": len(sp_values),
        "parameter_count": len(df_order),
    }


# -----------------------------------------------------------------------------
# CHECK 4: PARAMETER NAMING PATTERN
# -----------------------------------------------------------------------------
def check_expected_naming(df_order):
    print("\n" + "=" * 80)
    print("[CHECK 4] Parameter naming pattern")
    print("=" * 80)

    if "parnme" not in df_order.columns:
        print("[FAIL] No 'parnme' column available.")
        return False

    sample = df_order["parnme"].astype(str).head(20).tolist()
    print("[INFO] Sample names:")
    for s in sample[:10]:
        print(f"  {s}")

    parsed = df_order["parnme"].astype(str).apply(parse_parnme)
    parsed_df = pd.DataFrame(parsed.tolist(), columns=["r2", "c2", "sp2"])

    bad = parsed_df.isna().any(axis=1).sum()
    if bad > 0:
        print(f"[WARN] {bad} parameter names did not match expected pattern q_r:<n>_c:<n>_sp:<n>")
        print(df_order.loc[parsed_df.isna().any(axis=1), "parnme"].head(10).to_string(index=False))
        return False
    else:
        print("[OK] All parameter names match expected pattern.")
        return True


# -----------------------------------------------------------------------------
# CHECK 5: STRONG ORDERING TEST
# -----------------------------------------------------------------------------
def check_order_alignment(df_order, G):
    """
    Stronger test:
    - Reconstruct (r,c,sp) from parnme
    - Compare against the explicit r,c,sp columns in param_order.csv
    - Confirm ordering is self-consistent
    """
    print("\n" + "=" * 80)
    print("[CHECK 5] Strong parameter order alignment test")
    print("=" * 80)

    if not {"r", "c", "sp"}.issubset(df_order.columns):
        print("[FAIL] param_order.csv must contain columns: r, c, sp")
        return False

    if "parnme" not in df_order.columns:
        print("[FAIL] param_order.csv must contain column: parnme")
        return False

    parsed = df_order["parnme"].astype(str).apply(parse_parnme)
    parsed_df = pd.DataFrame(parsed.tolist(), columns=["r2", "c2", "sp2"])

    comparison = pd.concat([df_order[["r", "c", "sp"]].reset_index(drop=True), parsed_df], axis=1)

    mismatch = (
        (comparison["r"].astype("Int64") != comparison["r2"].astype("Int64")) |
        (comparison["c"].astype("Int64") != comparison["c2"].astype("Int64")) |
        (comparison["sp"].astype("Int64") != comparison["sp2"].astype("Int64"))
    )

    mismatch_count = mismatch.sum()

    if mismatch_count > 0:
        print(f"[FAIL] {mismatch_count} rows have mismatched explicit vs parsed ordering.")
        print("\n[INFO] First mismatches:")
        print(
            comparison.loc[mismatch, ["r", "c", "sp", "r2", "c2", "sp2"]]
            .head(10)
            .to_string(index=False)
        )
        return False

    print("[OK] Explicit r/c/sp columns match parsed parameter names for all rows.")

    unique_order = df_order[["r", "c", "sp"]].drop_duplicates().shape[0]
    if unique_order != len(df_order):
        print("[FAIL] Duplicate (r,c,sp) rows found in param_order.")
        return False
    else:
        print("[OK] Each (r,c,sp) tuple appears exactly once.")

    if not np.isfinite(G).all():
        print("[FAIL] G contains NaN or infinite values.")
        return False
    else:
        print("[OK] G contains only finite values.")

    print("\n[INFO] First 5 aligned rows:")
    print(comparison.head(5).to_string(index=False))

    print("\n[INFO] Last 5 aligned rows:")
    print(comparison.tail(5).to_string(index=False))

    return True


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    root_name = input("Enter root name (e.g. coloc): ").strip()

    ok, real_ws = check_workspace_files(root_name)
    if not ok:
        print("\n[ABORT] Missing required files. Fix these before running PESTPP-IES.")
        return

    df_order, df_truth, G = check_parameter_order(real_ws)
    summarize_basis_behavior(df_order, df_truth)
    check_expected_naming(df_order)
    check_order_alignment(df_order, G)

    print("\n" + "=" * 80)
    print("[DONE] inverse-check completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()