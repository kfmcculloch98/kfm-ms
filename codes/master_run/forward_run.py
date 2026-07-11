import traceback
from surrogate import run_surrogate

if __name__ == "__main__":
    try:
        run_surrogate()
    except Exception as e:
        print("[forward_run] ERROR:", repr(e), flush=True)
        traceback.print_exc()
        raise