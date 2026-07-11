import traceback
from pathlib import Path
import surrogate

print("[forward_run] this file:", Path(__file__).resolve(), flush=True)
print("[forward_run] surrogate module file:", surrogate.__file__, flush=True)

from surrogate import run_surrogate

if __name__ == "__main__":
    try:
        print("[forward_run] starting run_surrogate()", flush=True)
        run_surrogate()
    except Exception as e:
        print("[forward_run] ERROR:", repr(e), flush=True)
        traceback.print_exc()
        raise