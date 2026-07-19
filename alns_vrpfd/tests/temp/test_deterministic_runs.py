import subprocess
from pathlib import Path


def run_alns_once(seed: int = 42, iterations: int = 50):
    script = Path(__file__).resolve().parents[3] / "run_alns.py"
    instance = Path(__file__).resolve(
    ).parents[3] / "data/Instance10/R_30_10_1.txt"
    cmd = ["python3", str(script), str(instance), "--iterations", str(iterations),
           "--seed", str(seed), "--deterministic", "--enable-composite"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True)
    return proc.stdout.decode()


def test_deterministic_runs_match():
    out1 = run_alns_once(seed=123, iterations=100)
    out2 = run_alns_once(seed=123, iterations=100)
    # Elapsed time can vary between processes; remove that line and compare remaining output.

    def strip_elapsed(out: str) -> str:
        return "\n".join([l for l in out.splitlines() if not l.startswith("Elapsed time:")])
    assert strip_elapsed(out1) == strip_elapsed(out2)
