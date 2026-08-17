from pathlib import Path
import subprocess
import sys

SCRIPTS = [
    "models/pcoma_c/evaluate_pcoma_c.py",
    "models/pcoma_r/evaluate_pcoma_r.py",
    "models/pcoma_o/evaluate_pcoma_o_aft.py",
    "models/pcoma_o/evaluate_pcoma_o_landmark.py",
    "models/pcoma_e/evaluate_pcoma_e_binary.py",
    "models/pcoma_e/evaluate_pcoma_e_ordinal.py",
]


def main():
    failures = []
    for script in SCRIPTS:
        print(f"Running {script}")
        result = subprocess.run([sys.executable, script], check=False)
        if result.returncode != 0:
            failures.append(script)
    if failures:
        raise SystemExit(f"Temporal evaluation failed for: {failures}")
    print("Temporal evaluation and PComA-O stress-test summaries completed.")


if __name__ == "__main__":
    main()
