from __future__ import annotations

from pathlib import Path
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_PREFIXES = {
    "data/private/pcoma_registry.csv",
    "data/cleaned_pcoma.csv",
    "data/development.csv",
    "data/temporal.csv",
    "data/features_development.csv",
    "data/features_temporal.csv",
    "results/generated/",
    "calibration/generated/",
    "validation/generated/",
    "uncertainty/generated/",
    "explainability/generated/",
    "figures/generated/",
}


def excluded(relative: str) -> bool:
    parts = Path(relative).parts
    if "__pycache__" in parts or ".pytest_cache" in parts:
        return True
    if relative.endswith((".pyc", ".pyo")):
        return True
    return any(relative == prefix or relative.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def main():
    manifest = []
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.sha256":
            continue
        relative = path.relative_to(ROOT).as_posix()
        if excluded(relative):
            continue
        files.append(path)
        manifest.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}")
    (ROOT / "MANIFEST.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    files.append(ROOT / "MANIFEST.sha256")

    destination = ROOT.parent / f"PComA-CORE_v{(ROOT / 'VERSION').read_text().strip()}_public.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            archive.write(path, arcname=(Path("PComA-CORE") / path.relative_to(ROOT)).as_posix())
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="utf-8"
    )
    print(destination)
    print(digest)


if __name__ == "__main__":
    main()
