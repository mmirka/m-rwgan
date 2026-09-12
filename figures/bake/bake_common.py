"""Shared helpers for the ``figures/bake/`` scripts.

Each bake script reads one oversized, git-ignored input from ``figures/data/``
and writes one small, git-tracked artifact to ``figures/data/derived/``. The
figure scripts then prefer the derived artifact and fall back to the raw input,
so a plain clone reproduces every figure with no download while re-baking from
the raw archive stays possible.

Every artifact records the size and SHA-256 of the input it was derived from,
so the reduction is auditable rather than magic: ``describe(path)`` prints what
an artifact came from, and a re-bake of a different archive is visibly a
different artifact.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = FIGURES_DIR / "data"
DERIVED_DIR = DATA_DIR / "derived"

# Bumped when the layout of an artifact's arrays changes in a way the figure
# scripts' readers would not survive.
FORMAT_VERSION = 1

_CHUNK = 1 << 22  # 4 MiB — a 516 MB input hashes in a couple of seconds


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(sources) -> dict:
    """Per-source name / byte size / SHA-256, as arrays an ``.npz`` can hold.

    ``sources`` is one path or several. Names are recorded relative to
    ``figures/data/`` when they live there, so an artifact does not carry the
    absolute path of the machine that baked it.
    """
    if isinstance(sources, (str, Path)):
        sources = [sources]
    paths = [Path(p) for p in sources]
    names, sizes, digests = [], [], []
    for p in paths:
        try:
            name = str(p.resolve().relative_to(DATA_DIR.resolve()))
        except ValueError:
            name = p.name
        names.append(name)
        sizes.append(p.stat().st_size)
        digests.append(sha256_file(p))
    return {
        "_format_version": np.int64(FORMAT_VERSION),
        "_source_name": np.array(names),
        "_source_bytes": np.array(sizes, dtype=np.int64),
        "_source_sha256": np.array(digests),
    }


def write_artifact(out_rel, sources, arrays: dict, *, baked_by: str) -> Path:
    """Write ``derived/<out_rel>`` holding ``arrays`` plus source provenance.

    ``out_rel`` mirrors the raw input's path under ``figures/data/``, not just
    its basename: two runs in this archive share a filename across the
    ``uniform_3c`` and ``hotspot30_3c`` subdirectories, so a flat derived
    directory would have one silently overwrite the other.
    """
    out_path = DERIVED_DIR / out_rel
    # np.savez_compressed appends .npz when the name lacks it, which would make
    # the returned path a lie; every caller already names the extension.
    assert out_path.suffix == ".npz", out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload.update(provenance(sources))
    payload["_baked_by"] = np.array(baked_by)
    np.savez_compressed(out_path, **payload)
    return out_path


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def report(out_path: Path, sources) -> None:
    if isinstance(sources, (str, Path)):
        sources = [sources]
    raw = sum(Path(p).stat().st_size for p in sources)
    derived = out_path.stat().st_size
    print(
        f"wrote {out_path.relative_to(FIGURES_DIR)}  "
        f"{_human(raw)} -> {_human(derived)}  ({raw / max(derived, 1):.0f}x smaller)"
    )


def describe(path: Path) -> None:
    """Print what a derived artifact holds and what it was baked from."""
    with np.load(path, allow_pickle=False) as z:
        try:
            print(f"{path.resolve().relative_to(DERIVED_DIR.resolve())}")
        except ValueError:
            print(f"{path.name}")
        print(f"  baked by       {z['_baked_by']}  (format v{int(z['_format_version'])})")
        for name, size, digest in zip(z["_source_name"], z["_source_bytes"], z["_source_sha256"]):
            print(f"  from           {name}  ({int(size):,} bytes)")
            print(f"                 sha256 {digest}")
        for key in z.files:
            if key.startswith("_"):
                continue
            arr = z[key]
            print(f"  {key:<14} {arr.dtype} {arr.shape}")


def require(path: Path, what: str) -> Path:
    """Fail with a message naming the download, rather than a bare IOError."""
    if not path.exists():
        print(
            f"error: {what} not found at {path}\n"
            "This input is git-ignored — see figures/data/README.md for the download.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return path


if __name__ == "__main__":
    targets = [Path(a) for a in sys.argv[1:]] or sorted(DERIVED_DIR.rglob("*.npz"))
    if not targets:
        raise SystemExit("no derived artifacts found; run bake/bake_all.sh first")
    for t in targets:
        describe(t)
        print()
