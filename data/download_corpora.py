#!/usr/bin/env python3
"""Download and extract audio corpora for PS26052 ANC training.

Usage::

    python data/download_corpora.py --all
    python data/download_corpora.py --librispeech --musan

Downloads are idempotent — existing files are skipped.  Each corpus is
placed under ``data/corpora/{dataset_name}/``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parent / "corpora"


# ──────────────────────────────────────────────────────────────────────
# Corpus definitions
# ──────────────────────────────────────────────────────────────────────

LIBRISPEECH_URLS = {
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
}

VCTK_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"

MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Console progress bar for urllib downloads."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100.0)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
    else:
        mb_down = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  Downloaded {mb_down:.1f} MB")
    sys.stdout.flush()


def _download_file(url: str, dest: Path) -> Path:
    """Download a file if it doesn't already exist."""
    if dest.exists():
        print(f"  ✓ Already downloaded: {dest.name}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=_progress_hook)
        print()  # newline after progress bar
    except Exception as exc:
        print(f"\n  ✗ Download failed: {exc}")
        if dest.exists():
            dest.unlink()
        raise
    return dest


def _extract_tar_gz(archive: Path, dest_dir: Path) -> None:
    """Extract a .tar.gz archive."""
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")
    print(f"  Extracting: {archive.name} → {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=dest_dir)


def _extract_zip(archive: Path, dest_dir: Path) -> None:
    """Extract a .zip archive."""
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")
    print(f"  Extracting: {archive.name} → {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(path=dest_dir)


# ──────────────────────────────────────────────────────────────────────
# Per-corpus download functions
# ──────────────────────────────────────────────────────────────────────

def download_librispeech(
    subsets: list[str] | None = None,
    root: Path | None = None,
) -> Path:
    """Download LibriSpeech clean speech corpus.

    Parameters
    ----------
    subsets : list of str, optional
        Which subsets to download. Default: ``["train-clean-100"]``.
    root : Path, optional
        Root directory. Default: ``data/corpora/librispeech``.
    """
    root = root or DATA_ROOT / "librispeech"
    subsets = subsets or ["train-clean-100"]

    print(f"\n{'='*60}")
    print(f"LibriSpeech — subsets: {subsets}")
    print(f"{'='*60}")

    marker = root / ".download_complete"
    cache_dir = root / "_cache"

    for subset in subsets:
        url = LIBRISPEECH_URLS.get(subset)
        if url is None:
            print(f"  ⚠ Unknown subset '{subset}', skipping.")
            continue

        # Check if already extracted
        expected_dir = root / "LibriSpeech" / subset
        if expected_dir.exists() and any(expected_dir.rglob("*.flac")):
            print(f"  ✓ Already extracted: {subset}")
            continue

        archive = cache_dir / f"{subset}.tar.gz"
        _download_file(url, archive)
        _extract_tar_gz(archive, root)

    marker.write_text("done\n")
    print(f"  ✓ LibriSpeech ready at: {root}")
    return root


def download_vctk(root: Path | None = None) -> Path:
    """Download VCTK multi-speaker speech corpus."""
    root = root or DATA_ROOT / "vctk"

    print(f"\n{'='*60}")
    print("VCTK Corpus")
    print(f"{'='*60}")

    # Check if already extracted
    wav_dir = root / "VCTK-Corpus-0.92" / "wav48_silence_trimmed"
    if wav_dir.exists() and any(wav_dir.rglob("*.flac")):
        print(f"  ✓ Already extracted.")
        return root

    cache_dir = root / "_cache"
    archive = cache_dir / "VCTK-Corpus-0.92.zip"
    _download_file(VCTK_URL, archive)
    _extract_zip(archive, root)

    print(f"  ✓ VCTK ready at: {root}")
    return root


def download_musan(root: Path | None = None) -> Path:
    """Download MUSAN noise/music/speech corpus."""
    root = root or DATA_ROOT / "musan"

    print(f"\n{'='*60}")
    print("MUSAN Corpus")
    print(f"{'='*60}")

    noise_dir = root / "musan" / "noise"
    if noise_dir.exists() and any(noise_dir.rglob("*.wav")):
        print(f"  ✓ Already extracted.")
        return root

    cache_dir = root / "_cache"
    archive = cache_dir / "musan.tar.gz"
    _download_file(MUSAN_URL, archive)
    _extract_tar_gz(archive, root)

    print(f"  ✓ MUSAN ready at: {root}")
    return root


def download_esc50(root: Path | None = None) -> Path:
    """Download ESC-50 environmental sound classification dataset."""
    root = root or DATA_ROOT / "esc50"

    print(f"\n{'='*60}")
    print("ESC-50 Corpus")
    print(f"{'='*60}")

    audio_dir = root / "ESC-50-master" / "audio"
    if audio_dir.exists() and any(audio_dir.rglob("*.wav")):
        print(f"  ✓ Already extracted.")
        return root

    cache_dir = root / "_cache"
    archive = cache_dir / "ESC-50-master.zip"
    _download_file(ESC50_URL, archive)
    _extract_zip(archive, root)

    print(f"  ✓ ESC-50 ready at: {root}")
    return root


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download audio corpora for PS26052 ANC training."
    )
    parser.add_argument("--all", action="store_true", help="Download all corpora")
    parser.add_argument("--librispeech", action="store_true", help="Download LibriSpeech")
    parser.add_argument(
        "--librispeech-subsets", nargs="+",
        default=["train-clean-100"],
        help="LibriSpeech subsets (default: train-clean-100)",
    )
    parser.add_argument("--vctk", action="store_true", help="Download VCTK")
    parser.add_argument("--musan", action="store_true", help="Download MUSAN")
    parser.add_argument("--esc50", action="store_true", help="Download ESC-50")
    parser.add_argument(
        "--root", type=Path, default=None,
        help=f"Root directory for corpora (default: {DATA_ROOT})",
    )

    args = parser.parse_args()

    if not any([args.all, args.librispeech, args.vctk, args.musan, args.esc50]):
        parser.print_help()
        print("\n⚠ No corpus selected. Use --all or specify individual corpora.")
        sys.exit(1)

    root = args.root or DATA_ROOT

    if args.all or args.librispeech:
        download_librispeech(subsets=args.librispeech_subsets, root=root / "librispeech")
    if args.all or args.vctk:
        download_vctk(root=root / "vctk")
    if args.all or args.musan:
        download_musan(root=root / "musan")
    if args.all or args.esc50:
        download_esc50(root=root / "esc50")

    print(f"\n{'='*60}")
    print("All requested downloads complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
