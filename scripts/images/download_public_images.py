#!/usr/bin/env python3
"""Download and unpack PAL-Bench public image shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST_URL = (
    "https://sprproxy-1258344707.cos.ap-shanghai.myqcloud.com/"
    "pal-bench-images/public/manifest.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download PAL-Bench public image shards.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_URL, help="Manifest URL or local path.")
    parser.add_argument("--out-dir", default="data/images", help="Extraction root.")
    parser.add_argument("--archive-dir", default=None, help="Where to store downloaded shard archives.")
    parser.add_argument("--users", default=None, help="Comma-separated users to download. Defaults to user_0000.")
    parser.add_argument("--all", action="store_true", help="Download all users. This is a large download.")
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Load the manifest and print the selected shards.")
    parser.add_argument(
        "--check-remote",
        action="store_true",
        help="Check selected remote archive sizes with HTTP HEAD requests.",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")
    parser.add_argument("--download-retries", type=int, default=5, help="HTTP attempts for each download.")
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=10.0,
        help="Initial delay in seconds before retrying a failed HTTP request.",
    )
    args = parser.parse_args()

    if args.download_retries < 1:
        raise ValueError("--download-retries must be at least 1")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.retry_base_delay < 0:
        raise ValueError("--retry-base-delay must be non-negative")

    manifest, local_manifest_dir = _load_manifest(args.manifest, retries=args.download_retries, timeout=args.timeout)
    _validate_manifest(manifest)
    shards = list(manifest.get("shards") or [])
    selected = _select_shards(shards, args.users, args.all)
    out_dir = Path(args.out_dir).expanduser().resolve()
    archive_dir = Path(args.archive_dir).expanduser().resolve() if args.archive_dir else out_dir / "_archives"

    print(f"manifest: {args.manifest}")
    print(f"release: {manifest.get('release_name')} ({manifest.get('n_users')} users, {manifest.get('n_images')} images)")
    print(f"selected users: {', '.join(row['user_id'] for row in selected)}")
    print(f"selected archive bytes: {sum(int(row['archive_bytes']) for row in selected)}")
    if not args.all and args.users is None:
        print("Only user_0000 is selected by default. Use --all for the full image archive.")
    if args.check_remote:
        _check_remote_archives(
            manifest=manifest,
            rows=selected,
            local_manifest_dir=local_manifest_dir,
            retries=args.download_retries,
            timeout=args.timeout,
        )
    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    for row in selected:
        archive_path = archive_dir / Path(row["file"]).name
        user_dir = out_dir / "public_images" / row["user_id"]
        if args.skip_existing and user_dir.exists():
            found = len(list(user_dir.glob("photo_*.png")))
            if found == int(row["n_images"]):
                print(f"[skip] {row['user_id']} already extracted ({found} images)")
                continue
        local_source = (local_manifest_dir / str(row["file"])) if local_manifest_dir else None
        expected_sha = str(row.get("sha256") or "")
        if local_source is not None and local_source.exists():
            if _archive_matches(archive_path, expected_sha):
                print(f"[archive] using verified {archive_path}")
            elif archive_path.resolve() != local_source.resolve():
                print(f"[copy] {local_source}")
                shutil.copyfile(local_source, archive_path)
            else:
                print(f"[archive] using local {archive_path}")
        else:
            url = row.get("url") or _join_url(str(manifest.get("base_url") or ""), str(row["file"]))
            if _archive_matches(archive_path, expected_sha):
                print(f"[archive] using verified {archive_path}")
            else:
                _download(
                    url,
                    archive_path,
                    retries=args.download_retries,
                    timeout=args.timeout,
                    base_delay=args.retry_base_delay,
                )
        actual_sha = _sha256_file(archive_path)
        if expected_sha and actual_sha != expected_sha:
            raise RuntimeError(f"sha256 mismatch for {archive_path.name}: {actual_sha} != {expected_sha}")
        _extract_archive(archive_path, out_dir)
        found = len(list(user_dir.glob("photo_*.png")))
        if found != int(row["n_images"]):
            raise RuntimeError(f"{row['user_id']}: expected {row['n_images']} images, found {found}")
        print(f"[ok] {row['user_id']} -> {user_dir}")
        if not args.keep_archives:
            archive_path.unlink()

    return 0


def _load_manifest(value: str, *, retries: int, timeout: float) -> tuple[dict[str, Any], Path | None]:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme in {"http", "https", "file"}:
        with _urlopen_with_retries(value, retries=retries, timeout=timeout, label=value) as response:
            manifest = json.loads(response.read().decode("utf-8"))
        local_dir = Path(parsed.path).expanduser().resolve().parent if parsed.scheme == "file" else None
        return manifest, local_dir
    path = Path(value).expanduser()
    with path.open(encoding="utf-8") as handle:
        return json.load(handle), path.resolve().parent


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("release_name") != "pal-bench-images":
        raise RuntimeError("unexpected image release name in manifest")
    if manifest.get("layout") != "public_images/{user_id}/{photo_id}.png":
        raise RuntimeError("unexpected image layout in manifest")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise RuntimeError("manifest has no shards")
    required = {"user_id", "file", "n_images", "archive_bytes", "sha256"}
    for index, row in enumerate(shards):
        missing = sorted(required - set(row))
        if missing:
            raise RuntimeError(f"manifest shard {index} is missing: {', '.join(missing)}")


def _select_shards(shards: list[dict[str, Any]], users: str | None, all_users: bool) -> list[dict[str, Any]]:
    if all_users:
        return shards
    wanted = {"user_0000"} if users is None else {item.strip() for item in users.split(",") if item.strip()}
    selected = [row for row in shards if row.get("user_id") in wanted]
    missing = sorted(wanted - {str(row.get("user_id")) for row in selected})
    if missing:
        raise RuntimeError(f"users not found in manifest: {', '.join(missing)}")
    return selected


def _check_remote_archives(
    *,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    local_manifest_dir: Path | None,
    retries: int,
    timeout: float,
) -> None:
    for row in rows:
        local_source = (local_manifest_dir / str(row["file"])) if local_manifest_dir else None
        expected_bytes = int(row["archive_bytes"])
        if local_source is not None and local_source.exists():
            actual_bytes = local_source.stat().st_size
            if actual_bytes != expected_bytes:
                raise RuntimeError(f"{row['file']}: expected {expected_bytes} bytes, found {actual_bytes}")
            print(f"[check] {row['file']} local size ok ({actual_bytes} bytes)")
            continue
        url = row.get("url") or _join_url(str(manifest.get("base_url") or ""), str(row["file"]))
        request = urllib.request.Request(url, method="HEAD")
        with _urlopen_with_retries(request, retries=retries, timeout=timeout, label=url) as response:
            actual_header = response.headers.get("Content-Length")
        actual_bytes = int(actual_header) if actual_header is not None else -1
        if actual_bytes != expected_bytes:
            raise RuntimeError(f"{row['file']}: expected {expected_bytes} bytes, remote has {actual_bytes}")
        print(f"[check] {row['file']} remote size ok ({actual_bytes} bytes)")


def _download(url: str, path: Path, *, retries: int, timeout: float, base_delay: float) -> None:
    print(f"[download] {url}")
    part_path = path.with_name(f"{path.name}.part")
    if part_path.exists():
        part_path.unlink()
    for attempt in range(1, retries + 1):
        try:
            with _urlopen_with_retries(url, retries=1, timeout=timeout, label=url) as response, part_path.open("wb") as handle:
                _copy_with_progress(response, handle)
            part_path.replace(path)
            return
        except Exception:
            if part_path.exists():
                part_path.unlink()
            if attempt >= retries:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            print(f"[retry] download failed, retrying in {delay:.1f}s ({attempt + 1}/{retries})", file=sys.stderr)
            time.sleep(delay)


def _copy_with_progress(response: Any, handle: Any) -> None:
    total_header = response.headers.get("Content-Length")
    total = int(total_header) if total_header and total_header.isdigit() else None
    copied = 0
    next_report = 256 * 1024 * 1024
    while True:
        chunk = response.read(1024 * 1024)
        if not chunk:
            break
        handle.write(chunk)
        copied += len(chunk)
        if total and copied >= next_report:
            pct = copied * 100 / total
            print(f"  downloaded {copied}/{total} bytes ({pct:.1f}%)")
            next_report += 256 * 1024 * 1024


def _urlopen_with_retries(request_or_url: Any, *, retries: int, timeout: float, label: str) -> Any:
    for attempt in range(1, retries + 1):
        try:
            return urllib.request.urlopen(request_or_url, timeout=timeout)
        except Exception:
            if attempt >= retries:
                raise
            delay = min(60.0, 2 ** (attempt - 1))
            print(f"[retry] HTTP request failed for {label}; retrying in {delay:.1f}s ({attempt + 1}/{retries})", file=sys.stderr)
            time.sleep(delay)


def _extract_archive(archive_path: Path, out_dir: Path) -> None:
    name = archive_path.name
    if name.endswith(".tar.zst"):
        if shutil.which("zstd") is None:
            raise RuntimeError("zstd is required to extract .tar.zst archives")
        cmd = ["sh", "-c", f"zstd -dc \"$1\" | tar -xf - -C \"$2\"", "sh", str(archive_path), str(out_dir)]
    elif name.endswith(".tar.gz"):
        cmd = ["tar", "-xzf", str(archive_path), "-C", str(out_dir)]
    elif name.endswith(".tar"):
        cmd = ["tar", "-xf", str(archive_path), "-C", str(out_dir)]
    else:
        raise RuntimeError(f"unknown archive suffix: {archive_path}")
    subprocess.run(cmd, check=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_matches(path: Path, expected_sha: str) -> bool:
    return path.exists() and bool(expected_sha) and _sha256_file(path) == expected_sha


def _join_url(base: str, file_name: str) -> str:
    return f"{base.rstrip('/')}/{file_name.lstrip('/')}"


if __name__ == "__main__":
    raise SystemExit(main())
