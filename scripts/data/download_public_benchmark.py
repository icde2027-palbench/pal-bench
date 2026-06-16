#!/usr/bin/env python3
"""Download and unpack the PAL-Bench full JSON benchmark artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST_URL = (
    "https://sprproxy-1258344707.cos.ap-shanghai.myqcloud.com/"
    "pal-bench-json/public/manifest.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download PAL-Bench full JSON benchmark.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_URL, help="Manifest URL or local path.")
    parser.add_argument("--out-dir", default="data/full", help="Extraction root.")
    parser.add_argument("--archive-dir", default=None, help="Where to store the downloaded archive.")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Load the manifest and print artifact metadata.")
    parser.add_argument("--check-remote", action="store_true", help="Check remote archive size with HTTP HEAD.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")
    parser.add_argument("--download-retries", type=int, default=5, help="HTTP attempts for the archive download.")
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=5.0,
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
    archive = (manifest.get("archives") or [])[0]
    out_dir = Path(args.out_dir).expanduser().resolve()
    archive_dir = Path(args.archive_dir).expanduser().resolve() if args.archive_dir else out_dir / "_archives"
    archive_path = archive_dir / Path(str(archive["file"])).name

    print(f"manifest: {args.manifest}")
    print(
        "release: "
        f"{manifest.get('release_name')} "
        f"({manifest.get('n_users')} users, "
        f"{manifest.get('n_public_photo_records')} photo records, "
        f"{manifest.get('n_evaluation_targets')} targets)"
    )
    print(f"archive bytes: {archive.get('bytes')}")
    print(f"extract to: {out_dir}")

    if args.check_remote:
        _check_archive(
            manifest=manifest,
            archive=archive,
            local_manifest_dir=local_manifest_dir,
            retries=args.download_retries,
            timeout=args.timeout,
        )
    if args.dry_run:
        return 0

    users_root = out_dir / "users"
    if args.skip_existing and _looks_extracted(users_root, manifest):
        print(f"[skip] benchmark already extracted at {users_root}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    local_source = (local_manifest_dir / str(archive["file"])) if local_manifest_dir else None
    expected_sha = str(archive.get("sha256") or "")
    if local_source is not None and local_source.exists():
        if _archive_matches(archive_path, expected_sha):
            print(f"[archive] using verified {archive_path}")
        elif archive_path.resolve() != local_source.resolve():
            print(f"[copy] {local_source}")
            shutil.copyfile(local_source, archive_path)
        else:
            print(f"[archive] using local {archive_path}")
    elif _archive_matches(archive_path, expected_sha):
        print(f"[archive] using verified {archive_path}")
    else:
        url = archive.get("url") or _join_url(str(manifest.get("base_url") or ""), str(archive["file"]))
        _download(
            str(url),
            archive_path,
            retries=args.download_retries,
            timeout=args.timeout,
            base_delay=args.retry_base_delay,
        )

    actual_sha = _sha256_file(archive_path)
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError(f"sha256 mismatch for {archive_path.name}: {actual_sha} != {expected_sha}")

    _extract_archive(archive_path, out_dir)
    _verify_extracted(users_root, manifest)
    print(f"[ok] full benchmark -> {users_root}")
    if not args.keep_archive:
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
    if manifest.get("release_name") != "pal-bench-json":
        raise RuntimeError("unexpected JSON release name in manifest")
    if manifest.get("layout") != "users/{user_id}/{user_id}_{agent_album,eval_gt,export_audit}.json":
        raise RuntimeError("unexpected JSON benchmark layout in manifest")
    if manifest.get("users_root_after_extract") != "data/full/users":
        raise RuntimeError("unexpected extraction root in manifest")
    archives = manifest.get("archives")
    if not isinstance(archives, list) or len(archives) != 1:
        raise RuntimeError("manifest must contain exactly one archive")
    required_archive = {"file", "bytes", "sha256"}
    missing = sorted(required_archive - set(archives[0]))
    if missing:
        raise RuntimeError(f"manifest archive is missing: {', '.join(missing)}")
    users = manifest.get("users")
    if not isinstance(users, list) or not users:
        raise RuntimeError("manifest has no users")


def _check_archive(
    *,
    manifest: dict[str, Any],
    archive: dict[str, Any],
    local_manifest_dir: Path | None,
    retries: int,
    timeout: float,
) -> None:
    expected_bytes = int(archive["bytes"])
    local_source = (local_manifest_dir / str(archive["file"])) if local_manifest_dir else None
    if local_source is not None and local_source.exists():
        actual_bytes = local_source.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(f"{archive['file']}: expected {expected_bytes} bytes, found {actual_bytes}")
        print(f"[check] {archive['file']} local size ok ({actual_bytes} bytes)")
        return
    url = archive.get("url") or _join_url(str(manifest.get("base_url") or ""), str(archive["file"]))
    request = urllib.request.Request(str(url), method="HEAD")
    with _urlopen_with_retries(request, retries=retries, timeout=timeout, label=str(url)) as response:
        actual_header = response.headers.get("Content-Length")
    actual_bytes = int(actual_header) if actual_header is not None else -1
    if actual_bytes != expected_bytes:
        raise RuntimeError(f"{archive['file']}: expected {expected_bytes} bytes, remote has {actual_bytes}")
    print(f"[check] {archive['file']} remote size ok ({actual_bytes} bytes)")


def _download(url: str, path: Path, *, retries: int, timeout: float, base_delay: float) -> None:
    print(f"[download] {url}")
    part_path = path.with_name(f"{path.name}.part")
    if part_path.exists():
        part_path.unlink()
    for attempt in range(1, retries + 1):
        try:
            with _urlopen_with_retries(url, retries=1, timeout=timeout, label=url) as response, part_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
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
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            target = (out_dir / member.name).resolve()
            if not str(target).startswith(str(out_dir.resolve()) + "/"):
                raise RuntimeError(f"unsafe archive member: {member.name}")
            if not member.name.startswith("users/"):
                raise RuntimeError(f"unexpected archive member: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(f"unsupported archive member type: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tf.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot extract archive member: {member.name}")
            with source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)


def _verify_extracted(users_root: Path, manifest: dict[str, Any]) -> None:
    users = manifest.get("users") or []
    for row in users:
        user_id = str(row["user_id"])
        user_dir = users_root / user_id
        for suffix in ("agent_album", "eval_gt", "export_audit"):
            path = user_dir / f"{user_id}_{suffix}.json"
            if not path.exists():
                raise RuntimeError(f"missing extracted file: {path}")
        audit = json.loads((user_dir / f"{user_id}_export_audit.json").read_text(encoding="utf-8"))
        if audit.get("passed") is not True:
            raise RuntimeError(f"{user_id}: export audit did not pass")


def _looks_extracted(users_root: Path, manifest: dict[str, Any]) -> bool:
    if not users_root.exists():
        return False
    users = manifest.get("users") or []
    if not users:
        return False
    for row in users:
        user_id = str(row["user_id"])
        user_dir = users_root / user_id
        if not user_dir.is_dir():
            return False
        for suffix in ("agent_album", "eval_gt", "export_audit"):
            if not (user_dir / f"{user_id}_{suffix}.json").exists():
                return False
    return True


def _archive_matches(path: Path, expected_sha: str) -> bool:
    return bool(expected_sha) and path.exists() and _sha256_file(path) == expected_sha


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _join_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


if __name__ == "__main__":
    raise SystemExit(main())
