#!/usr/bin/env python3
"""Download and extract every raw data artifact declared in config/source.json.

The script is deliberately dependency-free (standard library only) so that it can
run before `pip install -r requirements.txt` has been executed.

It is idempotent: a source whose marker files are already present on disk is
skipped, so re-running over an already-populated data/raw/ costs nothing and
never touches existing files.

Usage
-----
    python src/data/download.py                    # all 'core' sources
    python src/data/download.py --list             # show status, download nothing
    python src/data/download.py --group all        # core + optional
    python src/data/download.py --id ebnerd_testset
    python src/data/download.py --dataset mind
    python src/data/download.py --force --id ebnerd_demo

Authentication
--------------
The MIND sources live in a gated HuggingFace repo. Provide a read token via
`export HF_TOKEN=hf_...` or by running `huggingface-cli login` (which writes
~/.cache/huggingface/token). See the `auth` block in config/source.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = REPO_ROOT / "config" / "source.json"

CHUNK = 1 << 20  # 1 MiB
MAX_REDIRECTS = 10
USER_AGENT = "ir-assignment1-downloader/1.0"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def log(msg: str) -> None:
    print(msg, flush=True)


class DownloadError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# auth
# --------------------------------------------------------------------------- #

def resolve_token(auth_spec: dict) -> str | None:
    """Find a bearer token from the environment, then from well-known files."""
    # Environment wins over on-disk tokens so CI can override a stale login.
    for var in auth_spec.get("token_env", []):
        token = os.environ.get(var, "").strip()
        if token:
            return token
    # Fall back to the file huggingface-cli login writes.
    for raw in auth_spec.get("token_files", []):
        path = Path(raw).expanduser()
        if path.is_file():
            token = path.read_text(encoding="utf-8").strip()
            if token:
                return token
    return None


def auth_headers(source: dict, manifest: dict) -> dict:
    """Build request headers for a source, resolving its auth scheme if any."""
    scheme = source.get("auth")
    if not scheme:
        return {}  # public source, no credentials needed
    spec = manifest.get("auth", {}).get(scheme)
    if spec is None:
        raise DownloadError(
            f"source {source['id']!r} references unknown auth scheme {scheme!r}"
        )
    token = resolve_token(spec)
    if not token:
        raise DownloadError(
            f"source {source['id']!r} needs a {scheme} token but none was found.\n"
            f"  {spec.get('help', '')}\n"
            f"  You may also need to accept the dataset terms once at:\n"
            f"  {spec.get('accept_terms_url', '')}"
        )
    header = spec.get("header", "Authorization")
    value = spec.get("value_template", "Bearer {token}").format(token=token)
    return {header: value}


# --------------------------------------------------------------------------- #
# HTTP with manual redirect handling
# --------------------------------------------------------------------------- #

def open_url(url: str, headers: dict, offset: int = 0, strip_auth_cross_host: bool = True):
    """GET `url`, following redirects by hand.

    urllib's automatic redirect handler forwards every header to the redirect
    target. HuggingFace redirects to a pre-signed CDN URL that rejects a
    stray Authorization header, so we drop auth headers when the host changes.
    """
    current = url
    origin_host = urlsplit(url).hostname  # auth is only valid for this host
    hdrs = dict(headers)

    for _ in range(MAX_REDIRECTS):
        send = dict(hdrs)
        send["User-Agent"] = USER_AGENT
        if offset:
            send["Range"] = f"bytes={offset}-"  # open-ended range = resume from offset
        # The pre-signed CDN target 400s on a forwarded bearer token, so drop it.
        if strip_auth_cross_host and urlsplit(current).hostname != origin_host:
            send.pop("Authorization", None)

        req = Request(current, headers=send)
        try:
            resp = urlopen(req, timeout=60)
        except HTTPError as exc:
            # urllib raises rather than returns when it declines to auto-follow.
            if exc.code in (301, 302, 303, 307, 308):
                location = exc.headers.get("Location")
                if not location:
                    raise DownloadError(f"redirect without Location from {current}") from exc
                current = location
                continue
            if exc.code == 401:
                raise DownloadError(
                    f"HTTP 401 Unauthorized for {url}\n"
                    f"  The source is gated. Check your token and that you have "
                    f"accepted the dataset terms."
                ) from exc
            # Sentinel string, caught by download_file to discard the .part and retry.
            if exc.code == 416:  # requested range not satisfiable -> restart
                raise DownloadError("range-not-satisfiable") from exc
            raise DownloadError(f"HTTP {exc.code} for {current}") from exc
        except URLError as exc:
            raise DownloadError(f"network error for {current}: {exc.reason}") from exc

        # urlopen follows some redirects internally; detect and re-target.
        final = resp.geturl()
        if final != current and resp.status in (301, 302, 303, 307, 308):
            current = final
            resp.close()
            continue
        return resp

    raise DownloadError(f"too many redirects for {url}")


def download_file(url: str, dest: Path, headers: dict, expected_size: int | None) -> Path:
    """Download to `dest`, resuming a partial `.part` file when possible."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")  # only promoted to dest once complete

    # A leftover .part from an interrupted run is the resume point.
    offset = part.stat().st_size if part.exists() else 0
    if expected_size and offset > expected_size:
        part.unlink()  # corrupt or stale leftover, start clean
        offset = 0
    if offset:
        log(f"    resuming from {human(offset)}")

    try:
        resp = open_url(url, headers, offset=offset)
    except DownloadError as exc:
        if "range-not-satisfiable" not in str(exc):
            raise
        part.unlink(missing_ok=True)  # server rejected our offset, restart from zero
        offset = 0
        resp = open_url(url, headers, offset=0)

    # A server that ignores Range answers 200 and restarts the stream.
    if offset and resp.status != 206:
        offset = 0  # must not append, the body is the whole file again
        part.unlink(missing_ok=True)

    total = expected_size
    if total is None:
        length = resp.headers.get("Content-Length")
        # On a 206 Content-Length covers only the remainder, so add the offset back.
        total = (int(length) + offset) if length else None

    mode = "ab" if offset else "wb"  # append when resuming, truncate otherwise
    got = offset
    last_report = 0.0
    started = time.time()

    with resp, open(part, mode) as fh:
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            now = time.time()
            if now - last_report >= 0.5:  # throttle redraws to keep logs readable
                last_report = now
                rate = (got - offset) / max(now - started, 1e-6)  # this run only, not resumed bytes
                if total:
                    pct = 100.0 * got / total
                    print(
                        f"\r    {human(got)} / {human(total)} ({pct:5.1f}%) "
                        f"at {human(rate)}/s   ",
                        end="",
                        flush=True,
                    )
                else:
                    print(f"\r    {human(got)} at {human(rate)}/s   ", end="", flush=True)
    print("\r" + " " * 70 + "\r", end="", flush=True)  # wipe the progress line

    # Truncated transfer: keep the .part so the next run resumes instead of restarting.
    if total and got != total:
        raise DownloadError(
            f"size mismatch: got {got} bytes, expected {total}. Partial file kept "
            f"at {part} - re-run to resume."
        )

    part.replace(dest)  # atomic promote, so dest only ever exists complete
    log(f"    downloaded {human(got)} in {time.time() - started:.1f}s")
    return dest


def sha256_file(path: Path) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        # iter(callable, sentinel) reads until read() returns b"".
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #

def _safe_members(zf: zipfile.ZipFile, dest: Path, skip_macosx: bool):
    """Yield members that stay inside `dest` (guards against zip-slip)."""
    dest = dest.resolve()
    for info in zf.infolist():
        name = info.filename
        # macOS zips carry a parallel __MACOSX/ tree of resource forks; pure noise.
        if skip_macosx and ("__MACOSX/" in name or Path(name).name == ".DS_Store"):
            continue
        target = (dest / name).resolve()
        # Reject "../" entries that would write outside dest (zip-slip).
        if not str(target).startswith(str(dest) + os.sep) and target != dest:
            raise DownloadError(f"unsafe path in archive: {name!r}")
        yield info


def extract_zip(archive: Path, dest: Path, strip_top_level: bool, skip_macosx: bool = True) -> None:
    """Extract `archive` into `dest`, optionally collapsing a single wrapper folder."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = list(_safe_members(zf, dest, skip_macosx))

        prefix = ""
        if strip_top_level:
            # Only safe to strip when every entry shares one root folder.
            tops = {m.filename.split("/", 1)[0] for m in members if m.filename.strip("/")}
            if len(tops) == 1:
                prefix = tops.pop() + "/"
            else:
                log(f"    note: strip_top_level requested but archive has {len(tops)} roots")

        for info in members:
            name = info.filename
            if prefix:
                if not name.startswith(prefix):
                    continue
                name = name[len(prefix):]
            if not name:
                continue  # the stripped root directory entry itself
            target = dest / name
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)  # zips need not list dirs first
            # Stream rather than zf.extract so large parquet files stay off the heap.
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, CHUNK)


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #

def is_present(source: dict, raw_dir: Path) -> bool:
    """True when a source's outputs already exist on disk.

    Uses the declared marker files when available; falls back to "extract_dir
    exists and is non-empty" for sources whose internal layout is not pinned.
    """
    markers = source.get("markers") or []
    if markers:
        return all((raw_dir / m).exists() for m in markers)
    # No pinned markers: treat a non-empty extract dir as done.
    target = raw_dir / source["extract_dir"]
    return target.is_dir() and any(target.iterdir())


def check_disk(sources: list[dict], raw_dir: Path) -> None:
    """Warn when the pending downloads look too big for the filesystem."""
    needed = sum(s.get("size_bytes") or 0 for s in sources)
    if not needed:
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(raw_dir).free
    # Archive and extracted copy coexist until extraction finishes, hence >2x.
    estimate = int(needed * 2.2)
    log(f"  pending download: {human(needed)}; free disk: {human(free)}")
    if estimate > free:
        log(
            f"  WARNING: need roughly {human(estimate)} including extraction, "
            f"but only {human(free)} is free."
        )


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #

def load_manifest(path: Path) -> dict:
    """Read the provenance record of what has been downloaded so far."""
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A corrupt manifest must not block a rebuild; it is derived data.
            log(f"  note: {path} was unreadable, starting a fresh manifest")
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    """Persist the manifest with stable ordering so diffs stay readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# main flow
# --------------------------------------------------------------------------- #

def select_sources(spec: dict, args) -> list[dict]:
    """Resolve CLI filters to the list of sources to act on, in manifest order."""
    sources = spec["sources"]
    # An explicit --id overrides the group/dataset filters entirely.
    if args.id:
        wanted = set(args.id)
        known = {s["id"] for s in sources}
        missing = wanted - known
        # Fail loudly on typos rather than silently downloading nothing.
        if missing:
            raise SystemExit(
                f"unknown source id(s): {', '.join(sorted(missing))}\n"
                f"available: {', '.join(sorted(known))}"
            )
        return [s for s in sources if s["id"] in wanted]

    picked = sources
    if args.group != "all":
        picked = [s for s in picked if s.get("group", "core") == args.group]
    if args.dataset:
        # Validate against the manifest so a typo fails loudly, as --id does.
        known = {s.get("dataset") for s in sources if s.get("dataset")}
        if args.dataset not in known:
            raise SystemExit(
                f"unknown dataset: {args.dataset}\n"
                f"available: {', '.join(sorted(known))}"
            )
        picked = [s for s in picked if s.get("dataset") == args.dataset]
    return picked


def process(source: dict, spec: dict, raw_dir: Path, archive_dir: Path,
            manifest: dict, args) -> str:
    """Bring one source to a ready state; returns what action was taken."""
    sid = source["id"]
    log(f"\n[{sid}] {source.get('name', sid)}")

    # The idempotency gate: existing data is never re-fetched or overwritten.
    if is_present(source, raw_dir) and not args.force:
        log(f"    present at {source['extract_dir']}/ - skipping")
        return "skipped"

    # Dry-run exits before auth so it works without credentials.
    if args.dry_run:
        log(f"    would download {human(source.get('size_bytes') or 0)} from {source['url']}")
        return "would-download"

    headers = auth_headers(source, spec)
    archive = archive_dir / f"{sid}.zip"

    # A complete archive from a previous run means only extraction is left.
    if archive.is_file() and not args.force:
        log(f"    archive already downloaded: {archive.name}")
    else:
        log(f"    fetching {source['url']}")
        download_file(source["url"], archive, headers, source.get("size_bytes"))

    digest = sha256_file(archive)
    expected = source.get("sha256")
    # sha256 is null in the manifest until pinned; recorded below either way.
    if expected and digest != expected:
        raise DownloadError(
            f"checksum mismatch for {sid}: got {digest}, expected {expected}"
        )

    dest = raw_dir / source["extract_dir"]
    log(f"    extracting into {source['extract_dir']}/")
    extract_zip(archive, dest, source.get("strip_top_level", False))

    # Post-condition: catches a wrong extract_dir/markers guess immediately.
    if not is_present(source, raw_dir):
        markers = source.get("markers") or []
        raise DownloadError(
            f"{sid}: extraction finished but expected files are missing.\n"
            f"  markers: {markers}\n"
            f"  Inspect {dest} and correct 'markers'/'strip_top_level' in config/source.json."
        )

    # Provenance for the report: what was fetched, from where, and its digest.
    manifest[sid] = {
        "url": source["url"],
        "size_bytes": archive.stat().st_size,
        "sha256": digest,
        "extract_dir": source["extract_dir"],
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    # Archives are redundant once extracted; dropping them matters on a tight disk.
    if not args.keep_archives:
        archive.unlink(missing_ok=True)
        log("    removed archive (pass --keep-archives to retain)")

    return "downloaded"


def cmd_list(spec: dict, raw_dir: Path) -> None:
    """Print a status table for every declared source without downloading."""
    log(f"raw dir: {raw_dir}")
    log(f"{'id':<30} {'dataset':<8} {'group':<9} {'size':>10}  status")
    log("-" * 74)
    for source in spec["sources"]:
        status = "present" if is_present(source, raw_dir) else "MISSING"
        log(
            f"{source['id']:<30} {source.get('dataset', '-'):<8} "
            f"{source.get('group', 'core'):<9} "
            f"{human(source.get('size_bytes') or 0):>10}  {status}"
        )


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, resolve the selection, and process each source in turn."""
    parser = argparse.ArgumentParser(
        description="Download raw datasets declared in config/source.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,  # surfaces the usage/auth notes under --help
    )
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES,
                        help="path to the source manifest (default: config/source.json)")
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="override the raw data root declared in the manifest")
    parser.add_argument("--group", default="core", choices=["core", "optional", "all"],
                        help="which group of sources to fetch (default: core)")
    # No `choices=` here: valid datasets come from the manifest, not from this file.
    parser.add_argument("--dataset", default=None,
                        help="restrict to one dataset (values come from config/source.json)")
    parser.add_argument("--id", action="append", default=[],
                        help="fetch a specific source id (repeatable); overrides --group/--dataset")
    parser.add_argument("--list", action="store_true",
                        help="print the status of every source and exit")
    parser.add_argument("--force", action="store_true",
                        help="re-download and re-extract even if files are present")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would happen without downloading")
    parser.add_argument("--keep-archives", action="store_true",
                        help="keep the downloaded .zip files (deleted after extraction by default)")
    args = parser.parse_args(argv)

    if not args.sources.is_file():
        raise SystemExit(f"source manifest not found: {args.sources}")
    spec = json.loads(args.sources.read_text(encoding="utf-8"))

    # Paths resolve relative to the repo root, so the cwd does not matter.
    raw_dir = args.raw_dir or (REPO_ROOT / spec.get("raw_dir", "data/raw"))
    raw_dir = Path(raw_dir).resolve()
    archive_dir = raw_dir / spec.get("archive_dir", "_archives")
    manifest_path = raw_dir / spec.get("manifest_file", "manifest.json")

    if args.list:
        cmd_list(spec, raw_dir)
        return 0

    selected = select_sources(spec, args)
    if not selected:
        log("no sources matched the selection")
        return 0

    # Report the real workload up front, before any slow transfer starts.
    pending = [s for s in selected if not is_present(s, raw_dir) or args.force]
    log(f"selected {len(selected)} source(s); {len(pending)} need work")
    if pending:
        check_disk(pending, raw_dir)

    manifest = load_manifest(manifest_path)
    counts: dict[str, int] = {}
    failures: list[tuple[str, str]] = []

    for source in selected:
        try:
            outcome = process(source, spec, raw_dir, archive_dir, manifest, args)
            counts[outcome] = counts.get(outcome, 0) + 1
        except DownloadError as exc:
            # One bad source must not abort the rest; collect and report at the end.
            log(f"    ERROR: {exc}")
            failures.append((source["id"], str(exc)))
        except KeyboardInterrupt:
            log("\ninterrupted - partial downloads are resumable, just re-run")
            return 130

    # Only write when something was actually recorded, so --list/no-op stay read-only.
    wrote_manifest = bool(manifest) and not args.dry_run
    if wrote_manifest:
        save_manifest(manifest_path, manifest)

    log("\n" + "=" * 60)
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    log(f"summary: {summary}" if summary else "summary: nothing to do")
    if failures:
        log(f"{len(failures)} source(s) failed:")
        for sid, msg in failures:
            log(f"  - {sid}: {msg.splitlines()[0]}")
        return 1
    if wrote_manifest:
        log(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
