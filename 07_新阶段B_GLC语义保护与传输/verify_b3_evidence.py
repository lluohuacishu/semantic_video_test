"""Verify archived evidence, restored bytes, sources and optional Git blobs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def check_payload(payload, row):
    assert len(payload) == row["stored_bytes"], row["path"]
    assert digest(payload) == row["stored_sha256"], row["path"]
    assert row["compression"] in {"identity", "gzip"}, row["path"]
    original = gzip.decompress(payload) if row["compression"] == "gzip" else payload
    assert len(original) == row["original_bytes"], row["path"]
    assert digest(original) == row["original_sha256"], row["path"]


def git_blobs(repo, specs):
    result = subprocess.run(["git", "cat-file", "--batch"], cwd=repo,
                            input=("\n".join(specs) + "\n").encode("utf-8"),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    pos = 0
    blobs = []
    for spec in specs:
        end = result.stdout.index(b"\n", pos)
        header = result.stdout[pos:end].split()
        assert len(header) == 3 and header[1] == b"blob", (spec, header)
        size = int(header[2])
        payload = result.stdout[end+1:end+1+size]
        assert result.stdout[end+1+size:end+2+size] == b"\n", spec
        blobs.append(payload)
        pos = end + size + 2
    assert pos == len(result.stdout)
    return blobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--verify-source", action="store_true")
    parser.add_argument("--git-ref", help="INDEX or a committed revision such as HEAD")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.resolve()
    manifest_bytes = (archive / "archive_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    rows = manifest["files"]
    assert len({r["path"] for r in rows}) == len(rows)
    for row in rows:
        target = (archive / row["path"]).resolve()
        assert target.is_relative_to(archive)
        check_payload(target.read_bytes(), row)
        if args.verify_source:
            source = (archive / row["path"] if row["source_path"] == "local_media_index.json"
                      else Path(manifest["run"]) / row["source_path"])
            assert digest(source.read_bytes()) == row["original_sha256"], str(source)
    assert sum(r["stored_bytes"] for r in rows) == manifest["stored_bytes"]
    assert sum(r["original_bytes"] for r in rows) == manifest["original_bytes"]
    result = {"status": "passed", "archive": str(archive), "records": len(rows),
              "manifest_sha256": digest(manifest_bytes),
              "stored_bytes": manifest["stored_bytes"], "original_bytes": manifest["original_bytes"],
              "source_bytes_verified": args.verify_source,
              "git_ref": args.git_ref, "git_blobs_verified": 0}
    if args.git_ref:
        repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                           cwd=archive).decode("utf-8").strip())
        relative = archive.relative_to(repo).as_posix()
        paths = [relative + "/archive_manifest.json"] + [relative + "/" + r["path"] for r in rows]
        prefix = ":" if args.git_ref == "INDEX" else args.git_ref + ":"
        blobs = git_blobs(repo, [prefix + path for path in paths])
        assert blobs[0] == manifest_bytes, "Git manifest differs from local manifest"
        for payload, row in zip(blobs[1:], rows):
            check_payload(payload, row)
        result["git_blobs_verified"] = len(blobs)
        result["head_at_verification"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
