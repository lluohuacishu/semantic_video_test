"""One pre-viewing byte-only midpoint refinement, with original evidence intact."""
import argparse
import copy
from pathlib import Path
import shutil
import sys
import time
from b2_common import Journal, git, read, require, sha, write
from encode_b3_controls import geometry_search, closest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, required=True)
    args = parser.parse_args()
    require(not git("status", "--porcelain"), "Commit amendment before execution")
    original = read(args.search / "selection.json")
    require(not original["matched"] and len(original["trials"]) == 9, "Unexpected prior search")
    observed = {r["crf"]: r["supplement_bytes"] for r in original["trials"]}
    require(observed[30.75] > original["target_supplement_bytes"] > observed[31.0], "Expected observed bracket missing")
    saved = read(args.search / "config.json")
    config = copy.deepcopy(saved["b3"])
    config["geometry"].update(first_crf=30.875, step=0.125, max_trials=1)
    output = args.search.parent / "geometry_refinement"
    output.mkdir()
    write(output / "config.json", {"b3": config, "baseline": saved["baseline"], "code_commit": git("rev-parse", "HEAD"), "amendment": "B3冻结协议_20261018.md, precision addendum"})
    journal = Journal(output)
    entry = journal.begin([sys.executable, *sys.argv], code_commit=git("rev-parse", "HEAD"))
    tick = time.perf_counter()
    try:
        geometry_search(config, saved["baseline"], output, journal)
        refinement = read(output / "selection.json")
        combined = original["trials"] + refinement["trials"]
        target = original["target_supplement_bytes"]
        chosen = closest(combined, target, "supplement_bytes")
        preserved = args.search / "selection_before_precision_refinement.json"
        require(not preserved.exists(), "Original selection already preserved")
        shutil.copyfile(args.search / "selection.json", preserved)
        result = {**original, "selected": chosen, "trials": combined,
                  "relative_supplement_difference": chosen["supplement_bytes"]/target-1,
                  "relative_complete_file_difference": chosen["file_bytes"]/config["target_file_bytes"]-1,
                  "matched": abs(chosen["supplement_bytes"]/target-1) <= config["budget_tolerance_fraction"],
                  "stop_reason": "one_pre_viewing_precision_refinement_complete",
                  "prior_selection_path": str(preserved), "prior_selection_sha256": sha(preserved),
                  "refinement_selection": str(output/"selection.json"), "refinement_commit": entry["code_commit"]}
        write(args.search / "selection.json", result)
        journal.finish(entry, 0, time.perf_counter()-tick)
    except BaseException as exc:
        write(output/"failure.json", {"type": type(exc).__name__, "message": str(exc)})
        journal.finish(entry, 1, time.perf_counter()-tick)
        raise


if __name__ == "__main__":
    main()
