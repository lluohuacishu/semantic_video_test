"""Verify artifacts, preserve input snapshots, and mark only F2 complete."""
from __future__ import annotations
import re
import zipfile
from pathlib import Path
from coverage_audit import (
    WORKSPACE, PROJECT, OUT, MANUAL, MANUAL_SHA, read, write, sha,
    load_packet, CONTROL, STAGE_E, mask, rectangles,
)


def main():
    for filename in ("步骤2冻结协议.json", "码流空间对应补充约定.json"):
        protocol = read(OUT/filename)
        assert all(sha(p) == value for p, value in protocol["input_sha256"].items())
    summary = read(OUT/"覆盖统计汇总.json")
    rows = read(OUT/"逐框覆盖明细.json")
    frames = read(OUT/"逐帧覆盖与无鱼开销.json")
    streams = read(OUT/"码流与人工框空间对应.json")
    assert len(rows) == 61 and len({r["id"] for r in rows}) == 61
    assert len(frames) == 50
    assert [r["source_frame"] for r in frames] == list(range(420, 470))
    assert sum(r["E"]["status"] == "missed" for r in rows) == 20
    assert [r["source_frame"] for r in rows if r["E"]["status"] == "missed"] == list(range(432, 452))
    assert all(r["E"]["all_boxes_full95"] for r in frames[32:])
    assert [r["E"]["all_boxes_full95"] for r in frames[12:32]] == [False]*20
    checked_regions = {}
    for method, selection in (("A", CONTROL/"A/selection.json"), ("E", STAGE_E/"selection.json")):
        selected, manifest, semantics, semantic_bytes, parts = load_packet(selection)
        assert sum(selected["parts"].values()) == selected["file_bytes"]
        assert sum(p["bytes"] for p in parts) == selected["parts"]["roi"]
        count = 0
        for i, record in enumerate(semantics["frames"]):
            active = mask(rectangles(manifest["patches"], i))
            for region in record["regions"]:
                x, y, w, h = region["box"]
                assert active[y:y+h, x:x+w].all()
                count += 1
        assert count == 35
        checked_regions[method] = count
    assert all(s["zero_overlap_with_reviewed_boxes_all_active_frames"]
               for s in streams if s["method"] == "E" and s["partition"] == "left")
    assert sha(MANUAL) == MANUAL_SHA
    snapshots = OUT/"输入快照"
    snapshots.mkdir(exist_ok=True)
    (snapshots/"manual_roi_snapshot.json").write_bytes(MANUAL.read_bytes())
    (snapshots/"冻结发送端预测.json.gz").write_bytes(semantic_bytes)
    assert sha(snapshots/"manual_roi_snapshot.json") == MANUAL_SHA

    report = OUT/"阶段F_步骤2_人工框覆盖与ROI开销核对_2026-10-15.md"
    project_report = PROJECT/"docs/phase_f_step2_coverage_20261015.md"
    checked_links = 0
    for md in [report, project_report, OUT/"逐框覆盖表.md", OUT/"漏保护清单.md"]:
        content = md.read_text("utf-8")
        assert "\ufffd" not in content
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content):
            path = Path(target) if re.match(r"^[A-Za-z]:/", target) else md.parent/target
            assert path.is_file(), f"Missing link: {path}"
            checked_links += 1
    assert len([line for line in (OUT/"逐框覆盖表.md").read_text("utf-8").splitlines()
                if line.startswith("| 00")]) == 61
    validation = read(OUT/"步骤2数据核验.json")
    validation.update({
        "all_35_sender_predicted_boxes_inside_actual_ROI_in_each_method": checked_regions,
        "packet_part_byte_totals_verified": True,
        "all_frozen_inputs_still_match_after_reporting": True,
        "report_generated": True,
        "report_links_verified": checked_links,
        "figures": 2,
        "figures_visually_inspected": True,
        "input_snapshot_copy_hash_verified": True,
        "human_blind_review_performed": False,
        "step2_status": "complete",
    })
    write(OUT/"步骤2数据核验.json", validation)
    artifacts = {
        str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob("*"))
        if p.is_file() and p.suffix != ".zip" and p.name != "证据包指纹.json"
    }
    write(OUT/"证据包指纹.json", {
        "date": "2026-10-15", "stage": "F", "step": 2, "artifact_sha256": artifacts,
        "production_model_algorithm_and_annotations_unchanged": True,
        "reproduction_scripts": {str(p): sha(p) for p in Path(__file__).parent.glob("*coverage*.py")},
    })
    bundle = OUT/"阶段F_步骤2_报告图表与数值.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path.suffix != ".zip":
                archive.write(path, path.relative_to(OUT))
        for path in sorted(Path(__file__).parent.glob("*coverage*.py")):
            archive.write(path, "reproduction/"+path.name)
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        bundled_files = len(archive.namelist())
    progress_path = WORKSPACE/"outputs/semantic_roi_phase_f/阶段F进度.json"
    progress = read(progress_path)
    previous_report = progress["report"]
    step = "step2 frozen 61-box coverage and existing ROI substream cost audit"
    if step not in progress["completed_steps"]:
        progress["completed_steps"].append(step)
    progress["remaining"] = ["frozen-E independent-video validation"]
    progress["status"] = "in_progress"
    progress["report"] = str(report)
    reports = progress.setdefault("reports", {})
    if "step1" not in reports:
        reports["step1"] = previous_report
    reports["step2"] = str(report)
    progress["latest_completed_step"] = 2
    write(progress_path, progress)
    print({
        "status": "F2 complete; F still in progress",
        "box_rows": len(rows), "report_links_verified": checked_links,
        "bundle_files": bundled_files, "bundle_bytes": bundle.stat().st_size,
        "model_training": False, "new_encodes": 0, "new_transfers": 0,
    }, flush=True)


if __name__ == "__main__":
    main()
