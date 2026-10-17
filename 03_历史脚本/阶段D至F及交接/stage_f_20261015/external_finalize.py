"""Final consistency checks, evidence bundle, and a bounded Stage F closeout."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile
from external_preflight import WORKSPACE, PROJECT, RUN, OUT, DATA, read, write, digest


def main():
    cases = [
        ("source_00000_00049", RUN, OUT, DATA),
        ("source_00050_00099", RUN/"positive_case", OUT/"positive_case", DATA/"positive_case"),
    ]
    rows = []
    for label,root,output,folder in cases:
        protocol = read(root/"experiment_protocol.json")
        assert all(digest(path) == value for path,value in protocol["input_sha256"].items())
        data = read(folder/"manifest.json")
        source = read(root/"initial_crf32/source_audit.json")
        assert source["source_frame_indices"] == data["source_frame_indices"]
        assert source["rgb_sha256"] == data["source_rgb_sha256"]
        assert all(b-a == 1 for a,b in zip(source["source_frame_indices"],source["source_frame_indices"][1:]))
        selected = {name:read(root/name/"selection.json") for name in ("E","H265")}
        pair_difference = abs(selected["E"]["file_bytes"]-selected["H265"]["file_bytes"])/selected["E"]["file_bytes"]
        assert pair_difference <= .02
        for name,value in selected.items():
            path = Path(value["file"])
            decoded = read(root/name/"network/receiver/decode.json")
            transfer = read(root/name/"network/transfer.json")
            end = read(root/name/"network/end_to_end.json")
            received = root/name/"network/receiver"/("message.scv" if name == "E" else "video.mp4")
            assert digest(path) == digest(received) == value["sha256"] == decoded["file_sha256"] == transfer["sha256"]
            assert decoded["frames"] == 50 and len(decoded["decoded_rgb_sha256"]) == 50
            assert end["separate_sender_receiver_processes"] and end["packet_sha256_equal"]
            assert sum(value["parts"].values()) == value["file_bytes"]
            assert transfer["application_bytes"] >= value["file_bytes"]
            assert transfer["limit_kbps"] == 20
            guard = read(root/name/"network/receiver/receiver_read_audit.json")
            assert all(r["allowed"] for r in guard["python_reads"])
            if name == "E":
                with zipfile.ZipFile(received) as archive:
                    assert archive.testzip() is None
                    manifest = json.loads(archive.read("manifest.json"))
                    for filename,info in manifest["files"].items():
                        payload = archive.read(filename)
                        assert len(payload) == info["bytes"] and hashlib.sha256(payload).hexdigest() == info["sha256"]
                assert manifest["profile"] == "semantic-video-layered-v2"
                assert manifest["teacher"]["threshold"] == .9
                assert manifest["configuration"]["min_pixels"] == 64
                assert manifest["configuration"]["segment_frames"] == 32
                assert manifest["configuration"]["padding"] == 16
                assert manifest["configuration"]["background_crf"] == 42
            rows.append({
                "case": label, "method": name, "file_bytes": value["file_bytes"],
                "packet_and_receive_sha256_match": True, "independent_decoder": True,
                "frames": 50, "same_case_file_size_difference": pair_difference,
                "all_receiver_read_audit_entries_allowed": True,
            })
        structure = read(output/"实际码流结构核验.json")
        assert next(r for r in structure if r["method"] == "H265")["frame_types"].get("B",0) > 0
        assert all(r["frame_types"].get("B",0) == 0 for r in structure if r["method"] == "E")
        for annotation in data["sparse_gt"]:
            assert digest(annotation["fish_mask"]) == annotation["fish_mask_sha256"]
            assert digest(annotation["published_mask"]) == annotation["published_mask_sha256"]
    negative = read(OUT/"新视频实测评价.json")
    positive = read(OUT/"positive_case/新视频实测评价.json")
    target = read(OUT/"positive_case/含鱼GT保护与逐连通域核对.json")
    assert negative["gt_positive_frames"] == 0 and positive["gt_positive_frames"] == 3
    assert positive["all50_actual_received_RGB_hashes_verified_each_method"]
    assert target["pooled_GT_fish_pixels"] == 2010 and target["E_covered_GT_fish_pixels"] == 1940
    assert target["all_components"] == 10 and target["components_full95"] == 9
    assert all(r["sparse_gt_metrics"]["tp"] == 0 for r in positive["rows"] if r["condition"] != "original")
    assert next(r for r in positive["rows"] if r["condition"] == "original")["sparse_gt_metrics"]["tp"] == 615
    first_indices = read(DATA/"manifest.json")["source_frame_indices"]
    second_indices = read(DATA/"positive_case/manifest.json")["source_frame_indices"]
    assert not set(first_indices) & set(second_indices)
    assert first_indices+second_indices == list(range(100))

    summary = WORKSPACE/"阶段F_首轮总结与后续方向_2026-10-15.md"
    project_summary = PROJECT/"docs/phase_f_summary_20261015.md"
    shutil.copy2(summary,project_summary)
    primary = OUT/"阶段F_步骤3_冻结方案新保留视频对照_2026-10-15.md"
    md_files = [primary,PROJECT/"docs/phase_f_step3_holdout_20261015.md",summary,project_summary]
    link_count = 0
    for path in md_files:
        text = path.read_text("utf-8")
        assert "\ufffd" not in text
        for destination in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",text):
            if destination.startswith(("https://","http://")): continue
            link = Path(destination) if re.match(r"^[A-Za-z]:/",destination) else path.parent/destination
            assert link.is_file(),str(link)
            link_count += 1
    validation = {
        "date": "2026-10-15", "stage": "F", "step": 3,
        "source_video_count": 1, "clips": 2, "distinct_source_frames": 100,
        "transmissions": rows, "total_received_frames_across_four_transmissions": 200,
        "all_protocol_input_hashes_unchanged": True,
        "no_model_or_production_algorithm_change": True,
        "all_sparse_GT_hashes_and_RGB_alignment_verified": True,
        "all_reconstructed_RGBs_verified_during_evaluation": True,
        "all_file_budgets_within_2_percent_pairwise": True,
        "H265_B_frames_verified_and_E_P_only_verified": True,
        "positive_clip_selected_by_publisher_GT_presence_not_gain": True,
        "source_is_official_train_new_local_holdout": True,
        "capture_independence_established": False,
        "reliable_semantic_protection_demonstrated": False,
        "human_blind_review_performed": False,
        "figures_visually_inspected": 3,
        "local_report_links_verified": link_count,
        "bounded_stage_F_first_cycle": "complete",
    }
    write(OUT/"步骤3最终核验.json",validation)
    write(OUT/"执行记录说明.json",{
        "initial_case_console_encoding": "first parent process stopped while printing child output using GBK; production encoding and JSON had already completed",
        "resume": "PYTHONUTF8=1 used for subsequent orchestration; same source, weights and encoded initial packet retained",
        "scientific_configuration_changed_for_resume": False,
        "evaluator_generalization": "source-frame labels generalized for 50..99 supplement; original0..49 numerical computations unchanged",
    })

    pack = OUT/"阶段F_步骤3_报告图表与实测证据.zip"
    hashes = {str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob("*"))
              if p.is_file() and p.suffix != ".zip" and p.name != "证据包指纹.json"}
    scripts = sorted(Path(__file__).parent.glob("external_*.py"))
    write(OUT/"证据包指纹.json",{
        "date":"2026-10-15","artifact_sha256":hashes,
        "scripts":{str(path):digest(path) for path in scripts},
        "source_RGB_and_model_weights_remain_in_project":True,
    })
    with zipfile.ZipFile(pack,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path.suffix != ".zip":
                archive.write(path,path.relative_to(OUT))
        archive.write(summary,summary.name)
        for script in scripts:
            archive.write(script,"reproduction/"+script.name)
        for label,root,output,folder in cases:
            for method in ("E","H265"):
                receiver = root/method/"network/receiver"
                name = "message.scv" if method == "E" else "video.mp4"
                archive.write(receiver/name,f"received_media/{label}/{method}/{name}")
                for filename in ("decode.json","receiver_read_audit.json"):
                    archive.write(receiver/filename,f"received_media/{label}/{method}/{filename}")
                archive.write(receiver/"reconstructed.mp4",f"received_media/{label}/{method}/viewing_copy.mp4")
            for label_row in read(folder/"manifest.json")["sparse_gt"]:
                index = label_row["source_frame"]
                archive.write(label_row["published_mask"],f"publisher_GT/{index:05d}.png")
                archive.write(label_row["fish_mask"],f"fish_masks/{index:05d}.png")
    with zipfile.ZipFile(pack) as archive:
        assert archive.testzip() is None
        entries = len(archive.namelist())

    progress_path = WORKSPACE/"outputs/semantic_roi_phase_f/阶段F进度.json"
    progress = read(progress_path)
    step = "step3 frozen-E local holdout video6: disjoint empty-GT and positive-GT clips, actual byte-matched TCP comparisons"
    if step not in progress["completed_steps"]: progress["completed_steps"].append(step)
    progress["status"] = "first_cycle_complete"
    progress["completion_scope"] = "initial three-step diagnostic cycle; not reliable semantic protection or generalization"
    progress["remaining"] = []
    progress["future_work"] = [
        "human readability checks on fixed failure examples",
        "one controlled background/ROI budget experiment using development data",
        "new capture-independent videos for any revised method",
    ]
    progress["research_goal_achieved"] = False
    progress["independent_capture_validation"] = "not established"
    progress["report"] = str(summary)
    progress["reports"]["step3"] = str(primary)
    progress["reports"]["summary"] = str(summary)
    progress["latest_completed_step"] = 3
    progress["new_video_evidence_scope"] = "one local holdout official-train video, two disjoint clips; conditional positive-GT supplement"
    write(progress_path,progress)
    print(json.dumps({
        "stage_F_first_cycle":"complete","reliable_semantic_protection":False,
        "local_links":link_count,"received_frames":200,"evidence_files":entries,
        "bundle_bytes":pack.stat().st_size,"bundle_sha256":digest(pack),
        "summary":str(summary)},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
