"""CPU-only regression for C1 pixel rules, bounded preview and task isolation."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from c1_io import MemoryFrames, RawPreview
from c1_runtime import ResidentRunner, TaskCancelled, receiver_audit, ref
from glc_container import frame_layout
from run_c0 import FFMPEG


class FakeCodec:
    """Validate control flow without neural inference; real hashes tested on GPU."""
    instances = []
    fail_index = None

    def __init__(self, repo, weights, device, decoder_only=False):
        self.profile = {"fake": True}
        self.entropy = SimpleNamespace(table_sha256="fake-table")
        self.loading = {}
        self.dpb = {"ref_latent": None, "ref_y": None}
        self.decoder_only = decoder_only
        self.__class__.instances.append(self)

    def frame(self, header, index, x=None, packet=None):
        if index == 0:
            assert self.dpb == {"ref_latent": None, "ref_y": None}, "DPB leaked across tasks"
        if self.decoder_only:
            assert x is None and packet is not None
        self.dpb = {"ref_latent": index, "ref_y": index}
        if index == self.fail_index:
            raise RuntimeError("injected frame failure")
        rgb = np.full((header["height"], header["width"], 3), index % 256, np.uint8)
        kind, z_size, parts, symbols = frame_layout(header, index)
        result = {"kind": kind, "z": bytes(z_size), "y": [struct.pack(">I", symbols) + bytes(12)] * parts,
                  "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                  "latent_sha256": hashlib.sha256(str(index).encode()).hexdigest()}
        return rgb, result


class C1Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.avi"
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*"FFV1"), 25, (96, 64))
        self.assertTrue(writer.isOpened())
        random = np.random.default_rng(42)
        for _ in range(7):
            writer.write(random.integers(0, 256, (64, 96, 3), dtype=np.uint8))
        writer.release()
        FakeCodec.instances = []
        FakeCodec.fail_index = None

    def tearDown(self):
        self.temporary.cleanup()

    def runner(self, role="encode"):
        return ResidentRunner(role, self.root, self.root, FFMPEG, "cpu", codec_factory=FakeCodec)

    def task(self, name, **kwargs):
        return {"input": str(self.source), "output": str(self.root / name),
                "preset": "compact", "frames": 4, "output_mode": "display", **kwargs}

    def test_memory_pixels_mapping_and_eof_equal_reference(self):
        for stride, start, width in [(1, 0, 0), (2, 1, 64), (3, 2, 47)]:
            with self.subTest(stride=stride):
                args = argparse.Namespace(input=self.source, frames=10, start_frame=start, frame_stride=stride, width=width)
                folder = self.root / str(stride)
                folder.mkdir()
                original = ref.extract_frames(args, folder, cv2, Image)
                reader = MemoryFrames(args)
                try:
                    frames = list(reader)
                    metadata = reader.metadata()
                finally:
                    reader.close()
                self.assertEqual({k: v for k, v in metadata.items() if k != "rgb_sha256"}, original)
                for i, rgb in enumerate(frames):
                    with Image.open(folder / "source" / f"im{i + 1:05d}.png") as expected:
                        self.assertEqual(rgb.tobytes(), expected.convert("RGB").tobytes())

    def test_empty_and_beyond_eof_fail(self):
        args = argparse.Namespace(input=self.source, frames=2, start_frame=20, frame_stride=1, width=0)
        with self.assertRaises(ValueError):
            MemoryFrames(args)

    def test_resident_reset_identical_stream_and_research_pixel_evidence(self):
        runner = self.runner()
        states = []
        runner.run(self.task("a"), on_event=lambda e: states.append(e["state"]))
        runner.run(self.task("b", output_mode="research", png_level=1))
        self.assertEqual(len(FakeCodec.instances), 1)
        self.assertEqual((self.root / "a/stream.glcv").read_bytes(), (self.root / "b/stream.glcv").read_bytes())
        self.assertEqual(runner.codec.dpb, {"ref_latent": None, "ref_y": None})
        self.assertEqual(states, ["running", "preview_pending", "ready"])
        self.assertEqual(list((self.root / "a").rglob("*.png")), [])
        self.assertEqual(len(list((self.root / "b").rglob("*.png"))), 8)
        with self.assertRaises(FileExistsError):
            runner.run(self.task("a"))

    def test_independent_decode_and_eof_shorter_than_request(self):
        self.runner().run(self.task("encode", frames=20))
        runner = self.runner("decode")
        task = self.task("decode", input=str(self.root / "encode/stream.glcv"))
        runner.run(task)
        result = json.loads((self.root / "decode/receiver/decode.json").read_text())
        self.assertEqual(result["header"]["frames"], 4)  # seven source frames, stride two
        self.assertTrue(result["all_frame_checksums_match"])
        self.assertEqual(result["source_media_reads"], [])
        self.assertTrue(runner.codec.decoder_only)
        with self.assertRaises(ValueError):
            runner.run({**task, "role": "encode"})

    def test_failed_task_does_not_poison_next_task(self):
        runner = self.runner()
        FakeCodec.fail_index = 1
        with self.assertRaisesRegex(RuntimeError, "injected"):
            runner.run(self.task("failed"))
        self.assertEqual(runner.codec.dpb, {"ref_latent": None, "ref_y": None})
        self.assertFalse((self.root / "failed/stream.glcv").exists())
        FakeCodec.fail_index = None
        runner.run(self.task("recovered"))
        self.assertEqual(len(FakeCodec.instances), 1)

    def test_cancellation_retains_status_and_allows_next_task(self):
        runner = self.runner()
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(TaskCancelled):
            runner.run(self.task("cancelled"), cancel_event=cancelled)
        self.assertEqual(json.loads((self.root / "cancelled/status.json").read_text())["state"], "cancelled")
        cancelled.clear()
        runner.run(self.task("after_cancel"), cancel_event=cancelled)

    def test_mid_frame_cancellation_resets_dpb_and_closes_previews(self):
        runner = self.runner()
        cancelled = threading.Event()
        original = FakeCodec.frame

        def frame(codec, header, index, **kwargs):
            result = original(codec, header, index, **kwargs)
            if index == 1:
                cancelled.set()
            return result

        with patch.object(FakeCodec, "frame", frame), self.assertRaises(TaskCancelled):
            runner.run(self.task("mid_cancel"), cancel_event=cancelled)
        self.assertFalse((self.root / "mid_cancel/stream.glcv").exists())
        self.assertFalse(list((self.root / "mid_cancel").rglob("*_preview.mp4")))
        self.assertEqual(runner.codec.dpb, {"ref_latent": None, "ref_y": None})
        cancelled.clear()
        runner.run(self.task("after_mid_cancel"), cancel_event=cancelled)

    def test_concurrent_callers_are_serialized_on_one_model(self):
        runner, errors = self.runner(), []
        start = threading.Barrier(3)

        def call(name):
            start.wait()
            try:
                runner.run(self.task(name))
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=call, args=(name,)) for name in ("thread1", "thread2")]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join(timeout=30)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(FakeCodec.instances), 1)
        self.assertEqual((self.root / "thread1/stream.glcv").read_bytes(), (self.root / "thread2/stream.glcv").read_bytes())

    def test_receiver_guard_is_scoped_and_rejects_media_reads(self):
        with receiver_audit() as reads:
            with self.assertRaisesRegex(RuntimeError, "attempted"):
                self.source.read_bytes()
            self.assertEqual(len(reads), 1)
        self.assertGreater(len(self.source.read_bytes()), 0)

    def test_raw_preview_keeps_all_frames_and_publishes_only_when_finished(self):
        output = self.root / "preview.mp4"
        preview = RawPreview(FFMPEG, output, 47, 31, 12.5)
        try:
            for i in range(9):
                preview.add(np.full((31, 47, 3), 10 * i, np.uint8))
            self.assertFalse(output.exists())
            self.assertEqual(preview.pending.maxsize, 2)
            preview.finish()
        finally:
            preview.abort()
        probe = subprocess.check_output([str(FFMPEG.with_name("ffprobe.exe")), "-v", "error", "-count_frames",
            "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate", "-of", "json", str(output)], text=True)
        stream = json.loads(probe)["streams"][0]
        self.assertEqual((stream["width"], stream["height"], int(stream["nb_read_frames"])), (48, 32, 9))
        self.assertEqual(stream["avg_frame_rate"], "25/2")

    def test_preview_launch_failure_and_corrupt_stream_are_not_ready(self):
        with self.assertRaises(OSError):
            RawPreview(self.root / "missing.exe", self.root / "failed.mp4", 64, 64, 25)
        stream = self.root / "broken.glcv"
        stream.write_bytes(bytes(64))
        runner = self.runner("decode")
        with self.assertRaises(ValueError):
            runner.run(self.task("corrupt", input=str(stream)))
        status = json.loads((self.root / "corrupt/status.json").read_text())
        self.assertEqual(status["state"], "failed")
        self.assertEqual(runner.codec.dpb, {"ref_latent": None, "ref_y": None})

    def test_bad_sequence_checksum_cannot_publish_a_preview(self):
        self.runner().run(self.task("sender"))
        original = self.root / "sender/stream.glcv"
        raw = bytearray(original.read_bytes())
        raw[-96] ^= 1
        raw[-32:] = hashlib.sha256(raw[:-32]).digest()
        corrupt = self.root / "wrong_sequence.glcv"
        corrupt.write_bytes(raw)
        runner = self.runner("decode")
        with self.assertRaisesRegex(ValueError, "aggregate differs"):
            runner.run(self.task("bad_sequence", input=str(corrupt)))
        self.assertFalse((self.root / "bad_sequence/receiver/reconstruction_preview.mp4").exists())
        self.assertEqual(json.loads((self.root / "bad_sequence/status.json").read_text())["state"], "failed")
        diagnostic = json.loads((self.root / "bad_sequence/receiver/reconstruction_diagnostics.json").read_text())
        self.assertEqual(diagnostic["mismatched_channels"], ["latent_sha256"])
        self.assertTrue(diagnostic["sequence_matches"]["rgb_sha256"])
        self.assertEqual(diagnostic["completed_frames"], diagnostic["total_frames"])
        self.assertFalse(diagnostic["cross_device_accepted"])
        runner.run(self.task("after_bad_sequence", input=str(original)))

    def test_rgb_only_difference_publishes_explicit_preview_and_resets_next_task(self):
        self.runner().run(self.task("preview_sender"))
        stream = self.root / "preview_sender/stream.glcv"
        original = FakeCodec.frame
        def differing(codec, header, index, x=None, packet=None):
            rgb, checked = original(codec, header, index, x, packet)
            if index == 0:
                rgb[0, 0, 0] += 1
                checked["rgb_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
            return rgb, checked
        runner = self.runner("decode")
        with patch.object(FakeCodec, "frame", differing):
            for policy in ("strict", "device_pair_test"):
                with self.assertRaisesRegex(ValueError, "aggregate differs"):
                    runner.run(self.task(policy, input=str(stream), compatibility_policy=policy))
                self.assertFalse((self.root / policy / "receiver/reconstruction_preview.mp4").exists())
            runner.run(self.task("rgb_preview", input=str(stream), compatibility_policy="reference_preview"))
        folder = self.root / "rgb_preview/receiver"
        result = json.loads((folder / "decode.json").read_text())
        self.assertTrue((folder / "reconstruction_preview.mp4").is_file())
        self.assertFalse(result["all_frame_checksums_match"])
        self.assertFalse(result["arithmetic_compatibility"]["reconstruction_verified"])
        self.assertEqual(result["reconstruction_verification"]["status"], "reference_verified_preview")
        self.assertEqual(result["source_media_reads"], [])
        diagnostic = json.loads((folder / "reconstruction_diagnostics.json").read_text())
        self.assertEqual(diagnostic["mismatched_channels"], ["rgb_sha256"])
        self.assertTrue(all(r["sender_checksums_match"] is None for r in result["frames"]))
        runner.run(self.task("strict_after_preview", input=str(stream)))
        restored = json.loads((self.root / "strict_after_preview/receiver/decode.json").read_text())
        self.assertTrue(restored["all_frame_checksums_match"])
        self.assertIsNone(restored["reconstruction_verification"]["warning"])

    def test_reference_preview_still_blocks_latent_difference(self):
        self.runner().run(self.task("latent_sender"))
        original = FakeCodec.frame
        def differing(codec, header, index, x=None, packet=None):
            rgb, checked = original(codec, header, index, x, packet)
            checked["latent_sha256"] = "0" * 64
            return rgb, checked
        with patch.object(FakeCodec, "frame", differing), self.assertRaisesRegex(ValueError, "aggregate differs"):
            self.runner("decode").run(self.task("latent_preview", input=str(self.root / "latent_sender/stream.glcv"),
                                               compatibility_policy="reference_preview"))
        self.assertFalse((self.root / "latent_preview/receiver/reconstruction_preview.mp4").exists())

    def test_ffmpeg_exit_is_reported_and_leaves_no_ready_video(self):
        target = self.root / "killed.mp4"
        preview = RawPreview(FFMPEG, target, 64, 64, 25)
        try:
            preview.process.kill()
            preview.process.wait(timeout=5)
            with self.assertRaises((RuntimeError, BrokenPipeError)):
                preview.add(np.zeros((64, 64, 3), np.uint8))
                preview.finish()
            self.assertFalse(target.exists())
        finally:
            preview.abort()

    def test_worker_rejects_existing_output_without_modifying_its_evidence(self):
        from c1_worker import execute
        runner = self.runner()
        task = self.task("existing", id="existing")
        runner.run(task)
        folder = Path(task["output"])
        before = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in folder.rglob("*") if p.is_file()}
        with patch("c1_worker.gpu_snapshot", return_value={}), self.assertRaises(FileExistsError):
            execute(runner, task, time.perf_counter(), False, {})
        after = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in folder.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_cold_process_endpoints_include_pre_timer_preparation(self):
        from c1_worker import execute
        for implementation in ("c1", "legacy"):
            runner = self.runner()
            options = runner.options

            def delayed_options(*args):
                time.sleep(0.05)  # stand in for first CUDA/context preparation
                return options(*args)

            task = self.task("cold_" + implementation, id=implementation, implementation=implementation)
            with patch.object(runner, "options", delayed_options), patch("c1_worker.gpu_snapshot", return_value={}):
                timing = execute(runner, task, time.perf_counter(), True, {})
            # The old formula omitted this delay despite correct inner timings.
            self.assertGreaterEqual(timing["process_launch_to_view_ready_seconds"] -
                                    timing["task_seconds"]["offline_view_ready_seconds"], 0.045)


if __name__ == "__main__":
    unittest.main()
