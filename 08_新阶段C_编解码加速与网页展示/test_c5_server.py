"""Real HTTP/TCP endpoint lifecycle; synthetic decoder, no CUDA or rate waits."""
import copy
import http.client
import json
from pathlib import Path
import socket
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from c5_config import load_settings
from c5_server import Endpoint, DataServer, Server, Peer, inspect_stream, write
from c5_transport import send_file, wire_header
from glc_container import StreamWriter, frame_layout


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = {"arithmetic": {"implementation": "test", "device": "test"}, "weights": {},
                    "model_revision": "test", "entropy": "test", "table_sha256": "test",
                    "backends": {"native": True, "python": True}}
        self.apps, self.servers, self.threads = [], [], []
        for role in ("receiver", "sender"):
            settings = load_settings(role)
            settings.update(data_dir=self.root / role, host="127.0.0.1", http_port=0, data_port=0, token="t" * 32)
            app = Endpoint(settings)
            app.environment = copy.deepcopy(self.env)
            self.apps.append(app)
            server = Server(("127.0.0.1", 0), app)
            self.servers.append(server)
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .02}, daemon=True)
            thread.start()
            self.threads.append(thread)
        self.rx, self.tx = self.apps
        self.data = DataServer(("127.0.0.1", 0), self.rx)
        self.rx.settings["data_port"] = self.data.server_address[1]
        thread = threading.Thread(target=self.data.serve_forever, kwargs={"poll_interval": .02}, daemon=True)
        thread.start()
        self.threads.append(thread)
        self.servers.append(self.data)
        self.tx.set_peer("http://127.0.0.1:" + str(self.servers[0].server_port), self.rx.token)
        self.stream = self.root / "valid.glcv"
        header = {"width": 64, "height": 64, "frames": 1, "fps": 12.5, "q": 0,
                  "intra_period": 64, "reset_interval": 32, "revision": "test", "entropy": "test",
                  "table_sha256": "test", "weights": {}, "arithmetic": self.env["arithmetic"], "container_version": 2}
        kind, zbytes, parts, symbols = frame_layout(header, 0)
        writer = StreamWriter(self.stream, header, version=2)
        writer.add({"kind": kind, "z": b"\0" * zbytes, "y": [struct.pack(">I", symbols) + b"\0"*12] * parts,
                    "rgb_sha256": "12"*32, "latent_sha256": "34"*32})
        writer.finish()
        self.description = inspect_stream(self.stream)
        self.rx.execute = self.fake_decode

    def tearDown(self):
        for server in self.servers:
            server.shutdown()
        for app in self.apps:
            app.close()
        for server in self.servers:
            server.server_close()
        for thread in self.threads:
            thread.join(2)
        self.temp.cleanup()

    def fake_decode(self, role, task, folder):
        self.assertEqual(role, "decode")
        self.assertEqual(Path(task["input"]), folder / "received.glcv")
        self.assertFalse((folder / "source.mp4").exists())
        stream = inspect_stream(task["input"])
        out = folder / "decode/receiver"
        out.mkdir(parents=True)
        write(out / "decode.json", {"input_sha256": stream["sha256"], "independent_decoder": True,
              "source_media_reads": [], "all_frame_checksums_match": True, "whole_file_checksum_verified": True,
              "sequence_reconstruction_checksums": stream["integrity"]["sequence_reconstruction_checksums"],
              "frames": [{}], "header": stream["header"], "arithmetic_compatibility": None})
        write(folder / "decode/timing.json", {"synthetic": True})

    def call(self, method, route, value=None, raw=None, headers=None, sender=False):
        server = self.servers[1 if sender else 0]
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        body = json.dumps(value).encode() if value is not None else raw
        connection.request(method, route, body, headers or {})
        response = connection.getresponse()
        result = response.status, response.read()
        connection.close()
        return result

    def await_terminal(self, app, identifier):
        for _ in range(300):
            job = app.get_job(identifier)
            if job["status"] in ("done", "error", "cancelled", "interrupted"):
                return job
            time.sleep(.01)
        self.fail("Job did not reach terminal state")

    def reserve(self, sender_id="a"*32):
        return self.rx.reserve_receive({"stream": self.description, "sender_id": sender_id})

    def test_open_does_not_run_preflight_or_inference(self):
        with patch.object(self.rx, "preflight") as preflight, patch.object(self.rx.executor, "submit") as submit:
            self.assertEqual(200, self.call("GET", "/")[0])
            self.assertEqual(200, self.call("GET", "/api/config")[0])
            preflight.assert_not_called()
            submit.assert_not_called()

    def test_control_auth_and_browser_origin(self):
        self.assertEqual(403, self.call("POST", "/control/preflight", {})[0])
        self.assertEqual(403, self.call("POST", "/control/preflight", {}, headers={
            "Authorization": "Bearer " + self.rx.token, "Origin": "http://evil"} )[0])
        self.assertEqual(403, self.call("GET", "/api/pairing", headers={"Host": "evil:80"})[0])
        self.assertEqual(200, self.call("POST", "/control/preflight", {}, headers={"Authorization": "Bearer " + self.rx.token})[0])

    def test_receiver_rate_snapshot_no_throttled_transfer(self):
        self.rx.preferences({"rate_mode": "20"})
        first = self.reserve()
        self.rx.preferences({"rate_mode": "40"})
        second = self.reserve("b"*32)
        self.rx.preferences({"rate_mode": "max"})
        self.assertEqual("20", first["rate_mode"])
        self.assertEqual("40", second["rate_mode"])
        self.assertEqual(first, self.reserve())
        self.rx.cancel(first["id"])
        self.rx.cancel(second["id"])

    def test_max_end_to_end_reuse_no_encoding_and_download(self):
        data = self.stream.read_bytes()
        status, body = self.call("POST", "/api/jobs", {"filename": "same.glcv", "bytes": len(data),
            "options": {"preset": "compact"}}, sender=True)
        self.assertEqual(201, status)
        identifier = json.loads(body)["id"]
        self.assertEqual(200, self.call("PUT", f"/api/jobs/{identifier}/upload", raw=data, sender=True)[0])
        with patch.object(self.tx, "execute") as encode:
            self.tx.start(identifier)
            job = self.await_terminal(self.tx, identifier)
            self.assertEqual("done", job["status"], job)
            encode.assert_not_called()
        self.assertFalse(job["result"]["encoded_this_task"])
        self.assertTrue(job["result"]["transfer"]["network_transfer_performed"])
        self.assertEqual(data, self.call("GET", f"/jobs/{identifier}/stream.glcv?download=1", sender=True)[1])
        repeat = self.tx.resend(identifier)
        self.assertEqual("stream", repeat["kind"])
        self.assertEqual(data, self.tx.source(repeat).read_bytes())
        self.tx.cancel(repeat["id"])

    def test_invalid_ticket_cannot_poison_valid_reservation(self):
        reservation = self.reserve()
        header = {**{k: self.description[k] for k in ("bytes", "sha256")}, "version": 2,
                  "job_id": reservation["id"], "ticket": "x"*40, "rate_mode": "max"}
        with socket.create_connection(self.data.server_address) as connection:
            connection.sendall(wire_header(header))
            connection.settimeout(2)
            self.assertEqual(b"", connection.recv(1))
        self.assertEqual("staged", self.rx.store.get(reservation["id"])["status"])
        result = send_file(self.stream, *self.data.server_address, reservation)
        self.assertTrue(result["verified"])
        self.assertEqual("done", self.await_terminal(self.rx, reservation["id"])["status"])

    def test_ack_lost_still_decodes_and_is_queryable(self):
        reservation = self.reserve()
        with patch("c5_server.acknowledge", side_effect=ConnectionError("lost")):
            with self.assertRaises(ConnectionError):
                send_file(self.stream, *self.data.server_address, reservation)
            job = self.await_terminal(self.rx, reservation["id"])
        self.assertEqual("done", job["status"])
        self.assertTrue(job["receipt_verified"])
        self.assertEqual("done", self.reserve()["status"])

    def test_truncated_transfer_keeps_partial_no_decode(self):
        reservation = self.reserve()
        header = {**{k: self.description[k] for k in ("bytes", "sha256")}, "version": 2,
                  "job_id": reservation["id"], "ticket": reservation["ticket"], "rate_mode": "max"}
        with patch.object(self.rx, "execute") as decode:
            with socket.create_connection(self.data.server_address) as connection:
                connection.sendall(wire_header(header) + b"short")
            job = self.await_terminal(self.rx, reservation["id"])
            self.assertEqual("error", job["status"])
            decode.assert_not_called()
        self.assertTrue((self.rx.directory / job["id"] / "received.glcv.part").exists())
        self.assertFalse((self.rx.directory / job["id"] / "received.glcv").exists())

    def test_restart_marks_reserved_interrupted(self):
        job = self.reserve()
        self.rx.cancel(job["id"])
        self.rx.store.update(job["id"], status="staged")
        # Preserve crash state without normal close's cooperative cancellation.
        self.rx.lease.close()
        replacement = Endpoint(self.rx.settings)
        try:
            self.assertEqual("interrupted", replacement.store.get(job["id"])["status"])
            self.assertFalse(replacement.tickets)
        finally:
            replacement.close()


if __name__ == "__main__":
    unittest.main()
