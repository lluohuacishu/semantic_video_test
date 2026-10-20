from contextlib import ExitStack
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from c4_config import load_settings
from c5_loopback import create_endpoints


class LoopbackTests(unittest.TestCase):
    def test_pairs_local_endpoints_without_starting_gpu(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            with patch("c5_loopback.Endpoint.preflight") as preflight, patch("c5_loopback.Endpoint.execute") as execute:
                servers, urls = create_endpoints(stack, load_settings()["runtime"], temp, 0, 0, 0)
                tx, rx, data = servers
                self.assertTrue(all(server.server_address[0] == "127.0.0.1" for server in servers))
                self.assertEqual(urls["receiver"].rstrip("/"), tx.app.peer.url)
                self.assertEqual(rx.app.token, tx.app.peer.token)
                self.assertNotEqual(tx.app.directory, rx.app.directory)
                self.assertIs(data.app, rx.app)
                self.assertEqual("max", rx.app.rate_mode)
                self.assertEqual("cold", tx.app.config()["worker_mode"])
                self.assertEqual(urls, rx.app.config()["loopback_urls"])
                preflight.assert_not_called()
                execute.assert_not_called()

    def test_occupied_port_does_not_touch_job_stores(self):
        with tempfile.TemporaryDirectory() as temp, socket.socket() as busy, ExitStack() as stack:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            root = Path(temp) / "untouched"
            with self.assertRaises(OSError):
                create_endpoints(stack, load_settings()["runtime"], root, 0, busy.getsockname()[1], 0)
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
