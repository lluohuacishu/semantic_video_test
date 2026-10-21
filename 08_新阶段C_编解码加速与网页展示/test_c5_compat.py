import copy
import unittest

from c5_compat import check_arithmetic, reconstruction_verdict


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.sender = {"implementation": "glcv-lab-f32-v1", "torch": "2.5.1+cu124",
                       "cuda": "12.4", "cudnn": 90100, "device": "GPU A",
                       "capability": [8, 9], "tf32": False, "deterministic": True}
        self.receiver = {**self.sender, "device": "GPU B"}

    def test_strict_identity_is_not_decode_success(self):
        result = check_arithmetic(self.sender, self.sender)
        self.assertEqual("exact_profile", result["admission"])
        self.assertFalse(result["reconstruction_verified"])

    def test_default_rejects_different_gpu(self):
        with self.assertRaisesRegex(ValueError, "profiles differ"):
            check_arithmetic(self.sender, self.receiver)

    def test_explicit_diagnostic_admits_name_only_without_mutation(self):
        original = copy.deepcopy(self.receiver)
        result = check_arithmetic(self.sender, self.receiver, "device_pair_test")
        self.assertEqual("device_pair_diagnostic", result["admission"])
        self.assertFalse(result["reconstruction_verified"])
        self.assertEqual(original, self.receiver)

    def test_core_differences_never_admitted(self):
        for field, value in (("torch", "other"), ("cuda", "other"), ("cudnn", 0),
                             ("capability", [8, 6]), ("tf32", True), ("deterministic", False),
                             ("implementation", "other")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_arithmetic(self.sender, {**self.receiver, field: value}, "device_pair_test")

    def test_incomplete_or_unknown_profiles_rejected_in_diagnostic(self):
        for profile in ({"device": "GPU B"}, {**self.receiver, "unrecognized": True}):
            with self.assertRaises(ValueError):
                check_arithmetic(self.sender, profile, "device_pair_test")

    def test_v1_and_cpu_do_not_enter_pair_diagnostic(self):
        with self.assertRaises(ValueError):
            check_arithmetic(self.sender, self.receiver, "device_pair_test", 1)
        with self.assertRaises(ValueError):
            check_arithmetic({**self.sender, "capability": None},
                             {**self.receiver, "capability": None}, "device_pair_test")

    def test_unknown_policy_rejected_even_for_identical_profiles(self):
        with self.assertRaises(ValueError):
            check_arithmetic(self.sender, self.sender, "ignore_checks")

    def test_reference_preview_admits_only_same_arithmetic_and_v2(self):
        result = check_arithmetic(self.sender, self.receiver, "reference_preview")
        self.assertEqual(result["admission"], "reference_preview_candidate")
        self.assertFalse(result["reconstruction_verified"])
        for receiver in ({**self.receiver, "torch": "different"}, {**self.receiver, "tf32": True}):
            with self.assertRaises(ValueError):
                check_arithmetic(self.sender, receiver, "reference_preview")
        with self.assertRaises(ValueError):
            check_arithmetic(self.sender, self.sender, "reference_preview", 1)

    def test_rgb_difference_is_explicit_preview_only(self):
        expected = {"latent_sha256": "1" * 64, "rgb_sha256": "2" * 64}
        actual = {**expected, "rgb_sha256": "3" * 64}
        for policy in ("strict", "device_pair_test"):
            with self.assertRaisesRegex(ValueError, "aggregate differs"):
                reconstruction_verdict(expected, actual, policy, completed_frames=8, total_frames=8)
        verdict = reconstruction_verdict(expected, actual, "reference_preview", completed_frames=8, total_frames=8)
        self.assertFalse(verdict["verified"])
        self.assertFalse(verdict["rgb_verified"])
        self.assertTrue(verdict["reference_verified"])
        self.assertIn("未验收", verdict["warning"])
        self.assertEqual(verdict["status"], "reference_verified_preview")

    def test_preview_never_accepts_latent_partial_or_missing_hashes(self):
        expected = {"latent_sha256": "1" * 64, "rgb_sha256": "2" * 64}
        for actual, count in (({**expected, "latent_sha256": "0" * 64}, 8), (expected, 7),
                              ({"latent_sha256": expected["latent_sha256"]}, 8), (expected, True)):
            with self.assertRaises(ValueError):
                reconstruction_verdict(expected, actual, "reference_preview", completed_frames=count, total_frames=8)

    def test_preview_policy_preserves_exact_matches_as_fully_verified(self):
        expected = {"latent_sha256": "1" * 64, "rgb_sha256": "2" * 64}
        result = reconstruction_verdict(expected, expected, "reference_preview", completed_frames=8, total_frames=8)
        self.assertTrue(result["verified"])
        self.assertIsNone(result["warning"])


if __name__ == "__main__":
    unittest.main()
