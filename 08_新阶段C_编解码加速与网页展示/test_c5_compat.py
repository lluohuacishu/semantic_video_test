import copy
import unittest

from c5_compat import check_arithmetic


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


if __name__ == "__main__":
    unittest.main()
