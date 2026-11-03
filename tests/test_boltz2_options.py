import unittest

from sandbox.broker import MAX_BOLTZ2_OPTIONS_BYTES, validate_boltz2_options


class Boltz2OptionsTest(unittest.TestCase):
    def test_accepts_complete_and_future_options(self):
        options = {
            "recycling_steps": 1,
            "sampling_steps": 50,
            "diffusion_samples": 1,
            "recycling_steps_affinity": 2,
            "sampling_steps_affinity": 50,
            "diffusion_samples_affinity": 1,
            "step_scale": 1.638,
            "skip_affinity_structure": True,
            "skip_affinity_confidence": False,
            "max_parallel_samples": 1,
            "token_precision": "float32",
            "batched_diffusion": True,
            "future_boltz2_option": {"enabled": True, "schedule": [1, 2, 3]},
        }
        self.assertEqual(validate_boltz2_options(options), options)

    def test_rejects_non_object_and_non_json_values(self):
        with self.assertRaisesRegex(ValueError, "object"):
            validate_boltz2_options(["sampling_steps", 50])
        with self.assertRaisesRegex(ValueError, "finite JSON"):
            validate_boltz2_options({"sampling_steps": float("nan")})
        with self.assertRaisesRegex(ValueError, "finite JSON"):
            validate_boltz2_options({"callback": object()})

    def test_rejects_oversized_configuration(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            validate_boltz2_options({"value": "x" * MAX_BOLTZ2_OPTIONS_BYTES})


if __name__ == "__main__":
    unittest.main()
