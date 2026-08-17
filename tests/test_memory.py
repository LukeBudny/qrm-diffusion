import unittest

from qrm_diffusion.config import MemoryConfig, PROJECT_MAX_VRAM_GIB
from qrm_diffusion.memory import GIB, allocator_fraction, apply_cuda_memory_policy


class MemoryTests(unittest.TestCase):
    def test_28_gib_budget_on_32_gib_gpu(self):
        self.assertAlmostEqual(allocator_fraction(28.0, 32 * GIB), 28 / 32)

    def test_budget_is_capped_at_full_device(self):
        self.assertEqual(allocator_fraction(40.0, 32 * GIB), 1.0)

    def test_project_policy_rejects_more_than_28_gib_before_cuda_import(self):
        with self.assertRaisesRegex(ValueError, "project maximum"):
            apply_cuda_memory_policy(
                MemoryConfig(max_vram_gib=PROJECT_MAX_VRAM_GIB + 0.1)
            )


if __name__ == "__main__":
    unittest.main()
