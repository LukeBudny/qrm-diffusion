from pathlib import Path
import unittest

from qrm_diffusion.config import PROJECT_MAX_VRAM_GIB, load_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_sd35_config(self):
        config = load_config(ROOT / "configs/models/sd35-medium.toml")
        self.assertEqual(config.model.backend, "sd35_native")
        self.assertEqual(config.memory.max_vram_gib, PROJECT_MAX_VRAM_GIB)
        self.assertFalse(config.controller.enabled)
        self.assertEqual(
            config.controller.config, "configs/agents/sd35-qrm-timestep.toml"
        )
        self.assertEqual(config.root, ROOT)
        self.assertEqual(
            config.resolve_path(config.model.options["checkpoint"]),
            ROOT / "models/sd3.5_medium.safetensors",
        )

    def test_diffusers_config(self):
        config = load_config(ROOT / "configs/models/sdxl-base.toml")
        self.assertEqual(config.model.backend, "diffusers")
        self.assertFalse(config.qrm.enabled)


if __name__ == "__main__":
    unittest.main()
