from __future__ import annotations

import re
import unittest
from pathlib import Path

from enhanced_mae.cli import build_parser
from enhanced_mae.runner import TRAINERS, TRAINING_ROOT


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class RepositoryQualityTests(unittest.TestCase):
    def test_registered_trainers_exist(self) -> None:
        for name, spec in TRAINERS.items():
            with self.subTest(name=name):
                self.assertTrue((TRAINING_ROOT / spec.script).is_file())

    def test_research_scripts_have_no_machine_paths(self) -> None:
        offenders: list[str] = []
        fixed_gpu = re.compile(r'device:\s*str\s*=\s*"cuda:\d+"')
        for path in (REPOSITORY_ROOT / "experiments").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if "/home/ubuntu/" in source or fixed_gpu.search(source):
                offenders.append(str(path.relative_to(REPOSITORY_ROOT)))
        self.assertEqual(offenders, [])

    def test_cli_exposes_expected_commands(self) -> None:
        help_text = build_parser().format_help()
        for command in ("train", "evaluate", "check-data", "smoke-test"):
            self.assertIn(command, help_text)


if __name__ == "__main__":
    unittest.main()
