import math
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

import torch

from pipe_surrogate_demo.data import make_synthetic_pipe_data
from pipe_surrogate_demo.list_task_sets import group_task_sets, main as list_task_sets_main
from pipe_surrogate_demo.loss import TaskwiseMSELoss
from pipe_surrogate_demo.schema import TARGET_NAMES, PipeSchema, all_task_sets
from pipe_surrogate_demo.train import DemoConfig, _write_outputs, train_demo


class DemoSmokeTest(unittest.TestCase):
    def test_cpu_training_is_finite_and_reproducible(self) -> None:
        config = DemoConfig(epochs=1, samples=8, links=4, batch_size=2, seed=11, output_dir=None)
        first = train_demo(config)
        second = train_demo(config)
        self.assertEqual(first["schema"]["y"], [8, 6, 4, 5])
        self.assertTrue(math.isfinite(first["best_val_loss"]))
        self.assertEqual(first["history"], second["history"])
        self.assertEqual(first["tasks"], ["Rate", "Depth", "Vel", "Vol", "Cap"])
        self.assertEqual(first["checkpoint_selection"], "lowest_validation_loss")

    def test_task_grouping_has_all_non_empty_subsets(self) -> None:
        task_sets = all_task_sets()
        self.assertEqual(len(task_sets), 31)
        self.assertEqual(len({tuple(task_set) for task_set in task_sets}), 31)
        self.assertEqual(sum(len(task_set) == 1 for task_set in task_sets), 5)
        self.assertEqual(sum(2 <= len(task_set) <= 4 for task_set in task_sets), 25)
        self.assertEqual(sum(len(task_set) == 5 for task_set in task_sets), 1)

    def test_task_listing_prints_all_31_combinations(self) -> None:
        groups = group_task_sets()
        self.assertEqual(
            [len(groups[name]) for name in ("STL", "Selective", "Full")],
            [5, 25, 1],
        )
        output = StringIO()
        with redirect_stdout(output):
            list_task_sets_main()
        lines = output.getvalue().splitlines()
        combinations = [line for line in lines if line in TARGET_NAMES or "+" in line]
        self.assertEqual(len(combinations), 31)
        self.assertIn("Rate+Depth+Vel", combinations)
        self.assertIn("Rate+Depth+Vel+Vol", combinations)

    def test_selected_tasks_change_prediction_width(self) -> None:
        config = DemoConfig(
            epochs=1,
            samples=8,
            links=4,
            batch_size=2,
            seed=11,
            tasks=("Rate", "Vel"),
            output_dir=None,
        )
        result = train_demo(config)
        self.assertEqual(result["tasks"], ["Rate", "Vel"])
        self.assertEqual(result["schema"]["prediction"], [8, 6, 4, 2])

    def test_synthetic_static_and_dynamic_channels(self) -> None:
        x, _ = make_synthetic_pipe_data(3, PipeSchema(links=4), seed=11)
        self.assertTrue(torch.allclose(x[0, :, 0, 0], x[0, :, 1, 0]))
        self.assertTrue(torch.allclose(x[0, :, :, 2:], x[1, :, :, 2:]))
        self.assertTrue(torch.all(x[:, 1:, :, 1] >= x[:, :-1, :, 1]))

    def test_taskwise_mse_is_sum_not_mean(self) -> None:
        prediction = torch.zeros((1, 2, 3, 2))
        target = torch.zeros_like(prediction)
        target[..., 0] = 1.0
        target[..., 1] = 2.0
        self.assertAlmostEqual(float(TaskwiseMSELoss()(prediction, target)), 5.0)

    def test_output_guard_refuses_non_empty_checkpoint_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "real_checkpoints"
            output_dir.mkdir()
            (output_dir / "best_model.pth").write_bytes(b"sentinel")
            config = DemoConfig(output_dir=output_dir)
            with self.assertRaises(FileExistsError):
                _write_outputs(config, {}, {})
            self.assertEqual((output_dir / "best_model.pth").read_bytes(), b"sentinel")


if __name__ == "__main__":
    unittest.main()
