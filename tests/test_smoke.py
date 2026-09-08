import math
import unittest

from pipe_surrogate_demo.train import DemoConfig, train_demo


class DemoSmokeTest(unittest.TestCase):
    def test_cpu_training_is_finite_and_reproducible(self) -> None:
        config = DemoConfig(epochs=1, samples=8, links=4, batch_size=2, seed=11, output_dir=None)
        first = train_demo(config)
        second = train_demo(config)
        self.assertEqual(first["schema"]["y"], [8, 6, 4, 5])
        self.assertTrue(math.isfinite(first["best_val_loss"]))
        self.assertEqual(first["history"], second["history"])


if __name__ == "__main__":
    unittest.main()
