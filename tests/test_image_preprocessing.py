"""Train/probing equivalence and diffusion preprocessing regression checks."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torchvision.transforms import v2 as transforms
import stable_pretraining as spt
from utils import get_img_preprocessor, get_eval_img_preprocessor


def load_transforms():
    # Load production factories without initializing robot/environment imports.
    path = Path(__file__).resolve().parents[1] / "eval_real_robot.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ("img_transform", "dp_img_transform")]
    namespace = dict(torch=torch, transforms=transforms, spt=spt,
                     get_eval_img_preprocessor=get_eval_img_preprocessor)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class ImagePreprocessingTests(unittest.TestCase):
    def test_train_probe_equivalence(self):
        factories = load_transforms()
        cfg = SimpleNamespace(eval=SimpleNamespace(img_size=32))
        probe = factories["img_transform"](cfg)
        torch.manual_seed(42)
        image = torch.randint(0, 256, (3, 48, 64), dtype=torch.uint8)
        outputs = []
        for label, x in [("uint8 [0,255]", image),
                         ("float32 [0,255]", image.float()),
                         ("float32 [0,1]", image.float() / 255)]:
            with self.subTest(label=label):
                train = get_img_preprocessor("pixels", "pixels", 32)
                expected = train({"pixels": x.clone()})["pixels"]
                actual = probe(x.clone())
                diff = (expected - actual).abs()
                print(f"{label}: max_abs_diff={diff.max().item()}, mean_abs_diff={diff.mean().item()}")
                self.assertLess(diff.max().item(), 1e-5)
                outputs.append(actual)
        for output in outputs[1:]:
            torch.testing.assert_close(outputs[0], output, rtol=0, atol=0)

    def test_old_float_scaling_regression(self):
        from utils import scale_to_unit_range
        image = torch.tensor([0., 128., 255.]).reshape(3, 1, 1)
        old = transforms.ToDtype(torch.float32, scale=True)(image)
        new = scale_to_unit_range(image)
        print(f"Before Normalize: old=[{old.min().item()}, {old.max().item()}], "
              f"new=[{new.min().item()}, {new.max().item()}]")
        torch.testing.assert_close(old, image)
        torch.testing.assert_close(new, image / 255.)

    def test_invalid_ranges(self):
        probe = get_eval_img_preprocessor(32)
        for value in (-1., 256., float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                probe(torch.full((3, 16, 16), value))

    def test_diffusion_transform_unchanged(self):
        cfg = SimpleNamespace(eval=SimpleNamespace(img_size=32))
        actual = load_transforms()["dp_img_transform"](cfg)
        expected = transforms.Compose([
            transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=32),
        ])
        for dtype in (torch.uint8, torch.float32):
            x = torch.randint(0, 256, (3, 48, 64)).to(dtype)
            torch.testing.assert_close(actual(x), expected(x), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
