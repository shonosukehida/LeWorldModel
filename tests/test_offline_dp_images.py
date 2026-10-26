"""Verify dataset pixels all the way through the actual DP preprocessing API."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from offline_dp_eval import dataset_images_to_hwc, load_episode, evaluate_episode, evaluate_multi_sample
from utils import get_img_preprocessor
from stable_worldmodel.policy import BasePolicy
from test_image_preprocessing import load_transforms


class OfflineImageTests(unittest.TestCase):
    def test_shape_dtype_and_range_validation(self):
        actual_shape = np.full((1, 3, 480, 640), 127.5, np.float32)
        self.assertEqual(dataset_images_to_hwc(actual_shape, 'pixels').shape, (1, 480, 640, 3))
        for dtype in (np.float32, np.uint8):
            chw = np.full((2, 3, 8, 10), 21, dtype=dtype)
            hwc = dataset_images_to_hwc(chw, 'pixels')
            self.assertEqual(hwc.shape, (2, 8, 10, 3))
            self.assertEqual(hwc.dtype, dtype)
            np.testing.assert_array_equal(hwc, np.moveaxis(chw, 1, -1))
            self.assertIs(dataset_images_to_hwc(hwc, 'pixels'), hwc)
        for shape in ((2, 1, 8, 10), (3, 8, 10), (2, 3, 0, 10), (2, 3, 8, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                dataset_images_to_hwc(np.zeros(shape, np.float32), 'pixels')
        for dtype in (np.float64, np.int16, np.bool_):
            with self.assertRaises(ValueError):
                dataset_images_to_hwc(np.zeros((2, 3, 8, 10), dtype=dtype), 'pixels')
        for value in (-.01, 255.01, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite raw'):
                dataset_images_to_hwc(np.full((2, 3, 8, 10), value, np.float32), 'wrist_pixels')

    def test_both_modes_match_training_at_encoder_input(self):
        # Fractional values expose lossy uint8 casts. Non-square images expose
        # incorrect axes; distinct channels/views/time expose mixing and padding.
        raw = np.linspace(0, 254.75, 4*3*8*10, dtype=np.float32).reshape(4, 3, 8, 10)
        wrist = 21 + raw * (234 / 255)
        cfg = SimpleNamespace(eval=SimpleNamespace(img_size=6))
        # Use the factory selected by eval_real_robot's offline branch.
        transform = load_transforms()['img_transform'](cfg)
        seen = []
        policy = SimpleNamespace(action_dim=8, obs_horizon=3, pred_horizon=5,
                                 action_horizon=2, num_inference_steps=2,
                                 process={'action_cartesian': object()},
                                 transform={'pixels': transform, 'wrist_pixels': transform})
        def prepare_and_capture(info):
            result = BasePolicy._prepare_info(policy, dict(info))
            seen.append(result)
            return result
        def get_action(info):
            prepare_and_capture(info)
            return np.array([[0., 0, 0, 0, 0, 0, 1, .5]])
        def sample(info, num_samples, denormalize):
            self.assertTrue(denormalize)
            prepare_and_capture(info)
            result = torch.zeros(1, num_samples, 5, 8)
            result[..., 6] = 1
            return result
        policy.get_action, policy.sample_action_sequences = get_action, sample
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.h5'
            with h5py.File(path, 'w') as f:
                f['pixels'], f['wrist_pixels'] = raw, wrist
                f['episode_idx'], f['step_idx'] = [7]*4, np.arange(4)
                gt = np.zeros((4, 8)); gt[:, 6] = 1
                f['action_cartesian'] = gt
            data, _, _ = load_episode(path, 7)
            np.testing.assert_array_equal(data['pixels'], raw.transpose(0, 2, 3, 1))
            with patch('matplotlib.figure.Figure.savefig'):
                evaluate_episode(policy, path, 7, Path(tmp) / 'episode', {})
                evaluate_multi_sample(policy, path, 7, 1, 2, 42, Path(tmp) / 'multi', {})
            for output, row in zip(seen, (0, 1, 2, 3, 1)):
                rows = np.maximum(np.arange(row-2, row+1), 0)
                for key, images in (('pixels', raw), ('wrist_pixels', wrist)):
                    train = get_img_preprocessor(key, key, 6)
                    expected = train({key: torch.from_numpy(images[rows].copy())})[key]
                    self.assertEqual(tuple(output[key].shape), (1, 3, 3, 6, 7))
                    torch.testing.assert_close(output[key][0], expected, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(output[key]).all())

    def test_float_scaling_and_existing_robot_transform_are_distinct(self):
        cfg = SimpleNamespace(eval=SimpleNamespace(img_size=8))
        factories = load_transforms()
        image = torch.full((3, 8, 10), 127.5)
        offline = factories['img_transform'](cfg)(image)
        train = get_img_preprocessor('pixels', 'pixels', 8)({'pixels': image})['pixels']
        torch.testing.assert_close(offline, train, rtol=0, atol=0)
        robot = factories['dp_img_transform'](cfg)(image)
        self.assertFalse(torch.allclose(offline, robot))


if __name__ == '__main__':
    unittest.main()
