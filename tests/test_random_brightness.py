"""Regression against frozen pre-change production code and new feature checks."""
import ast
import importlib.util
from pathlib import Path
import random
from unittest.mock import patch
import h5py
import numpy as np
from PIL import Image
import pytest
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf, open_dict
from brightness import BrightnessConfig, RandomBrightness, TransformDataset, apply_brightness, save_brightness_preview
from utils import get_img_preprocessor, get_column_normalizer, get_eval_img_preprocessor

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tests/baseline_brightness'
spec = importlib.util.spec_from_file_location('brightness_legacy_utils', BASE / 'utils.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def data_setup(path, cfg):
    """Execute production run() through loaders, without constructing models."""
    tree = ast.parse(path.read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    stop = next(i for i, n in enumerate(run.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'overhead_encoder' for t in n.targets))
    run.body = run.body[:stop] + [ast.Return(ast.Call(ast.Name('locals', ast.Load()), [], []))]
    run.decorator_list = []
    ast.fix_missing_locations(run)
    import lightning as pl
    ns = dict(torch=torch, spt=spt, swm=swm, Path=Path, open_dict=open_dict, pl=pl,
              get_column_normalizer=legacy.get_column_normalizer if path.parent == BASE else get_column_normalizer,
              get_img_preprocessor=legacy.get_img_preprocessor if path.parent == BASE else get_img_preprocessor,
              BrightnessConfig=BrightnessConfig, TransformDataset=TransformDataset)
    exec(compile(ast.Module(body=[run], type_ignores=[]), str(path), 'exec'), ns)
    return ns['run'](OmegaConf.create(cfg))


@pytest.fixture
def hdf5(tmp_path):
    path = tmp_path / 'observations.h5'
    rng = np.random.default_rng(12)
    with h5py.File(path, 'w') as f:
        f['ep_len'], f['ep_offset'] = [24, 24], [0, 24]
        for key in ('pixels', 'wrist_pixels'):
            f[key] = rng.integers(20, 180, (48, 16, 20, 3), dtype=np.uint8)
        for key in ('action', 'action_cartesian', 'proprio'):
            f[key] = rng.normal(size=(48, 4)).astype(np.float32)
    return path


def config(path, enabled=None, frameskip=1, action='action_cartesian', workers=0):
    cfg = dict(learning_seed=41, seed=3072, img_size=12, train_split=.75,
               data=dict(dataset=dict(path=str(path), num_steps=3, frameskip=frameskip,
                                      keys_to_load=['pixels', 'wrist_pixels', action, 'proprio'],
                                      keys_to_cache=[action, 'proprio'])),
               wm=dict(prop_dim=4), loader=dict(batch_size=3, num_workers=workers))
    if enabled is not None:
        cfg['augmentation'] = dict(random_brightness=dict(enabled=enabled))
    return cfg


@pytest.mark.parametrize('key', ['pixels', 'wrist_pixels'])
@pytest.mark.parametrize('encoding', ['uint8', 'float255', 'unit'])
def test_A_preprocessor_bitwise(key, encoding):
    x = torch.arange(3*3*16*20).reshape(3, 3, 16, 20).remainder(256).to(torch.uint8)
    if encoding == 'unit':
        x = x.float()/255
    elif encoding == 'float255':
        x = x.float()
    expected = legacy.get_img_preprocessor(key, key, 12)({key: x.clone()})[key]
    for kwargs in ({}, dict(random_bright_enabled=False)):
        actual = get_img_preprocessor(key, key, 12, **kwargs)({key: x.clone()})[key]
        assert torch.equal(expected, actual)


def assert_batches_equal(first, second):
    count = 0
    first, second = list(first), list(second)
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert a.keys() == b.keys()
        for key in a:
            assert torch.equal(a[key], b[key]), key
        count += 1
    assert count >= 2


@pytest.mark.parametrize('enabled', [None, False, True])
@pytest.mark.parametrize('action,frameskip', [('action_cartesian', 1), ('action', 2)])
def test_B_C_D_F_J_data_regression(hdf5, enabled, action, frameskip):
    old = data_setup(BASE/'train.py', config(hdf5, frameskip=frameskip, action=action))
    old_global_rng = torch.get_rng_state().clone()
    new = data_setup(ROOT/'train.py', config(hdf5, enabled, frameskip, action))
    assert torch.equal(old_global_rng, torch.get_rng_state())
    assert torch.equal(old['rnd_gen'].get_state(), new['rnd_gen'].get_state())
    for key in ('train_set', 'val_set'):
        subset = new[key].dataset if enabled else new[key]
        assert subset.indices == old[key].indices
        if not enabled:
            assert type(new[key]) is type(old[key])
            assert type(new[key].dataset) is type(old[key].dataset)
            assert new[key].dataset.transform is new['transform']
        else:
            assert isinstance(new[key], TransformDataset)
            assert new['dataset'].transform is None
    assert_batches_equal(old['val'], new['val'])
    assert_batches_equal(new['val'], new['val'])
    if not enabled:
        assert_batches_equal(old['train'], new['train'])
        assert torch.equal(old['rnd_gen'].get_state(), new['rnd_gen'].get_state())
    else:
        a, b = new['train_set'][0], new['train_set'][0]
        for key in ('pixels', 'wrist_pixels'):
            assert not torch.equal(a[key], b[key])
        for key in (action, 'proprio'):
            assert torch.equal(a[key], b[key])
            assert torch.equal(a[key], old['train_set'][0][key])


def test_E_eval_bitwise():
    image = torch.arange(3*16*20).reshape(3, 16, 20).remainder(256).to(torch.uint8)
    expected = legacy.get_img_preprocessor('pixels', 'pixels', 12)({'pixels': image})['pixels']
    assert torch.equal(expected, get_eval_img_preprocessor(12)(image))


def test_H_controlled_factors_and_views():
    images = torch.full((4, 3, 4, 5), .4)
    factors = torch.tensor([.5, .75, 1., 1.5]).reshape(4, 1, 1, 1)
    with patch('torch.Tensor.uniform_', autospec=True, return_value=factors) as sampler:
        with patch('brightness.torch.empty', wraps=torch.empty) as allocate:
            actual = RandomBrightness()(images)
    sampler.assert_called_once()
    allocate.assert_called_once_with((4, 1, 1, 1), dtype=images.dtype, device=images.device)
    assert torch.equal(actual, apply_brightness(images, factors))
    torch.manual_seed(23)
    transform = RandomBrightness()
    a, b = transform(images), transform(images)
    assert torch.unique(a[:, 0, 0, 0]).numel() == 4
    assert not torch.equal(a, b)
    assert torch.equal(a[:, 0], a[:, 1])


def test_I_range_fixed_clamping():
    image = torch.full((1000, 3, 2, 2), .1)
    factors = RandomBrightness(.5, 1.5)(image) / .1
    assert factors.min() >= .5 and factors.max() <= 1.5
    assert torch.equal(RandomBrightness(2, 2)(image), image*2)
    output = RandomBrightness(3, 3)(torch.ones_like(image))
    assert output.min() == 1 and output.max() == 1


@pytest.mark.parametrize('values', [dict(min_factor=0), dict(min_factor=-1), dict(max_factor=float('inf')),
    dict(min_factor=float('nan')), dict(min_factor=2, max_factor=1), dict(min_factor=True),
    dict(preview_steps=1), dict(preview_steps=2.5), dict(preview_steps=True), dict(enabled='false')])
def test_L_invalid_config(values):
    with pytest.raises(ValueError, match='random_brightness'):
        BrightnessConfig.from_config(dict(augmentation=dict(random_brightness=values)))


def test_L_config_defaults():
    assert BrightnessConfig.from_config({}) == BrightnessConfig()
    assert not BrightnessConfig.from_config(dict(augmentation=dict(random_brightness=dict(enabled=False)))).enabled
    assert BrightnessConfig.from_config(dict(augmentation=dict(random_brightness=dict(enabled=True)))).enabled


def test_K_preview_pixels_and_rng(hdf5, tmp_path):
    dataset = swm.data.HDF5Dataset(path=hdf5, num_steps=3)
    cfg = BrightnessConfig(enabled=True, preview_steps=5, min_factor=.25, max_factor=2)
    torch_before, np_before, py_before = torch.get_rng_state(), np.random.get_state(), random.getstate()
    save_brightness_preview(dataset, tmp_path, cfg)
    assert torch.equal(torch_before, torch.get_rng_state())
    assert np.array_equal(np_before[1], np.random.get_state()[1])
    assert np_before[0:1]+np_before[2:] == np.random.get_state()[0:1]+np.random.get_state()[2:]
    assert py_before == random.getstate()
    with h5py.File(hdf5) as f:
        for key, filename in [('pixels', 'overhead_gradient.png'), ('wrist_pixels', 'wrist_gradient.png')]:
            raw = torch.from_numpy(f[key][0]).float()/255
            with Image.open(tmp_path/'brightness_preview'/filename) as image:
                arr = np.asarray(image)
            assert arr.shape == (40, 100, 3)
            for i, factor in enumerate(np.linspace(.25, 2, 5)):
                expected = (apply_brightness(raw, float(factor))*255).round().byte().numpy()
                assert np.array_equal(arr[24:, i*20:(i+1)*20], expected)
    off = tmp_path/'off'
    save_brightness_preview(dataset, off, BrightnessConfig())
    assert not off.exists()
    with pytest.raises(RuntimeError, match='Failed to save brightness preview'):
        save_brightness_preview(dataset, hdf5, cfg)


def test_H_pipeline_position():
    x = torch.full((4, 3, 16, 20), 200, dtype=torch.uint8)
    expected = legacy.get_img_preprocessor('pixels', 'pixels', 12)({'pixels': torch.ones_like(x).float()})['pixels']
    actual = get_img_preprocessor('pixels', 'pixels', 12, random_bright_enabled=True,
                                  min_factor=2, max_factor=2)({'pixels': x})['pixels']
    assert torch.equal(expected, actual)


def test_real_multiview_preview_and_preprocessing(tmp_path):
    path = Path('/home/shonosukehida/.stable_worldmodel/datasets/flip_mug/val_ep10_tm300_multiview_play/push.h5')
    if not path.exists():
        pytest.skip('Local real multiview HDF5 unavailable')
    dataset = swm.data.HDF5Dataset(path=path, num_steps=3,
                                  keys_to_load=['pixels', 'wrist_pixels', 'action_cartesian', 'proprio'])
    sample = dataset[0]
    for key in ('pixels', 'wrist_pixels'):
        expected = legacy.get_img_preprocessor(key, key, 32)({key: sample[key]})[key]
        actual = get_img_preprocessor(key, key, 32)({key: sample[key]})[key]
        assert torch.equal(expected, actual)
    before = torch.get_rng_state()
    save_brightness_preview(dataset, tmp_path, BrightnessConfig(enabled=True))
    assert torch.equal(before, torch.get_rng_state())
    for filename in ('overhead_gradient.png', 'wrist_gradient.png'):
        with Image.open(tmp_path/'brightness_preview'/filename) as image:
            assert image.size == (640*9, 480+24)


def test_E_existing_wm_checkpoint_exact_inference():
    path = Path('/home/shonosukehida/.stable_worldmodel/checkpoints/flip_mug/ep100_tm300_multiview_play/seed_41/lewm20261006_145420/lewm_epoch_2_object.ckpt')
    if not path.exists():
        pytest.skip('Local multiview WM checkpoint unavailable')
    model = torch.load(path, map_location='cpu', weights_only=False).eval()
    image = torch.arange(3*3*32*32).reshape(3, 3, 32, 32).remainder(256).byte()
    def inputs(factory):
        batch = {key: factory(key, key, 224)({key: image.clone()})[key].unsqueeze(0)
                 for key in ('pixels', 'wrist_pixels')}
        batch['proprio'] = torch.zeros(1, 3, 8)
        batch['action_cartesian'] = torch.zeros(1, 3, 8)
        return batch
    with torch.inference_mode():
        old, new = model.encode(inputs(legacy.get_img_preprocessor)), model.encode(inputs(get_img_preprocessor))
        for key in ('emb', 'act_emb'):
            assert torch.equal(old[key], new[key])
        assert torch.equal(model.predict(old['emb'], old['act_emb']), model.predict(new['emb'], new['act_emb']))


def test_multiworker_wrapper(hdf5):
    old_cfg, new_cfg = config(hdf5, workers=2), config(hdf5, enabled=True, workers=2)
    old_cfg['loader']['timeout'] = new_cfg['loader']['timeout'] = 10
    old = data_setup(BASE/'train.py', old_cfg)
    new = data_setup(ROOT/'train.py', new_cfg)
    assert_batches_equal(old['val'], new['val'])
    batches = list(new['train'])
    assert len(batches) >= 2
    assert batches[0]['pixels'].shape == (3, 3, 3, 12, 15)


def test_yaml_default_disabled():
    cfg = OmegaConf.load(ROOT/'config/train/lewm.yaml')
    assert BrightnessConfig.from_config(cfg) == BrightnessConfig()
