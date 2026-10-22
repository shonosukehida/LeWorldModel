"""DP action statistics shared by training and checkpoint restoration."""
import numpy as np
import torch
from normalization_stats import SafeStandardScaler


def validate_action_stats(state, action_dim):
    mode = state.get('action_normalization', 'zscore')
    if mode not in ('zscore', 'min_max'):
        raise ValueError(f'Unknown action_normalization: {mode!r}')
    names = ('action_mean', 'action_std') if mode == 'zscore' else ('action_min', 'action_max')
    values = []
    for name in names:
        if name not in state:
            raise ValueError(f'Missing action statistic: {name}')
        try:
            value = torch.as_tensor(state[name]).detach().cpu()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f'{name}: invalid action statistic') from exc
        if not value.is_floating_point():
            raise ValueError(f'{name}: expected floating point statistics')
        if tuple(value.shape) not in ((action_dim,), (1, action_dim)):
            raise ValueError(f'{name}: expected ({action_dim},) or (1, {action_dim}), got {tuple(value.shape)}')
        if not torch.isfinite(value).all():
            raise ValueError(f'{name}: NaN/Inf statistics')
        values.append(value)
    if values[0].shape != values[1].shape:
        raise ValueError('Action statistic shapes disagree')
    if mode == 'zscore' and (values[1] <= 0).any():
        raise ValueError('action_std must be positive')
    if mode == 'min_max' and (values[1] < values[0]).any():
        raise ValueError('action_max must be >= action_min')
    return mode


def fit_action_stats(dataset, key, mode='zscore'):
    if mode not in ('zscore', 'min_max'):
        raise ValueError(f'Unknown action_normalization: {mode!r}')
    data = torch.from_numpy(np.array(dataset.get_col_data(key)))
    if data.ndim != 2 or not data.shape[1] or not len(data):
        raise ValueError('Expected nonempty (samples, action_dim) actions')
    if not torch.isfinite(data).all():
        raise ValueError('Actions contain NaN/Inf')
    state = {'action_normalization': mode}
    if mode == 'zscore':
        mean = data.mean(0, keepdim=True).clone()
        std = data.std(0, keepdim=True).clone()
        state.update(action_mean=mean, action_std=torch.where(std < 1e-4, torch.ones_like(std), std))
    else:
        state.update(action_min=data.amin(0, keepdim=True), action_max=data.amax(0, keepdim=True))
    validate_action_stats(state, data.shape[1])
    return state


def min_max_parameters(low, high):
    # Float64 arithmetic avoids overflow and cancellation for tiny float32 ranges.
    low, high = low.double(), high.double()
    span = high - low
    # Nonconstant ranges map exactly to endpoints, including subnormal ranges.
    return low, torch.where(span == 0, torch.ones_like(span), span)


class MinMaxScaler:
    def __init__(self, state):
        self.low, self.span = min_max_parameters(state['action_min'], state['action_max'])
        self.low, self.span = self.low.numpy(), self.span.numpy()

    def transform(self, x):
        return (np.asarray(x) - self.low) / self.span * 2 - 1

    def inverse_transform(self, x):
        return (np.asarray(x) + 1) / 2 * self.span + self.low


def restore_action_processor(state, action_dim):
    mode = validate_action_stats(state, action_dim)
    if mode == 'min_max':
        return MinMaxScaler(state)
    processor = SafeStandardScaler(eps=1e-4)
    processor.mean_ = state['action_mean'].cpu().numpy()
    processor.scale_ = state['action_std'].cpu().numpy()
    return processor


class ActionTransform:
    def __init__(self, state, key):
        self.state, self.key = state, key

    def __call__(self, sample):
        x = sample[self.key]
        if self.state['action_normalization'] == 'zscore':
            result = (x - self.state['action_mean'].to(x.device)) / self.state['action_std'].to(x.device)
        else:
            low, span = min_max_parameters(self.state['action_min'], self.state['action_max'])
            result = (x.double() - low.to(x.device)) / span.to(x.device) * 2 - 1
        sample[self.key] = result.float()
        return sample


def normalizer_artifact(state, key):
    """Keep historical mean/std aliases in the standalone artifact."""
    result = dict(state, action_key=key)
    for name in ('mean', 'std', 'min', 'max'):
        if 'action_' + name in state:
            result[name] = state['action_' + name]
    return result
