"""Same-row training windows and episode-local inference padding."""
import h5py
import numpy as np
import torch
from stable_worldmodel.data import HDF5Dataset
from train_imle_policy import IMLEProprioDataset
from offline_imle_eval import load_proprio_episode
from offline_imle_core import build_observation_history
from test_imle_integration import dataset


def test_training_and_inference_windows_do_not_cross_episodes(tmp_path):
    path=tmp_path/'episodes.h5'
    dataset(path,40)
    with h5py.File(path,'a') as f:
        for key,value in dict(ep_len=[20,20],ep_offset=[0,20],
                              episode_idx=np.repeat([0,7],20),step_idx=np.tile(np.arange(20),2)).items():
            del f[key]
            f[key]=value
        for row in range(40):
            f['pixels'][row]=row
            f['wrist_pixels'][row]=row
            f['proprio'][row]=row
            f['action_cartesian'][row]=row
    raw=HDF5Dataset(path=str(path),num_steps=16,frameskip=1,
                    keys_to_load=['pixels','wrist_pixels','proprio','action_cartesian'])
    wrapped=IMLEProprioDataset(raw,'action_cartesian',8,lambda x:x,lambda x:x)
    assert len(wrapped)==10
    for index,(episode,start) in enumerate(raw.clip_indices):
        sample=wrapped[index]
        rows=torch.arange(episode*20+start,episode*20+start+16)
        torch.testing.assert_close(sample['proprio'][:,0],rows[:8].float())
        torch.testing.assert_close(sample['action'][:,0],rows.float())
        for key in ('pixels','wrist_pixels'):
            np.testing.assert_array_equal(sample[key][:,0,0,0],rows[:8])
        assert rows[-1] < (episode+1)*20
    data,steps,_=load_proprio_episode(path,7)
    history=build_observation_history(data,0,8,('pixels','wrist_pixels','proprio'))
    np.testing.assert_array_equal(history['proprio'][0,:,0],np.full(8,20))
    np.testing.assert_array_equal(history['pixels'][0,:,0,0,0],np.full(8,20))
    assert steps[0]==0
