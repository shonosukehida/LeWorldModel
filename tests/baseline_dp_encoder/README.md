Frozen sources for the encoder-selection compatibility tests:

- LeWorldModel: fff51a42d4b22057ae29972d3e3ec3f1bf90df70
- stable-worldmodel: 95fc0dfe4f890de610593e55440d27d0059b2db4

obs_encoder.py is an exact pre-change copy. eval_real_robot.py and train_policy.py
contain the pre-change loader/writer functions extracted using ast.unparse.
Tests execute these functions without importing robot hardware modules. Policy,
normalization, U-Net and DDPM implementations are shared because they are unchanged.
