# Random brightness regression baseline

`utils.py` and `train.py` are byte-for-byte snapshots of LeWorldModel commit
`d6cb1dc1179770fdae48f4433c1b8d40c2194b70` on `real_robot_multiview`, captured
before the brightness implementation. Do not update these when changing the new
implementation: the numerical regression tests intentionally compare against
this historical behavior.

`test_random_brightness.py` imports the frozen utility functions and executes the
actual dataset/DataLoader section of the frozen and current `run()` functions
through AST extraction. This avoids constructing a model or importing robot
hardware while still checking the production split and loader code.

The synthetic fixtures are real HDF5 files loaded with the unmodified
stable-worldmodel HDF5Dataset. Optional local tests additionally read a real
multiview recording and a pre-existing multiview WM checkpoint; these are skipped
with explicit reasons on machines where those files are absent.
