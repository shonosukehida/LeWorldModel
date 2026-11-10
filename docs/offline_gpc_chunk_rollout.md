# GPC chunk_rollout オフライン評価

`policy_type=gpc`、`eval.real_robot.dry_run=true`、`dry_run_mode=chunk_rollout` の場合だけ有効。
`eval.probing.exe_probe=false` を指定する。既存 Probing の動作は変更していない。
保存済み goal / goal wrist 画像と goal proprio、実測 proprio を含むデータセットが必要。
実機・カメラ初期化前に保存して return する。

## 変更範囲と互換性

- `eval_real_robot.py`: GPC chunk_rollout のオフライン分岐のみ追加。
- `offline_gpc_eval.py`: 専用 runner、評価、描画を追加。既存 RNG 保護の対象に新モードを追加。
- `tests/test_offline_gpc_chunk_rollout.py`: 候補選択、保存、描画、分岐、RNG、実チェックポイント、変更前 DP との比較。
- 本ドキュメント。

`offline_dp_eval.py` は tqdm の import とループのラップだけを変更。`stable-worldmodel`、`eval_real_robot.py`（今回の幅修正）、既存 GPC API、正規化、モデル、学習、実機ループ、既存 config のデフォルトは変更していない。作業前からの `config/eval/flip_mug.yaml` の変更は保持。commit / push / ブランチ切り替えは実施していない。

## 処理と時間軸

1. `load_episode` と `build_observation_history` で既存 DP と同じ履歴を構築。
2. 各チャンクで `dry_run_num_candidates` 本を `sample_candidates` により1回生成。seed は `dry_run_seed + chunk_index`。DP checkpoint の逆正規化を1回だけ適用。
3. 対象行の overhead / wrist / 実測 proprio、保存済み goal / goal wrist / goal proprio、実際の step_idx を `_policy_observation` に渡す。GT や未来観測は渡さない。
4. 既存 `select_candidates` を再利用。コンテナの一時コピーで sampler を保存済み候補の再生に置き換え、元の `GPCPolicy.get_action_chunk()` を実行する。既存の切り出し、WM 正規化、rollout、goal encoding、reward、torch.argmax をそのまま使用する。DP モデルの再実行はない。
5. rewards の `(1,K)` shape と有限値、選択結果と元候補の完全一致を検証。
6. GPC のチャンク幅・オフライン stride は **cfg.eval.real_robot.dp_execute_steps**（省略時1）。runner から評価関数へ明示的に渡す。整数 `1 <= dp_execute_steps <= DP action_horizon` を検証。Diffusion の幅は従来の DP action_horizon のまま。WM 評価区間は予測 index `[obs_horizon - 1, min(obs_horizon - 1 + GPC action_horizon, pred_horizon))`。可視化区間は `[obs_horizon - 1, obs_horizon - 1 + dp_execute_steps)`。WM 評価区間が表示区間を覆わないカスタム構成はエラー。
7. 実機では `dp_execute_steps` ステップ実行して再計画する。この意味は変更せず、GPC オフラインの幅も同じ実行単位に合わせる。WM 評価区間は短縮しない。
8. 末尾は valid mask で短縮。WM は既存 API と同じ全評価区間を評価する。HDF5 の mask 外だけ NaN padding、step index は -1。これは無効領域であり JSON・評価・描画には含めない。候補・reward・実測 proprio に非有限値があればエラー。

K は `cfg.eval.real_robot.dry_run_num_candidates`（省略時20）だけから取得する。`cfg.gpc.num_candidates` はオフラインでは不要。通常実機では従来どおり同設定を使う。既存 multi_sample の実装は変更していない。

## 保存内容

```
<output_dir>/offline_gpc/chunk_rollout/episode_<id>_<unique>/
  chunk_rollout.h5
  chunk_rollout_metrics.json
  chunk_rollout_position.png
  chunk_rollout_orientation.png
  chunk_rollout_gripper.png
```

HDF5: `predicted_chunks (C,K,E,8)`、`ground_truth_actions`、`step_idx`、`chunk_start_steps`、`chunk_step_idx`、`chunk_valid_mask`、`selected_candidate_indices (C,)`、`candidate_rewards (C,K)`、`selected_chunks (C,E,8)`（E = dp_execute_steps）、`representative_index`、`wm_evaluation_intervals (C,2)`。

JSON: 各チャンクの全 reward / 選択 index、既存 DP のチャンク診断指標、dataset / episode、K、seed / 各チャンク seed、各 horizon、WM 評価区間、表示区間、dp_execute_steps、設定、DP checkpoint、具体的 WM checkpoint の path / size / mtime、DP / WM の実際の正規化統計を保存する。JSON は NaN / Inf を許さない。

PNG: GT 黒、全候補薄灰、代表 index 0 青、WM 選択赤（太さ3、zorder10）。チャンクごとに独立した Line2D とし境界を接続しない。末尾が1点の赤線は点でも表示する。横軸は `Episode step (step_idx)`。

## 検証コマンド

ワークスペース `/home/shonosukehida/work` から実行:

```bash
PYTHONPATH=LeWorldModel LeWorldModel/.venv/bin/python -m pytest -q -s \
  LeWorldModel/tests/test_offline_gpc_chunk_rollout.py --tb=short

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=LeWorldModel LeWorldModel/.venv/bin/python -m pytest -q \
  LeWorldModel/tests/test_offline_gpc_eval.py \
  LeWorldModel/tests/test_offline_dp_eval.py \
  LeWorldModel/tests/test_offline_dp_images.py \
  LeWorldModel/tests/test_dp_action_chunks.py \
  LeWorldModel/tests/test_action_normalization.py \
  LeWorldModel/tests/test_image_preprocessing.py \
  LeWorldModel/tests/test_dp_encoder_selection.py --tb=short
```

新テストは開始時 HEAD `437aec5d059b4000cdaf994fc597d6f32047a335` から Git で DP コードを読み、同じモデル代替・データ・seed の実出力を比較。候補、GT、開始位置、mask、指標、HDF5 datasets、JSON（出力固有ID除外）、PNG バイト列が一致。既存 multi_sample の変更前比較、episode の履歴・保存・描画、実 GPC API との数値一致は既存テストも実行。

実チェックポイントテストは CPU で実 DP / JEPA / データ / 統計を使用し、17フレーム、20候補で複数チャンクと末尾を確認する。これはタスク達成精度の評価ではない。CUDA と全300ステップの実重量評価、実機制御、学習再開は未検証。Hydra エントリポイントのモデルロード device は従来どおり CUDA。CPU 検証は evaluator / runner の直接呼び出し。

## GPU 環境での実行例（このコマンド自体は未実行）

```bash
cd /home/shonosukehida/work/LeWorldModel
.venv/bin/python eval_real_robot.py --config-name flip_mug \
  policy_type=gpc \
  policy=/home/shonosukehida/.stable_worldmodel/checkpoints/flip_mug/ep100_tm300_multiview_play_normal_brightness/lewm \
  gpc.diffusion_checkpoint=/home/shonosukehida/.stable_worldmodel/checkpoints/flip_mug/policy/diffusion_policy/ep200_tm300_multiview_demo/seed_41/20261021_135400/latest.ckpt \
  gpc.num_candidates=5 \
  eval.real_robot.dry_run=true \
  eval.real_robot.dry_run_mode=chunk_rollout \
  eval.real_robot.dry_run_num_candidates=20 \
  eval.real_robot.dry_run_seed=42 \
  eval.real_robot.dp_execute_steps=4 \
  eval.real_robot.dry_run_episode_idx=0 \
  eval.real_robot.dry_run_dataset_path=/home/shonosukehida/.stable_worldmodel/datasets/flip_mug/ep200_tm300_multiview_demo/push.h5 \
  eval.normalization_stats_path=/home/shonosukehida/.stable_worldmodel/datasets/flip_mug/ep200_tm300_multiview_demo/process_stats.npz \
  eval.real_robot.execute=false \
  eval.probing.exe_probe=false
```

goal 画像2枚・goal proprio は利用するデータに対応した保存済み設定にする。既存 config の他マシンのパスも環境に合わせる。

## 初回実装時の結果（幅修正前の記録）

- 新テスト7件成功（17フレームの実 DP / WM / 20候補を含む）。その後追加した DP episode の開始時 Git コードとの HDF5 / metadata / JSON / 全 PNG 比較も1件成功。新テスト計8件成功。
- 既存回帰: **58 passed / 2 skipped / 10 failed / 38 subtests passed**。10 failed は既存実機ループ3テスト（うち1テストの8 subtests）で、古い FakeEnv / SafetyEnv の `_last_sdk_qpos` 欠落による。同じ開始時 Git コードを `namespace` に渡して LoopTests 4件を実行し、同じ10 errors を再現した。既存テストは削除・無効化・緩和していない。既存の補完 fixture による3テストは成功。
- Diffusion chunk_rollout の同 seed 比較は18ステップ・3候補・3チャンク（末尾2ステップ）。全候補、GT、step 対応、mask、指標、全 HDF5 datasets、JSON、PNG バイト列が一致。episode も同じ実出力比較に成功。multi_sample は既存の変更前比較テストが成功。
- 実チェックポイント: DP `ep200_tm300_multiview_demo/seed_41/20261021_135400/latest.ckpt`、WM `ep100_tm300_multiview_play_normal_brightness/lewm_object.ckpt`。候補 `(3,20,8,8)`、rewards `(3,20)`、開始 step `[0,8,16]`、valid 長 `[8,8,1]`、選択 index `[19,6,19]`。
- 実出力: `/tmp/gpc_real_checkpoint_ojuuhhaj/offline_gpc/chunk_rollout/episode_0_wfgw0x5g/`。position PNG を目視し、赤線・境界の分離・末尾の点を確認。
- 最終 Git diff / diff --check を確認。`run` 以外の全関数・クラスの AST は開始時 HEAD と一致。既存 config の Git diff も開始時の保存内容と完全一致。stable-worldmodel の作業ツリーは clean。

ログ:

- 新機能・実 checkpoint: `/tmp/gpc_chunk_final_test.log`
- episode 変更前比較: `/tmp/gpc_chunk_episode_regression.log`
- 既存回帰: `/tmp/gpc_chunk_regression_final.log`
- 開始時コードの既存 LoopTests: `/tmp/gpc_chunk_baseline_loops.log`

episode の追加比較を実行したコマンド:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=LeWorldModel \
  LeWorldModel/.venv/bin/python -m pytest -q \
  LeWorldModel/tests/test_offline_gpc_chunk_rollout.py::test_dp_episode_artifacts_against_starting_revision \
  --tb=short
```

最初の無制限 CPU thread の既存回帰実行は完走せず終了コード143となったため、上記の CPU thread 数を2にした同じ全テストで最終結果を確認した。テスト対象を除外して成功扱いにはしていない。

## チャンク幅修正・進捗表示の追加

変更関数:

- `run_offline_gpc_chunk_rollout`: `cfg.eval.real_robot.dp_execute_steps` を型変換せず評価関数へ渡す。不正な bool / 小数を整数化して受け入れない。
- `evaluate_gpc_chunk_rollout`: 整数幅を検証し、その幅で再推論・配列確保・切り出し・mask を構築。WM 選択関数と評価区間は変更しない。metadata に `action_horizon` / `dp_execute_steps` / `chunk_width` を区別して記録。
- `evaluate_chunk_rollout` (Diffusion): ローカルの `tqdm.auto` import と既存ループのラップだけを追加。数値処理は変更なし。

双方のループに `total=len(rows)`、`unit='chunk'` を指定。tqdm の iterator はループ本体が完了して次に進む際に更新されるため、GPC は WM 評価前に完了扱いにならない。例外は元のまま伝播する。依存関係は既存 `stable-worldmodel/pyproject.toml` の `tqdm` を使用し、依存追加なし。

CLI で確認した表示例（GPC は実 checkpoint、DP は合成候補）:

```
GPC chunk rollout:  40%|████      | 2/5 [00:27<00:41, 13.71s/chunk]
GPC chunk rollout: 100%|██████████| 5/5 [01:06<00:00, 13.31s/chunk]
DP chunk rollout: 100%|██████████| 3/3 [00:00<00:00, 1537.88chunk/s]
```

今回の変更ファイル:

- `offline_gpc_eval.py`
- `offline_dp_eval.py`
- `tests/test_offline_gpc_chunk_rollout.py`
- `tests/test_offline_gpc_eval.py`
- `tests/test_gpc_chunk_execute_steps.py`（新規）
- `docs/offline_gpc_chunk_rollout.md`

既存テストのソース完全一致チェックは、許可された tqdm import とラップだけを取り除いて比較する形に変更。それ以外のコード差分は引き続き拒否し、実出力比較も維持する。旧幅を前提とする GPC テストには幅を明示。実 checkpoint テストは幅4に設定。

検証コマンド（ワークスペースから）:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=LeWorldModel \
  LeWorldModel/.venv/bin/python -m pytest -q \
  LeWorldModel/tests/test_gpc_chunk_execute_steps.py \
  LeWorldModel/tests/test_offline_gpc_chunk_rollout.py \
  -k 'not real_checkpoints' --tb=short

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=LeWorldModel \
  LeWorldModel/.venv/bin/python -m pytest -q -s \
  LeWorldModel/tests/test_offline_gpc_chunk_rollout.py::test_real_checkpoints_cpu_chunk_rollout \
  --tb=short
```

追加の幅8互換テストは単独でも実行した。開始時 Git リビジョン `daadc29273f52e9ed3b37a8903f9bfbd219e3d0f` の GPC evaluator と比較し、全 HDF5 datasets・PNG・指標・JSON 相当値が一致（新たに明記した dp_execute_steps、説明文、実行固有ディレクトリを除く）。

幅修正後の最終結果:

- 専用・互換テスト: **20 passed / 1 deselected**。deselected の実 checkpoint テストは別実行で **1 passed**。
- 幅8/4/1、幅9拒否、300ステップで75チャンク、末尾、全20候補の8ステップ評価、argmax、候補の実行区間との完全一致、単一サンプリング、GT 非依存、RNG、例外伝播を確認。
- 進捗は最初と2番目の WM 評価で例外を発生させ、評価中は未完了であり、完了済みチャンクのみカウントされることを確認。
- Diffusion chunk_rollout の修正前後の候補・mask・指標・HDF5・JSON・PNG が一致。episode の保存出力比較、multi_sample の既存比較も成功。
- 既存回帰: **58 passed / 2 skipped / 10 failed / 38 subtests passed**。失敗は初回実装時と同じ実機ループ3テスト（うち8 subtests）で、FakeEnv / SafetyEnv の `_last_sdk_qpos` 欠落。開始時 Git コードでも再現。元テストは除外・緩和していない。
- 実 checkpoint: CPU、17フレーム、K=20、WM horizon=8、実行幅4。開始 step `[0,4,8,12,16]`、有効長 `[4,4,4,4,1]`。`predicted_chunks (5,20,4,8)` / `selected_chunks (5,4,8)`。WM interval は全チャンク `[7,15)`、表示 interval は `[7,11)`。
- 実出力: `/tmp/gpc_real_checkpoint_6vqfj5t4/offline_gpc/chunk_rollout/episode_0_y494wbrj/`。
- 保存ディレクトリとファイル名・他の保存項目は維持。`eval_real_robot.py` / `stable-worldmodel` / 実機推論 / checkpoint 形式に今回の差分なし。Git diff / diff --check 確認済み。commit / push / ブランチ切り替えなし。
- 未検証: CUDA、実機制御、実 checkpoint での全300ステップ。300ステップ75チャンクの検証には合成データ・候補と実 GPC 評価 API を使用。

幅修正後ログ:

- `/tmp/gpc_stride_final_unit.log`
- `/tmp/gpc_stride_compat8.log`
- `/tmp/gpc_stride_real.log`
- `/tmp/gpc_stride_regression.log`
- `/tmp/gpc_stride_baseline_loops.log`
