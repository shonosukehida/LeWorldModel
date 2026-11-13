# JEPA Encoderを条件入力に使うDiffusion Policy

両リポジトリの作業ブランチは `real_robot_jepa_dp`。
起点はローカルの `real_robot_multiview`。作業開始時は両方とも未コミット変更なし。

## 構成

`encoder.type: jepa` を明示した場合にのみ有効。

```text
pixels       → overhead_encoder → overhead_projector ┐
wrist_pixels → wrist_encoder    → wrist_projector    ├→ Concat → 時間方向Flatten → U-Net.global_cond
proprio      → 事前学習時のzscore → prop_encoder       ┘
```

元の `jepa.JEPA.encode()` を再利用し、CLS token・projector・連結順序も共通。
次元は復元したモデルの実際の出力から取得し、付属の世界モデル設定と照合する。
確認した実物では overhead=192、wrist=192、proprio=32、合計416。
`obs_horizon=8` なら `global_cond` は3328次元。追加の学習可能な射影層はない。

`FrozenJEPAEncoder` は5つの特徴抽出モジュールを `requires_grad_(False)` にし、
`train()` を呼んでも子モジュールを含めevalを維持する。forwardは `torch.no_grad()`。
OptimizerにはU-Netのパラメータだけを渡す。predictor/action_encoder/pred_projは
Identityに置換し、重みを持ち回らず、特徴抽出でも呼び出さない。
互換性ハンドル `EmptyWristEncoder` の次元は0。Wristの実処理は複合Encoder内で行う。

DPの変更は観測辞書を渡す `_encode_observation_info` フックだけ。
既存型では従来の `_encode_observation(pixels, wrist_pixels)` に委譲する。
U-Net、scheduler、loss、行動系列サンプリング、action normalizationは既存実装を共有する。
既存ResNet18/Conv設定ではJEPA関連のロード・前処理・インスタンス化を行わない。

## 事前学習アーティファクト

同じJEPA事前学習に対応する3ファイルを指定する。

- `encoder.checkpoint`: `ModelObjectCallBack` が `torch.save(model, ...)` で保存した `*_object.ckpt`。
- `encoder.world_config`: その学習runの `config.yaml`。画像サイズ・潜在次元・proprio次元を照合する。
- `encoder.process_stats`: **JEPA事前学習データ**の `process_stats.npz`（proprioが必須）。

Lightningのstate_dictのみのCheckpointには対応しない。部分ロードやランダム初期化で
補完せずエラーにする。必要モジュール・次元・統計の欠落/不整合もエラーにする。
既存モデルオブジェクトには正規化統計が含まれないため、明示的な統計の指定が必要。
統計がなければ、元のJEPA事前学習データを使って作成する。

```bash
.venv/bin/python create_process_stats.py \
  --dataset /absolute/path/to/jepa_pretraining/push.h5 \
  --output /absolute/path/to/jepa_pretraining --keys action_cartesian proprio --no-overwrite
```

`process_stats.npz` は既存世界モデル評価と共通のzscore形式（ddof=1、std<1e-4なら1）。
統計とCheckpointが同じ学習に由来することは利用者が指定時に確認する。
DP教師データからproprio統計を再計算しない。
画像は世界モデルの `get_img_preprocessor`（範囲スケーリング、ImageNet正規化、Resize）を共有する。
画像サイズは世界モデル設定が優先される。学習時と推論時に同じ値を使い、推論のeval設定で上書きしない。
ProprioはDataset/実機からの生値をEncoder内で一度だけ正規化する。
JEPA-DP学習時に追加のランダム明るさ拡張は適用しない。

## 学習

以下はLeWorldModelディレクトリから実行する。`stable-worldmodel` の同ブランチがimportされる環境を使う。

```bash
.venv/bin/python train_policy.py --config-name jepa_diffusion_policy \
  encoder.checkpoint=/absolute/path/to/lewm_epoch_10_object.ckpt \
  encoder.world_config=/absolute/path/to/config.yaml \
  encoder.process_stats=/absolute/path/to/pretraining/process_stats.npz \
  data.dataset.path=/absolute/path/to/demonstrations/push.h5
```

`pred_horizon`/`obs_horizon`/`action_horizon`、`action_key`（既定action_cartesian）、
`action_normalization`、`model`、`diffusion`、`loader`、`optimizer`、`trainer`の設定は従来どおり。
CPUでは `device=cpu loader.num_workers=0 wandb.enabled=false` を追加できる。
`pred_horizon >= obs_horizon - 1 + action_horizon` を満たすように設定する。

既定保存先は `$STABLEWM_HOME/checkpoints/<task>/policy/jepa_diffusion_policy/<dataset>/seed_<seed>/<timestamp>/`。
`STABLEWM_HOME` 未設定時は既存キャッシュ先を使う。`best.ckpt`/`latest.ckpt`/epoch別Checkpointを保存する。
通常DPの `diffusion_policy` ディレクトリと区別される。

## dry run

既存 `policy_type=diffusion` とCheckpointのEncoder種別による自動復元を利用する。
JEPA-DPはCUDAが使えない環境ではCPUに復元する。

```bash
.venv/bin/python eval_real_robot.py --config-name flip_mug \
  policy_type=diffusion gpc.diffusion_checkpoint=/absolute/path/to/jepa_dp/best.ckpt \
  eval.real_robot.dry_run=true eval.real_robot.dry_run_mode=episode \
  eval.real_robot.dry_run_dataset_path=/absolute/path/to/demonstrations/push.h5 \
  eval.real_robot.dry_run_episode_idx=0 eval.real_robot.output_dir=/tmp/jepa_dp_eval
```

同じコマンドの `eval.real_robot.dry_run_mode` を以下に変更できる。

- `episode`: 各時刻の予測。`eval.real_robot.dp_execute_steps` も既存どおり。
- `multi_sample`: `eval.real_robot.dry_run_timestep=1 eval.real_robot.dry_run_num_candidates=20` を指定。
- `chunk_rollout`: 観測履歴からチャンクを予測し、既存の軌道可視化を出力。

3モードとも `pixels`/`wrist_pixels` に加え `proprio` 列を読み、履歴先頭を左パディングする。
図・HDF5・JSONの形式は既存と共通。通常DPではproprioを要求しない。
dry run入口で未使用の `FrankaSimEnv` importによるEGL初期化が起きていたため、そのimportを削除した。
実機環境の登録処理は維持している。

## 実機推論

実機の既存接続・カメラ・制御設定を確認して実行する。

```bash
.venv/bin/python eval_real_robot.py --config-name flip_mug \
  policy_type=diffusion gpc.diffusion_checkpoint=/absolute/path/to/jepa_dp/best.ckpt \
  eval.real_robot.dry_run=false
```

既存DPの画像履歴に加え、現在姿勢・グリッパーから組み立てたproprio履歴を渡す。
出力の正規化復元・行動実行・チャンク制御は既存DPと共通。
実機を動かす検証は今回実施していない。

## 保存・復元

従来のDP保存フィールドを維持し、JEPAの場合のみ以下を追加する。

- `encoder_type: jepa` / `jepa_format_version: 1`
- `jepa_encoder`: 元の `JEPA.encode` と学習済み特徴抽出モジュールを含む、CPUモデルオブジェクト。
- `obs_encoder_state_dict`: 同じEncoderの重みとproprio統計。復元時 `strict=True` でロードする。
- `jepa_feature_dims` / `jepa_image_size`: 出力次元と前処理情報。
- `jepa_source`: 元のCheckpoint・世界モデル設定・統計ファイルの絶対パス、SHA-256、世界モデル設定の写し。

U-Net重み、action正規化統計、horizon、scheduler、Hydra設定も共に保存する。
推論時は元のJEPAアーティファクトがなくても復元できる。
元の世界モデルと同様、モデルオブジェクトの復元には互換性のあるPythonクラスと依存環境が必要。
ResNet18/Convの保存形式とstate_dictキーは変えない。Encoder種別なしの旧CheckpointはResNet18として復元する。

## 検証

検証コマンド（LeWorldModelディレクトリ）:

```bash
MPLCONFIGDIR=/tmp/jepa-mpl OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q \
  tests/test_jepa_dp.py tests/test_dp_encoder_selection.py tests/test_dp_action_chunks.py \
  tests/test_action_normalization.py tests/test_image_preprocessing.py tests/test_offline_dp_images.py \
  tests/test_offline_dp_eval.py tests/test_gpc_chunk_execute_steps.py \
  tests/test_offline_gpc_eval.py tests/test_offline_gpc_chunk_rollout.py
```

新規テストは小型の合成ViT/JEPAモデルとローカル実物Checkpointを区別して検証する。
実物アーティファクト不在の場合、そのテストはskipする。
既存AST分離テストには観測フックの依存を追加した。
既存ソース一致テストはJEPA限定ディスパッチと観測フックを展開して比較し、
変更を許容する範囲を限定している。

詳細な実行結果は本ファイル末尾に記録する。
長時間学習、GPUでの再現性、実機DP/GPCロールアウト、明るさ耐性の定量比較は未実施。
小規模学習とdry runの成功はロバスト性の仮説を実証するものではない。

## 変更ファイル

LeWorldModel:

- `jepa_dp.py`: JEPA事前学習アーティファクトの復元、provenance、JEPA-DP自己完結Checkpoint。
- `config/train/policy/jepa_diffusion_policy.yaml`: JEPA明示選択用の学習設定。
- `train_policy.py`: proprio列、Frozen JEPA、モデル限定Optimizer、Checkpoint追加情報。
- `eval_real_robot.py`: Encoder種別での復元、実機proprio履歴、未使用EGL依存の削除。
- `offline_dp_eval.py`: JEPA選択時だけproprioを読み、各評価モードの履歴に含める。
- `tests/test_jepa_dp.py`: 新規統合・異常系・完全一致回帰テスト。
- `tests/test_offline_dp_eval.py`: 分離fixtureの観測フック、JEPA追加分だけを除く比較helper。
- `tests/test_offline_gpc_eval.py` / `tests/test_offline_gpc_chunk_rollout.py`: 既存出力比較を維持し、JEPA追加分を限定して区別。
- `JEPA_DP.md`: 本ドキュメント。

stable-worldmodel:

- `stable_worldmodel/diffusion/jepa.py`: Frozen複合EncoderとDPサブクラス。
- `stable_worldmodel/diffusion/obs_encoder.py`: JEPA設定の解決のみ追加。
- `stable_worldmodel/policy.py`: 3箇所の観測辞書フック。
- `docs/jepa_diffusion_policy.md`: ライブラリ側の構成説明。

起点commit: LeWorldModel `5beec045e9d17f0511f7b64c187a07148422b4a4`、
stable-worldmodel `c393f481a4430e75f06aa42bb5ae13f9150fbb37`。
ソース構造の比較はこの実装の起点に固定する。過去の基準のまま既存IMLE追加を
差分として扱っていたテスト1件も、この起点とJEPA追加範囲に合わせた。

## 実データのスモーク検証（2026-11-13）

使用した実物:

- JEPA: `/home/shonosukehida/.stable_worldmodel/checkpoints/flip_mug/ep100_tm300_multiview_play_normal_brightness/seed_41/lewm20261103_155756/lewm_epoch_10_object.ckpt`
- 世界モデル設定: 上記と同じディレクトリの `config.yaml`。
- 統計: `/home/shonosukehida/.stable_worldmodel/datasets/ep100_tm300_multiview_play_normal_brightness/process_stats.npz`。
- DP教師データ: `/home/shonosukehida/.stable_worldmodel/datasets/flip_mug/ep200_tm300_multiview_demo/push.h5` の先頭16行。

元データは読み取り専用で、スモーク検証用コピーを `/tmp/jepa_real_subset.h5` に作成した。
CPU、batch=2、obs_horizon=2、pred_horizon=8、action_horizon=3、
U-Net down_dims=[16,32]、diffusion_step_embed_dim=16、kernel_size=3、
diffusion学習step=4/推論step=2、1 epoch、WandB無効で、実際の `train_policy.py` を完走。
train_loss=1.157674、train_eval_loss=1.126583、val_loss=0.974786。
短い学習動作確認であり、性能評価用の学習済みDPではない。

検証用Checkpoint:

`/tmp/jepa-validation-cache/checkpoints/flip_mug/policy/jepa_diffusion_policy/ep100_tm300_multiview_play_normal_brightness/seed_41/20261113_114632/best.ckpt`

同Checkpointと同16行を使い、実際の `eval_real_robot.py --config-name flip_mug` の
`episode` / `multi_sample` / `chunk_rollout` をCPUで実行。
共通指定は `policy_type=diffusion eval.real_robot.dry_run=true`、
`eval.real_robot.dp_execute_steps=1`、`eval.real_robot.dry_run_episode_idx=0`。
候補数2、multi_sampleのtimestep=1。3モードすべて完走し、
`/tmp/jepa-real-offline/offline_dp/` 以下に図12枚、HDF5とmetrics JSON各3個を生成した。

旧実物DP Checkpointの復元・特徴量・行動の比較には、既存テストが参照する
`/home/shonosukehida/.stable_worldmodel/checkpoints/flip_mug/policy/diffusion_policy/ep200_tm300_multiview_demo/seed_41/20261021_135400/latest.ckpt`
も使用した。

## 最終テスト結果

上記の関連10ファイルをまとめて実行し、**86 passed、2 skipped、46 subtests passed**（230.63秒）。
失敗なし。skipはCUDA不在と、旧テストが固定参照する別ユーザー配下のCheckpoint不在。
後者は前述の実在する旧実物Checkpointによる比較で別途カバーした。
環境はCPU / PyTorch 2.5.1+cu121。リポジトリ全テストの実行結果ではなく、上記関連テストの結果。

- 合成JEPA: 3入力、動的特徴次元、U-Net条件形状、正常/異常ロード、凍結、BNを含む状態不変、U-Net更新、保存・復元後の特徴/行動完全一致。
- 実物JEPA: 同じ凍結・学習更新・保存/復元・特徴/行動一致テストを成功。
- 既存ResNet18/Conv: 同じ入力・seedでLoss、生成行動、複数候補、乱数状態が完全一致。
- 既存ResNet18/Convの実際のAdamW更新後の重み・バッファも完全一致
  （追加した更新比較を含めた2件の対象テストも別途実行し、2 passed）。
- 旧実物/合成Checkpoint、新形式の既存Encoder Checkpointを復元し、特徴量・条件・行動一致。
- 既存DPのepisode/multi_sample/chunk_rollout、GPC候補生成・報酬・選択・チャンク処理・RNG復元・出力ファイルを回帰検証。
- 既存の描画PNGとHDF5の完全一致比較も成功。
- 両リポジトリの `git diff --check` 成功。

テスト出力ログ: `/tmp/jepa-final-suite.log`。
実データCLIログ: `/tmp/jepa-real-training.log`、`/tmp/jepa-real-episode.log`、
`/tmp/jepa-real-multi.log`、`/tmp/jepa-real-chunk.log`。
変更は両作業ブランチの作業ツリーに保持しており、commit/pushはしていない。
