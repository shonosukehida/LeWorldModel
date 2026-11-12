# IMLE proprio conditioning 実装・検証報告

2026-11-12 / `feature/imle-proprio-conditioning`

## 作業の保護と調査基準

- LeWorldModel基準: `e6048c9` (`real_robot_multiview`)
- stable-worldmodel基準: `cfe90c7` (`real_robot_multiview`)
- 開始時は両リポジトリともclean。各々 `../LeWorldModel-imle-proprio`、`../stable-worldmodel-imle-proprio` worktreeで作業。
- 保護ブランチのソース、既存checkpoint、既存datasetに変更なし。push・mergeなし。
- 実機・カメラへの接続なし。実機ループの検証はAST抽出した関数とFakeEnvのみ。

調査対象: `train_imle_policy.py`、`imle_checkpoint.py`、`imle_runtime.py`、`eval_real_robot.py`、`offline_imle_eval.py`、両YAML、既存IMLEテスト、およびstable-worldmodelのIMLE policy/model/networks、共通policy/data。

変更前のCPU基準:

- `tests/imle` とIMLE integration/multiview regression/dispatch: **22 passed / 16 subtests passed**。
- DP action chunks / GPC chunk execution / GPC proprio weight / gripper reinforcement / process stats / offline DP: **70 passed, 1 failed, 1 skipped / 67 subtests passed**。
- 既存失敗: `ProcessStatsTests.test_diffusion_evaluation_without_datasets_or_world_checkpoint`。テストが`run`関数の後半をmoduleとしてAST実行するため、既存`return`が `SyntaxError: 'return' outside function` になる。今回の変更前にも再現。対象外のテストや製品コードは修正しない。

## A. 変更ファイル

LeWorldModel:

| ファイル | 内容 |
|---|---|
| `train_imle_policy.py` | IMLE専用Dataset wrapper、Train観測からの統計計算、分布診断 |
| `imle_checkpoint.py` | input設定・統計・規約の保存と厳密な復元 |
| `offline_imle_eval.py` | raw proprio読出し、履歴保存、3モードとIMLE-GPCへの入力 |
| `offline_imle_core.py` (新規) | proprio有効時だけの評価ループ。DPループから必要部分を分離 |
| `eval_real_robot.py` | IMLE専用 `proprio_history` の追加 |
| `config/train/policy/imle_policy.yaml` | `imle.use_proprio: false`（明示的opt-in） |
| `config/eval/flip_mug.yaml` | checkpointが入力構成を決める旨のコメントのみ |
| `tests/test_imle_proprio.py` (新規) | 学習、正規化、checkpoint、互換性、WM、実機モック |
| `tests/test_imle_proprio_time.py` (新規) | 複数episodeでの時刻整列とpadding |
| `docs/IMLE_PROPRIO_CONDITIONING.md` (新規) | 本報告 |

stable-worldmodel:

| ファイル | 内容 |
|---|---|
| `stable_worldmodel/imle/model.py` | proprio時だけgenerator/gripper条件幅を拡張 |
| `stable_worldmodel/imle/policy.py` | condition生成で画像と正規化済み状態を連結 |
| `stable_worldmodel/imle/proprio.py` (新規) | 統計計算、検証、正規化・逆正規化 |
| `tests/imle/test_proprio.py` (新規) | CPU forward/backward、seed、統計エラー |
| `IMLE_PROPRIO_CONDITIONING.md` (新規) | ライブラリ側の仕様 |

共通の `stable_worldmodel/policy.py`、DP encoder/trainer/checkpoint、`jepa.py`、reward、WM正規化、`offline_dp_eval.py`、GPC visualizationには変更なし。共通化のリファクタリングなし。

## B. 調査結果とアーキテクチャ

1. 学習: HDF5Datasetはepisode内部の長さ`pred_horizon`の窓を切り出す。画像はその先頭`obs_horizon`フレーム、teacher actionは窓全体。画像と状態は同じraw sampleから切り出す。従来同様、学習窓の先頭paddingは行わない。
2. 推論: 各制御周期で画像履歴を更新し、開始時は最初の観測を繰り返す。新しい状態履歴も同じ周期・同じ回数で更新。chunk実行中も履歴を更新する。
3. 既存画像特徴: `_encode_observation`は各時刻でoverhead/wristを連結した後flattenする。従来IMLEはそれを`(B,2*To,F)`へreshapeする。
4. proprio有効時: flattenされた画像特徴を`(B,To,2F)`へ戻し、正規化した`(B,To,8)`を末尾に連結する。
5. Cartesian状態は`[x,y,z,qx,qy,qz,qw,gripper]`。位置m、Quaternion xyzw、gripper 0=open/1=closed。
6. DP actionは既存zscore/min-maxを保持。WMは既存SafeStandardScalerとprocess statsを保持。IMLE proprioは専用zscoreを追加。
7. checkpointはgenerator、2 encoder、optimizer、EMA、config/architecture、horizons、action統計を従来形式で保存。状態用フィールドを追加。
8. GPCはIMLEの`sample_action_sequences(..., denormalize=True)`から候補を受け取り、既存WM用にactionを正規化し、既存rollout/reward/argmaxを経て物理chunkを返す。この処理は変更していない。

デフォルト`F=128, To=8, Tp=16, Ta=8`の場合:

| 項目 | 画像のみ | 画像＋proprio |
|---|---|---|
| 各カメラ特徴 | `(B,8,128)` | 同左 |
| proprio | 不使用 | `(B,8,8)` |
| IMLE condition | `(B,16,128)` | `(B,8,264)` |
| Generatorのflatten条件 | `(B,2048)` | `(B,2112)` |
| Gaussian noise | `(B,16,7)` | 同左 |
| Gripper head条件（mean over tokens/time） | `(B,128)` | `(B,264)` |
| full action | `(B,16,8)` | 同左 |
| 実行chunk | `(B,8,8)` | 同左 |
| 複数候補 | `(B,N,16,8)` | 同左 |

Separate gripperを無効化すると従来どおりnoise_dim=8。RS-IMLE、gripper損失、Gaussian分布、EMA、action逆正規化、`To-1`からのchunk切出しは不変。

[公式のprocess_batch](https://github.com/krishanrana/imle_policy/blob/main/imle_policy/train.py) も画像特徴とlow-dimensional observationを各時刻で連結し履歴をflattenしている。本変更はこの結合方式に合わせる。一方、現在のローカルIMLEはs3fmp由来のgenerator/RS loss/独立gripper headを使っており、公式のネットワークへ置換はしない。従来の画像のみcamera-token構成はそのまま残す。

## C. 正規化とcheckpoint

- 既存と同じseedのrandom_splitを先に行い、**Trainに選ばれた窓の先頭To状態だけ**を統計計算に使用。
- meanと母標準偏差（`correction=0`）をfloat64で計算、float32として保存。
- std < 1e-4の次元はscale=1。定数次元はmean差分が0になる。clippingなし。
- `normalize_proprio`を学習loss、単体推論、候補生成の全てで使う。wrapperの返却値はraw状態であり、二重正規化しない。
- validationも同一Train統計を使用。`process_stats.npz`、`eval.normalization_stats_path`はIMLE状態正規化に使用しない。
- 入力NaN/Inf、統計欠落、非正のscale、次元・順序・単位・規約の不一致は明示的ValueError。
- 学習出力 `proprio_diagnostics.json` にTrain/Validation別のraw/normalized min/max/mean/stdを8次元別に記録。

保存フィールド:

```text
use_proprio: bool
proprio_stats:
  proprio_dim: 8
  proprio_normalization: zscore
  proprio_mean: tensor(8)
  proprio_std: tensor(8)  # safe scale
  proprio_layout: [x,y,z,qx,qy,qz,qw,gripper]
  proprio_position_units: m
  proprio_quaternion_order: xyzw
  proprio_gripper_convention: 0=open,1=closed
```

config.imleにもuse_proprioを保存。top-level設定との不一致は拒否。既存checkpointで設定がなければfalse。保存された設定だけでモデルを構築するため、評価YAMLで画像のみcheckpointを状態入力モデルに変えることはできない。

Quaternion: 既存dataset converter (`robot/utils/data_conversion.py`) はepisode内で単位化し、先頭qw>=0と以降の符号連続性を適用している。IMLEは格納値および既存実機観測をそのまま読み、成分別zscoreのみ行う。符号の再canonicalizeやQuaternionの再単位化は追加しない。実データの分布・実機側との符号整合は、この環境では未検証。

## D. 後方互換性と数値検証の範囲

| 対象 | 検証結果 |
|---|---|
| DP | 基準`cfe90c7`と同じsynthetic checkpoint/入力、seed 0/42/99でchunk・乱数状態が完全一致（許容誤差0） |
| DP＋WM/GPC | 候補、WM入力、mock予測、reward、選択index、actionが完全一致。実機モックでbuffer、gripper強化、履歴、ログも一致 |
| JEPA | 変更前`e6048c9`のJEPAクラスと同じ重みの小型ViTモデルでrolloutのtensor出力が完全一致。実JEPA＋状態条件IMLEでも既存argmaxからのchunk一致を確認 |
| 画像のみIMLE | 変更前クラスと同じモデル・encoder重み、入力、seed 0/42/99で候補出力・乱数状態が完全一致。設定欠落checkpointの読み込みも確認 |
| IMLE＋WM | 状態条件候補で既存GPCを実行。TinyWorldModelでepisode dry_runとvisualization保存、実JEPAで候補評価と選択を確認 |

本番学習済みDP/JEPA/旧IMLE checkpointの数値比較とは区別すること。上記は小型実ネットワーク、synthetic checkpoint、または明記したmockによるCPU検証。YAMLの実データ/学習済みDP checkpointパスはこの環境に存在しない。本番重みでの比較を検証済みとはしない。

## E. 実行とテスト

使用環境: 既存 `../LeWorldModel/.venv/bin/python` (Python 3.10 / PyTorch 2.5.1+cu121)。`PYTHONPATH`に新しいstable-worldmodelとLeWorldModelのworktreeを明示。`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1`を使用。

新規学習例（実機IOなし）:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/../stable-worldmodel-imle-proprio:$PWD" \
../LeWorldModel/.venv/bin/python train_imle_policy.py \
  imle.use_proprio=true device=cpu wandb.enabled=false \
  data.dataset.path=/absolute/path/to/push.h5 \
  loader.num_workers=0
```

CPUテスト対象:

- stable-worldmodel `tests/imle`。
- LeWorldModel `test_imle_{proprio,proprio_time,integration,multiview_regression,dispatch,legacy_checkpoint,encoder_selection,training_logging}.py`。
- `test_dp_action_chunks.py`, `test_gpc_chunk_execute_steps.py`, `test_gpc_proprio_weight.py`, `test_gripper_reinforcement.py`, `test_process_stats.py`, `test_offline_dp_eval.py`。

新規テストでforward/loss/backward/optimizer/EMA、候補多様性・再現性、checkpoint復元、Train統計の再計算一致、raw状態の学習/推論正規化完全一致、正規化逆変換、NaN/Inf、paddingと複数episode境界、chunk境界、単体全3モード、IMLE＋WM、FakeEnvの実機経路を検証。

オフライン状態履歴は評価rootの`proprio_histories.npz`に全stepのraw/normalized `(T,To,8)`とstep_idxを保存する。multi_sample/chunk_rolloutもstep_idxから使用履歴を特定できる。既存teacher/full候補/実行chunkのartifact形式を維持。IMLE＋WMは従来どおりepisode限定。

最終テスト結果は下記「最終実行結果」に記録する。

## F. 残存リスクとマージ評価

- 実データでの長時間学習、学習済み状態条件IMLEのタスク成功率、実機実行性能は未検証。
- CUDAは `torch.cuda.is_available() == False, device_count == 0` のため未実施。
- 既存のwindow単位random_splitを維持しており、隣接Train/Validation窓が同じraw時刻を共有する場合がある。統計の計算対象はTrain窓のみだが、episode独立validationではない。
- dataset converterのQuaternion符号規約と各実データ/実機観測の分布整合は運用前に確認が必要。今回の経路では既存Quaternionを変更していない。
- proprio専用評価loopはDPコードを変更しないため分離した。DP評価に将来仕様変更があればIMLE側loopの追従を確認する。
- 2リポジトリを組にして使用する。旧stable-worldmodelインストールのまま新trainerを起動しないこと。

既存処理への影響はIMLE選択時の入力構築に限定。CPUで検証した範囲はレビュー・統合候補にできるが、本番checkpointでの完全互換性と実タスク性能の承認までは判断しない。`real_robot_multiview`へのマージ・pushは行っていない。

## 最終実行結果

最終統合実行（上記テスト一覧からtime専用ファイルを除く）:

```text
1 failed, 119 passed, 2 skipped, 40 warnings, 85 subtests passed in 169.93s
```

別実行 `tests/test_imle_proprio_time.py`:

```text
1 passed, 2 warnings in 5.00s
```

合計: **120 passed, 1既存failed, 2 skipped / 85 subtests passed**。

- 失敗は変更前と同じProcessStatsTestsのAST構文エラー1件のみ。新規失敗0。
- skip: CUDA unavailable、`IMLE_LEGACY_DP_CHECKPOINT`未指定での実学習済みDP比較。
- warningsはpytest/依存ライブラリ警告とNVML利用不可等。CUDAテストを成功扱いにしない。
- 最終source照合で、共通policy/data、DP trainer、JEPA、reward、WM/action正規化、IMLE損失/networks、offline DP/GPC、GPC visualization、IMLE runtimeの保護対象ファイルは基準commitとバイト単位で一致。
- 両元worktreeはclean。変更は指定featureブランチのworktreeに限定。

最終統合コマンド（workspace rootから）:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/stable-worldmodel-imle-proprio:$PWD/LeWorldModel-imle-proprio" \
OMP_NUM_THREADS=1 LeWorldModel/.venv/bin/python -m pytest \
 stable-worldmodel-imle-proprio/tests/imle \
 LeWorldModel-imle-proprio/tests/test_imle_proprio.py \
 LeWorldModel-imle-proprio/tests/test_imle_integration.py \
 LeWorldModel-imle-proprio/tests/test_imle_multiview_regression.py \
 LeWorldModel-imle-proprio/tests/test_imle_dispatch.py \
 LeWorldModel-imle-proprio/tests/test_imle_legacy_checkpoint.py \
 LeWorldModel-imle-proprio/tests/test_imle_encoder_selection.py \
 LeWorldModel-imle-proprio/tests/test_imle_training_logging.py \
 LeWorldModel-imle-proprio/tests/test_dp_action_chunks.py \
 LeWorldModel-imle-proprio/tests/test_gpc_chunk_execute_steps.py \
 LeWorldModel-imle-proprio/tests/test_gpc_proprio_weight.py \
 LeWorldModel-imle-proprio/tests/test_gripper_reinforcement.py \
 LeWorldModel-imle-proprio/tests/test_process_stats.py \
 LeWorldModel-imle-proprio/tests/test_offline_dp_eval.py -q
```
