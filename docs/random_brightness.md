# WM学習へのランダム明るさ拡張

2026-11-03実装・検証。両リポジトリの作業ブランチは `real_robot_multiview`。
変更前のLeWorldModelは `d6cb1dc1179770fdae48f4433c1b8d40c2194b70`、
stable-worldmodelは `7e1b7b03f707a1b5648c818b84808904f5db64eb`。
作業開始時に未コミット変更なし。stable-worldmodelへの変更はない。

## 使用方法と変更内容

`config/train/lewm.yaml` に次の設定を追加した。既存のmultiview学習コマンドに
`augmentation.random_brightness.enabled=true` を追加すると有効になる。

```yaml
augmentation:
  random_brightness:
    enabled: false
    min_factor: 0.5
    max_factor: 1.5
    preview_steps: 9
```

倍率は正の有限値かつ `min_factor <= max_factor`、段階数は2以上の整数。
不正値は学習準備前にValueErrorになる。設定なし・OFF時は従来のDatasetに
transformを設定してから分割する経路を維持し、Wrapperもプレビューも使わない。

| ファイル | 変更内容 |
| --- | --- |
| `config/train/lewm.yaml` | デフォルトOFFの設定を追加 |
| `utils.py` | キーワード専用の `random_bright_enabled`, `min_factor`, `max_factor` を追加。既存の範囲変換を同じ計算の共通関数へ移動 |
| `brightness.py` | 設定検証、時刻ごとの明るさ変換、Dataset Wrapper、PNGプレビュー |
| `train.py` | ON時だけtransformを分離し、学習開始前にプレビューを保存 |
| `tests/test_random_brightness.py` | 変更前の実装との比較、新機能、実画像、実checkpoint、複数workerのテスト |
| `tests/baseline_brightness/{utils.py,train.py,README.md}` | 変更前の読み取り用スナップショットと説明。Gitの元ファイルとバイト単位で一致することを確認 |
| `docs/random_brightness.md`, `docs/brightness_validation/*.log` | この報告書と検証ログ |

モデル・損失・checkpoint形式・DP学習/推論コードは変更していない。

## 変換位置とtrain/valの分離

OFF時は従来と同じ `scale_to_unit_range → ToImage(ImageNet正規化) → Resize`。
ON時は `[0,1]` への変換とImageNet正規化の間に明るさ変換を挿入する。
`torch.clamp(image * factor, 0, 1)` を適用し、倍率の形状は `(T,1,1,1)`。
各時刻・各視点で独立に一様分布からサンプリングし、RGBには同じ倍率を使う。

ON時はtransformなしの単一HDF5Datasetを従来のGenerator/条件で分割した後、
各SubsetをTransformDatasetで包む。trainには明るさあり、valには従来のtransformを渡す。
行動・proprioの正規化は従来と同じ統計・関数・順序を使い、両側で共用する。
通常の `action` キーについては、HDF5Datasetのframeskip整形を一度戻して
正規化してから再整形し、従来の「正規化→整形」の結果を維持する。
`action_cartesian` 等の別名キーにはこの追加整形を行わない。

cfg.seed、learning_seed、分割Generator、DataLoaderの引数は変更していない。
Wrapperにはバッチ読み取りを素通しする属性転送を設けず、複数workerでも
必ず各サンプルへtransformを適用する。

## プレビュー

ON時に既存の `run_dir/config.yaml` 保存後、Trainer/manager実行前に次を保存する。

- `run_dir/brightness_preview/overhead_gradient.png`
- `run_dir/brightness_preview/wrist_gradient.png`

最初の有効なclipの最初の観測をHDF5から直接読む。各カメラの同じ元画像を
minからmaxまでpreview_steps等分し、倍率ラベル付きで横に並べる。
範囲変換と明るさ演算は実際の前処理と共通。正規化前のRGBを0–255へ丸めてPNGにする。
乱数は使用しない。保存エラーは保存先と原因を含むRuntimeErrorとして送出する。

## 検証コマンドと結果

以下のLeWorldModel用コマンドはリポジトリのルートで実行する。
CPUスレッド数を制限したのはテスト実行時の過剰な並列化を避けるためで、学習実装には影響しない。

回帰・新機能・既存前処理テスト:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest tests/test_random_brightness.py tests/test_image_preprocessing.py -q
```

**PASS: 35 passed、8 subtests passed、FAIL/SKIPなし**。
[ログ](brightness_validation/features.log)。複数workerはsandbox内のソケット制限で
一度失敗したため、許可された通常環境でこのコマンドを再実行しPASSした。

| 指示書の検証 | 結果・確認範囲 |
| --- | --- |
| A | PASS。両視点、uint8/floatの0–255、floatの0–1で、変更前・OFF・省略の出力がtorch.equal |
| B | PASS。分割インデックス、OFF時のDataset/Subset型・transformの取り付け位置が一致 |
| C | PASS。複数の学習バッチの全キーが完全一致。通常actionのframeskip=2も確認 |
| D/J | PASS。ON時のval全バッチが従来と完全一致し、反復アクセスでも不変 |
| E | PASS。WMの既存前処理テスト、DP推論の既存変換テスト、DP学習が使用するデフォルト前処理の完全一致。実WM checkpointの潜在表現・predict出力も完全一致。既存DP checkpoint/推論回帰テストも全体実行でPASS |
| F | PASS。OFF時の分割、shuffle順序を含む全学習バッチ、Generator状態、セットアップ前後のtorch RNG状態が一致 |
| G | FAIL（既存不具合・環境不足）。全体テストは後述のとおり変更前後で同じ失敗・エラー・SKIP |
| H | PASS。(T,1,1,1)のサンプリング形状、制御した倍率列、時刻間/視点間の独立性、RGB倍率共有、変換の挿入位置 |
| I | PASS。倍率範囲、min=max、clamp後の0–1範囲 |
| K | PASS。両PNG、段階数、各タイルの画素値と両端倍率、torch/NumPy/Python乱数状態、OFF時に出力しないこと、保存エラー。実HDF5画像も確認 |
| L | PASS。設定なし/OFF/ON、デフォルトYAML、不正範囲・非有限値・不正段階数 |

既存テスト全体（新規テストは上のコマンドで別途実行。既存テストの除外・緩和はない）:

```bash
# 変更前のutilsを先に読み込んだ比較実行
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=tests/baseline_brightness:.:../robopy/src .venv/bin/python -m pytest tests --ignore=tests/test_random_brightness.py --continue-on-collection-errors -q
# 変更後
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:../robopy/src .venv/bin/python -m pytest tests --ignore=tests/test_random_brightness.py --continue-on-collection-errors -q
```

**両方同じ結果: 80 PASS / 2 FAIL / 2 SKIP / 3収集ERROR、107 subtests PASS**。
[変更前ログ](brightness_validation/legacy_suite.log)、[変更後ログ](brightness_validation/current_suite.log)。
全体テストの比較実行ではutilsだけを差し替え、学習コードの変更前後は新規回帰テストが
スナップショットの実際のrun本体を実行して比較している。

FAIL（どちらも変更前から同じ）:

- `test_eval_dataset_path.py::DatasetPathTests::test_dry_run_dispatch_unchanged`:
  eval_real_robot.pyの既存dispatch ASTが古い期待値と不一致。
- `test_process_stats.py::ProcessStatsTests::test_diffusion_evaluation_without_datasets_or_world_checkpoint`:
  既存の抽出テスト内で `dp_img_transform` が未定義。

ERROR（変更前後で同じ）: xarm_admittance_config、xarm_gripper_wait、xarm_robot_inferenceの
3ファイルの収集。robopyをPYTHONPATHに追加しても `digit_interface` / `dynamixel_sdk` が未導入。

SKIP（変更前後で同じ）: 古いハードコード先 `/home/hida/...` のDP checkpointが存在しないテストと、
CUDAが必要なテスト。別の実在するDP checkpointを使う `test_dp_encoder_selection.py` のテストはPASS。

stable-worldmodelの関連既存テスト（同リポジトリのルートで実行）:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=../LeWorldModel ../LeWorldModel/.venv/bin/python -m pytest tests/data/test_data.py tests/data/test_data_utils.py tests/data/test_datasets.py tests/data/test_protocol.py tests/wm tests/test_policy.py -q
# 比較用: PYTHONPATHの先頭に ../LeWorldModel/tests/baseline_brightness を追加して同じコマンドを実行
```

**両方同じ結果: 225 PASS / 22 FAIL**。13件はVideoDatasetの `decord` 未導入、
9件はWorldModelPolicyの既存テストが空のaction_spaceを渡すためのValueError。
失敗テストIDの集合が変更前後で一致することもスクリプトで確認した。
[変更後ログ](brightness_validation/stable_worldmodel_suite.log)、
[比較ログ](brightness_validation/stable_worldmodel_legacy_suite.log)。
環境収集・stressを含む、より広いdataテスト実行はsandbox内で中断しており、成功とは数えていない。

## 実データ・checkpointと確認の限界

実画像の確認には次を読み取り専用で使用した。

`~/.stable_worldmodel/datasets/flip_mug/val_ep10_tm300_multiview_play/push.h5`

WM実checkpointの読み込み・固定入力での潜在表現と予測結果の完全一致には次を使用した。

`~/.stable_worldmodel/checkpoints/flip_mug/ep100_tm300_multiview_play/seed_41/lewm20261006_145420/lewm_epoch_2_object.ckpt`

GPU/bf16での短時間学習、初期loss・更新後weightの比較、実機での動作は未実施。
CUDAを使用できない環境であり、この範囲の完全保証はしていない。
既存モデル・loss・推論・checkpoint入出力コードに差分がなく、CPU上の完全一致試験と
既存テストの失敗集合が不変であることを根拠に、検証した範囲で後方互換性を確認した。
全体テストが完全に成功したという主張はしていない。
