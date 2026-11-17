# DP画像エンコーダの選択と互換性検証

## 使用方法

`config/train/policy/diffusion_policy.yaml`:

```yaml
encoder:
  type: conv # resnet18 | conv（省略時resnet18）
  pretrained: false # 従来どおりResNet18にのみ適用
  feature_dim: 512 # ConvNetにのみ適用
```

学習開始時に構造、1ビューあたりのパラメータ数、出力特徴量次元を表示する。
両ビューは独立したエンコーダを持つ。

ConvNetはConv2d(3→32, k5,s2,p2) → ReLU →
Conv2d(32→64,k3,s2,p1) → ReLU → Conv2d(64→128,k3,s2,p1) →
ReLU → AdaptiveAvgPool2d(1) → Flatten → Linear(128,feature_dim)。
入力(B,3,H,W)、出力(B,feature_dim)。パラメータ数は94,784 + 129×feature_dim。

| エンコーダ | 特徴量次元 | 1ビューのパラメータ数 | 2ビュー合計 |
|---|---:|---:|---:|
| ResNet18 | 512 | 11,176,512 | 22,353,024 |
| ConvNet | 512 | 160,832 | 321,664 |

ResNet18のクラス本体、DiffusionPolicy、正規化、scheduler、horizon処理は変更していない。

## 保存と復元

既存フィールドを維持し、`encoder_type`と`encoder_config`を追加する。
ResNet18設定は`{pretrained: bool}`、ConvNet設定は`{feature_dim: int}`。
ネットワークのその他の構造は上記クラスで固定。
学習・推論とも`create_obs_encoder()`を使用する。型未指定はResNet18、未知の型はValueError。
復元はcheckpointのトップレベル設定を優先し、YAMLやcheckpoint内config.encoderを参照しない。
既存checkpointのencoder_type欠落時もResNet18を生成する。
復元ではpretrained重みのダウンロードを行わず、両エンコーダとU-Netの保存済み重みを
load_state_dictの既定値strict=Trueでロードする。
U-Net条件次元は両エンコーダのoutput_shape()の和×obs_horizon。
実機DP単体・GPCとoffline評価は同一のload_diffusion_policyを使用する。

## 実行した検証（2026-11-03）

LeWorldModelディレクトリで実行:

```bash
MPLCONFIGDIR=/tmp/dp-encoder-mpl OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q -s tests/test_dp_encoder_selection.py
MPLCONFIGDIR=/tmp/dp-encoder-mpl OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q tests/test_offline_dp_eval.py tests/test_offline_dp_images.py tests/test_dp_action_chunks.py tests/test_action_normalization.py tests/test_image_preprocessing.py tests/test_process_stats.py
```

新規互換性テストは以下を検証する:

1. 変更前の凍結ResNet18と新ファクトリのstate_dictキー・shape、output_shape、forward形状。
2. 凍結した旧保存関数・旧エンコーダで生成した旧形式checkpointのロード。
3. 同じcheckpoint・入力・seed=42で旧/新ローダーの画像特徴、U-Net条件、行動列、推論後の乱数状態を完全一致で比較。
4. 新ResNet18 checkpointの保存・復元・推論。
5. ConvNetのforward/backward、両ビューの勾配、Adam更新、保存、strictロード、出力一致、行動生成。feature_dim=37と512を検証。
6. 学習設定のtype省略と旧checkpointのencoder_type省略がResNet18になること、未知型エラー、復元メタデータがconfig.encoderより優先されること。
7. ConvNetを実際のoffline episode / multi_sample評価へ渡し、推論・出力ファイル作成。

比較元はtests/baseline_dp_encoderに保存した変更前ソース。
合成checkpointに加えて、以下の実物旧形式checkpointも読み取り専用で検証:

`/home/shonosukehida/.stable_worldmodel/checkpoints/flip_mug/policy/diffusion_policy/ep200_tm300_multiview_demo/seed_41/20261021_135400/latest.ckpt`

実物にencoder_typeがないことも確認する。保存済みscheduler設定・推論ステップ数を変更せず、
CPU/PyTorch 2.5.1+cu121で比較。画像は決定的な合成観測（32×32）を使用。
実物・合成旧checkpoint、新ResNet18、新ConvNetの全比較で、
overhead特徴・wrist特徴・U-Net条件特徴・予測行動列の最大絶対誤差はすべて **0**。
許容誤差0でtorch.equalを使用。乱数状態も完全一致。

関連既存テストは **45 passed, 2 skipped, 43 subtests passed, 1 failed**。
失敗はtest_process_statsのtest_diffusion_evaluation_without_datasets_or_world_checkpointで、
ASTで切り出した実行名前空間にdp_img_transformがないことによるNameError。
変更前eval_real_robot.pyを/tmpに取り出して同じテストを実行し、同じ失敗を再現（5 passed, 1 failed）。
今回追加の関数を使うASTテストには新しい依存を注入した。
スキップはCUDA利用不可の検証と、旧テストが固定参照する/home/hida配下のcheckpoint不在。
後者は上記の実物checkpointテストで別途カバーした。

stable-worldmodelで追加実行:

```bash
MPLCONFIGDIR=/tmp/dp-encoder-mpl OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q tests/test_policy.py
```

**36 passed, 9 failed**。すべてWorldModelPolicyのテストfixtureでaction_spaceが不足するValueError。
変更前のobs_encoder.pyとdiffusion/__init__.pyを持つパッケージコピーを/tmpに作成し、
PYTHONPATHをそのコピーに向け同じテストを実行、同一9件の失敗を確認（36 passed）。
今回変更していないWorldModelPolicyの既存不整合として扱い、対象外の修正は行っていない。

未実施: 実機を動かすDP/GPCロールアウト、GPU比較、長時間の実データ学習。
実機経路は共通ローダーと既存のディスパッチ・GPC/DP回帰テストで確認し、
物理ロボットでの動作確認は行っていない。

## 変更ファイル

- stable-worldmodel/stable_worldmodel/diffusion/obs_encoder.py: ConvObsEncoder、設定解決、共通ファクトリを追加。
- stable-worldmodel/stable_worldmodel/diffusion/__init__.py: 新クラス・関数を公開。
- LeWorldModel/config/train/policy/diffusion_policy.yaml: typeとConvNet特徴量次元を追加。
- LeWorldModel/train_policy.py: 共通生成、構造表示、checkpointメタデータを追加。
- LeWorldModel/eval_real_robot.py: checkpointから自動選択しoutput_shapeから条件次元を計算。
- LeWorldModel/tests/test_dp_encoder_selection.py: 実モデルを使う互換性・学習更新・offline評価テスト。
- LeWorldModel/tests/baseline_dp_encoder/: 変更前エンコーダと保存・ロード関数、および出典。
- LeWorldModel/tests/test_action_normalization.py、tests/test_offline_dp_eval.py: AST分離実行に新しい依存を注入。
- LeWorldModel/DP_ENCODER_SELECTION.md: 本報告書。

最終の新規テスト実行結果: **5 passed, 3 subtests passed**（15.30秒）。
上記の必須互換性検証はすべて成功。ResNet18クラスのASTも変更前と同一であることを確認。
両リポジトリの`git diff --check`も成功。

## Spatial Softmax追加（2026-11-17）

`encoder.pooling: gap | spatial_softmax`で集約方式を選択する。省略時は`gap`。
不正値は`ValueError`。Spatial Softmax版は新規学習する。

| encoder | GAP | Spatial Softmax | 最終特徴次元 |
|---|---|---|---|
| resnet18 | 既存avgpool + fc=Identity | avgpool=SpatialSoftmax + fc=Linear(1024,512) | 512 |
| conv | AdaptiveAvgPool2d + Flatten + Linear(128,D) | SpatialSoftmax + Flatten + Linear(256,D) | D=feature_dim |

Spatial Softmaxは各チャネルの空間softmax（固定温度1）から、[-1,1]の格子上で
期待座標を計算する。出力順は(x0,y0,x1,y1,...)。長さ1の軸は座標0とする。
格子は入力のdevice/dtypeで生成し、パラメータやpersistent bufferを持たない。
入力解像度は固定しないが、ResNet18の32×32入力は最終マップが1×1になるため
位置情報を持たない。テストでは64×64と224×224で座標と勾配を確認する。

GAPのモジュール構造、初期化順、forward経路、state_dictキー・shapeは維持する。
設定解決の戻り値も従来どおりで、`pooling=gap`を明示しても保存辞書に追加しない。
Spatial Softmax時のみ既存の`encoder_config`に`pooling: spatial_softmax`を保存する。
トップレベル項目の追加はない。復元は既存ローダーのメタデータを優先する。
旧checkpointはGAPになり、full configの異なるpooling指定は復元構造を上書きしない。
strictロードは維持し、GAP/Spatial Softmax間の重み自動変換は行わない。

train_policy.py、eval_real_robot.py、DiffusionPolicy、ConditionalUnet1D、GPC、IMLEは変更しない。
凍結比較ソースtests/baseline_dp_encoderも変更しない。

実行コマンド（各リポジトリ直下、CPU）:

```bash
# LeWorldModel
MPLCONFIGDIR=/tmp/spatial-mpl OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q tests/test_dp_encoder_selection.py tests/test_dp_action_chunks.py tests/test_offline_dp_eval.py tests/test_offline_dp_images.py tests/test_gpc* tests/test_offline_gpc* tests/test_imle* tests/test_action_normalization.py tests/test_image_preprocessing.py tests/test_process_stats.py
# stable-worldmodel
MPLCONFIGDIR=/tmp/spatial-mpl OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q tests/test_policy.py tests/test_gpc_execute_first_candidate.py
```

最新ブランチとの一致をfetch後に確認:
LeWorldModel `b01ee178333790ea414e6eef813f81a2fa4e609c`、
stable-worldmodel `81f8e230ed76ca9dff4e50a7bf7168e2c5c39f33`。

互換性・追加機能テスト単独の最終結果: **10 passed, 1 skipped, 5 subtests passed**。
凍結実装と旧実物checkpointで、overhead/wrist特徴、U-Net条件、行動列、推論後RNGは
`torch.equal`で一致（最大絶対差0）。同一seedのGAP初期化・学習loss・更新後の全重みと
RNGも一致。ConvNet GAPは旧Sequentialのキー・shape・strictロード・出力・初期化RNGを比較。
Spatial Softmaxは手計算、座標範囲、64/224入力、両ビューの非ゼロ勾配、Adam更新、保存、
strict復元、メタデータ優先、保存前後の推論完全一致を確認。
offline episode/multi_sample（64×64画像）と実GPCPolicyからの候補生成・評価・行動選択も成功。
誤ってpoolingメタデータを削除したSpatial Softmax重みはstrictロードで失敗することも確認。

stable-worldmodel回帰テストは変更前後とも **40 passed, 9 failed**。
失敗一覧と理由は同一で、WorldModelPolicy fixtureのaction_space不足によるValueError。

未検証: GPU（利用不可につきskip）、実機動作、長時間の実データ学習、
pretrained=Trueでのネットワークからの重み取得。

LeWorldModel広域回帰の比較:

| 実行 | passed | failed | skipped | subtests passed |
|---|---:|---:|---:|---:|
| 変更前 | 167 | 16 | 3 | 64 |
| 変更後（追加Spatialテストを含む） | 171 | 16 | 4 | 66 |
| エンコーダテスト最終版単独（GAP学習一致検証も含む） | 10 | 0 | 1 | 5 |

変更前後のFAILED一覧とエラー詳細（E行）は完全一致し、新規失敗は0。
16件はsubtestの失敗を含む。既存失敗はIMLE proprio統計不足、過去revisionとの
設定・ソース固定比較の不一致、test_process_statsのAST切り出しによる
`SyntaxError: return outside function`。対象外のコードやテストは修正しない。
追加skipはCUDA不在。他のskipは既存のCUDA/外部checkpoint条件による。
ログは`/tmp/spatial-{baseline,current}-{lewm,swm}.log`と
`/tmp/spatial-feature-final.log`。
両リポジトリの`git diff --check`は成功。
