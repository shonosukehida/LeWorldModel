> 移植元 `real_robot_imle` の履歴資料です。ここに記載された過去のテスト結果・CLI実行実績は今回の統合の検証結果ではありません。今回の変更・実行結果・制約は [IMLE_INTEGRATION_REPORT.md](IMLE_INTEGRATION_REPORT.md) を参照してください。

# IMLE画像エンコーダ選択の実装報告

両repoとも既存の`real_robot_imle`上で作業。新規ブランチ・コミット・pushは行っていない。
s3fmpは読み取り専用。既存のユーザー差分と前回のIMLE実装差分は保持した。

## 今回の変更ファイル

LeWorldModel、既存ファイルへの変更:

- `config/train/policy/imle_policy.yaml`: `encoder.type`とConvNet構造設定を追加。
- `imle_checkpoint.py`: エンコーダ構築の最小分岐、checkpointへの構造保存と復元。
- `docs/IMLE_IMPLEMENTATION.md`: 本報告へのリンクを追加。

LeWorldModel、新規:

- `tests/test_imle_encoder_selection.py`: 両encoderの学習・復元・推論・dry_run/GPC、
  legacy checkpoint、ResNetの変更前数値比較。
- `tests/fixtures/imle_resnet18_before_encoder_selection.npz`: 変更前実装から保存した
  ResNet推論・損失・学習更新後出力・RNG。元ソースSHA256とseedも保存。
- `docs/IMLE_ENCODER_SELECTION.md`: 本報告。

stable-worldmodel、新規:

- `stable_worldmodel/imle/obs_encoder.py`: 独立したConvNet画像encoder。
- `tests/imle/test_obs_encoder.py`: 参照出力・パラメータ・勾配とshapeの検証。
- `tests/imle/generate_convnet_reference.py`: 読み取り専用の参照ASTからfixtureを生成。
- `tests/imle/convnet_reference.npz`: s3fmpが使用するml-networksの実装による数値fixture。

stable-worldmodel、既存ファイルへの変更:

- `stable_worldmodel/imle/provenance.md`: ConvNetの対応・適応を追記。

この拡張では既存DP、JEPA、GPC、実機制御、offline評価、IMLE生成器・損失・学習ループの
コードは変更していない。前回の作業から存在していた他の差分も変更していない。

## 設定例

```yaml
encoder:
  type: resnet18  # resnet18 | convnet
  pretrained: false
  convnet:
    channels: [16, 32, 64]
    kernel_sizes: [3, 3, 3]
    strides: [2, 2, 2]
    paddings: [1, 1, 1]
    activation: Mish
    pool_size: 1
imle:
  feature_dim: 128
```

ResNet18を選ぶ場合は`convnet`設定を使用しない。
ConvNetは提供されている事前学習重みがないため、`pretrained: true`は明確なエラーにする。

LeWorldModel cwdでの学習例（データパスは置き換える）:

```bash
python train_imle_policy.py encoder.type=resnet18 data.dataset.path=/absolute/path/push.h5
python train_imle_policy.py encoder.type=convnet data.dataset.path=/absolute/path/push.h5
```

CPU最小実行例:

```bash
python train_imle_policy.py encoder.type=convnet device=cpu \
  data.dataset.path=/absolute/path/push.h5 img_size=32 \
  loader.batch_size=2 loader.num_workers=0 \
  imle.n_samples_per_condition=2 trainer.max_epochs=1 wandb.enabled=false
```

`encoder.type=resnet18`への置き換えで同じCPU実行が可能。
その他のIMLE設定・train/validation split・画像処理・正規化は従来どおり。

推論時にencoderを指定する必要はない。checkpointの保存設定から自動で構築する。

```bash
python eval_real_robot.py --config-name flip_mug \
  policy_type=imle gpc.diffusion_checkpoint=null \
  gpc.imle_checkpoint=/absolute/path/latest.ckpt gpc.imle_device=cpu \
  eval.real_robot.dry_run=true eval.real_robot.dry_run_mode=episode \
  eval.real_robot.dry_run_dataset_path=/absolute/path/push.h5
```

`dry_run_mode=multi_sample` / `chunk_rollout`も同じ。
IMLE-GPCも既存コマンドのcheckpointを変更するだけで両encoderを使用できる。

## s3fmpとの対応と差異

参照: s3fmp `7b795a36f4f640c2b7ad2a192790a5513b70b23d`。

- 実際の設定: `models/cfg/xarm/imle.yaml`の`policy.encoder`。
- 呼び出し: `PolicyBase`の`ml.Encoder(...)`。
- 構造の実装: s3fmp/uv.lockが固定するml-networks
  `c672fb376b28b96c7c58cf6bbceb351176b0642d`の
  `vision.py::Encoder / ConvNet`、`layers.py::ConvNormActivation / LinearNormActivation`。

ConvNetは、biasありConv2d + Mishを3段、AdaptiveAvgPool2d(1)、Flatten、
biasありLinear(64,128)。normなし、dropoutなし、attentionなし。
参照のIdentity層や汎用ラッパを省略したが、演算と乱数を消費する層の初期化順序は一致する。
同じseed・画像で初期パラメータ、特徴出力、入力勾配、全パラメータ勾配が完全一致した。
参照コードを学習・推論時にimportする依存はない。

| 項目 | ResNet18（既存） | ConvNet（新規） |
|---|---|---|
| backbone | 既存ResNet18、fcをIdentity化 | Conv2d 3→16→32→64 + Mish |
| 射影 | 512→128 | 64→128 |
| 1カメラのパラメータ数 | 11,242,176 | 31,904 |
| 2カメラ合計 | 22,484,352 | 63,808 |
| 最終特徴次元 | 128 | 128 |
| 画像前処理 | 既存IMLEのImageNet正規化・resize | 同じ処理を使用 |
| 既定画像サイズ | 224 | 224 |

パラメータ数は`imle.feature_dim=128`・既定構造で実測。生成器は含まない。
特徴次元は`imle.feature_dim`で変更可能。ConvNetの層幅・kernel・stride・padding・
activation・pool sizeもconfigとcheckpointに保存する。

overhead/wristに独立したencoderを1つずつ持ち、観測履歴の各画像を既存の処理で
エンコードする。生成器へ渡す特徴の並び・形状は両方式とも同じ。

s3fmpとの残る差異:

- s3fmpは単一カメラと状態MLP。ここでは既存IMLEと同じ2カメラの画像履歴。
- s3fmpのxarm設定は入力画像96×128、画素値/255。本実装は公平な比較と
  ResNet後方互換性のため、両方式とも既存IMLEのImageNet正規化とresizeを維持する。
- s3fmpは観測履歴1。本実装の既定は8、ただしconfigで変更可能。
- ConvNetの数値比較は同一入力tensorでのencoder単体比較。
  前処理・条件入力が異なるため、s3fmp全体と同じ学習結果とは主張しない。

## Checkpoint互換性

新規checkpointには、full configに加えて`encoder_architecture`を保存する。
型、特徴次元、backbone/射影情報、ConvNetの全解決済み構造、norm/dropout/attentionと
biasの設定を記録する。ConvNetの省略値も保存時に解決し、保存configに埋め込む。
復元時に保存configと構造情報の整合を検証し、正しいencoderを構築して重みを読み込む。

`encoder.type`と`encoder_architecture`を持たない旧IMLE checkpointはResNet18扱い。
ResNet18分岐内のencoder・射影の生成順序、state_dictキー、特徴次元は維持した。
encoder未指定・明示ResNet18の双方で、変更前fixtureとの数値・RNG完全一致を確認。
変更前コードで実際に保存した旧checkpointも、旧loader/新loaderの物理action chunkと
RNG状態が完全一致した。

## 検証

最終実行は **26 passed（追加subtest 2件も成功）**、失敗0。
CPUのtorch 2.5.1環境で実行。実機・カメラ操作は行っていない。
ログ: `/tmp/imle-encoder-final.log`。変更ファイルの静的チェックとdiff空白検査も成功。

- 両encoderのCPU 1epoch学習、backward/optimizer step、latest/best保存。
- 両encoderのoptimizer/EMA/正規化を含むcheckpoint復元と固定seed候補一致。
- 両encoderの単体推論、episode/multi_sample/chunk_rollout、action buffer、
  IMLE-GPC候補評価・選択・可視化、実際のJEPAクラスとの接続。
- ConvNetの参照画像96×128でのパラメータ・出力・全勾配完全一致。
  32×32、224×224、非既定構造・特徴次元でのforward/復元。
- 旧ResNet18の型なしcheckpoint、型なしconfig、明示型の互換性。
- 既存DP/DP-GPCの固定seed数値比較とRNG、既存学習済みDP checkpointの100step推論。
- 既存DP/WorldModelPolicy/GPC評価ロジック、DP学習、JEPA、offline評価のソース不変性。

再実行（workspace root）:

```bash
PYTHONPATH=./LeWorldModel:./LeWorldModel/tests MPLCONFIGDIR=/tmp/imle-mpl OMP_NUM_THREADS=1 \
  LeWorldModel/.venv/bin/python -m pytest \
  LeWorldModel/tests/test_imle_encoder_selection.py \
  LeWorldModel/tests/test_imle_integration.py \
  stable-worldmodel/tests/imle/test_obs_encoder.py \
  stable-worldmodel/tests/imle/test_imle.py -q
```

ConvNet参照fixtureの明示的な再生成:

```bash
python stable-worldmodel/tests/imle/generate_convnet_reference.py \
  --ml-networks /path/to/ml-networks-at-c672fb3
```

実データでの長時間学習・CUDA検証・実機性能比較は未実施。
既知の既存テスト失敗・依存不足については前回の実装報告を参照。

既存DPの実checkpointを含む26件を再実行する場合は、上のコマンドに
`IMLE_LEGACY_DP_CHECKPOINT=/absolute/path/existing_dp_latest.ckpt`の環境変数と
次のテスト対象を追加する:

```text
LeWorldModel/tests/test_imle_legacy_checkpoint.py
LeWorldModel/tests/test_dp_action_chunks.py::NumericalTests
```

これらの数値比較では変更前と差0、RNGも一致。
全既存テストの再実行は行っていない（変更対象外の既知failureは前回報告に記載）。
