# IMLE統合・実資産/CUDA追加検証（2026-11-12）

**判定：マージ保留を継続する。** 実JEPAとCUDAへの接続経路は確認できた。
実学習済みIMLE checkpointはサーバに存在しないとユーザーから確認があり、
そのcheckpointの復元互換性・実運用性能は未検証のまま。
今回のIMLEは未学習の診断専用モデルであり、学習済みIMLEの検証成功とは扱わない。

## 対象と保護

- 作業場所：`/home/hida/workspace/imle-integration/`。
- 両リポジトリの `integration/imle-into-multiview` だけに検証スクリプト・文書を追加。
- 今回、DP/GPC/IMLEの推論実装、評価設定、正規化処理は変更していない。
  `eval_real_robot.py`、`stable_worldmodel/policy.py`、その他の推論ソースは前回統合コミットと同一。
- 元worktreeには書き込み操作を行わず、未コミット変更を保持。
  `real_robot_multiview` と `origin/real_robot_imle` の参照は変更していない。
- 実機・カメラには接続していない。オフライン専用スクリプトでは
  `robopy`、`xarm`、`pyrealsense2`、`robot`、`env`、`eval_real_robot` のimportを禁止。
  検証後にも該当モジュールが未importであることをassertした。
- push、PR作成、最終マージは行っていない。

## 資産の再探索

`/home/hida` を再帰探索した。venv、git、認証・ブラウザ設定、ツールキャッシュ等は除外。
初回探索で116 checkpointを列挙し、Torch ZIP内のpickleメタデータを読み、
`imle` と `generator_state_dict` のマーカーを調べたが該当なし。
探索時のエラーは0件。後に作成した診断用checkpointは探索結果から区別している。
他ユーザーのホームや外部サーバは探索していない。

ユーザー確認：「このサーバにはIMLE ckptがなかった」。
そのため既存の実学習済みIMLEの読み込み・推論は実施不能。
診断用モデルは既定のIMLE ConvNet設定で固定seedから生成したもので、
checkpoint epoch=-1、optimizer状態0件。診断用と明記し、実運用checkpointと混同しない。
診断時のIMLE min/max範囲は既存WM統計のraw min/maxを借用したものであり、
実IMLEの学習統計の整合性を検証したわけではない。

## 実JEPAの特定

現在の評価設定および11月10日のHydra記録は `policy: lewm_weights` だが、
この名前で解決される `_object.ckpt` は存在しない。
設定は書き換えず、シェル履歴の最後の明示的な `policy=` overrideを採用した：

```
flip_mug/ep100_tm300_multiview_play_normal_brightness/lewm
```

読み込みに使用した実checkpoint：

```
/home/hida/.stable_worldmodel/checkpoints/flip_mug/ep100_tm300_multiview_play_normal_brightness/lewm_object.ckpt
SHA256: b530905cc48d93f7f944c02e6da3fd608658cec37b176da1de96b4972ed39d26
```

`AutoCostModel` で `jepa.JEPA`（25,506,010 parameters）をロードした。
二視点ViT、projector、proprio encoder、action encoder、ARPredictorを含む実モデルである。
「現在使用中」とユーザーに再確認できたわけではなく、履歴から確認できる最新の明示指定である。
欠けている `lewm_weights` aliasの実体は引き続き不明。

正規化統計は同じデータ系列の既存process_stats.npzを読み取り専用で使用：

```
/home/hida/.stable_worldmodel/datasets/flip_mug/ep100_tm300_multiview_play_normal_brightness/process_stats.npz
SHA256: 70ee997f6d8613d76d863177a918ca2f4484d1c016a3a6cd704a8b1b94f942f1
```

検証前後のSHA256が一致。元の統計・checkpointは変更していない。

## オフライン入力と結果

保存済みの実観測データ
`/home/hida/.stable_worldmodel/datasets/flip_mug/val_ep10_tm300_multiview_play/push.h5`
から先頭16フレームを切り出した。画像・proprio・actionは既存列の値をそのままコピー。
学習用の画像前処理、IMLE checkpoint復元、実GPCPolicy、実JEPA rollout、LatentGoalReward、
argmax選択、既存offline evaluator・recorderを使った。
画像サイズ224、20候補、8step実行チャンク、lambda_prop=1.0。
評価goalは切り出した16フレームの最終観測であり、エピソード全体の最終goalではない。

| 確認項目 | CPU | CUDA（RTX 4080） |
|---|---|---|
| 診断用IMLE checkpoint復元 | 成功 | 成功、同じcheckpoint |
| IMLE単体チャンク生成 | 成功 `(1,8,8)` | 成功 `(1,8,8)` |
| 実JEPAによる20候補rollout | 成功 `(1,20,9,416)` | 成功 `(1,20,9,416)` |
| 候補報酬・argmax・物理行動選択 | 成功、期待候補と完全一致 | 成功、期待候補と完全一致 |
| IMLE単体episode評価・描画 | 成功 | 成功 |
| config経由IMLE+WM評価・描画・保存 | 成功 `(16,8)` | 成功 `(16,8)` |
| NaN/Inf検査 | 成功 | 成功 |
| 既存統計が未変更 | 成功 | 成功 |
| アプリケーションimport先検査 | 47モジュール成功 | 47モジュール成功 |

GPU環境：NVIDIA RTX 4080、driver 580.126.09、PyTorch 2.5.1+cu121、
PyTorch CUDA runtime 12.1。sandbox内ではGPUが見えず、承認済みの制約外実行で確認した。
CUDA実行の最大PyTorch割当量は353,073,152 bytes。
実機の周期・遅延・長時間安定性やタスク成功率を測ったものではない。

## CPU/CUDAの差の切り分け

同じcheckpoint・seed=42でも、CPUとCUDAのGaussian乱数列は一致しなかった。
そのため生成候補の最大差は0.0253798012、選択indexはCPU=7、CUDA=8。
さらに、同一Gaussian入力と同一物理候補を与える診断を実施した：

| 比較 | 観測結果 |
|---|---|
| 同一ノイズでのIMLE候補の最大絶対差 | 0.0000215768814 |
| 同一物理候補での実JEPA報酬の最大絶対差 | 0.000152587890625 |
| 同一物理候補での最適index | CPU=7、CUDA=7、一致 |

同一入力・同一重みでも、CPU/CUDAバックエンド間でbitwise一致にはならないことを確認した。
候補集合が変わる差と、同一入力で生じるデバイス計算の差を分離して記録している。
許容誤差を設定・緩和して成功扱いにする判定は行っていない。
各デバイス内での「実際の選択行動＝argmax候補」は完全一致で検証した。
同一物理候補での最適index一致はこの1条件の確認であり、すべての入力で保証するものではない。
既存DP/GPCの変更前後比較とは別の比較である。

## import保証の範囲

検証スクリプトが統合worktreeの両ルートを先頭に設定し、実際にロードされたモジュールの
`__file__` とSHA256を記録する。次のモジュールが対応する統合ルート外なら失敗する：

- `stable_worldmodel` 配下すべて。
- `jepa`、`module`、`utils`、IMLE checkpoint/offline、正規化モジュール。
- pickleから復元したJEPAクラスのソースファイルも統合LeWorldModel配下であることを確認。

元venvのeditable install先を変更する方法は使っていない。
この検証ランナーの起動では統合実装を使うことを検査しているが、
別シェルから任意に起動したコマンドまで保証するものではない。

## 残存リスクとマージ判定

今回解消した未検証項目は「実JEPAとの接続」「診断用IMLEでのCUDA単体/WM推論」
「検証プロセスが統合実装を読むこと」。

残る項目：

1. 実学習済みIMLEのcheckpoint/EMA/encoder/学習統計の復元互換性、実データでの品質。
2. `lewm_weights` の正しい実体と、履歴WMが現行運用モデルであることの確認。
3. 実checkpointと実観測を組み合わせたCUDA再検証。今回は未学習IMLEの接続試験に限る。
4. CPU/CUDAでの丸め差、異なるノイズ列による候補・選択差。候補が拮抗する場合の挙動は未評価。
5. 前回報告の既存テストの失敗群。今回、既存処理を変えて修復していない。
6. 全体CLIの依存込み起動、長時間・全エピソード評価、実機の閉ループ挙動。
   実機・カメラ試験は実施禁止のため未実施。

既存DP/GPCの厳密回帰テストは再実行し、**4 passed、16 subtests passed**。
ただし実学習済みIMLE不在と上記の未検証が残るため、無条件のマージ推奨はしない。
実学習済みIMLEを用意できた時点で同じランナーに `--imle-checkpoint` を指定して再検証する。

## 再現方法と証跡

追加した検証専用スクリプト：`scripts/validate_imle_offline_assets.py`。
`--imle-checkpoint` を省略すると**未学習診断用**checkpointを生成する。
実学習済みcheckpointを検証する場合は必ずパスを明示する。

```bash
# workspaceルート。元worktree内のvenvを読むだけで、install先は変更しない。
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/imle-followup-mpl \
LeWorldModel/.venv/bin/python \
  imle-integration/LeWorldModel/scripts/validate_imle_offline_assets.py \
  --device cuda \
  --imle-checkpoint /absolute/path/to/trained-imle.ckpt \
  --world-model /home/hida/.stable_worldmodel/checkpoints/flip_mug/ep100_tm300_multiview_play_normal_brightness/lewm \
  --stats /home/hida/.stable_worldmodel/datasets/flip_mug/ep100_tm300_multiview_play_normal_brightness/process_stats.npz \
  --dataset /home/hida/.stable_worldmodel/datasets/flip_mug/val_ep10_tm300_multiview_play/push.h5 \
  --output /home/hida/workspace/imle-integration/validation/followup/trained-cuda
```

GPUが見える実行環境が必要。`--device cpu` でも同じ検証を実行できる。
`--world-model` は `_object.ckpt` の前までのprefixを指定する。

証跡：`/home/hida/workspace/imle-integration/validation/followup/`。
`checkpoint_inventory.json`、`discovery.json`、`preservation.json`、`cpu.log`、`cuda.log`、
各デバイスの `result.json`・`selection.npz`・offline出力、
CUDA側の `controlled_device_comparison.npz`、`regression.log` を参照。
モデル資産や画像データはリポジトリへ追加していない。

今回の変更ファイル：

- LeWorldModel/scripts/validate_imle_offline_assets.py（新規、検証専用）
- LeWorldModel/docs/IMLE_OFFLINE_FOLLOWUP.md（本報告）
- LeWorldModel/docs/IMLE_INTEGRATION_REPORT.md（追加報告へのリンク）
- stable-worldmodel/IMLE_INTEGRATION_REPORT.md（追加検証結果の追記）
