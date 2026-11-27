# Flip-mug GPC のサブゴール

## 学習と推論

LeWorldModel と隣接する stable-worldmodel の作業ツリーを組み合わせて使う。LeWorldModel ディレクトリから次のコマンドで学習できる。

```bash
export PYTHONPATH="$PWD:$PWD/../stable-worldmodel${PYTHONPATH:+:$PYTHONPATH}"
python train_subgoal.py subgoal.output_path="$PWD/outputs/subgoal.pt"
```

[学習設定](../config/eval/train_subgoal.yaml) は [flip_mug.yaml](../config/eval/flip_mug.yaml) の画像前処理、正規化統計、ゴール画像2枚、ゴール姿勢を継承する。`policy` に WM の AutoCostModel 参照、`subgoal.dataset_path` にデモ HDF5 を指定する。`subgoal.horizons`、`hidden_dim`、`encode_batch_size`、`batch_size`、`epochs`、`learning_rate`、`device` はすべて `subgoal` 配下で上書きできる。

教師データの作成時は WM を評価モードで凍結し、行動を与えずに全実観測を一度ずつエンコードする。CPU メモリに保存するのは各観測の結合潜在で、画像はバッチ単位で読み込む。同じ `episode_idx` 内で `step_idx` がちょうど H 離れた実測ペアだけを使う。共通の最終ゴールは一度だけエンコードし、全ペアで共有する。MLP は `[z_t, z_g, H]` から `z_(t+H)` を MSE で回帰する。保存ファイルは MLP の重み、形式バージョン、入出力を復元するための次元を含む。

同じ WM、正規化統計、ゴール設定で推論する。以下は Diffusion 候補を使う dataset dry_run の例。

```bash
python eval_real_robot.py --config-name flip_mug \
  policy=flip_mug/ep100_tm300_multiview_play_normal_brightness/lewm \
  gpc.goal_mode=subgoal +gpc.subgoal_checkpoint="$PWD/outputs/subgoal.pt" \
  eval.real_robot.dry_run=true eval.real_robot.execute=false \
  eval.real_robot.dry_run_mode=chunk_rollout
```

IMLE 候補を使う場合は `gpc.diffusion_checkpoint=null` と `gpc.imle_checkpoint=/path/to/imle.ckpt` を指定する。IMLE-GPC の subgoal の dataset dry_run は `episode` に対応する。この経路でも学習と同じ設定済み最終ゴールを使用する。

GPC は既存の候補列の評価区間を切り出した後、その実際の長さを H とする。`dp_execute_steps` は選んだチャンクから実行するステップ数を指定する。subgoal のコストは overhead、wrist、proprio を結合した全次元の終端二乗 L2 距離であり、各次元の重みは 1。最小コストの候補を実行するため、`rank.enabled=true`、`execute_first_candidate=false` を使用する。既存のゴールモードでは MLP の構築・ロード・呼び出しを行わない。

## 比較対象と変更ファイル

| リポジトリ | 実装直前のコミット |
|---|---|
| LeWorldModel | `5bf2474ae3a34fa1dbfebc293fbc540771bebcae` |
| stable-worldmodel | `795a18ce94e57d8b1a862e77078e8ae37e4d5606` |

| リポジトリ | ファイル | 変更理由 |
|---|---|---|
| LeWorldModel | `config/eval/train_subgoal.yaml` | デモ、固定ゴール、WM、MLP 学習条件を Hydra で指定 |
| LeWorldModel | `train_subgoal.py` | 凍結エンコード、エピソード内の教師ペア生成、MLP のみの学習と保存 |
| LeWorldModel | `saved_robot_goal.py`, `eval_real_robot.py` | 設定画像の RGB 変換と目標姿勢のクォータニオン処理を学習と推論で共有。subgoal 用に保存ゴールと学習済みエンコーダーを確認 |
| LeWorldModel | `offline_imle_eval.py` | IMLE-GPC の subgoal 評価に設定済みゴールを渡し、保存する評価条件を記録 |
| LeWorldModel | `offline_latent_dist_eval.py` | 最終ゴールへの診断曲線を維持し、subgoal の候補報酬を実際の予測サブゴールで照合 |
| LeWorldModel | `tests/test_subgoal_training.py` | 時刻とエピソードの照合、固定ゴール、教師、WM 凍結、Predictor の未使用、MLP の復元を検証 |
| LeWorldModel | `tests/test_subgoal_regression.py` | 固定した変更前コードと、候補生成から実行境界までを比較 |
| LeWorldModel | `docs/subgoal_gpc.md` | 使用方法、比較条件、検証結果を記録 |
| stable-worldmodel | `stable_worldmodel/subgoal.py` | H 条件付き MLP と単独チェックポイント |
| stable-worldmodel | `stable_worldmodel/policy.py`, `stable_worldmodel/reward.py` | 明示したモードで MLP を復元し、既存の rollout、終端評価、選択へ接続 |
| stable-worldmodel | `tests/test_subgoal.py` | 評価区間の H、最小コスト選択、他モードでのロード抑止を検証 |

## 後方互換性の比較

CPU の別プロセスに両版のソースを読み込み、同じ実チェックポイント、設定、デモ観測、固定ゴール、PyTorch seed 42 で比較した。WM は `ep100_tm300_multiview_play_normal_brightness/lewm_object.ckpt`、DP は `20261021_135400/latest.ckpt`、IMLE は `20261104_110824_634382/latest.ckpt`。ゴールと観測前処理は実装当日の flip_mug 設定を使用した。候補数は 2、WM の評価長と実行バッファは各 8 ステップ。CPU と NVIDIA GeForce RTX 3070 Ti でそれぞれ決定論的アルゴリズムを有効にし、seed 42 で比較した。

| 経路 | 完全一致を確認した値 |
|---|---|
| DP / IMLE 単体 | 入力観測、前処理・正規化後の入力、ポリシーの予測、物理単位のチャンク、バッファから順番に取り出した行動、クリップ後の目標、模擬 follower に渡す指令、推論後の Torch RNG 状態 |
| DP-GPC / IMLE-GPC の各 `final_goal`, `current_observation`, `latent_smoothness` | 上記の適用項目、候補行動列、WM 用に正規化した行動、WM の予測潜在状態列全体、評価値、選択インデックス、選択チャンク |

配列の dtype、shape、配列全体のバイト列の SHA-256 を照合し、数値の許容誤差は使用しなかった。実機側は実際の `execute` メソッドを通し、状態取得、IK、FK、follower への送信先をテストダブルに置き換えて照合した。GPU では CUDA RNG 状態も照合した。変更前後の CPU 同士、GPU 同士のいずれにも不一致はなかった。

```bash
export SUBGOAL_REGRESSION_IMLE="$HOME/.stable_worldmodel/checkpoints/flip_mug/policy/imle/20261104_110824_634382/latest.ckpt"
python -m pytest -q tests/test_subgoal_regression.py tests/test_subgoal_training.py \
  ../stable-worldmodel/tests/test_subgoal.py
```

## dry_run と学習確認

変更後はデモの先頭 10 観測を使った短い HDF5 と実チェックポイントで、以下の全 28 条件が HDF5 を保存するところまで確認した。HDF5 の画像は縦横を 4 分の 1 に間引いた。DP-GPC の episode は設定済み開始画像を使う既存ロボット dry_run で 2 制御ステップを実行した。この 28 条件の比較では検証ハーネスでロード先を CPU、出力先を一時領域に設定した。EGL を要求する未使用のシミュレータ import をハーネス内で置き換え、既存の評価関数を呼び出した。

| 候補生成 | goal_mode | 成功した dry_run_mode |
|---|---|---|
| DP / IMLE 単体 | 該当なし | episode, multi_sample, chunk_rollout |
| DP-GPC | 既存 3 モードと subgoal | episode, multi_sample, chunk_rollout, latent_dist |
| IMLE-GPC | final_goal, current_observation, subgoal | episode |
| IMLE-GPC | latent_smoothness | episode, multi_sample, chunk_rollout |

同じ実デモ抜粋で H=1〜8 の 44 ペアを作り、実 WM と既定の画像前処理で Hydra 学習を 1 エポック実行した。MSE は 0.90983456。保存した MLP を DP-GPC と IMLE-GPC の dry_run でロードした。ホストの RTX 3070 Ti でも同じ 44 ペアで 1 エポック学習し、MSE は 0.90983462。GPU 上の通常の Hydra CLI から、DP の chunk_rollout と IMLE の episode の両方で新しく学習した重みをロードし、評価結果を保存した。

GPU で直接比較を再現するには、上記の比較テストに環境変数 `SUBGOAL_REGRESSION_DEVICE=cuda` を指定する。

## テストスイート

Python 3.10.16、PyTorch 2.5.1+cu121 を使用した。変更前と変更後で同じ仮想環境、依存パッケージ、スレッド設定を使った。以下は CPU で各リポジトリの `tests/` 全体を実行した件数。失敗・収集エラーはどちらのリポジトリでも変更前後のテスト ID が一致した。既存テストの期待値は編集していない。

| 対象 | 成功 | 失敗 | スキップ | 収集エラー |
|---|---:|---:|---:|---:|
| LeWorldModel 変更前 | 313 | 17 | 5 | 3 |
| LeWorldModel 変更後、追加分込み | 318 | 17 | 6 | 3 |
| stable-worldmodel 変更前 | 673 | 14 | 0 | 0 |
| stable-worldmodel 変更後、追加分込み | 679 | 14 | 0 | 0 |

- LeWorldModel の失敗には、過去のソース・既定値との同一性を要求するテスト、IMLE の dispatch や proprio 統計の fixture、関数の途中を切り出す処理統計テストが含まれる。3 モジュールの収集は Robopy とその依存で停止した。ローカルの依存を補って再試行しても、チェックアウトした Robopy に `XArmAdmittanceConfig` がなく、`blosc2` も利用できず、両版で同じ 3 モジュールが未収集だった。
- stable-worldmodel ではローカルキャッシュの動画・シミュレータ依存を検証プロセスだけに追加して再実行した。残った 14 件は既存の CEM の action space・正規化に関する 5 件と、WorldModelPolicy の空の action_space に関する 9 件。
- LeWorldModel で増えたスキップは実 IMLE パスを環境変数で指定する比較テスト。パスを指定した別実行で成功した。今回追加したテストはパラメータ展開後、両リポジトリ合わせて 12 件で、個別検証を含めてすべて成功した。全体の件数は pytest の集計に従い、成功した subtest は表の件数に加算していない。
- CPU 実行でスキップされた既存 CUDA テストのうち、CUDA シード復元、SpatialSoftmax の勾配、学習済みチェックポイントを用いた GPC-OPT の 3 件を GPU 上で再実行し、変更前・変更後とも 3 件成功した。
- ディスク容量に合わせ、全体実行では大きな一時 HDF5 の出力先だけをメモリディスクへ向けた。候補数・期待値・入力データ・チェックポイントは変更前後で揃えた。


## 確認範囲

指定された 7 項目のうち各ポリシーに適用される値は、直接比較した条件ですべて一致した。単体ポリシーには WM rollout、候補ごとの GPC コスト、ランキングの選択インデックスは存在しない。

実機、カメラ、実 IK/FK・SDK、閉ループの時間特性とタスク成功率は未検証。学習確認に使った 44 ペアはデモ全体の一部であり、全デモでの収束、未知状態でのサブゴール精度、実機投入用の重みの品質は評価していない。異なる WM を使うと同じ次元でも潜在空間が異なり得るため、学習と運用では同一の WM チェックポイントと正規化統計を指定する。
