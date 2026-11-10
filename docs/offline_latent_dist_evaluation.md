# GPC latent_dist オフライン診断

`policy_type=gpc`、`eval.real_robot.dry_run=true`、`dry_run_mode=latent_dist` の組み合わせだけで起動する。`eval.probing.exe_probe=false` を指定する。保存済み Goal / wrist Goal / goal proprio と、実測 proprio のある HDF5 が必要。モデル読込後、実機・カメラ初期化より前に return する。Diffusion / world_model と新モードを組み合わせるとモデル・実機初期化前にエラーにする。Probing 自体の既存処理は変更していない。

## 変更ファイル

- `offline_latent_dist_eval.py`（新規）: 評価、保存、4図の描画、専用 runner。
- `eval_real_robot.py`: 新モードの専用分岐と他 policy との組み合わせの拒否。
- `offline_gpc_eval.py`: 初期ロードを含む RNG 保存対象に latent_dist を追加する1行のみ。
- `tests/test_offline_latent_dist_eval.py`（新規）。
- 本ドキュメント。

`offline_dp_eval.py`、stable-worldmodel（policy / reward / モデル / 学習）、既存 GPC 評価関数、実機ループは未変更。既存テストも変更していない。開始時の YAML の `dry_run_episode_idx: 1` の未コミット変更は保持した。commit / push / ブランチ切り替えなし。

## 指標と既存 API

Goal は `GPCPolicy.encode_goal()` で1回だけ計算。観測は同じ `_policy_observation()` から現在の overhead / wrist / 実測 proprio を取り出し、既存 `_prepare_info()` と `world_model.encode()['emb']` で全時刻を直接 encoding する。rollout は使わない。両者の shape `(1,1,D)`、次元、有限値を検証する。独自の pooling / flatten / concat / 潜在正規化は加えない。モデルは eval モードを要求し、全計算は no_grad。実際の runner は既存の eval 済みモデルを受け取る。

観測 latent `z[t]` と Goal `g` から `||z[t]-g||₂` を全時刻で保存する。

rollout 開始行は `0,S,2S,...`、S=`dp_execute_steps`。`1 <= S <= action_horizon` の整数のみ受け入れる。H=`policy.action_horizon` は既存 GPC の WM horizon。

1. 既存 `build_observation_history()` / `sample_candidates()` で K 本を1回だけ生成。K は `dry_run_num_candidates`、seed は `dry_run_seed + evaluation_index`。DP checkpoint の逆正規化は既存 sampler で1回。
2. 一時コピーした GPC の Goal encoder を計算済み latent の再利用に置き換える。モデル自体は変更しない。
3. 既存 `select_candidates()` を呼ぶ。元の `GPCPolicy.get_action_chunk()` による候補スライス、正規化、rollout、reward、torch.argmax をそのまま実行する。rollout 出力を捕捉して選択終端を取り出すので、DP 再サンプリングも選択後の WM 再計算もない。
4. 候補評価完了後、独立した比較として物理 GT 行動を既存 `normalize_action_for_world_model()` で1回正規化し、同じ開始観測から `GPCPolicy.rollout()` を実行。
5. 既存 reward と同じ `predicted_emb[0,k,-1]` と `goal_emb[0,-1]` を使って、GT / GPC の終端 L2 距離を求める。reward 自体は二乗距離の負の値のまま。保存 reward が実際に `-L2²` と一致することも検証する。
6. 終端距離から開始時刻の observed distance を引いて delta を計算。GT 予測終端と実観測 `z[t+H]` の L2 距離を WM prediction error とする。

通常の GPC 実機候補数 `gpc.num_candidates` と、他の dry_run の挙動は維持する。Goal の一時キャッシュと rollout 捕捉は専用コピー上だけで行う。

## 時刻対応と末尾

`robot/utils/data_conversion.py::calculate_action_cartesian()` は、`action_cartesian[t] = leader_target[t+1]` として保存し、最後の target を繰り返す。したがって GT 入力は保存済み **`action_cartesian[t:t+H]`** とし、追加シフトしない。既存 DP との対応は予測配列 `[obs_horizon-1:obs_horizon-1+H]`。これらの定義と前提を metadata に記録する。

実観測の将来 latent は同一エピソードの行 `t+H` を使い、GT / 候補 rollout の入力には渡さない。WM の JEPA rollout は現在の1観測から H 行動で H 個の予測を進めるため、終端を `t+H` と対応させる。

行 `t+H` が存在しない開始時刻は、たとえ H 本の GT 行動だけが揃っていても評価しない。補完やエピソード越境はせず、スキップ時刻と理由を保存する。Observed encoding は独立して全時刻を評価する。短いエピソードでは rollout 配列が空になっても正しい shape で保存・描画する。

外部データも同じ観測→行動の保存規約である必要がある。Flip Mug の学習設定は frameskip=1。異なる時間間引きで学習した WM への一般化は未検証であり、今回の診断は保存データ1行を1 step とする。

## 出力と図

```
<eval.real_robot.output_dir>/offline_gpc/latent_dist/episode_<id>_<unique>/
  observed_latent_distance.png
  rollout_latent_distance.png
  latent_distance_delta.png
  wm_prediction_error.png
  latent_distance.h5
  latent_distance_metrics.json
```

- Observed: 全時刻の Goal L2 距離、横軸は Episode step。
- Rollout: 青が GT、赤が GPC の H-step 予測終端距離、黒が開始時刻の観測距離。横軸は rollout 開始 step。
- Delta: 終端距離 − 開始時刻の距離。青 GT、赤 GPC、ゼロの水平線。
- Prediction error: GT 予測終端と実際の `t+H` 観測 latent の L2 距離。

単一開始時刻なら点で表示し、存在しない将来を補間しない。全図に対応する数値を HDF5 に保存する。HDF5 は仕様の全必須項目に加え、`rollout_future_steps`、`current_observed_distance`、`future_observed_distance`、Goal / observed / GT terminal / GPC terminal latent を保存する。JSON は checkpoint、統計パスと数値、Goal パスと proprio、設定、各 horizon、K、seed、latent dimension、距離方式、時刻規約、スキップ理由などを保存。NaN / Inf は許さずエラーにする。

進捗は `Encoding observed latents` と `Evaluating latent distances` の2段階。total は実際の対象数。画像は1フレームずつ GPU に渡し、全時刻の latent は CPU へ移して保持する。HDF5 の画像は既存 load_episode と同じく CPU に読み込む。

## 検証

専用テストは **18 passed**（実 checkpoint テストを除く）。別実行の実 checkpoint テストは **1 passed**。

- 同一観測 / Goal で距離0、有限・非負、shape 整合、共通前処理、Goal encoding 1回。
- H=8 に対して S=8/4/1、正しい GT スライス、正規化、未来観測の非混入、全 K=20 候補の評価。
- reward argmax、既存 GPC API と同一候補・観測で選択行動の完全一致、再サンプリングなし、GT を変更しても選択 / rewards 不変。
- delta / WM error / future observed distance の数値整合、末尾スキップ、短いエピソード、他エピソードの巨大値を混入させないこと。
- 他 policy の新モード拒否、実機初期化前の return、RNG の保存・失敗時復元、学習モード拒否・非有限 reward 拒否。
- 開始時 Git `ac2a9381c6bc26ed040d120927d8c2727c73636d` と offline_dp_eval.py の完全一致、既存 GPC 評価関数・実機関数の AST 一致。

実 checkpoint は DP `ep200_tm300_multiview_demo/seed_41/20261021_135400/latest.ckpt`、WM `ep100_tm300_multiview_play_normal_brightness/lewm_object.ckpt` を CPU で読み、実データ10フレーム / K=20 / H=8 / S=4 で評価。latent dimension は416。Observed 10時刻、rollout は開始step0の1件、残りstep4/8は未来不足でスキップ。全保存値が有限で、選択 reward と `-distance²` が一致した。

出力: `/tmp/gpc_real_checkpoint_5vuecd_7/offline_gpc/latent_dist/episode_0__zozr3ds/`。rollout PNG も目視確認。

既存9ファイルの回帰初回は **76 passed / 2 skipped / 13 failed / 38 subtests passed**。うち1件は追加分岐の位置による AST dispatcher テストの失敗で、既存テストを変えずに分岐位置を修正し、そのテストを含む再実行 **17 passed** を確認。残る12件は以下の既存問題:

- FakeEnv / SafetyEnv の `_last_sdk_qpos` 欠落による3つの実機ループテスト（うち8 subtests）、計10 failures。
- 既存実 checkpoint テスト2件は YAML の episode_idx=1 を引き継ぐが、切り出したデータは episode0のみなので失敗。既存テストやユーザー設定は変更していない。

開始時 Git の GPC / 実機コードで同じ6テストを実行し、同じ12 errors を再現した。既存の同 seed Diffusion 保存出力比較（episode / multi_sample / chunk_rollout）と GPC 幅8の修正前比較は初回回帰で成功している。

最終全回帰の再実行中にディスク容量が尽きたため、その一括結果は利用不可。承認を得て今回の最終回帰ログだけを空にしてコマンド実行を復旧した。残り容量が約196MBのため、全回帰の再反復は行っていない。成功扱いに置き換えず、上記の初回結果と修正後の対象テスト結果を報告する。

未検証: CUDA、実機制御、実 checkpoint の全約300 step、他タスク / frameskip の異なるモデル。単一エピソードの距離曲線だけで H1、成功、物理的進捗、DP 候補不足を断定しない。

## 実行コマンド

ワークスペース `/home/shonosukehida/work` から実行したテスト:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=LeWorldModel \
  LeWorldModel/.venv/bin/python -m pytest -q \
  LeWorldModel/tests/test_offline_latent_dist_eval.py -k 'not real_checkpoints' --tb=short

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=LeWorldModel \
  LeWorldModel/.venv/bin/python -m pytest -q -s \
  LeWorldModel/tests/test_offline_latent_dist_eval.py::test_real_checkpoints_cpu_latent_dist --tb=short

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=LeWorldModel \
  LeWorldModel/.venv/bin/python -m pytest -q \
  LeWorldModel/tests/test_offline_gpc_eval.py \
  LeWorldModel/tests/test_offline_gpc_chunk_rollout.py \
  LeWorldModel/tests/test_gpc_chunk_execute_steps.py \
  LeWorldModel/tests/test_offline_dp_eval.py \
  LeWorldModel/tests/test_offline_dp_images.py \
  LeWorldModel/tests/test_dp_action_chunks.py \
  LeWorldModel/tests/test_action_normalization.py \
  LeWorldModel/tests/test_image_preprocessing.py \
  LeWorldModel/tests/test_dp_encoder_selection.py --tb=short
```

GPU 環境での全エピソード実行例（このコマンド自体は未実行。Goal 画像 / goal proprio は対象データに合わせた保存設定が必要）:

```bash
cd /home/shonosukehida/work/LeWorldModel
.venv/bin/python eval_real_robot.py --config-name flip_mug \
  policy_type=gpc \
  policy=/home/shonosukehida/.stable_worldmodel/checkpoints/flip_mug/ep100_tm300_multiview_play_normal_brightness/lewm \
  gpc.diffusion_checkpoint=/home/shonosukehida/.stable_worldmodel/checkpoints/flip_mug/policy/diffusion_policy/ep200_tm300_multiview_demo/seed_41/20261021_135400/latest.ckpt \
  eval.real_robot.dry_run=true \
  eval.real_robot.dry_run_mode=latent_dist \
  eval.real_robot.dry_run_num_candidates=20 \
  eval.real_robot.dp_execute_steps=8 \
  eval.real_robot.dry_run_episode_idx=0 \
  eval.real_robot.dry_run_seed=42 \
  eval.real_robot.dry_run_dataset_path=/home/shonosukehida/.stable_worldmodel/datasets/flip_mug/ep200_tm300_multiview_demo/push.h5 \
  eval.normalization_stats_path=/home/shonosukehida/.stable_worldmodel/datasets/flip_mug/ep200_tm300_multiview_demo/process_stats.npz \
  eval.probing.exe_probe=false \
  eval.real_robot.execute=false
```

ログ: `/tmp/latent_dist_final_unit.log`、`/tmp/latent_dist_real_final.log`、`/tmp/latent_dist_regression.log`、`/tmp/latent_dist_dispatch_final.log`、`/tmp/latent_dist_baseline_failures.log`。
