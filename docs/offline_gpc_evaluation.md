# GPC multi_sample オフライン評価

`policy_type=gpc`、`eval.real_robot.dry_run=true`、`dry_run_mode=multi_sample` の組み合わせで有効になる。実機・カメラの初期化より前に結果を保存して return する。`execute=true` でもこの経路で実機制御には入らない。

## 変更範囲

- `eval_real_robot.py`: 対象モードだけ DP のデータセット用前処理・候補数を選び、既存の WM／正規化統計の読み込み後にオフライン評価へ分岐。対象モード全体の PyTorch／NumPy RNG を保存・復元する。
- `offline_gpc_eval.py`: 保存済み候補を既存 GPC API に再入力して評価し、選択情報の保存と専用描画を追加。
- `tests/test_offline_gpc_eval.py`: 変更前 Diffusion との比較、既存 GPC API／実 JEPA／実チェックポイントとの整合性、実機を初期化しない分岐、RNG、保存・描画などの検証。
- `stable-worldmodel/stable_worldmodel/policy.py`: ユーザーが例外として許可した1行修正。`GPCPolicy.rollout()` の候補方向への展開結果を tuple ではなく Tensor として保持する。修正前は JEPA／LeWM の `info['pixels'].size(2)` が失敗する。既存実機 GPC にもこの不具合修正が適用される。

`offline_dp_eval.py` は変更していない。既存 DP／GPC の公開 API、`get_action_chunk()`、学習・損失・EMA・モデル構造・チェックポイント形式・正規化・実機制御ループは変更していない。`config/eval/flip_mug.yaml` の作業前からあった未コミット変更は保持した。

## 処理フロー

1. 既存の設定から WM、正規化統計、DP を読み込む。データセット画像には既存 Diffusion オフライン経路と同じ `img_transform()` を使う。実機 DP の `dp_img_transform()` は変更しない。
2. `load_episode()` と `build_observation_history()` により、同一の観測履歴と左端パディングを構築する。
3. WM には対象フレームの実測 `proprio` を渡す。`action_cartesian` を proprio の代替にしない。既存 `_load_or_capture_goal()` の保存画像読み込み部分と `_policy_observation()` を再利用する。保存済み goal 画像・goal proprio が必要で、実機での goal 撮影にはフォールバックしない。
4. 既存 `evaluate_multi_sample()` → `sample_candidates()` により DP を1回だけ呼び、物理単位の候補 `(K,H,8)` を生成する。
5. 一時コピーした GPC／DP コンテナで、DP sampler だけを生成済み候補を返すアダプターへ差し替える。モデル・正規化器は変更しない。元の `GPCPolicy.get_action_chunk()` を呼び、既存の行動区間抽出 → WM 正規化 → rollout → reward → argmax をそのまま実行する。元の DP sampler は再実行しない。
6. `(1,K)` の rewards を捕捉し、選択インデックスと元候補との完全一致を確認する。GT は診断指標・可視化だけに使う。
7. GPC 専用ディレクトリに保存し、全予測区間の選択軌道を赤色で描画する。WM が評価した区間は黄色の帯で示す。

評価区間は予測系列の **`[obs_horizon - 1, min(obs_horizon - 1 + action_horizon, H))`**。右端を含まない。黒い破線の `Selected action index` は実行開始位置であり、候補番号ではない。

## 出力

通常の保存先:

```
<eval.real_robot.output_dir>/offline_gpc/multi_sample/episode_<id>_step_<t>_<unique>/
  multi_sample_rollout.h5
  multi_sample_metrics.json
  multi_sample_position.png
  multi_sample_orientation.png
  multi_sample_gripper.png
```

Diffusion 用の `offline_dp/` とは独立し、実行ごとに一意なディレクトリを作る。

HDF5 には既存の候補・GT・valid mask・step 対応・実行開始位置に加え、以下を保存する。

- `selected_candidate_index`: scalar
- `candidate_rewards`: `(1,K)`
- `selected_candidate_actions`: 元候補をそのまま取り出した `(H,8)`
- `wm_evaluation_interval`: `[start,end]`、意味は右端を含まない `[start,end)`

JSON には診断指標・選択インデックス・全候補スコア・メタデータを保存する。HDF5 の metadata 属性と JSON の metadata に、DP チェックポイント情報、WM の具体的チェックポイントパス／サイズ／mtime、正規化統計・goal 画像のパス、設定、seed、各 horizon、区間、位置対応を記録する。全設定は未使用の旧 interpolation を強制解決せず保存する。

## 検証と制限

- 変更前の LeWorldModel リビジョン `351dbd30365a14fe8fe1219752a3db0db9d4257c` に対し、Diffusion の候補、GT 対応、指標、全 HDF5 dataset、metadata、JSON（実行固有ディレクトリ・run ID 除外）、PNG バイト列が完全一致。
- 同じ候補・観測・goal で、オフライン選択と既存 GPC API の結果が完全一致。K 本すべての評価、argmax の同点時の先頭選択、評価区間の切り詰め、単一サンプリング、GT 非依存、実測 proprio、RNG 復元、全8成分の描画を確認。
- 小型の実 JEPA と、ローカルにある実 DP／WM チェックポイントを CPU で評価。実データから切り出した5フレームを使用。実チェックポイントで候補2本・20本の両方の実行に成功した。タスク達成精度を評価するものではない。
- 既存の `episode`／`chunk_rollout`、前処理・正規化、旧 DP チェックポイント読み込み、既存 GPC API の回帰テストを実施。
- 既存 `test_dp_action_chunks.py::LoopTests` の3テストは、そのままでは古い `FakeEnv` に `_last_sdk_qpos` がなく失敗する。変更前コードでも同じ失敗を再現した。元テストは変更せず、追加テスト内だけでこの模擬状態を補完すると3テストすべて成功する。実機ループを含む既存関数・クラスの AST が変更前と一致することも確認。
- CUDA は利用できないため CUDA 固有のテストは skip。実機接続・制御、全エピソードでの実重量評価、既存 checkpoint の学習再開は未実施。
- `run()` の Hydra 経路は既存どおり CUDA を使う。CPU の実チェックポイント検証は新しいオフライン評価関数を直接呼ぶ。Hydra／実 GPU を含む一括実行は未検証。
- テスト中の実チェックポイントは DP `ep200_tm300_multiview_demo/seed_41/20261021_135400/latest.ckpt` と WM `ep100_tm300_multiview_play_normal_brightness/lewm_object.ckpt`。既存設定の `policy: lewm_weights` および `/home/hida/...` のパスをそのまま検証したものではない。

実行結果: 回帰コマンドは **54 passed / 2 skipped / 4 deselected / 38 subtests passed**。その後の追加テスト最終実行は **15 passed**（実チェックポイント20候補、補完 fixture による既存ループ3テストを含む）。最終の画像メモリ保持量調整後にも、単体テスト **11 passed**、実チェックポイント20候補テスト **1 passed** を再確認した。テスト群は一部重複しており、件数を単純合算しない。

実チェックポイント20候補の検証出力:

```
/tmp/gpc_real_checkpoint_6ur_manm/offline_gpc/multi_sample/episode_0_step_2_otlmgiir/
```

## 実行したコマンド

ワークスペース `/home/shonosukehida/work` から:

```bash
PYTHONPATH=LeWorldModel LeWorldModel/.venv/bin/python -m pytest -q -s \
  LeWorldModel/tests/test_offline_gpc_eval.py --tb=short

PYTHONPATH=LeWorldModel LeWorldModel/.venv/bin/python -m pytest -q \
  LeWorldModel/tests/test_offline_gpc_eval.py \
  LeWorldModel/tests/test_offline_dp_eval.py \
  LeWorldModel/tests/test_offline_dp_images.py \
  LeWorldModel/tests/test_dp_action_chunks.py \
  LeWorldModel/tests/test_action_normalization.py \
  LeWorldModel/tests/test_image_preprocessing.py \
  LeWorldModel/tests/test_dp_encoder_selection.py \
  -k 'not LoopTests' --tb=short
```

最終調整後の再確認:

```bash
PYTHONPATH=LeWorldModel LeWorldModel/.venv/bin/python -m pytest -q \
  LeWorldModel/tests/test_offline_gpc_eval.py \
  -k 'not production_checkpoints_cpu and not current_sdk_state_fixture' --tb=short

PYTHONPATH=LeWorldModel LeWorldModel/.venv/bin/python -m pytest -q -s \
  LeWorldModel/tests/test_offline_gpc_eval.py::test_production_checkpoints_cpu --tb=short
```

2番目の回帰コマンドでは古い fixture の4つの `LoopTests` を除外する。失敗していた3つは1番目の追加テスト内で fixture を補完して実行する。テストログは `/tmp/gpc_final_tests.log`、追加テストのログは `/tmp/gpc_final_new_tests.log`、最終単体テストは `/tmp/gpc_final_unit_tests.log`、最終実チェックポイントテストは `/tmp/gpc_final_production_test.log`。

GPU 環境での実行例（このコマンド自体は未実行）:

```bash
cd /home/shonosukehida/work/LeWorldModel
.venv/bin/python eval_real_robot.py --config-name flip_mug \
  policy_type=gpc \
  policy=/home/shonosukehida/.stable_worldmodel/checkpoints/flip_mug/ep100_tm300_multiview_play_normal_brightness/lewm \
  gpc.diffusion_checkpoint=/home/shonosukehida/.stable_worldmodel/checkpoints/flip_mug/policy/diffusion_policy/ep200_tm300_multiview_demo/seed_41/20261021_135400/latest.ckpt \
  eval.real_robot.dry_run=true \
  eval.real_robot.dry_run_mode=multi_sample \
  eval.real_robot.dry_run_num_candidates=20 \
  eval.real_robot.dry_run_seed=42 \
  eval.real_robot.execute=false \
  eval.probing.exe_probe=false
```
