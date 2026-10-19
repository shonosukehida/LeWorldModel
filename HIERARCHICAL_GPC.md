# Step 4: H-JEPA GPC-RANK

## Phase 1: Policy

`HierarchicalGPCPolicy`は既存`GPCPolicy`の派生クラス。既存GPCとstable-worldmodelは変更しない。
`get_action(raw_info, dp_info_dict=raw_dp_history)`は物理値のnumpy `(B,8)`を返す。
`set_env()`とGPCの型判定を継承し、実ロボットループの履歴管理・グリッパ補強を再利用する。

DPはCheckpointのobs/pred/action_horizonを復元し、
`sample_action_sequences(..., num_samples=K, denormalize=True)`で物理値候補を生成する。
`start=obs_horizon-1`、`end=min(start+action_horizon,pred_horizon)`の未来区間のみを評価する。
実際に得られた区間の長さがUpperのHで割り切れなければエラー。Hに合わせた切り捨てはしない。
最小total_costの候補から最初の1 stepだけを返し、行動バッファへ未来8 stepを蓄積しない。

HWMには未変換画像・物理値Proprio/行動を渡す。`_prepare_info()`と
`normalize_action_for_world_model()`は呼ばない。画像のレイアウトだけを
明示設定`raw_image_layout=NHWC`に従ってNCHWへ変換する。
DP用履歴はNHWC、HWM用現在観測は1時点であり、混同しない。
DPはそのCheckpoint内config.img_sizeと学習時`get_img_preprocessor()`の変換、
Checkpoint内action_mean/stdを使う。Lower/Upperの前処理はHWM内部に任せる。

`last_diagnostics`から全候補の5種類のコスト、best_idx、selected_action、selection_id、
実際のLower/Upper Rolloutステップ数を参照できる。大きな潜在列は保持しない。

## Phase 2: 実Checkpointのオフライン検証

ロボット関連モジュールをimportしない`offline_hierarchical_gpc.py`を使用する。
DPローダーは既存の実装を`diffusion_checkpoint.py`へ移動して再利用した。
既存eval側も同じシグネチャのローダーをimportするため、旧DP/GPCの復元方法は維持する。

共通設定は`config/eval/hjepa/gpc.yaml`。Checkpoint/Lower統計の既定値はnullで、
**実際の学習済みファイルを明示指定する。パスの探索・推測・統計再計算は行わない。**
Upperの統計はCheckpoint埋め込み、Lowerは指定NPZと学習config.yamlを使用する。

以下の変数を手元の実ファイル・行番号に設定する。CURRENT_ROWはDPのobs_horizon分の
実際の連続履歴が存在する行、GOAL_ROWはゴールの行を選ぶ。過去履歴の複製・paddingは行わない。

```bash
DP=/absolute/path/to/dp.ckpt
LOWER=/absolute/path/to/lewm_epoch_10_object.ckpt
UPPER=/absolute/path/to/upper/last.ckpt
LOWER_STATS=/absolute/path/to/lower/process_stats.npz
LOWER_CONFIG=/absolute/path/to/lower/config.yaml
# CURRENT_ROW と GOAL_ROW は実データから選択した整数を設定する。
MODEL_ARGS=(
  "hjepa.diffusion_checkpoint=$DP"
  "hjepa.world_model.lower_checkpoint=$LOWER"
  "hjepa.world_model.upper_checkpoint=$UPPER"
  "hjepa.world_model.lower_stats_path=$LOWER_STATS"
  "hjepa.world_model.lower_config_path=$LOWER_CONFIG"
)
.venv311/bin/python offline_hierarchical_gpc.py "${MODEL_ARGS[@]}" \
  hjepa.device=cpu hjepa.num_candidates=3 \
  dataset_path=/data/hida/flip_mug/ep100_tm300_multiview_play/push.h5 \
  "current_index=$CURRENT_ROW" "goal_index=$GOAL_ROW"
```

画像は保存済みNHWC RGB、Proprioは8D、episode_idx必須、step_idxは存在する場合に検査する。
対応しない画像レイアウトは推測せずエラー。実データのフォーマットを明示的に変換して使用する。
出力`outputs/hierarchical_gpc_offline/results.npz`には全コスト、best_idx、物理値action、
Lower/Upperの実際の予測回数を保存する。H=4・未来8 stepなら8 / 2を検証する。
この検証はタスク成功率や実機安全性の評価ではない。

今回の環境では実DP/Upperのパスが提供されず、指定Flip Mug HDF5も存在しないため、
**実学習済み3モデルによるPhase 2は未実施**。スクリプト・設定・エラー処理のMock検証のみ。

## Phase 3: 既存実ロボットループ

`eval_real_robot.py --config-name flip_mug_hjepa`で新分岐を使用する。
既存`flip_mug`設定のDP単独・単層GPCのコードパスは維持する。
階層分岐は旧evalの統計自動生成へ入らず、指定Checkpoint/統計だけで構築する。
`_policy_observation()`は元々値を変換せず辞書を構築するためそのまま使用する。
カメラ・制御周期・安全なenv.execute()・終了処理・動画/HDF5保存は既存ループを再利用する。

新設定の既定値はexecute=false、dry_run=true。モデル構築の確認だけなら：

```bash
.venv311/bin/python eval_real_robot.py --config-name flip_mug_hjepa \
  "${MODEL_ARGS[@]}" hjepa.device=cpu
```

実機に指令を送らない既存Dry-runアダプターでの接続確認：

```bash
.venv311/bin/python eval_real_robot.py --config-name flip_mug_hjepa \
  "${MODEL_ARGS[@]}" hjepa.device=cpu hjepa.num_candidates=3 \
  eval.real_robot.execute=true eval.real_robot.dry_run=true \
  eval.real_robot.max_steps=2 eval.real_robot.non_interactive=true
```

既存のロボット・カメラ依存パッケージ、Dry-run/ゴール画像設定も必要。
今回実行したのはFakeEnvによる実ループのテストであり、このCLIによる実Checkpoint Dry-runは未実施。

実Checkpointオフライン検証とDry-runを現地で終えた後、実機で使用するコマンドは以下。
**このコマンドは今回自動実行していない。**

```bash
.venv311/bin/python eval_real_robot.py --config-name flip_mug_hjepa \
  "${MODEL_ARGS[@]}" hjepa.device=cuda \
  eval.real_robot.execute=true eval.real_robot.dry_run=false
```

候補数・評価horizon・重み・スケールはhjepa以下の設定で変更できる。
H=4・action_horizon=8が既定。HはUpper Checkpointから取得し、別の値で上書きしない。

グリッパ補強は既存`_reinforced_gpc_action()`をそのまま使用する。
補強中はDP/HWM推論を停止して観測EE姿勢を固定し、期限・再発動条件に従って復帰する。
既存rollout.h5のキー・gripper_policy_command/command/actualは維持する。
追加の`hierarchical_costs.jsonl`には候補別/選択コスト・best_idx・policy_action・
実際のexecuted_action・gripper_policy/command/actualを保存する。
補強保持中はinference_performed=falseとし、古いコストを新しい推論結果として記録しない。

## 検証範囲

Phase 1: Mock DP/HWMによる候補選択・shape/有限性・未来区間・1 step出力。
Phase 2: 明示パス・Checkpoint統計検証、学習時画像変換選択、保存データの境界検査。
Phase 3: 実Policyクラスを既存制御関数+FakeEnvへ接続し、6周期で3回だけ推論、
補強中の姿勢保持・再開・1周期1指令・既存ログと追加JSONLの保存を確認。
実ロボット接続・動作・タスク成功率は検証していない。
