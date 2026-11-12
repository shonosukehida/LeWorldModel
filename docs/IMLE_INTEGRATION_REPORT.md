# IMLE統合準備の検証報告（2026-11-12）

判定：**現時点では real_robot_multiview へのマージを推奨しない（保留）**。
IMLE移植と対象回帰検証は実施済み。既存テストの未解消失敗、実学習済みIMLE・設定で指定された実WM・CUDAの未検証が残る。
実機・カメラへ接続せず、動作指令を発行していない。push、PR作成、最終マージは実施していない。

## ブランチと既存作業の保護

両リポジトリとも `integration/imle-into-multiview` を別worktreeに作成した。
作業場所：`/home/hida/workspace/imle-integration/{LeWorldModel,stable-worldmodel}`。
元のworktreeは引き続き `real_robot_multiview`。元ブランチの参照も未変更。

| リポジトリ | multiview基準コミット | IMLE移植元 |
|---|---|---|
| LeWorldModel | cc06f7de7e1bc5dc6ee2440fca8131d39215a772 | a6bb419a6cc8d89ceb278490aa15491ffa0cd528 |
| stable-worldmodel | ab3e681039977bcd978989082f0b7654a90ef07b | 6f7ea56be3730219c0dc9f75154d03d7841afde1 |

fetchで上記リモート参照を確認した。ブランチ全体のマージは行わず、IMLEファイルだけを選択移植した。
開始時の未コミット変更は LeWorldModel/config/eval/flip_mug.yaml と
stable-worldmodel/stable_worldmodel/reward.py、未追跡 tests/test_latent_goal_reward.py。
元の3ファイルを保持した。設定のローカル変更は統合しない。

基準コミット同士は依存不整合：LeWorldModelは `LatentGoalReward`、
`latent_goal_costs`、`latent_reward_metadata` を要求するが、stable-worldmodelの基準コミットに存在しない。
ユーザーの明示承認により、元作業ツリーのreward.pyとそのテストを**内容を変えず**統合worktreeへコピーした。
変更前の比較用worktreeにも同じreward.pyを適用した。生の基準コミットのままでは既存テスト7ファイルが収集エラーになる。
この前提修復をIMLEによる変更と区別する。

## 既存処理と追加内容

- eval_real_robot.py：IMLEを明示選択した場合のみcheckpoint検証・ロード・オフラインdispatchを追加。
  実行直前のCartesian検証もIMLE候補生成器だけが対象。
- policy.py：GPCPolicyの既存diffusion_policy引数を保持し、action_policy引数を追加。
  候補生成以降のWM正規化、rollout、報酬、argmax、物理行動選択は変更していない。
  IMLE側にあった `wm_info` をtuple化する古い差分は移植していない。
- IMLEPolicyはDiffusionPolicyの観測・チャンクインターフェースを継承する。
  既存の観測履歴、dp_execute_stepsバッファ、Gripper補強をそのまま通る。
  DPのdenoising・schedulerはIMLE生成に使用しない。
- flip_mug.yaml：imle_checkpoint: null / imle_device: cuda のみ追加。
  policy_type、DP checkpoint、候補数、lambda_prop、正規化パス、execute/dry_run、チャンク長、補強・可視化設定を変更していない。
- load_diffusion_policy、DiffusionPolicy、正規化モジュール、既存offline評価・可視化モジュールは元と同一。
- 移植したoffline_imle_evalはmultiviewのLatentGoalRewardを使いlambda_propを保持する。
  IMLE単体はepisode / multi_sample / chunk_rolloutに対応。
  IMLE+WMのdataset dry_runはepisodeのみ。他モードは明示エラーにし、暗黙に別モードへ切り替えない。
- IMLE設定・trainer・checkpoint・encoder・参照fixture・テストを移植。
  移植元テストの比較基準を今回のmultiview SHAへ更新し、legacy ResNetテストのencoder指定を修正。
  logging比較はIMLE移植元SHAを固定して参照する。
- 既存ソース同一性テスト2件は、追加したIMLE専用ガードの正確な文字列だけを除外して比較。
  GPCPolicyのコンストラクタ拡張以外の実装はASTで厳密比較する。数値の許容誤差は緩めていない。
- 依存はstable-worldmodelのoptional extra `imle`（torch-ema==0.3、omegaconf>=2.3.0）に追加しuv.lockと整合。
  テストでは元のvenvを変更せず、torch-emaだけを隣接test-depsへ隔離配置した。
  両リポジトリを同時に使用する際は、統合側stable-worldmodelをimportする必要がある。

| 使用方法 | policy_type | gpc.diffusion_checkpoint | gpc.imle_checkpoint |
|---|---|---|---|
| DP単体 | diffusion | DPパス | null |
| DP + WM | gpc | DPパス | null |
| IMLE単体 | imle | null | IMLEパス |
| IMLE + WM | gpc | null | IMLEパス |

WMには既存のpolicy・正規化統計設定が必要。IMLE用dry_runでは
`eval.real_robot.dry_run=true`、`eval.real_robot.execute=false` を明示する。
IMLE+WMではさらに `eval.real_robot.dry_run_mode=episode` を指定する。
この報告ではトップレベルCLIの依存全体を含む起動成功は主張しない。

## テスト結果

| テスト | 結果・根拠 |
|---|---|
| 最終対象テスト一括（回帰・IMLE・dispatch・reward） | **42 passed、25 subtests passed**。最終実装で再確認 |
| multiview基準との厳密回帰 | 4 passed、16 subtests passed |
| 固定観測・seedのDP全チャンクとRNG | 3 seedで完全一致。実ResNet・U-Net・DDPM、小型合成重みを使用 |
| GPC候補・WM入力・報酬・選択行動 | 完全一致。WMは制御可能な代替モデル、候補生成は実DP |
| 正規化・逆正規化 | 基準とのDP/GPC出力完全一致、正規化実装の同一性確認、既存統計テストを実施 |
| チャンク・Gripper・観測履歴・保存ログ | DP/GPC × steps 1/3 × 補強有無 × 中断有無の16条件で完全一致 |
| 既存学習済みDP checkpoint | 基準・統合版の実loaderでCPU推論、物理チャンクとRNG完全一致 |
| IMLE本体・encoder・trainer・checkpoint・legacy互換 | focused実行27 passed。EMA、optimizer、合成HDF5学習、参照数値、候補生成・選択、PNG/NPZ出力を含む |
| IMLE dispatch・設定・WM統計 | 最終6 passed。実checkpoint復元、統計読込とfallback、lambda_prop=0、WM候補評価、保存統計が未変更であることを確認 |
| LeWorldModel全既存テスト（比較基準） | 203 passed / 6 failed / 7 skipped、123 subtests passed。同じ承認済みreward前提 |
| LeWorldModel全体＋IMLE・rewardテスト | 239 passed / 6 failed / 7 skipped、148 subtests passed。上記と同一の6失敗。最終追加のdispatchテストは別実行で確認 |
| stable-worldmodel policy / wm / probing | 変更前後とも78 passed / 9 failed |
| DataLoader multiworker再確認 | sandbox外で1 passed。実機接続なし |
| 差分の空白検証・依存metadata/lock確認 | 成功 |

全体テストで残った6件の内訳（すべて変更前にも同じ失敗を再現）：

1. test_eval_dataset_path：現在の基準からchunk_rolloutを除去して比較する古いテスト前提。
2. test_process_stats：runの一部分をモジュールとしてAST実行し `return outside function`。
3. test_random_brightness::test_multiworker_wrapper：sandbox内でDataLoader 10秒timeout。
   sandbox外の単独再確認は成功。timeout値は変更していない。
4. test_random_brightness::test_yaml_default_disabled：基準設定はbrightness有効、テストは無効を要求。
5. test_xarm_robot_inference::test_dry_run_two_views_history_and_rollout_files：fixtureのaction_horizonとdp_execute_stepsが不整合。
6. test_xarm_robot_inference::test_policy_setup_failure_cleanup：AST実行namespaceにOmegaConfがない。

stable-worldmodelの9件は既存WorldModelPolicyテストのaction_spaceが空文字であることが原因。
変更前後で同一失敗。既存推論の仕様を変えてテストを通す対応は行っていない。

## 未検証とマージ条件

- 実機制御・カメラ接続・CUDA学習/推論・タイミング・閉ループ成功率は未検証。実機接続テストは実施禁止に従い未実施。
- 学習済み実運用IMLE checkpointはローカルに見つからず、合成モデルとテスト内学習checkpointで検証した。
- 既定policy=lewm_weightsに対応するWMは存在しない。IMLE+WMは小型実JEPA/ViTと代替WMで確認したが、設定で指定された実WM・実データとの性能は未検証。
- 全体CLIは重い既存シミュレータ・robot依存をimportするため、今回のdispatch検証は実run関数のASTと実offline処理を使用。
  実機なしでのクリーン環境CLI起動の確認は別途必要。
- 別途9件＋既存5件（sandbox再確認後）についてテスト前提の修復・合意が必要。
- 必ず両リポジトリを組み合わせて導入し、承認済みreward前提も含める。
- IMLE+WMのdry_runはepisodeのみ。この制限を評価者が認識する必要がある。

対象としたDP/GPC処理に数値差は見つからず、IMLEが明示選択された場合だけ追加経路を使う。
ただし未解消の全体テストと実運用checkpoint検証が残るため、無条件にマージ可能とは判断しない。

## 再実行と証跡

workspaceルートで以下の環境を設定し、ハードウェア非接続のpytestだけを実行する。

```bash
export PYTHONPATH=/home/hida/workspace/imle-integration/test-deps:/home/hida/workspace/imle-integration/stable-worldmodel:/home/hida/workspace/imle-integration/LeWorldModel
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export IMLE_LEGACY_DP_CHECKPOINT=/home/hida/.stable_worldmodel/checkpoints/flip_mug/policy/diffusion_policy/ep200_tm300_multiview_demo/seed41/20261021_135400/latest.ckpt
LeWorldModel/.venv/bin/python -m pytest imle-integration/LeWorldModel/tests imle-integration/stable-worldmodel/tests/imle imle-integration/stable-worldmodel/tests/test_latent_goal_reward.py -q
```

checkpointの32×32固定画像は保存観測の代替であり、全実観測での同一性を保証しない。
証跡は `/home/hida/workspace/imle-integration/validation/` のログを参照。
比較基準worktreeは `/home/hida/workspace/imle-baseline/`（detached）に残している。
移植元ドキュメントの過去の成功記録は今回の証跡に数えない。

## 変更ファイル一覧

### LeWorldModel

- `config/eval/flip_mug.yaml`
- `config/train/policy/imle_policy.yaml`
- `docs/IMLE_ENCODER_SELECTION.md`
- `docs/IMLE_IMPLEMENTATION.md`
- `docs/IMLE_INTEGRATION_REPORT.md`
- `eval_real_robot.py`
- `imle_checkpoint.py`
- `imle_runtime.py`
- `offline_imle_eval.py`
- `tests/fixtures/imle_resnet18_before_encoder_selection.npz`
- `tests/test_imle_dispatch.py`
- `tests/test_imle_encoder_selection.py`
- `tests/test_imle_integration.py`
- `tests/test_imle_legacy_checkpoint.py`
- `tests/test_imle_multiview_regression.py`
- `tests/test_imle_training_logging.py`
- `tests/test_offline_gpc_eval.py`
- `tests/test_offline_latent_dist_eval.py`
- `train_imle_policy.py`

### stable-worldmodel

- `IMLE_INTEGRATION_REPORT.md`
- `pyproject.toml`
- `stable_worldmodel/imle/__init__.py`
- `stable_worldmodel/imle/losses.py`
- `stable_worldmodel/imle/model.py`
- `stable_worldmodel/imle/networks.py`
- `stable_worldmodel/imle/obs_encoder.py`
- `stable_worldmodel/imle/policy.py`
- `stable_worldmodel/imle/provenance.md`
- `stable_worldmodel/policy.py`
- `stable_worldmodel/reward.py`
- `tests/imle/convnet_reference.npz`
- `tests/imle/generate_convnet_reference.py`
- `tests/imle/generate_reference.py`
- `tests/imle/s3fmp_reference.npz`
- `tests/imle/test_imle.py`
- `tests/imle/test_obs_encoder.py`
- `tests/test_latent_goal_reward.py`
- `uv.lock`
