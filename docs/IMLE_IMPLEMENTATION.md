> 移植元 `real_robot_imle` の履歴資料です。ここに記載された過去のテスト結果・CLI実行実績は今回の統合の検証結果ではありません。今回の変更・実行結果・制約は [IMLE_INTEGRATION_REPORT.md](IMLE_INTEGRATION_REPORT.md) を参照してください。

# IMLE導入・GPC統合 実装報告

追補: [画像エンコーダ選択の実装報告](IMLE_ENCODER_SELECTION.md)（ResNet18 / ConvNet）。

## ブランチと変更ファイル

開始時・完了時とも LeWorldModel と stable-worldmodel は既存の
`real_robot_imle`。新規ブランチ・コミット・pushは行っていない。
開始時、LeWorldModelは差分なし。stable-worldmodelには既存の
`stable_worldmodel/probing/flip_mug/plot.py` の差分があり、そのまま保持した。
s3fmpは読み取り専用で参照し、変更していない。

LeWorldModel新規:

- `train_imle_policy.py`
- `config/train/policy/imle_policy.yaml`
- `imle_checkpoint.py`
- `imle_runtime.py`
- `offline_imle_eval.py`
- `tests/test_imle_integration.py`
- `tests/test_imle_legacy_checkpoint.py`
- `docs/IMLE_IMPLEMENTATION.md`

LeWorldModel変更:

- `eval_real_robot.py`: IMLEロード・選択・dry_run分岐、IMLEだけの実行前検証。
  明示的なIMLE CLI指定時は、未使用のFrankaSimEnv importを省略して
  headless環境のEGL初期化を避ける。既存CLI指定時のimport順序は維持。
- `config/eval/flip_mug.yaml`: `imle_checkpoint: null`、`imle_device: cuda`を追加。
  既存のdiffusion checkpoint値は維持。

stable-worldmodel新規:

- `stable_worldmodel/imle/{__init__,networks,model,losses,policy}.py`
- `stable_worldmodel/imle/provenance.md`
- `tests/imle/test_imle.py`
- `tests/imle/generate_reference.py`
- `tests/imle/s3fmp_reference.npz`

stable-worldmodel変更:

- `stable_worldmodel/policy.py`: GPCコンストラクタに`action_policy=`を追加。
  `diffusion_policy=`と属性は維持し、同じ候補生成器を参照する。
  rollout・正規化・報酬・argmax・action sliceは変更していない。
- `pyproject.toml`: 独立した`imle` extra (`torch-ema==0.3`, OmegaConf)。
- `uv.lock`: IMLE extraとtorch-emaのみ追加。既存パッケージversionは維持。
  `uv lock --check --offline`も成功。
- `stable_worldmodel/probing/flip_mug/plot.py`: 作業開始前からのユーザー差分。
  今回の実装対象ではない。

既存`train_policy.py`、DPモデル・scheduler・checkpoint loader、`jepa.py`、
World Model学習、`offline_dp_eval.py`は変更していない。
IMLEPolicyはDPの外部I/Oメソッドのみ継承し、DPの初期化・denoisingは実行しない。
これにより既存のaction buffer・gripper補強・制御経路をそのまま使う。

## s3fmpとの対応

参照はs3fmp `7b795a36f4f640c2b7ad2a192790a5513b70b23d`。
呼び出しは`main.py → build_policy → IMLEPolicy → _shared_step / inference`。
`ml-networks`はs3fmpのlockが固定する
`c672fb376b28b96c7c58cf6bbceb351176b0642d`を別途取得して確認した。

| 参照 | 移植先 |
|---|---|
| `policy.py::rs_imle_loss` | `stable_worldmodel/imle/losses.py` |
| `IMLEPolicy._generate / _gripper / _noise / inference` | `imle/model.py`, `imle/policy.py` |
| `IMLEPolicy._shared_step` | `IMLEPolicy.compute_loss` |
| ml-networksの1D U-Net・残差block・attention・sampling層 | `imle/networks.py` |
| `PolicyBase`のoptimizer・EMA・validation hooks | `train_imle_policy.py` |
| `main.py::train`、`models/cfg/xarm/imle.yaml` | 新規trainer・Hydra config |
| `eval.py`のcheckpoint復元・EMA適用 | `imle_checkpoint.py` |

RS-IMLEは軌道全体のL2距離。epsilon以下の候補を棄却し、選ばれた候補に勾配を流す。
epsilon=.01は固定、次元スケーリングあり。教師例を除外する`min_dist < dmax`の
境界挙動も参照のまま維持。単純なbest-of-K MSEとして実装していない。
独立gripper headは教師軌道を入力するL1学習、推論では生成軌道を入力する。
trajectory consistencyやepsilon更新則は参照にないため追加していない。

主な適応は画像encoderと条件入力。s3fmpのConvNet＋状態MLPを、
DPと同じoverhead/wrist ResNet18＋各128次元への線形射影に置き換えた。
状態の代わりに2カメラの特徴履歴を条件にする。生成器と損失は維持するが、
参照と同一encoder・同一条件入力であるとは主張しない。
画像前処理、HDF5の同一行観測/action窓、正規化統計、random_splitはDPに合わせた。
s3fmpのepisode splitとは異なる。

参照のN=128、Gaussian std=1、AdamW lr=1e-4 / weight_decay=.01、
U-Net幅64/128/256、kernel5、attention8 heads、gripper幅32/64/128、
EMA=.999、500epoch/patience20を維持。schedulerはなし。
実験指定に合わせhorizonを32/1から16/8、画像を224、batchを64から16に適応。
gripper係数は`.1 / pred_horizon`なのでT=16では.00625（参照T=32では.003125）。
すべてconfigで指定できる。

詳細と数値fixtureの出典はstable-worldmodelの
`stable_worldmodel/imle/provenance.md`を参照。
公開RSS実装はローカル参照の代用として移植していない。

## 実行方法

LeWorldModelディレクトリ、両repoをeditable installした環境を前提とする。
IMLE学習・EMA checkpoint推論にはtorch-ema extraが必要。

```bash
# 必要な場合のみ環境へ追加
python -m pip install -e '../stable-worldmodel[imle]'

# 学習: パスは実データへ置き換える（既定の /data/hida/... は本環境に存在しない）
python train_imle_policy.py \
  data.dataset.path=/absolute/path/push.h5 \
  output_dir=./imle_checkpoints
```

以下の`/absolute/path/latest.ckpt`を学習出力へ置き換える。
`gpc.diffusion_checkpoint=null`は必須。文字列`"null"`や`"none"`は使わない。
headless CLIでは以下の明示的な`policy_type=imle`または
`gpc.imle_checkpoint=...`指定により未使用シミュレータimportを回避する。

```bash
# IMLE単体 episode dry_run（ロボット・カメラを初期化しない）
python eval_real_robot.py --config-name flip_mug \
  policy_type=imle gpc.diffusion_checkpoint=null \
  gpc.imle_checkpoint=/absolute/path/latest.ckpt \
  eval.real_robot.dry_run=true eval.real_robot.dry_run_mode=episode \
  eval.real_robot.dry_run_dataset_path=/absolute/path/push.h5 \
  eval.real_robot.dry_run_episode_idx=0

# 上のコマンドに以下をそれぞれ指定すると別モード
# eval.real_robot.dry_run_mode=multi_sample eval.real_robot.dry_run_timestep=200
# eval.real_robot.dry_run_mode=chunk_rollout
# CPU実行: gpc.imle_device=cpu

# IMLE-GPC dataset dry_run（World Model checkpointとWM用統計が必要）
python eval_real_robot.py --config-name flip_mug \
  policy_type=gpc policy=/absolute/path/world_model_run \
  gpc.diffusion_checkpoint=null gpc.imle_checkpoint=/absolute/path/latest.ckpt \
  gpc.num_candidates=50 eval.real_robot.dry_run=true \
  eval.normalization_stats_path=/absolute/path/process_stats.npz \
  eval.real_robot.dry_run_dataset_path=/absolute/path/push.h5 \
  eval.real_robot.dry_run_episode_idx=0

# IMLE単体 実機推論（未実施。既存のロボット設定・安全制御を使う）
python eval_real_robot.py --config-name flip_mug \
  policy_type=imle gpc.diffusion_checkpoint=null \
  gpc.imle_checkpoint=/absolute/path/latest.ckpt \
  eval.real_robot.execute=true eval.real_robot.dry_run=false

# IMLE-GPC 実機推論（未実施）
python eval_real_robot.py --config-name flip_mug \
  policy_type=gpc policy=/absolute/path/world_model_run \
  gpc.diffusion_checkpoint=null gpc.imle_checkpoint=/absolute/path/latest.ckpt \
  eval.real_robot.execute=true eval.real_robot.dry_run=false
```

`dp_execute_steps`は名称・意味を維持しIMLEにも適用。
checkpointは自分の正規化統計、generator/両encoder/optimizer/EMA、epoch、
validation loss、architecture、noise dimension、horizon、full config、
`policy_type=imle`、8D action順序・m単位・xyzw/gripper規約を保存する。
推論はcheckpoint統計でのみ逆正規化する。

IMLE-GPC dry_runはデータセット観測と最後の観測をgoalに用いるオープンループ評価。
実際のJEPA rollout・候補評価・argmax・既存GPC recorderを通すが、
閉ループtask成功率を測るものではない。`proprio`列が必要。
可視化は既存DP/GPC関数を再利用するため、一部の凡例・ログ文字列にはDP表記が残る。
出力は`offline_imle/`、`offline_imle_gpc/`に分離する。

## 検証結果

実機・カメラ・robot commandの実行は行っていない。

- IMLE新規テスト10件成功（CPU, torch 2.5.1）。forward、loss、backward、optimizer、
  seed再現性、候補shape、chunk切出し、正規化、EMA/optimizer checkpoint復元を確認。
- 実際のtrainerで合成HDF5を1epoch学習し、latest/best checkpointの保存を確認。
- s3fmp参照ASTから出力・loss・全generator/headパラメータ勾配を保存。
  小型同条件モデルで完全一致。カメラencoder・状態条件を含むend-to-end同等性は未検証。
- IMLE単体3モードと複数action実行の評価、GPCへの候補入力・WM正規化・評価・
  最適候補・buffer・PNG/NPZ出力を確認。
- 実際のJEPA＋小型ViTでIMLE-GPC rolloutと選択を確認。
- `eval_real_robot.py` CLIから合成checkpoint/HDF5の単体episode dry_runが成功。
  同CLIで合成JEPA＋実際のARPredictorを保存・復元してIMLE-GPC dry_runも成功。
  生成物は`/tmp/imle-cli-smoke/`。
- 既存DP固定seed比較12条件で数値差0、RNG一致。
- DP-GPC固定seed比較3条件で候補・WM入力・mock予測・報酬・argmax・actionが数値差0、RNG一致。
- 実在する学習済みDP latest.ckpt（config記載の20261021_135400）について、
  変更前後のloader/Policyで、固定32×32画像（値17、変換x/255）・seed、checkpoint既定100stepの全推論を実行。
  逆正規化済みaction chunk `(1,8,8)` とRNG状態が完全一致。
  実行ログ`/tmp/imle-existing-dp.log`、結果`/tmp/imle-existing-dp-result.json`。
  ソース不変性とこの実checkpoint比較を再実行する独立テスト
  `tests/test_imle_legacy_checkpoint.py`も追加。2件とも成功。
- 既存のbuffer/gripper/safety mock回帰テストも実行し、比較対象のコマンドとログが一致。

### 成功・失敗・未実施の区別

別途実行した上記2件を除くLeWorldModelの全unittest discoveryは88件: 81成功、2skip、1failure、4error。
新規IMLE6件を含む。既存の失敗は次の通り:

1. `test_dry_run_dispatch_unchanged`: 既存テストが現在のHEADからchunk_rollout分岐を
   取り除いて比較するため失敗。変更前ソースでも同じfailureを再現。
2. `test_diffusion_evaluation_without_datasets_or_world_checkpoint`: テストnamespaceに
   `dp_img_transform`がなくNameError。変更前ソースでも同じerrorを再現。
3. ロボット関連3モジュール: robopyの依存`digit_interface`がなくimport error。
   PYTHONPATHに既存robopy/srcを加えて確認。機器を動かす形では補完していない。

skipは既存テストが`/home/hida/...`にDP checkpointを探すもの等。
実際の`/home/shonosukehida/...`checkpointの比較は上記の別検証で実行した。

stable-worldmodelのWM・policy・probingテストは87件: 78成功、9failure。
9件は既存WorldModelPolicyテストが空の`PlanConfig.action_space`を使い、
現在の実装がjoint/cartesianのみ受け付けるため。
変更前`policy.py`でも同じ9件のfailureを再現した。
WM単体のテスト群およびprobing historyは成功。
最初の実行環境ではOmegaConf/sklearn不足で収集エラーになったが、
依存の揃ったLeWorldModel環境で再実行して上記結果を得た。

未実施: CUDA学習、実データでの長時間学習、IMLEの学習済み実機checkpointでの性能評価、
学習済みJEPAを使ったIMLE-GPCのtask性能、実機推論。
「既存環境の全機能の互換性確認済み」とは扱わない。

再実行例:

```bash
# workspace root
PYTHONPATH=./LeWorldModel MPLCONFIGDIR=/tmp/imle-mpl OMP_NUM_THREADS=1 \
  LeWorldModel/.venv/bin/python -m pytest \
  stable-worldmodel/tests/imle/test_imle.py LeWorldModel/tests/test_imle_integration.py -q

# 変更前revisionとのソース比較＋実checkpoint推論（LeWorldModel cwd）
IMLE_LEGACY_DP_CHECKPOINT=/absolute/path/existing_dp_latest.ckpt \
  MPLCONFIGDIR=/tmp/imle-mpl OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover \
  -s tests -p test_imle_legacy_checkpoint.py -v

# LeWorldModel cwd: 既知の失敗・依存不足を含む全件
PYTHONPATH=../robopy/src MPLCONFIGDIR=/tmp/imle-mpl OMP_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v

# stable-worldmodel cwd: 既知の9failureを含む
MPLCONFIGDIR=/tmp/imle-mpl OMP_NUM_THREADS=1 ../LeWorldModel/.venv/bin/python -m pytest \
  tests/wm tests/test_policy.py tests/probing/test_flip_mug_history.py -q
```
