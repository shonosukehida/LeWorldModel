# 実ロボットFlip MugのUpperJEPA単独学習（Step 2）

```bash
python train_upper.py
```

設定は`config/train/upper_jepa.yaml`。`wm.horizon`（H、既定4）、
`wm.history_size`（K、既定3）、`wm.use_action_delta`、損失重み、IDM有効・無効を変更できる。
既存の下位モデルやCheckpointを渡す経路はなく、ネットワークを新規初期化する。
既定データ設定は`config/train/data/upper_flip_mug.yaml`で、入力は
`/data/hida/flip_mug/ep100_tm300_multiview_play/push.h5`。
別の場所にある同形式のデータは`data.dataset.path=/absolute/path/push.h5`で指定する。
`path=null`を明示した場合に限り、`dataset.name`に対応するキャッシュを探す。
既存`data/flip_mug.yaml`のdemo用パスは変更しない。
FrankaPush用の`data/upper_push.yaml`も従来用途のため保持する。

Flip Mug変換コードの`pixels`（俯瞰）、`wrist_pixels`（手先）、
`proprio`・`action_cartesian`（ともに`[x,y,z,qx,qy,qz,qw,gripper]`の8次元）、
`episode_idx`を使用する。`step_idx`は存在する場合に利用する。
ViT-tiny・224画素・patch14、潜在192+192+32=416次元、H=4、履歴3、差分ありを既定とする。
AdamW（lr=5e-5、weight_decay=1e-3）、Batch Size 16、10 epochs、
学習Seed 41・分割Seed 3072・Train比率0.9は`lewm.yaml`に合わせる。
Mixed PrecisionはStep 2の`bf16-mixed`を維持する。

CPUでの確認例（W&Bへの送信なし）：

```bash
python train_upper.py \
  trainer.accelerator=cpu trainer.precision=32-true wandb.enabled=false num_workers=0
```

学習は実行ごとに`outputs/upper_jepa/flip_mug/upper_jepa_<timestamp>/`を作成する。
出力先は`output_dir`で指定できる。

## データ・正規化

H=4、K=3では、観測は`[t,t+4,t+8,t+12]`、行動は
`[t:t+4]`、`[t+4:t+8]`、`[t+8:t+12]`。行動区間の終端は含まない。
最後の観測は最後のH先教師を作るために使用する。
全中間行を検査し、エピソード境界・欠損stepをまたぐ窓、短い窓、
非有限行動・必要観測の非有限proprioを含む窓を除外する。

`episode_idx`は必須。`step_idx`があれば同一エピソード内で厳密増加を要求し、
+1でない箇所をまたぐ窓を除外する。なければ保存行順の隣接を用いる。
分離したブロックに同じEpisode IDが再登場するデータはエラーにする。
画像はRGBのNHWC/NCHWに対応し、既存`get_img_preprocessor()`を用いる。

Episode IDを`seed`と`train_split`で分割してから、Train Episodeの行だけで
`action_cartesian`、`action_delta`、`proprio`の統計を計算する。
ValidationにはTrain側統計を適用し、統計計算へ含めない。
差分はStep 1の共有関数、統計は既存のddof=1 / safe standard deviation方式を用いる。
`process_stats.npz`は既存`load_normalization_process()`で読める。
`episode_split.json`に分割と入力パスを保存する。
学習内部で3種類の統計を生成するため、`create_process_stats.py --include-action-delta`を
別途実行する必要はない。このCLIは指定HDF5全体を集計するため、その出力を
Train専用統計の代わりに使わない。下位JEPA用の既存`process_stats.npz`は上書きしない。

Datasetは画像とproprioのみ変換し、行動は**物理値**の`(K,H,8)`を返す。
`UpperJEPA.actions_normalized=False`とし、`build_macro_action()`でのみ行動を標準化する。
絶対Quaternionは生値を維持。相対回転のみ単位長化・符号統一する。
差分ありのMacro-actionは`16H-8`次元（H=4で56）、差分なしは`8H`次元。
両者の統計は独立で、欠落・不正値に対する代替値は使わない。
この学習経路は3種類すべての統計を必須とする。

## 損失・ログ

- `pred_loss`：`predict(z[:,:-1], action_emb)[:, -1:]`と`z[:, -1:]`の416次元MSE。
  最終予測だけを教師とし、途中時点の予測損失は加算しない。H=4、K=3では教師は`z_(t+12)`。
- `overhead_pred_loss`、`wrist_pred_loss`、`prop_pred_loss`：同じ最終予測・教師の192 / 192 / 32次元ごとのMSE。
- `sigreg_loss`：K+1時刻の潜在を`(K+1,B,416)`として既存SIGRegへ入力。計算はfloat32。
- `idm_loss`：`(z[:, :-1],z[:,1:])`から正規化済みMacro-actionを推定するMSE。
  IDMは新規ランダム初期化後に凍結し、Optimizerから除外する。Forwardは勾配を追跡し、
  IDM損失でEncoderを更新する。IDM自身には勾配を蓄積しない。時間対応は全Kペアで維持する。

合計は`pred_loss + sigreg.weight*sigreg_loss + idm.weight*idm_loss`。
教師潜在は既存`lejepa_forward()`と同じEncoderで生成し、stop-gradientやEMAは追加しない。
下位`train.py`は全履歴位置に予測損失を掛けるが、上位層は最終時点のみを教師とする。
下位の学習処理は変更しない。固定IDMの重みもCheckpointに保存・復元する。
AdamWと既存LinearWarmupCosineAnnealingLRを使用し、Schedulerは各epoch終了時に進める。
LightningのMixed PrecisionとGradient Clippingを使用する。

既存`lejepa_forward()` / `spt.Module`と同じログ規約：

| 対象 | Train | Validation | 頻度 |
|---|---|---|---|
| 各損失 | `fit/<loss名>` | `validate/<loss名>` | epoch |
| 各損失 | `fit/<loss名>_step` | `validate/<loss名>_step` | step |

IDM無効時は`idm_loss`を記録しない。`log_dict`のstep/epoch/sync_dist設定も既存と一致する。
既定の`log_every_n_steps=50`は既存Trainerの既定値と同じ。
CSVにも記録し、W&Bは下位JEPAと別Run IDを作成する。
潜在空間・予測時間間隔が異なるため、上下層のMSE値を直接比較して優劣を判断しない。

## Checkpointと推論

毎epochの`upper_jepa-epoch=XXX.ckpt`と`last.ckpt`を保存する。
Lightning Checkpointにはモデル全体・IDM・SIGRegのstate_dict、Optimizer/Scheduler状態、
設定・Episode分割・3種類の統計値を含む。H/K/差分有無・物理値入力規約・
統計の相対パスも保存する。元データや外部統計ファイルを移動しても復元可能。

```python
import torch
from upper_training import load_upper_checkpoint

training_module = load_upper_checkpoint("/path/to/last.ckpt")  # CPU, eval mode
model = training_module.model
# raw_obs: pixels/wrist_pixels=(B,K,C,Y,X) or (B,K,Y,X,C), proprio=(B,K,8)
# physical_action_chunks: (B,K,H,8)
obs = training_module.observation_transform(raw_obs)
with torch.no_grad():
    z = model.encode(obs)["emb"]
    macro = model.build_macro_action(physical_action_chunks)
    predictions = model.predict(z, model.action_encoder(macro))
    next_upper_latent = predictions[:, -1]  # 最後の観測からH step先
```

Checkpointの観測変換はDatasetと同じ。既に変換済みの画像/proprioへ再適用しない。
`model.rollout()`にも同じ観測変換結果と物理値の行動列を渡す。
上下層のコスト統合とGPC接続は含まない。
