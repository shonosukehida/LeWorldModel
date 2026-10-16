# HierarchicalWorldModel（Step 3）

独立学習済みのLower JEPAとUpperJEPAを保持する推論用クラス。
モデルの構造・重みを変更せず、呼び出し時に両層をevalモードにする。
Tensorを共有している上下層はエラーにし、重みを黙ってコピーしない。
学習・結合Checkpoint保存・DP/GPCへの接続は実装しない。

## 復元

```python
from hierarchical_world_model import HierarchicalWorldModel

model = HierarchicalWorldModel.from_checkpoints(
    lower_checkpoint="/lower_run/lewm_epoch_10_object.ckpt",  # runディレクトリも可
    upper_checkpoint="/upper_run/last.ckpt",
    lower_stats_path="/lower_statistics/process_stats.npz",
    lower_config_path="/lower_run/config.yaml",  # 省略時はLowerの隣のconfig.yaml
    lower_weight=1.0, upper_weight=1.0,
    lower_cost_scale=1.0, upper_cost_scale=1.0,
    device="cpu",
)
```

`config/hierarchical_world_model.yaml`から`hydra.utils.instantiate()`でも生成できる。
必須パスのnullは実ファイルへ置き換える。

Lowerは`train.py`の`ModelObjectCallBack`が保存したモデルオブジェクトを、
既存`stable_worldmodel.policy.AutoCostModel`で復元する。
このローダーはファイルのprefixに`_object.ckpt`を付加するため、クラス側で
明示したファイル名からそのsuffixを外して渡す。ディレクトリは既存の選択方法を使う。
重みのみの`*_weights.ckpt`はこの既存ローダーでは復元できないため、
推測によるモデル構築はせずエラーとする。`*_object.ckpt`を指定すること。
Lower設定はCartesian action・frameskip=1・num_preds=1を要求する。

LowerのオブジェクトCheckpointは正規化統計を内包しない。
**そのLower学習に対応する統計ファイルを別途指定する必要がある。**
ファイルの由来が学習時と同じかどうかまでは自動検証できない。
欠落・8次元との不一致・非有限値・非正の標準偏差はエラーにする。

Upperは`upper_training.load_upper_checkpoint()`で復元し、そのモデルと
埋め込み統計・画像サイズ設定を使用する。外部のUpper統計は再計算・混用しない。
`actions_normalized=False`のStep 2 Checkpointを要求する。
既にモデルと統計を読み込んでいる場合は、コンストラクタへ上下層・各process・
`lower_img_size`・`upper_img_size`を直接渡せる。

## 入出力

`info`は**未変換の現在観測1時点**。候補軸を含めない。
画像はRGBのCHW形式で、uint8または学習で対応する未標準化floatを用いる。

| キー | shape | 内容 |
|---|---|---|
| pixels / wrist_pixels | (B,1,3,Y,X) | 現在の俯瞰・手先画像 |
| proprio | (B,1,8) | 物理値の現在状態 |
| goal / goal_wrist_pixels | (B,1,3,Y,X) | ゴール画像（get_cost時のみ必須） |
| goal_proprio | (B,1,8) | 物理値のゴール状態（get_cost時のみ必須） |

`action_sequences`は物理値の`[x,y,z,qx,qy,qz,qw,gripper]`、shape `(B,K,T,8)`。
Tは正でUpperのHの整数倍であること。切り捨て・パディングは行わない。
入力辞書・Tensor・ゴールを破壊的に変更しない。

- Lower画像・現在/ゴールproprioはLowerの画像サイズと統計で変換。
  行動はLowerのaction_cartesian統計で一度標準化して、既存JEPA.rolloutへ渡す。
- Upper画像・現在/ゴールproprioはUpperの画像サイズと統計で変換。
  行動は物理値のままUpperJEPA.rolloutへ渡し、その内部のbuild_macro_actionで
  絶対・差分をそれぞれのUpper統計で標準化する。絶対Quaternionの生値は維持する。

画像には各層とも学習時の`get_img_preprocessor()`を使用する。
行動・proprio・画像の変換済み入力を渡さない。値から変換済みかどうかは推測しない。
両モデルは同一deviceで利用する。移動は`model.to(device)`で行う。
推論時に勾配が不要な場合は、呼び出しを`torch.no_grad()`で囲む。
パラメータのrequires_grad設定自体は変更しない。

## API・コスト

```python
import torch
with torch.no_grad():
    trajectories = model.rollout(info, physical_action_candidates)
    costs = model.get_cost(info, physical_action_candidates)
# 上の2つは別の評価例。コストだけ必要ならget_costのみを呼ぶ。
best_candidate = costs["total_cost"].argmin(dim=1)
```

`rollout()`は`lower_rollout`・`upper_rollout`を返す。それぞれ既存モデルの
結果辞書で、`predicted_emb`に初期潜在を含む系列を保持する。
Lowerは`(B,K,T+1,D_lower)`、Upperは`(B,K,T/H+1,416)`。
H=4・T=8ならLowerは時刻t～t+8、Upperはt・t+4・t+8。
両層とも観測1時点から開始し、予測履歴を追加して直近最大3時点を使う。
Checkpointの履歴容量が3未満なら、その容量までとする。

`get_cost()`は内部で`rollout()`を1回だけ呼び、各層のencodeでゴール潜在を生成して
criterionへ渡す。各層のget_costを呼ばないため重複Rolloutは発生しない。
両層とも時刻t+Tの最終潜在とゴールの次元方向の二乗誤差和を計算する。

結果には上記2つのRollout辞書に加え、以下の`(B,K)` Tensorを含む。

| キー | 定義 |
|---|---|
| lower_cost / upper_cost | 各層の最終潜在と各層のゴール潜在の二乗誤差和 |
| normalized_lower_cost | lower_cost / lower_cost_scale |
| normalized_upper_cost | upper_cost / upper_cost_scale |
| total_cost | lower_weight * normalized_lower_cost + upper_weight * normalized_upper_cost |

重みとスケールは既定1.0。スケールは有限・正、重みは有限を要求する。
生コスト・正規化後・統合後のNaN/Infを検出すると明示的にエラーにする。
候補集合に依存するMin-Max正規化は行わない。
将来は検証データで求めた固定スケールを設定へ指定できる。

## 既存GPCとの境界

現行GPCPolicyは_prepare_infoで画像・proprioを変換し、World Model用行動も
事前に標準化する。またSolverのget_costはTensor単体を想定する。
本APIは未変換観測・物理値行動・内訳を含む辞書を扱うため、そのまま接続できない。
次のStepではGPC側の前処理を重ねず本APIへ物理値を渡し、選択処理でtotal_costを
取り出す明示的なアダプターが必要となる。本StepではGPCを変更しない。
