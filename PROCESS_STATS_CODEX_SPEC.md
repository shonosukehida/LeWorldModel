# `process_stats.npz` 生成スクリプト仕様書

## 目的

学習済みデータセットから、`eval_real_robot.py` が読み込める
`process_stats.npz` を事前生成する。

このスクリプトは diffusion policy の checkpoint を変更せず、評価時に
データセットを再読込して統計を生成する必要をなくすための standalone utility
とする。

## 対象リポジトリ

- `/home/hida/workspace/LeWorldModel`
- 主な既存実装: `eval_real_robot.py`
- 既存の統計形式実装:
  - `build_normalization_process()`
  - `save_normalization_process()`
  - `load_normalization_process()`

## 実装するファイル

次のファイルを追加する。

```text
LeWorldModel/create_process_stats.py
```

可能であれば、正規化処理本体は次の共通モジュールへ移動し、
`eval_real_robot.py` と新スクリプトの両方から利用する。

```text
LeWorldModel/normalization_stats.py
```

既存関数を単純にコピーして二重管理してはならない。共通モジュールへの移動が
大きすぎる場合は、まず既存関数を import して利用してもよい。ただし、
`eval_real_robot.py` の import によって実機・GUI・重いモデル依存を不要に
初期化しないこと。

## CLI

最低限、以下の引数を実装する。

```bash
python create_process_stats.py \
  --dataset /path/to/push.h5 \
  --output /path/to/process_stats.npz
```

引数仕様:

| 引数 | 必須 | 内容 |
|---|---:|---|
| `--dataset` | yes | 入力 HDF5 dataset のファイルパス。`~` を展開する |
| `--output` | yes | 出力する `process_stats.npz` のパス。親ディレクトリを作成する |
| `--keys` | no | 統計対象の列。既定値は `action_cartesian proprio` |

`--dataset` と `--output` は同一パスを許可しない。入力が存在しない場合は、
解決後の絶対パスを含む `FileNotFoundError` を発生させる。

## 入力データ

現在の `flip_mug` 用の標準実行例は次の通り。

```bash
cd /home/hida/workspace/LeWorldModel
python create_process_stats.py \
  --dataset /home/hida/.stable_worldmodel/datasets/flip_mug/ep200_tm300_multiview/push.h5 \
  --output /home/hida/.stable_worldmodel/datasets/flip_mug/ep200_tm300_multiview/process_stats.npz
```

デフォルトで必要な dataset column は以下とする。

```text
action_cartesian
proprio
```

`goal_proprio` は入力 HDF5 column として要求しない。`proprio` と同じ
processor を共有する goal alias として、既存の評価側フォーマットに従って
復元可能にする。

## 統計計算

各対象 column について、既存の評価実装と同じ規則で計算する。

- NaN を含む行を除外する
- `mean_`: 列ごとの平均
- `scale_`: 不偏標準偏差（`ddof=1`）
- 標準偏差が `1e-4` 未満の場合は `1.0` に置換する
- `raw_min_`, `raw_max_` を保存する
- min/max の正規化後の値も保存する
- `goal_proprio` は `proprio` と同じ processor を参照する

既存の `SafeStandardScaler` と serialization 形式を再利用し、独自の npz 形式を
追加してはならない。

## 出力形式

`eval_real_robot.py` の `load_normalization_process()` がそのまま読み込める
形式にする。

必須 metadata:

```json
{
  "version": 1,
  "keys": ["action_cartesian", "proprio"],
  "action_key": "action_cartesian"
}
```

各 processor について、少なくとも以下の配列を保存する。

```text
processor_{index}_mean
processor_{index}_scale
processor_{index}_raw_min
processor_{index}_raw_max
processor_{index}_normed_min
processor_{index}_normed_max
processor_{index}_eps
```

出力は `np.savez_compressed()` で保存する。

## 評価コードとの統合

`eval_real_robot.py` では、事前生成した statistics がある場合にそれを読み込む。

diffusion policy の推論では、action の normalization は checkpoint 内の
`action_mean` / `action_std` を使用する。一方、評価起動時の共通 `process` は
この `process_stats.npz` から復元する。

また、`policy_type: diffusion` かつ `exe_probe: false` の場合は、probing 用の
dataset をロードしないようにする。以下のロードは
`if cfg.eval.probing.exe_probe:` ブロック内に限定する。

```python
dataset = get_dataset(cfg, cfg.eval.probing.dataset_name)
val_dataset = get_dataset(cfg, cfg.eval.probing.val_dataset_name)
```

これにより diffusion policy 単体の実機推論では、probing 用 train/validation
dataset を不要にする。

## エラー処理

以下の場合は、原因と対象パスまたは column 名を含む明確な例外を発生させる。

- dataset file が存在しない
- HDF5 に指定 column が存在しない
- 有効な行が 0 件である
- `action_cartesian` が存在しない
- 出力先が入力 dataset と同一である
- 既存 npz の上書きが禁止されている場合

出力ファイルは、統計計算と validation が完了してから一度だけ保存する。
中途半端な npz を残さないこと。

## 検証

最低限、以下を実装・実行する。

1. fixture HDF5 から `process_stats.npz` を生成できる
2. 生成した npz を `load_normalization_process()` でロードできる
3. ロード結果に以下の key が含まれる
   - `action_cartesian`
   - `proprio`
   - `goal_proprio`
4. `action_cartesian` の mean/std が期待値と一致する
5. NaN 行が統計計算から除外される
6. 欠損 dataset column で明確に失敗する
7. `eval_real_robot.py` の diffusion policy 経路が world model checkpoint を要求しない
8. `exe_probe: false` の場合に probing 用 dataset を要求しない

利用可能な既存テストフレームワークに合わせ、テストを追加する。少なくとも
CLI の smoke test と npz の round-trip test を用意する。

## 完了条件

- 指定した HDF5 dataset から単独コマンドで `process_stats.npz` を生成できる
- 生成物を現在の評価コードで読み込める
- 学習時に作成された diffusion checkpoint 内の action statistics と矛盾しない
- diffusion policy 単体の評価で world model checkpoint と probing dataset が不要
- 既存の `world_model` / `gpc` / probing 有効時の挙動を変更しない