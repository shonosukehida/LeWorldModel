# DP/GPC 行動chunk実行の変更・検証報告（2026-10-27）

実装と、実機なしで実施可能な比較を完了した。ただし、学習済みロボットcheckpointによる数値比較と実機での挙動同等性は未検証である。以下の一致結果は、CPU上の合成checkpointおよびモック環境での結果であり、実機同等性の保証ではない。

## 1. 変更ファイル・関数

- `../stable-worldmodel/stable_worldmodel/policy.py`
  - `DiffusionPolicy.get_action_chunk()`を追加。既存の`predict_action()`を使用し、行動列を2次元に展開して従来と同じ優先順位（action_cartesian → action → action_joint）で逆正規化し、(B, Ta, D)へ戻す。
  - `DiffusionPolicy.get_action()`は変更していない。
  - `GPCPolicy.get_action_chunk()`へ従来の候補生成・評価・選択処理を移し、最良候補の全行動列を返す。
  - `GPCPolicy.get_action()`はchunkの先頭を返す互換APIとして維持。
- `eval_real_robot.py`
  - `_DPActionBuffer`を追加。整数の`1 <= dp_execute_steps <= policy.action_horizon`を検証。先頭N個をコピーして保存し、1stepずつ取り出す。
  - `_run_xarm_task_with_env()`のDP/GPC呼び出しをバッファ経由へ変更。値が1でも実際のpolicyには`get_action_chunk()`を呼ぶ。
  - 既存の補強関数へバッファを渡すため、発動条件、monotonic deadline、hysteresis/rearmの処理は変更していない。補強発動時に残りを破棄し、補強中は推論せず、終了後は最新履歴で推論する。
  - 観測履歴の更新位置、補強→env.executeによる安全制限の順序は維持。finallyで停止・例外・通常終了時のバッファを破棄。
- `config/eval/flip_mug.yaml`: `eval.real_robot.dp_execute_steps: 1`を追加。コード上も省略時は1。
- `tests/test_dp_action_chunks.py`: 旧新比較、実際のDDPM処理、複数step、停止・例外、安全制限の6テストを追加。
- `tests/test_gripper_reinforcement.py`、`tests/test_xarm_robot_inference.py`: モックpolicyをchunk APIに対応し、AST抽出にバッファクラスを追加。
- `tests/baseline_dp_chunks/`: 変更直前の2ファイルを保存。READMEにcommitとSHA-256を記録し、git showの内容と一致することも確認した。

## 2. 比較基準

- LeWorldModel: `00cbd1bc6e915fdd70c00a644af4397c38bb24e4`
- stable-worldmodel: `6aa6d7cbdea8fbd6c53f8da7b7e557bb7b7271f6`

保存した旧ソースから旧クラスと旧制御ループを実行して比較した。新しいget_action()同士だけの比較ではない。重量依存のあるモジュール全体のimportを避けるため、テストはASTで実際のクラス・関数を抽出して実行する。

## 3. DP数値一致

実際のResNet18ObsEncoder、ConditionalUnet1D、DDPMSchedulerを使用。テスト用の合成重みを保存・読み込みして旧新で共有した。CPU、float32、B=1、To=2、予測長8、Ta=3、D=8、DDPM学習timestep数8、推論4step。U-Netは小さい構成(16,32)であり、本番構成・本番重みではない。観測は同一の32×32画像tensor。seedは0、42、99。

action_cartesian / action / action_joint / 逆正規化なしの4条件×3 seed、計12条件で、旧get_action(info)と新get_action_chunk(info)[:,0]は**要素単位で完全一致、最大絶対誤差0**。各条件でtorch RNG状態を推論前に復元し、推論後のRNG状態も一致した。新get_action()も旧出力と完全一致。chunk形状は(1,3,8)。正規化にはStandardScalerを使用した。

## 4. GPC数値一致

実際のDP候補生成、世界モデル入力への正規化、報酬shape検証、argmax、最良候補抽出を実行。4候補、3 seedで比較した。目標も同一。JEPAの代わりに累積和による予測モックと、目標との二乗距離報酬を使用した。

以下はすべて旧新で**最大絶対誤差0**。

- DP候補列（逆正規化済み）
- 世界モデルへ渡す正規化済み行動列
- モック予測列
- 各候補の報酬
- argmaxで選ばれたindex
- 旧get_action()と新chunk先頭の行動値、および新get_action()の値
- 推論後のtorch RNG状態

実JEPA rolloutおよび本番報酬での出力一致は未検証。実装上、これらのコードには変更していない。

## 5. dp_execute_steps=1 制御ループ互換性

旧新の実際の制御ループを、同じ時計・観測・モックpolicy・モック環境で複数step実行。DP/GPCそれぞれについて、補強OFF、rearm_threshold=1の従来再発動、rearm_threshold=.1のhysteresis、step途中停止の計8シナリオを比較した。

**推論step番号・模擬時刻、観測履歴の全値、モック環境に渡した指令の全値、保存HDF5の全datasetが完全一致**。比較datasetには画像、proprio、commanded_action、target_qpos、safe_qpos、qpos/qvel、EE姿勢、gripper、補強active、raw policy gripper command、実測gripper、timestamp、goalを含む。NaN箇所も一致。

具体例：10Hz、補強継続.25秒、policy gripper出力 .9 → .1 → .8 の従来再発動では、推論はstep 0、3、4。補強activeは[1,1,1,0,1,1]。補強のgripper目標は64/84、固定するEE姿勢は発動stepで観測した姿勢。step 3の再推論時の画像履歴は[2,3]、wristは[12,13]。hysteresis条件ではstep 3の.1でrearmし、そのstepでは発動しない。補強OFFでは毎step推論する。

ここで一致した「時刻」はモック時計である。実CPU/GPUの計算時間や実機のdeadline付近の挙動までは証明していない。

## 6. 複数stepバッファ、補強、安全制限

- N=3、6制御step、補強OFFでは推論stepは[0,3]。行動は先頭3個を順に消費する。バッファ単体では各行動に異なる値0..23を与え、順序とコピー保持を検証した。
- バッファ使用中も観測履歴を更新し、step 3の再推論に[2,3]が渡ることを確認。
- step 0で補強が発動した場合、未実行の閉じ動作を破棄。step 1、2では推論なし、step 3に最新観測で再推論。activeは[1,1,1,0,0,0]。古い閉じ動作で再発動しないことを確認。
- 補強目標64/84に対してgripper max_delta=.1を適用すると、実行値は[.1,.2,.3]と段階制限されることを確認。補強後に安全制限が適用されている。
- 旧新の実際のclip_cartesian_actionをモックループに組み込み、位置delta、workspace、姿勢、gripperの制限を通した指令を比較。DP/GPCとも全step一致。最初の入力[.7,.1,.2,0,0,0,1,.9]に対し、制限後は[.51,.01,.29,0,0,0,1,.1]（float32丸めあり）で一致。
- 通常終了、SIGINT停止、env.execute例外の後にバッファが空であり、環境closeが1回行われることを確認。
- N=0,-1,4(Ta=3),1.5,True,"1"を拒否。chunkの不正次元、B≠1、長さ不足を拒否。

## 7. 回帰テストと既存推論への影響

- 新規比較テスト: **6件成功**。
- 既存gripper reinforcementテスト: **16件成功**。
- 既存offline DP（episode / multi_sample）: **13件中12件成功、CUDA依存1件skip**。
- LeWorldModel/tests全体: **68件中62件成功、2件skip、4件error**。全体成功とは報告しない。
  - 1件はtest_process_statsのASTテストnamespaceにdp_img_transformがない既存fixtureエラー。保存した旧eval_real_robot.pyで同じテストを実行し、同じNameErrorを確認済み。
  - 3件はXArm関連モジュールのimport失敗。robopy/srcをPYTHONPATHへ追加して再試行してもdigit_interface / dynamixel_sdk不足で実行不可。再試行は5件中2件成功、3件import error。
- git diff --check: 両リポジトリで成功。

ResNet、U-Net、scheduler、学習処理、checkpoint形式、JEPA構造・重み・rollout、本番報酬・候補選択処理を変更していない。DPのget_actionと候補生成も変更していない。DP/GPC以外の分岐、episode/multi_sampleのdry run経路も変更していない。N=1ではchunk全体の逆正規化・コピー分の処理が増えるが、DDPM推論回数は増えない。N>1では補強中を除きN制御stepごとの再推論になる。

再現コマンド（workspaceルート）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=LeWorldModel:robopy/src MPLCONFIGDIR=/tmp/codex-mpl \
  LeWorldModel/.venv/bin/python -m unittest discover -s LeWorldModel/tests -p test_dp_action_chunks.py -v
```

## 8. 未検証・残存リスク

設定で指定された学習済みDP checkpoint `/home/hida/.stable_worldmodel/checkpoints/flip_mug/policy/diffusion_policy/ep200_tm300_multiview_demo/seed41/20261003_085301/best.ckpt` は存在しない。したがって「同一本番checkpoint・本番前処理での旧新比較」は未完了。実JEPA checkpointを使った予測・本番報酬の比較も未実施。合成モデルテストはsingle-view、B=1、CPUのみであり、本番multiview・GPU数値一致を保証しない。

実機でのみ確認できるSDK指令到達、低レベル平滑化、IK/FKの実測結果、通信遅延、実測グリッパ、admittance、壁時計での推論完了時刻、補強deadline近傍の挙動は未検証。ロボットは動かしていない。安全制限関数とモック送信値の一致は確認したが、物理挙動が同じとは断定しない。実機投入前には、本番checkpointを使う数値比較と、上記の実機確認が残る。
