# Phase 2 実装・検証報告　2026-11-18

## A. Executive Summary

Flip Mugの実HDF5から、DIAMOND単体で学習、再開、1時刻予測、複数候補の自己回帰予測、画像・動画評価を実行した。CPU・FP32で公式RGB／離散行動の演算と全勾配がbitwise一致した後に、ロボット用の連続8次元行動と2視点6chを追加した。

納品する32 stepのcheckpointは単一窓の配線検証用。未使用episodeでは背景色は出るが、4時刻目のマグの輪郭を復元できなかった。画像生成APIはPhase 3のオフライン接続検証に使える。候補順位付けに必要な生成品質、GPUでのレイテンシ、実機周期との対応は未検証。作業はオフラインで完結した。

| 完了条件 | この環境で確認した範囲 |
|---|---|
| 固定source・差分記録・公式数値一致 | 固定Git blobとSHA-256、MIT、独立参照実行、CPUの厳密一致 |
| 8D・2視点・HDF5 sequence・学習 | 実100 episodeから6 frameを読み、FP32の順伝播、逆伝播、勾配clipと更新を確認 |
| checkpoint・1step・H-step・K候補 | 保存前後の同一画像、再開の完全一致、B>1かつK>1を含むshape・乱数のテストに合格 |
| 画像・動画・オフライン診断 | 2つの検証episode。64角GT／元解像度GT、行動診断、時刻別・視点別集計を保存 |
| 速度・VRAM | CPUは優先条件まで実測。CUDA用同期・VRAM計測を実装。GPU実測は未完 |
| 後方互換性 | 調べた既存テストの失敗8件は変更前でも再現。対象外のテスト、スキップ条件はH参照 |

## B. 実装内容

対象は両repoの `real_robot_multiview`。開始時に通常のcheckoutは両方とも `real_robot_jepa_dp`、LeWorldModel側に未コミット変更があったため、同じローカルrepoから `diamond-phase2/{LeWorldModel,stable-worldmodel}` に専用worktreeを作成した。適用元はLeWorldModel `afef844ca6fd23f418acffe4e6e09ca1c1a6bdd9`、stable-worldmodel `2b5cc0fdcfb72b64562630ca0ec504ae0e5a519a`。stable-worldmodelの納品commitは `0552164a1c578cc019da8a99668ea3c69d275da4`。

| repo / 新規ファイル | 役割 |
|---|---|
| stable-worldmodel · `stable_worldmodel/wm/diamond/source/` | upstreamの対象ソース、AdamW/schedulerの出典、公式agent/trainer YAML、MIT LICENSE。原本のPythonには `.source` を付け、runtimeのimportから隔離 |
| 同 · `manifest.json`, `upstream/{blocks,optim,types}.py`, `upstream/diffusion/{inner_model,denoiser,diffusion_sampler}.py`、各 `__init__.py` | 固定commit・各hash・変更一覧。Batch型を相対importし、公式optimizerの2関数を抽出。samplerの初期／churnノイズに任意の専用generatorを渡せる |
| 同 · `action_embedding.py`, `preprocessing.py`, `history.py`, `model.py` | 物理行動の正規化、共有Linear、letterbox・逆変換、4観測／3行動の履歴更新、1step・H-step・K候補API |
| 同 · `checkpoint.py`, `tests/diamond/test_{official,robot}.py` | schema検査、保存・復元、独立参照との一致、画像・行動・候補軸のテスト |
| LeWorldModel · `diamond_dataset.py`, `train_diamond.py`, `config/train/diamond.yaml` | HDF5 schema・fingerprint・episode split、固定WM統計、6 frame窓、Hydra学習・評価loss・再開 |
| 同 · `eval_diamond.py`, `benchmark_diamond.py` | 検証episodeの未来画像、時刻別評価、同一noiseの行動診断、動画、候補batchとmicrobatchの計測 |
| 同 · `tests/test_diamond_{dataset,checkpoint}.py`, 本書 | 時間indexと境界、将来画像を書き換えた場合、実際の学習入口経由の再開、厳格ロード |

公式固定commitは `5bcd1599755b4f2fae8e5e079e02f0728e174965`。演算の原本と実行コピーを同梱し、参照側のテストだけが原本を独立したmodule名で実行する。runtimeからAtari agent、policy、reward、terminationのimportは発生しない。[公式ツリー](https://github.com/eloialonso/diamond/tree/5bcd1599755b4f2fae8e5e079e02f0728e174965) · [source/patch manifest](../../stable-worldmodel/stable_worldmodel/wm/diamond/manifest.json)

ロボット側のモデル差分は、離散lookupを各時刻の共有 `Linear(8,64,bias=False)` に置き換え、画像chを3または6に設定すること。6ch時のconv入力は30ch、出力は6ch。4埋め込みをflattenした256次元にnoise条件を加え、公式のcond_projとAdaGroupNormに渡す。Linear重みには公式AdamWのLinear用decay規則が適用される。オフラインのepisode分割・window抽出・学習step数はロボット用の設定。

## C. 公式一致テスト

RGB各64×64、履歴4、離散18行動、公式UNet（各段64ch・深さ2、中央attention）、FP32、同一重み・RNG。復元経路を十分に検査するため、比較用重みでは通常0で初期化する残差・出力の一部も決定的な非ゼロ値にした。通常の学習は公式の初期化を使用する。比較は有限値、shape、dtype、値のバイト列。絶対／相対許容差を導入していない。

| 比較対象 | 検査内容 | CPU結果 |
|---|---|---|
| sigma、offset、noisy image | log-normal sample、同一ch内で空間一定のoffset、pixel noise込みの画像、4つのpreconditioner | bitwise、最大絶対差0 |
| 行動・UNet・denoiser | action embedding、条件入力、UNet出力と各段の特徴、予測残差、量子化前の画像、clamp・byte truncation後の画像 | 同上 |
| 6 frameの学習 | 教師x4/x5、各4行動、2番目の条件末尾が量子化した再構成、異なる有効mask、2損失の平均、全パラメータの勾配 | 同上。2番目のlossから1番目の再構成前出力への勾配なし |
| 推論 | 初期unit Gaussian、3段Eulerの全中間画像、生成結果を戻す3時刻の連鎖 | 同上 |

GPUでの一致、公式requirementsのTorch 2.4.1環境と今回のTorch 2.5.1間の一致は未検証。ここでのoracleは同じCPU・ライブラリで実行した固定commitそのもの。[ロボット拡張前の結果](../../results/parity.log) · [最終テスト結果](../../results/diamond-acceptance.log)

## D. ロボット・データ・学習の検証

- 行動は絶対target `[x_m,y_m,z_m,qx,qy,qz,qw,gripper]`。0=open、1=closed。datasetに隣接するWM `process_stats.npz` を既存loaderで読み、`(raw-mean)/scale` を学習・推論で共有する。既存scalerと同じ値、one-hotでのlookup等価性、入力・重みの勾配、同一noiseで行動変更による出力差を確認した。
- Overhead RGB→Wrist RGBの順で結合する。高さ480・幅640を高さ48・幅64へアスペクト比維持・bilinear antialias・align_corners=Falseで縮小し、byteに最近接丸め、上下8行を黒で埋め、[-1,1]へ写す。全視野を残すためcropしない。元解像度への表示はpad除去後にbilinearで復元。学習、推論用入力、評価GTに同一関数を使用する。
- 今回のHDF5は `ep100_tm300_multiview_play_normal_brightness/push.h5` のみ。両画像はfloat32 `(30000,3,480,640)`、`action_cartesian/proprio` はfloat32 `(30000,8)`。2視点版はこれら4列および `ep_offset/ep_len/episode_idx/step_idx` が必須。単眼版はwrist列を省ける。leader、episode名などは任意。proprioはschemaで確認し、画像予測の入力には使わない。
- 6観測と5行動だけを読む。1番目はx0..3・a0..3→x4、2番目はx1..3＋量子化再構成・a1..4→x5。推論では最初の4画像以降、生成画像だけで履歴を更新する。seed 123、90/10 episode、全train 26,550窓・val 2,950窓。300行の各episodeから295窓。学習窓は一様・復元抽出。全action列が同一のepisodeペアは検査した3データセット間で0件。部分的・物理的に関連する撮影の有無までは判定できない。
- 指紋はschema・全index・全action・各viewの先頭末尾画像のSHA-256。全HDF5のchecksumではない。今回の値は `c8c43382acb385b030bfc60e60561d6b538c93ae09b4626a7530656e775e81be`。WM統計ファイルのSHA-256は `fb8cec05c92411361dcacd74b6cbbd54d173daf160e7c7fe7073ca0e0b45c743`。この統計は全100 episodeで再計算した平均・scaleに完全一致した。train 90 episodeだけとは最大絶対差がそれぞれ0.017135、0.025895で、検証側の行動分布が正規化に含まれる。[分割・統計・schema](../../results/overfit/data.json) · [由来の検査](../../results/data-provenance.log)

| 実行 | 結果 |
|---|---|
| 実窓batch 1、4 step、公式100 step warmup | grad normはclip前6.793–7.555、step 4のtrain loss 1.232285、val 1.815840。公式schedulerにより最初の更新lrは0 |
| 4→5 step再開と独立した連続5 step | step 5のloss 1.399486、val 1.814855。モデル236 tensor、AdamW状態、scheduler、Python/NumPy/Torch/window RNG、実観測からの出力が完全一致 |
| 1訓練窓、batch 1、32 step、warmup 0 | 固定窓・固定noiseのlossは1.776794→1.274873（16）→1.075675（32）。最終val 1.136382。少数窓で損失が下がるところまで確認 |
| 新規テスト6件 | 公式一致、単眼・2視点、pack/unpack、行動感度、B/K対応、異常shape、episode分離、保存直前の出力とロード後の一致、中断・復元、異常checkpointでの失敗を確認 |

通常設定はAdamW lr 1e-4、eps 1e-8、対象パラメータのdecay 1e-2、grad norm 1、100 optimizer stepsのwarmup、batch 32、6 frame/2対象、EMAなし。保存名はstep単位で、同名を上書きしない。再開時は別の出力先に継続する。epochは累積抽出窓数を窓総数で割った整数。[smoke](../../results/smoke/metrics.jsonl) · [固定窓](../../results/overfit/metrics.jsonl) · [再開照合](../../results/real-resume-check.log)

## E. 実行方法

以下は今回使用した環境と引数。出力名は実績値で、再実行は未使用名を指定する。`output` を省略した学習では時刻付きディレクトリを作る。評価に限れば配布済みの32 step checkpointをそのまま使える。

~~~bash
cd /home/shonosukehida/work/diamond-phase2/LeWorldModel
export PYTHONPATH=/home/shonosukehida/work/diamond-phase2/stable-worldmodel:/home/shonosukehida/work/diamond-phase2/LeWorldModel
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/diamond-mpl
PY=/home/shonosukehida/work/LeWorldModel/.venv/bin/python
R=../results
CKPT=$R/overfit/step-000032.pt

"$PY" train_diamond.py --config-name diamond device=cpu batch_size=1 steps=4 validate_every=4 save_every=4 output=$R/smoke
"$PY" train_diamond.py --config-name diamond device=cpu batch_size=1 steps=5 validate_every=4 save_every=4 resume=$R/smoke/step-000004.pt output=$R/resumed
"$PY" train_diamond.py --config-name diamond device=cpu batch_size=1 steps=5 validate_every=4 save_every=5 output=$R/continuous
"$PY" train_diamond.py --config-name diamond device=cpu batch_size=1 train_windows=1 steps=32 warmup_steps=0 validate_every=16 save_every=32 output=$R/overfit

"$PY" eval_diamond.py --checkpoint "$CKPT" --horizon 1 --starts 10 --num-episodes 1 --output $R/eval-h1
"$PY" eval_diamond.py --checkpoint "$CKPT" --horizon 4 --starts 10 --num-episodes 2 --diagnostics --video --output $R/eval-final
"$PY" benchmark_diamond.py --checkpoint "$CKPT" --device cpu --k 1 5 --horizon 1 --steps 3 --microbatch 0 1 --warmup 1 --repeats 5 --output $R/bench-cpu.json
"$PY" benchmark_diamond.py --checkpoint "$CKPT" --device cpu --k 20 --horizon 8 --steps 3 --microbatch 0 --warmup 1 --repeats 3 --output $R/bench-priority.json

cd ../stable-worldmodel
"$PY" -m pytest -q -p no:cacheprovider tests/diamond ../LeWorldModel/tests/test_diamond_dataset.py ../LeWorldModel/tests/test_diamond_checkpoint.py
~~~

本格学習の候補（未実行）は、LeWorldModelのworktreeで以下。入力は上記100 episode、出力は `outputs/diamond/日時/step-*.pt`。configの既定は1,000 optimizer steps、500 step間隔の保存。checkpointは1個約50.8 MiB、2個で約102 MiB。CPU batch 2の2 step（読込、validation、保存込み）の実測は8.03秒、プロセス最大RSS約1.26 GiB。GPU batch 32の必要VRAM・所要時間は未測定。16/32-bit混合は使用していない。GPUで最初に小batchで余裕を計測してから本格設定を選べる。[CPU資源計測](../../results/train-resource.txt)

~~~bash
cd /home/shonosukehida/work/diamond-phase2/LeWorldModel
CUBLAS_WORKSPACE_CONFIG=:4096:8 "$PY" train_diamond.py --config-name diamond device=cuda batch_size=32 steps=1000
"$PY" benchmark_diamond.py --checkpoint "$CKPT" --device cuda --k 1 5 20 50 --horizon 1 4 8 --steps 1 3 5 --microbatch 0 5 --output ../results/bench-gpu.json
~~~

## F. 推論例と画像品質

32 step checkpointのSHA-256は `077a5912b0fd378a5508ee498bd504634b86e050d5853f56f80873b4fa3174e0`。評価はvalのepisode 21、23のrow 10開始。観測10–13、未来14–17。各窓の入力 `(1,4,6,64,64)`、正規化済み過去行動 `(1,3,8)`、未来行動 `(1,4,8)`、出力 `(1,4,6,64,64)`。全フレーム有限、出力範囲は2 episode合計で[-1, 0.678432]。1step専用実行のepisode 21はMSE 0.016136。2 episodeを集計すると1step 0.015987、H=4の全時刻平均0.016631。

| 未来時刻 | Overhead 64角 | Wrist 64角 | Overhead 元解像度 | Wrist 元解像度 |
|---:|---:|---:|---:|---:|
| 1 | .016094 | .015881 | .020806 | .017213 |
| 2 | .016479 | .016091 | .021257 | .017563 |
| 3 | .016809 | .017015 | .021573 | .018703 |
| 4 | .017260 | .017423 | .022166 | .019272 |

MSEはRGB強度[0,1]に対する画素平均。64角は黒padを含み、元解像度はpadを除いて拡大した予測を実画像と比較する。episode 21の4時刻目では元GTで明瞭なマグの輪郭が、両方の予測で失われる。現在の品質ではマグの姿勢や接触を読み取れない。[短期checkpoint](../../results/overfit/step-000032.pt) · [64角比較](../../results/eval-final/ep-021-start-0010/t004-64.png) · [元解像度比較](../../results/eval-final/ep-021-start-0010/t004-original.png) · [動画](../../results/eval-final/ep-021-start-0010/comparison.mp4)

同一の初期観測・過去3行動・noise seedを固定し、未来行動だけを変えた。記録行動による予測に対する出力差MSEの平均は、時刻shuffleで1.08e-6、別episodeで2.57e-5、正規化後のゼロで4.34e-5。ゼロは統計上の平均targetに相当する。未実行行動の真の未来画像がないため、この差は行動感度を示す。記録周期が未確定のため、動画の4 fpsは再生時の設定。各時刻のPNG、両episodeのMP4、per-window/per-viewの生の値を保存した。[行動別比較](../../results/eval-final/ep-021-start-0010/t004-actions.png) · [全指標](../../results/eval-final/metrics.json)

## G. 単体推論速度・メモリ

Intel Core i7-12700、Torch threads 2、Python 3.10.16、Torch 2.5.1+cu121、FP32、B=1。各組合せで1回warmup後、model APIの呼出しから候補出力の組立てまでを計測。モデルロード、HDF5、描画、DP候補生成、スコア計算は計測の外。CUDA経路は計測の前後で同期し、peak allocated/reserved両方を記録する。

| K | H | denoise段数 | 同時候補数 | n | 平均 ms | p50 ms | p95 ms | p99 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 3 | 1 | 5 | 94.22 | 94.15 | 94.45 | 94.47 |
| 5 | 1 | 3 | 5 | 5 | 493.31 | 493.92 | 495.61 | 495.87 |
| 5 | 1 | 3 | 1 | 5 | 471.85 | 471.86 | 472.22 | 472.23 |
| 20 | 8 | 3 | 20 | 3 | 18159.99 | 18191.42 | 18200.72 | 18201.55 |

CPUではこのK=5/H=1で1候補ずつのほうが速かった。少数反復のpercentileであり、制御時の上限値は得られていない。カーネルのデバイス情報にはGeForce RTX 3070 Tiがあるが、作業環境には `/dev/nvidia*` がなく `torch.cuda.is_available() == False`。ホスト側の追加確認は中断。GPUでの学習・推論、VRAM容量・ピーク、GPU全条件sweepはいずれも未測定。[CPU計測](../../results/bench-cpu.json) · [優先条件の生の3回分](../../results/bench-priority.json)

## H. 後方互換性

既存の実装・既存config・requirementsは追記や変更なし。既存JEPA、DP、IMLE loader、GPCをimportした新規Pythonプロセスで、DIAMOND moduleが `sys.modules` に入らないことを確認した。以下は選定した関連unit/integration suiteの結果。[import検査](../../results/import-isolation.log)

| 対象 | 実行した既存テスト | pass / fail / skip |
|---|---|---:|
| stable-worldmodel | `test_gpc_execute_first_candidate.py`、`test_latent_goal_reward.py`、`tests/imle/` | 23 / 0 / 0（別に28 subtests pass） |
| LeWorldModel | `test_action_normalization`、`test_process_stats`、`test_image_preprocessing`、`test_gpc_first_candidate_integration`、`test_imle_legacy_checkpoint`、`test_dp_action_chunks`、`test_imle_multiview_regression`、`test_gpc_visualization`、`test_imle_integration`、`test_imle_dispatch`、`test_offline_gpc_eval` | 59 / 8 / 2（別に42 subtests pass） |
| LeWorldModel追加 | skipのうち既存DP実checkpointのパスを `IMLE_LEGACY_DP_CHECKPOINT` で指定し、物理action chunkとRNGを比較 | 1 / 0 / 0 |

8失敗の全テストIDを両方の無変更の監査HEADで再実行し、同じ8件が失敗した。内訳は、normalizationテストが関数の一部を抽出してコンパイルする際の `return outside function` 1件、過去commitとのGPCソース／offline evaluator／YAML比較が3件、テストfixtureにIMLEのproprio統計がないものが4件。残るskipは既存テストの固定絶対パスに合うDP checkpointがないため。Phase 2で追加したコードによる新たな失敗は選定範囲で検出されなかった。[既存LeWorldModel](../../results/legacy-le-final.log) · [変更前8件の再現](../../results/baseline-eight.log) · [実DP](../../results/trained-dp.log) · [既存stable-worldmodel](../../results/legacy-stable-final.log)

## I. 未解決事項

1. 変換コードの `action_cartesian[t]=leader_target[t+1]` と末尾反復は、選択したHDF5の全行で最大絶対差0。カメラ設定30 fps、収録設定10 Hzが存在するが、保存列から実周期・指令到達・露光タイミングは復元できない。学習は保存済みrowのstride 1を使い、checkpointには `measured_hz=None` を記録した。[変換コード](../robot/utils/data_conversion.py)
2. split内の窓はepisode単位で分離した。厳密な汎化評価には、trainに限定して作った別versionの統計と、撮影・収録セッションの独立性の確認が必要。今回の比較は既存WM統計をそのまま使った条件。
3. 本格学習・GPU計測、実機周期に対する余裕、長期予測、候補順位への有用性は未検証。64角化で失われるマグの細部と、短期学習によるぼけの寄与は分離できていない。

## J. Phase 3への引き継ぎ

`load_for_inference` は信頼できるv1学習checkpointをロードし、eval状態のモデルとmetadataを返す。schemaは型・version・固定SHA／manifest・UNet／action／image仕様・EDM／sampler・alignment・正規化・dataset／split・config・optimizer／scheduler／RNG・epoch／step・ライブラリversionを保持する。構造・仕様・state_dictは厳格に検査する。raw物理行動の境界がある呼び出し側ではWMの正規化を1回だけ適用する。

~~~python
import torch
from stable_worldmodel.wm.diamond.action_embedding import normalize_actions
from stable_worldmodel.wm.diamond.checkpoint import load_for_inference
from stable_worldmodel.wm.diamond.preprocessing import pack_views

# raw_viewsの各値は(B,4,3,H,W)、物理行動は末尾8次元。
model, meta = load_for_inference(checkpoint_path, device)
obs = pack_views(raw_views, meta["spec"]["image"]["views"]).to(device)
past = normalize_actions(raw_past, meta["normalizer"]).to(device)      # (B,3,8)
future = normalize_actions(raw_future, meta["normalizer"]).to(device)  # (B,H,8)
window = torch.cat((past, future[:, :1]), dim=1)
image = model.predict_next(obs, window, torch.Generator(device=device).manual_seed(17))
sequence = model.rollout_images(obs, past, future, torch.Generator(device=device).manual_seed(17))
# raw_candidates (B,K,H,8) → predictions (B,K,H,6,64,64)
candidates = normalize_actions(raw_candidates, meta["normalizer"]).to(device)
predictions = model.rollout_images(obs, past, candidates, torch.Generator(device=device).manual_seed(17))
~~~

今後の接続候補は `stable_worldmodel/policy.py` の `GPCPolicy.rollout` 境界。最初に保存済み候補で履歴4観測＋実際に適用した過去3行動の順を照合し、物理候補にWM統計を適用してK軸を確認する。次に同一seedのオフラインrolloutを比較し、生成画像を評価器に渡す際のview順・解像度とgoal側の変換を固定する。その後GPUでK/H/Nとmicrobatch、評価器込みの時間と候補順位の安定性を測る。現段階でGPCへの接続、選択処理、実機評価には進んでいない。
