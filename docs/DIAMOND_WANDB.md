# DIAMOND の学習モニタリング

既定は `wandb.enabled=false`。従来のローカル学習と metrics.jsonl をそのまま使用する。有効時は既存環境に wandb SDK が必要。オンラインでは事前に同じ環境で `wandb login` を実行するか、実行環境の秘密環境変数 WANDB_API_KEY を使う。

~~~bash
cd /home/shonosukehida/work/LeWorldModel
export PYTHONPATH="$PWD:$PWD/../stable-worldmodel"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python train_diamond.py device=cuda batch_size=32 steps=10000 wandb.enabled=true wandb.name=flip-mug-diamond wandb.log_images=true
~~~

既存runの再開は元の学習設定のまま、`resume=/絶対パス/step-XXXXXX.pt` と元stepより大きい `steps`、新しい `output` を指定する。最新のオンラインcheckpointは同じrun IDで追記する。古い地点から別の実験に分岐する際に同じオンライン履歴を巻き戻す機能はない。W&B導入前のcheckpointでは新規runが復元global stepの次から始まる。offlineでは各起動が独立したローカルrunとなり、必要なときに `wandb sync <offline-runディレクトリ>` を実行する。disabledモードまたはenabled=falseではSDKをimportしない。SDK未導入・初期化失敗・記録失敗は警告し、以降ローカルで学習を続ける。

| 指標 | 定義 |
|---|---|
| global_step | 完了したoptimizer stepの累計。W&Bの履歴をstepごとに1回送信 |
| train/loss_step, train/learning_rate, train/grad_norm | metrics.jsonlのloss、今回の更新に使用したlr、clip前の全パラメータ勾配の総ノルム |
| system/step_time_sec | 訓練窓の抽出開始から更新・scheduler更新とloss/勾配値のCPU取得までの実時間。検証、通信、画像生成、保存を除く |
| epoch, epoch_progress | Equivalent Epochの1始まり番号と連続進捗。復元抽出のため、窓の網羅は保証されない。L=ceil(使用train窓数/batch_size)、番号=ceil(step/L)、進捗=step/L。26,550窓とbatch 32ではL=830、10,000 stepで12.048 epoch |
| train/loss_epoch | epoch内のstep lossのサンプル数加重平均。最終部分epochも記録し、epoch_partialで識別。epoch_samplesは集計したサンプル数 |
| val/loss_epoch | 検証episode上の更新なしのEDM loss。fixedでは従来どおり固定seedのval_batches個のbatch、fullでは全検証窓を一度ずつ評価し端数もサンプル数加重。検証noiseは訓練用RNGから分離 |

W&B WorkspaceのLine plotでXをglobal_stepにしてstep loss、lr、grad norm、step時間を表示する。別のLine plotではXをepoch、Yをtrain/loss_epochとval/loss_epochの2系列にする。既定ではW&B有効時に毎epoch末と最終stepで検証し、validate_every_epochsで間隔を変更できる。従来のvalidate_everyと最終stepのローカル評価は継続し、同時に必要な場合は結果を共有する。未評価点にはval値を送信しない。validation_mode=fullはstep検証にも適用される。

新checkpointはepoch内の加重和・サンプル数とオンラインrun IDをconfig.epoch_monitorに保存する。途中終了時の部分平均を表示した後も分子・分母は保持する。旧版ではcheckpointと同じディレクトリのmetrics.jsonlから該当区間を復元する。不足時はepoch_history_complete=falseとし、そのepochのtrain平均を省く。次の完全なepochから通常集計に戻る。学習checkpoint既存のepochフィールドは従来の定義を維持する。

wandb.log_images=trueでは、image_every_epochs間隔で固定val窓・固定行動・固定noiseからimage_horizon時刻を自己回帰生成する。val/predictionsにはカメラ別に、各行がt+1からt+H、左から最後の実観測・GT・予測の画像を送信する。既定では画像生成は無効。

### 開始日時と再開先

学習入口で取得したAsia/Tokyoの日時を使う。既定の出力は `outputs/diamond/YYYYMMDD_HHMMSS/`、表示名は `diamond_YYYYMMDD_HHMMSS`（先頭はwandb.nameで変更可）。同一秒に同じ既定パスを取得する場合は、ディレクトリと表示名に順に `_02` 等が付く。明示したoutputは従来どおりそのパスを占有する。異なる出力ルートで同じ表示名になってもrun_idは独立する。

config.yaml、checkpointのconfig、W&B configでrun_timestamp、started_at、run_id、run_nameを確認できる。再開時には元の情報を引き継ぎ、既定では元timestamp付きの別ディレクトリ（_resume、_resume_02等）に保存する。旧checkpointに日時がない場合は警告して再開時の日時を使い、保存済みオンラインID・表示名があれば継承する。サーバー側にしかない旧表示名はSDKに委ねる。IDがなければ新しいIDを割り当てる。offlineでは同じ識別子を記録し、各起動のログをローカルに保存する。
