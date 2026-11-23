# nnwa コマンドと設定値

以下の既定値は同梱の `experiment=pilot_d5` を選んだ場合の値。`null` は未指定を表す。`validate`、`generate`、`train`、`evaluate`、`analyze` は Hydra の `グループ.項目=値` を受け取る。`pbs`、`export`、`extract`、`reanalyze` は通常のコマンドライン引数を使う。

## 入出力からコマンドを選ぶ

| コマンド | 主な入力 | 出力と用途 |
|---|---|---|
| `nnwa validate` | 実験プリセットと上書き値 | `runs/<run_name>/hydra/resolved_config.yaml`、`overrides.yaml`、`metadata.yaml`。導出した形状・宣言したCPU予算・更新ログ見積もりを事前確認する。学習パラメータが一部未定でも使える。 |
| `nnwa generate` | 回路、乱数、split、ショット数、シャードサイズ | `data/<data.id>/manifest.json` とシャード。物理観測・正解・IDを一度生成して保存する。 |
| `nnwa train` | 保存済みtrainデータまたはその場で生成するストリーム | `runs/<run_name>/checkpoint.pt`、`run.json`、`train-attempts/`。モデル・Adam等の状態を更新する。 |
| `nnwa evaluate` | 評価用データ、終端チェックポイント | `runs/<run_name>/manifest.json` とシャードごとのスコア・判定。重みを固定して推論する。 |
| `nnwa analyze` | 同じ評価用データ、1個以上の予測manifest | `runs/<run_name>/report.json`、`rates.csv`、`contrasts.csv`。正解と照合し、ショット単位で集計する。 |

設定グループの既定の組み合わせは `code=surface_rotated_z noise=uniform_circuit window=fixed_d model=query_gnn_rnn training=full_closed data=paired evaluation=G analysis=primary runtime=cpu experiment=pilot_d5`。

### 共通の識別子・パス・CPU設定

| 設定 | 既定値 | 意味・変更したときの影響 |
|---|---|---|
| `experiment` / `study_stage` | `pilot_d5` / `pilot` | `paired_d5`、`paired_d7` は `main` 用。距離・データ系列の設定も変わる。 |
| `schema_version` / `seed` | `1` / `101` | 記録形式の版と実験の乱数シード。seedは生成ショット、初期化、学習の乱数系列に影響する。 |
| `run_name` | `run-<UUID>` をCLIが生成 | runの出力名。再現・再開時は明示すると指定しやすい。新規runには新しい名前を使う。 |
| `paths.data_root` / `paths.run_root` | `NNWA_DATA_ROOT` / `NNWA_RUN_ROOT`。未設定時 `./data` / `./runs` | データと実行結果の親。共有FSかローカルディスクかで読書き時間が変わる。相対パスの基準はコマンドを実行したディレクトリ。 |
| `runtime` / `runtime.device` / `runtime.dtype` | `cpu` / `cpu` / `float32` | `runtime=local_cpu` は1スレッド、`test_cpu` は1スレッド・CPU予算8のテスト用。モデル容量は共通。 |
| `runtime.allocated_cpus` | `null` | 数値実行時は利用可能なCPU数を明示する。PBSでは資源要求から設定される。 |
| `runtime.processes` | `1` | DDP学習のランク数。ランクごとにモデルと実行環境を持つ。評価ワーカー数は `evaluation.workers` で指定。 |
| `runtime.intraop_threads` / `runtime.interop_threads` | `16` / `1`（`local_cpu` は `1` / `1`） | プロセス内の演算スレッド数と演算間並列数。CPU数を上回る並列化は競合を招く。DDPのCPU所要数は通常「ランク数 × intraop」。 |
| `runtime.benchmark_threads` | `[1,8,16,32,48,96]`（local/testは `[1]`） | 比較用の候補値。1回のtrainで使うスレッド数は `runtime.intraop_threads`。 |

距離、ノイズ、窓幅、モデル構造は全コマンドの共通条件。[回路・窓・モデルの既定値と構成手順](experiment_setup.md)

### generate と学習データ入力

| 設定 | 既定値 | 意味・変更したときの影響 |
|---|---|---|
| `data.family_id` / `data.replicate` | `pilot_d5` / `0` | データの乱数名前空間。独立反復を区別する。保存データで学習するときは生成時と揃える。 |
| `data.id` | `null` | 保存先・読込先のディレクトリ名。生成時に省略すると計画由来のID。保存済みデータを選ぶコマンドでは指定する。 |
| `data.split` | generate・validateは `validation`、trainは `train`、evaluate・analyzeは `test` | 学習、設定選択用の検証、最終評価を別系列に分ける。validationを評価・解析するときは両コマンドに `data.split=validation` を渡す。 |
| `data.shots` / `data.shard_shots` | `null` / `null`（生成時必須） | 総ショット数と1シャードの行数。シャード数は切り上げ除算。大きくするとファイル数と検証の固定費が減り、ワーカーの作業単位・中断時のやり直し量が増える。シャード境界はサンプリング計画の一部。 |
| `data.sampling_processes` / `data.sampling_max_in_flight` | `1` / `null` | generateのサンプラ数と同時に投入するシャード数。既定の上限は直列で1、プールでワーカー数の2倍。各サンプラは1スレッド。ディスク書込や親側の検証が律速ならCPU増加の効果は小さい。 |
| `data.train_source` / `data.training_chunk_shots` | `generated_chunks` / `null` | 単一プロセスで逐次生成する場合は一度に生成する量を指定。`stored_shards` は保存済みtrainの `data.id` が必要。DDPは保存済み方式を使う。 |
| `data.num_workers` | `0` | 追加のデータローダー数。現行ワークフローは0を使用。 |
| `data.storage` / `data.bitorder` | `packed_npy` / `little` | 検出器やラベルのビット詰め形式。ディスク量と読み込み後の展開方法を決める。 |
| `data.keep_zero_event_shots` / `data.label_schema` | `true` / `clean_cut_core_parity_v1` | 無イベントのショットも母数に含め、12コアのXORが全体ラベルになる形式を使う。 |

### train

| 設定 | 既定値 | 意味・変更したときの影響 |
|---|---|---|
| `training` / `training.name` | `full_closed` | 新規学習。`continued_full`（S+）、`fixed_window`（W）、`dynamic_window`（D）は共通の親チェックポイントから継続学習する。 |
| `training.max_steps` / `training.batch_size` | `null` / `null`（必須） | フェーズ内の最大更新数と、1回の更新に全ランク合計で使う軌跡数。積が消費ショット数。保存データはその件数以上が必要。バッチを変えると同じ総ショット数でも更新回数と学習結果が変わる。 |
| `training.microbatch_size` | `null`（1ランクのバッチ全体） | 1回の順伝播・逆伝播で扱う軌跡数。小さくすると中間テンソルのピークが下がる。分割した勾配を蓄積して全体で1回更新する。 |
| `training.execution` / `training.ddp_timeout_seconds` | `single` / `120` 秒 | `ddp` はCPU/Glooの同期学習。グローバルバッチをランク数で割り切る。タイムアウトは通信・終了待ちの期限。 |
| `training.optimizer` / `training.learning_rate` | `null` / `null`（必須） | `adam` または `sgd` と正の学習率。Adamは勾配モーメントもRAMとチェックポイントに保持する。 |
| `training.scheduler` / `training.scheduler_step_size` / `training.scheduler_gamma` | すべて `null` | 実行時は `none` または `step`。stepでは指定した更新回数ごとに学習率をgamma倍にする。 |
| `training.log_every_updates` / `training.force_large_log` | **`100` / `false`** | 各ランクがフェーズ内で指定間隔の倍数に達したときに、処理時間・CPU時間・RSSなどを含む `update` を `train-attempts/<attempt>/rank-<rank>.jsonl` に1行書く。間隔を広げると計測と書き込みが減る。forceは下記の行数上限の明示的な解除。 |
| `checkpoint.save_every_updates` | `100` | 定期保存の間隔。大きくすると書込と同期待ちが減り、中断後に再計算する区間が長くなる。`null` は初期・終了・明示的な一時停止等の必須保存のみ。 |
| `checkpoint.init_from` / `checkpoint.resume_from` | `null` / `null` | initは継続アームの親を指定して最適化状態をリセット。resumeは元のrunの学習状態を復元し、元の更新予算の続きから進める。 |
| `training.validation_data_id` | `null` | 継続アームの終端で別のvalidationセットをS方式で評価。指定すると推論コストが加わる。 |
| `training.initialization` / `training.data_stream_phase` | `random` / `base` | 初期学習の重み・データ系列。継続プリセットでは `full_checkpoint` / `continuation`。 |
| `training.global_context` / `training.core_context` | `full_closed` / `full_closed` | 全体とコアのクエリが使う観測範囲。Wのコアは `deployment_window`、Dのコアは `sampled_window`。 |
| `training.sampled_cores_per_trajectory` / `training.core_sampling` | `1` / `uniform` | 各軌跡の学習に全体クエリと一様に選んだ1コアのクエリを使う。 |
| `training.core_loss_weight` / `training.loss_normalization` | `1.0` / `one_plus_core_loss_weight` | 全体損失とコア損失の重みと正規化。主実験の目的関数として固定。 |
| `training.reset_optimizer_at_start` / `training.checkpoint_policy` | `true` / `fixed_updates` | 新しいフェーズでは最適化状態を初期化し、規定の更新回数で終端モデルを決める。 |
| `training.dynamic_buffers.enabled` / `.ratios` / `.distribution` / `.sampling_unit` | `false` / `[0,0.5,1.0]` / `independent_uniform` / `minibatch` | Dではenabledがtrue。1グローバルバッチごとに左右を独立抽選し、ランクとマイクロバッチで共有。 |

**更新ログの上限**は `ランク数 × (max_steps // log_every_updates)` が **1,000を超える**場合に適用する。`//` は整数除算。単一プロセスは1ランク。開始・終了・チェックポイント等のイベントは別途記録される。たとえば `max_steps=250`、間隔100なら各ランクの更新行は100と200の2行。250での終了と終端保存は専用イベントに残る。再開時もフェーズの更新番号を引き継ぐ。見積もりは1フェーズを予定通り実行したときの件数で、失敗後の再実行や別runの行数を合算するディスク制限ではない。`training.force_large_log=true` は間隔・学習内容・チェックポイント頻度に影響しない。

### evaluate と analyze

| 設定 | 既定値 | 意味・変更したときの影響 |
|---|---|---|
| `checkpoint.path` | `null`（evaluateで必須） | 凍結評価する終端チェックポイント。各ワーカーは開始時に読み、複数シャードでモデルを再利用する。 |
| `evaluation` / `evaluation.id` | `G` | Gは全体を直接予測。Fは全履歴で各コアを予測しXOR、Sは窓内の観測で各コアを予測しXOR。`paired` は同じショットに3方式を適用し、計算量と保存するスコアが増える。 |
| `evaluation.query` / `evaluation.context` / `evaluation.aggregation` | `global` / `full_closed` / `identity` | 選んだ方式のクエリ、観測範囲、集約方法。組み合わせはプリセットで切り替える。 |
| `evaluation.threshold` / `evaluation.reset_recurrent_state_per_query` / `evaluation.update_model_state` | `0.5` / `true` / `false` | 確率の判定閾値、独立クエリ間の状態リセット、凍結推論の契約。 |
| `evaluation.workers` / `evaluation.worker_threads` | `1` / `null` | シャードを処理するプロセス数と各ワーカーの演算スレッド数。既定スレッド数は直列ではruntimeのintraop、プールでは1。ワーカー数に応じてモデル等のRAMと初期ロードが増加。 |
| `evaluation.max_in_flight` / `evaluation.batch_size` | `null` / `1` | 未完了シャード数の上限（既定は直列1、プールはワーカー数の2倍）とワーカー内推論バッチ。大きなバッチは中間テンソルのRAMを増やす。 |
| `evaluation.matching` / `evaluation.resume` | `false` / `false` | trueでそれぞれMWPM比較を追加、既存runの検証済みシャードを再利用。再開では同じデータ・モデル・数値条件を使う。 |
| `analysis.predictions` / `analysis.contrasts` | `[]` / `[S_minus_G,F_minus_G,S_minus_F]` | 読む評価manifestと比較する対。`analysis=adaptation` は `[Splus_minus_W,W_minus_D]`。Bashではリスト全体を引用する。 |
| `analysis.unit` / `analysis.pool_training_seeds_as_independent_shots` | `whole_trajectory` / `false` | ショット全体を1標本とし、同じ軌跡上の方式間差を集計。 |
| `analysis.confidence_level` / `analysis.ler_interval` / `analysis.paired_interval` / `analysis.sparse_interval` | `0.95` / `wilson` / `paired_bootstrap` / `conservative_exact_discordances` | LER・対応のある差・不一致が少ない場合の区間推定方針。 |
| `analysis.bootstrap_replicates` / `analysis.bootstrap_seed` / `analysis.sparse_rule` | すべて `null`（必須） | 再標本化回数・乱数・疎な場合の切替閾値。不一致数が閾値以下で疎な場合の区間を使う。反復数は集計の時間・一時メモリに影響する。 |
| `analysis.ratio_min_denominator_failures` | `null` | 相対比も求める場合に必要な分母側の最小失敗数。既定は差の推定。 |

### PBS、可搬アーカイブ、再解析

| コマンド | 引数と既定値 | 処理・資源への影響 |
|---|---|---|
| `nnwa pbs render` | `--spec JSON`、`--output PBSファイル` ともに必須 | 設定を検証し、単一ノード用スクリプトを新規作成。投入は別操作。 |
| `nnwa pbs status` | `job_id`、`--qstat 実行ファイル` は必須。`--receipt JSON` は省略可 | スケジューラ上の状態を照会。receipt指定時は成果物も照合する。 |
| `nnwa export` | `--dataset ディレクトリ`、`--report report.json`、`--output アーカイブ` が必須 | 最小構成でもラベル・予測・解析結果等を検証して収録。圧縮時には一時領域も使う。 |
| 同上の任意引数 | `--include-detectors`、`--include-checkpoints` は既定off。`--training-dataset ディレクトリ`、`--log ファイル` は既定空、複数指定可 | 再復号には検出器とチェックポイントの両方を含める。追加したデータの分だけ転送・保存量が増える。 |
| `nnwa extract` | 位置引数 `archive destination` は必須 | アーカイブを指定先に検証・展開する。展開後の容量を確保する。 |
| `nnwa reanalyze` | `--bundle 展開先`、`--output 新規出力先` ともに必須 | 保存済みのラベル・予測から統計を再計算。学習やNN推論の計算は不要。 |

`pbs render --spec` のJSONでは以下を指定する。具体例は [PBSの運用手順](../04_cluster_workflow.md) にある。

| JSONの項目 | 既定値・役割 |
|---|---|
| `stage`、`job_name`、`run_name` | 必須。stageは `probe/generate/profile/train/resume/evaluate`。PBS側の名前と実験側の名前を分けて指定する。 |
| `repository`、`source_revision`、`conda_executable`、`conda_prefix` | 必須。リポジトリ・Condaのパスは絶対パス、revisionは完全なコミットID。 |
| `data_root`、`run_root`、`receipt_dir` | 必須の絶対パス。計算ノードでアクセスできる保存先・ジョブ記録先。 |
| `resources.ncpus`、`resources.memory`、`resources.walltime`、`threads` | 必須。CPU・メモリ・壁時計時間とプロセス内スレッド数。`resources.queue` と `resources.account` は既定未指定。 |
| `overrides` / `profile` | 既定 `[]` / `{}`。overridesはHydraの上書き文字列の配列。profileステージの既定は `batch_size=2, warmup=1, steps=2`。run名・ルート・CPU割当・intraopはPBS仕様側が設定する。 |
