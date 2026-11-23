# 96 CPUで約1,000万ショットを学習し、独立な1,000万ショットで評価する

距離5・60ラウンド・ノイズ確率0.003のモデルを1つ学習し、G/F/Sを比較する場合の開始設定。**学習は10,000,128ショット、評価は10,000,000ショット**とする。グローバルバッチ192、52,084更新なら96ランクに均等配分でき、1ランクは1更新につき2ショットを処理する。10,000,000以下に収める必要がある場合は52,083更新・9,999,936ショットを使い、生成数も揃える。評価にはランク数に対する整除条件はない。

各段階をPBSに個別投入する場合は、コメント付きYAMLとシェルスクリプトを使える。[PBSでの実行手順](../../scripts/pbs/README.md)

## 割り当てと実行例

| 段階 | 設定案 | 選んだ理由 |
|---|---|---|
| train/testデータ生成 | 8サンプラ、未完了16シャードまで。trainは10,000行/シャード、testは2,500行/シャード | 生成は書き込み・検証にも時間を使う。trainはファイル数を抑え、testは96ワーカーに配分できる4,000シャードにする。 |
| 学習 | 96 DDPランク×演算1スレッド、グローバルバッチ192、マイクロバッチ1 | 1ランクにモデル1組を置き、2ショットを1つずつ処理して勾配を蓄積する。全ランクが同期して1回更新する。 |
| 保存 | 更新ログは5,000更新間隔、チェックポイントは1,000更新間隔 | 更新ログは各ランク10行、計960行。チェックポイントはランク0が書き、中断時は直近の保存から再開する。 |
| 評価 | 96ワーカー×1スレッド、推論バッチ1、同時投入192シャード、MWPMなし | 各ワーカーが1シャードのG/F/Sを処理する。同じ軌跡を3条件で比較でき、再開時には完了済みシャードを利用できる。 |
| 解析 | 1プロセス、1スレッド | 保存済みラベルと予測をシャード単位で読み、ショット全体を単位として差を計算する。 |

以下は**96 CPUを割り当てた単一Linuxノード内**で、Conda環境を有効にし、保存に使うファイルシステムの作業ディレクトリから実行するBash例。`$PWD/nnwa-d5-96` が保存先になる。既存runと区別できる新しいディレクトリを使う。例の `experiment=paired_d5` は本実験の系列。Adam、学習率0.001、固定学習率、および解析時の閾値・反復数・シードは実験者が決める値で、パッケージの既定値は未指定。バッチを128から192に変えると、同じショット予算でも更新数と勾配が変わるので、学習設定は事前にvalidationで選ぶ。

```bash
set -eu
export NNWA_DATA_ROOT="$PWD/nnwa-d5-96/data" NNWA_RUN_ROOT="$PWD/nnwa-d5-96/runs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
common=(experiment=paired_d5 runtime=local_cpu runtime.allocated_cpus=96)
sampling=(data.sampling_processes=8 data.sampling_max_in_flight=16)
train=(training.execution=ddp runtime.processes=96 data.train_source=stored_shards data.id=train10m
       training.batch_size=192 training.microbatch_size=1 training.max_steps=52084
       training.optimizer=adam training.learning_rate=0.001 training.scheduler=none
       training.log_every_updates=5000 checkpoint.save_every_updates=1000)

nnwa validate "${common[@]}" "${train[@]}" data.split=train run_name=check96
nnwa generate "${common[@]}" "${sampling[@]}" data.split=train data.id=train10m \
  data.shots=10000128 data.shard_shots=10000
nnwa train "${common[@]}" "${train[@]}" run_name=base96
nnwa generate "${common[@]}" "${sampling[@]}" data.split=test data.id=test10m \
  data.shots=10000000 data.shard_shots=2500
nnwa evaluate "${common[@]}" evaluation=paired evaluation.workers=96 evaluation.worker_threads=1 \
  evaluation.max_in_flight=192 evaluation.batch_size=1 data.id=test10m run_name=paired96 \
  "checkpoint.path=$NNWA_RUN_ROOT/base96/checkpoint.pt"
nnwa analyze "${common[@]}" data.id=test10m run_name=summary96 \
  "analysis.predictions=[$NNWA_RUN_ROOT/paired96/manifest.json]" \
  analysis.sparse_rule=1 analysis.bootstrap_replicates=9999 analysis.bootstrap_seed=17
```

生成だけを別ジョブにするなら要求CPUを8に下げてよい。PBSではジョブ仕様の `resources.ncpus` と `threads` から実行時の割当を設定する。保存先がジョブをまたいで同じファイルを指すようにする。評価の中断後は同一run名に `evaluation.resume=true` を加える。学習再開は同じ96ランク・学習設定に `checkpoint.resume_from=$NNWA_RUN_ROOT/base96/checkpoint.pt` を加える。[PBSジョブの構成と再開](../04_cluster_workflow.md)

## RAM、ディスク、ログ

| 対象 | 見積もり | 内訳 |
|---|---|---|
| 学習RAM | 約39～40 GiB＋親プロセス等 | 小規模ローカル測定のランク別ピーク約412～422 MiBを96倍した参考値。Python・PyTorch、重み・勾配・Adam状態、活性、Gloo通信、データ読み込みを含むプロセス全体のRSS。 |
| 評価RAM | 約34 GiB＋親プロセス等 | 約360 MiB/ワーカー×96。推論用モデル・ライブラリ・その時点のグラフ等を含む。 |
| train/test入力 | 各約1.91 GB | 各ショットは検出器180 B、全体ラベル1 B、コアラベル2 B、ID 8 B。データ系列は2つ生成する。 |
| G/F/Sの予測 | 約1.15 GB | 1ショット115 B。共通ID、Gの1スコアとF/S各12スコア、ビット詰めの判定と集約結果を保存する。 |
| 学習チェックポイント | 最新1個は約24 MBを目安 | ローカル8ランクの終端Adamファイルは23,583,637 B。ランク状態の分だけ構成による差が出る。ショット本体は別ファイル。 |
| 更新ログ | 960行、ローカルの行長で約0.74 MB | `96 × (52,084 // 5,000)`。既定の100更新間隔なら49,920行となり開始前エラー。明示的に許す設定は `training.force_large_log=true`。 |

RAM見積もりの単純加算では共有ページを重複計上する一方、親、一時ピーク、全シャードの管理情報、OSページキャッシュは十分に見積もれない。初回の割当案は**ノードメモリ96 GiB**。生成・学習・評価を順に実行する場合は段階ごとのピークを見る。対象ノードのジョブ全体のピーク使用量と制限値を比較して再調整する。[ローカル資源監査とRSS測定の条件](../verification/local_multicore_audit.md)

保存する配列の合計は約4.97 GB（10進）。この分割ではtrain 1,001シャード×5ファイル、test 4,000×5、予測4,000×10で、シャード配下だけで**65,005ファイル**。さらに上位manifest、ログ、回路、チェックポイント、一時ファイルがある。容量と併せてinodeと共有FSのメタデータ性能も確認する。特に各評価ワーカーは起動時にデータセットのシャード目録を読むので、4,000シャード×96ワーカーの初期読込も測定に含める。

チェックポイントは初期・定期52回・終端の計54回。基本学習では保存先を置換するため保持量はほぼ一定だが、累計書き込みはおよそ1.3 GBに達し、保存時に一時領域も使う。開始・終了・チェックポイントなどのイベントは更新ログの1,000行制限とは別枠で、例ではチェックポイント関連だけで54×96行。上の容量見積もりは基本学習1本とG/F/S評価1本のもの。

## 対象ノードで時間を見積もる

96ランク・バッチ192のスループットは未測定。学習は「52,084 × 1更新の平均秒数」、評価は「10,000,000 ÷ 全ワーカー合計のshots/s」で見積もる。以下は処理速度を仮定した換算で、ジョブ待ち・起動・周期保存・終端処理は別に計上する。

| 段階 | ノードで観測したと仮定する速度 | 本番の処理時間 |
|---|---|---|
| 学習 | 0.5 / 1 / 2 秒/更新 | 7.2 / 14.5 / 28.9時間 |
| G/F/S評価 | 150 / 200 / 250 ショット/秒 | 18.5 / 13.9 / 11.1時間 |

本番前に、同じノードと保存先で200更新（38,400ショット）を別run名で測る。校正用は `training.max_steps=200 training.log_every_updates=25` なら全ランク計768更新行。評価は独立な `validation` 系列を使い、本番と同じ2,500行のシャードを少なくとも192個用意すると、96ワーカーに2巡分の仕事を渡せる。起動時間、定常時の更新秒数、評価の合計shots/s、ジョブ全体のメモリピーク、シャードの読み書き時間を記録する。時間を取りすぎる場合の比較候補は**48ランクで同じグローバルバッチ192**。ランク当たり4ショットになり、モデル複製数と同期参加数が半分になる。同じ学習予算のもとで96ランクとの差を測れる。[設定リファレンス](cli_parameters.md)
