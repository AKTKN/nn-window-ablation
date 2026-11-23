# 実験条件の設定と小規模な動作確認

## 環境と設定ファイル

Linux上のリポジトリのルートで、一度だけCPU用環境を作る。既に導入済みなら `conda activate nn_window_decoding` から始められる。[環境の再構築手順](../../environments/README.md)

```bash
conda env create -f environment.yml
conda activate nn_window_decoding
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[experiment,dev]'
python -m pip check
```

設定の起点は [config.yaml](../../src/nn_window_ablation/conf/config.yaml)。Hydraが `defaults` の順に `code`、`noise`、`window`、`model`、`training`、`data`、`evaluation`、`analysis`、`runtime` の各グループを読み、`_self_` で共通項目を、最後に `experiment` で距離・窓・データ系列等を設定する。CLIの `experiment=paired_d7` はプリセットの選択、`training.max_steps=4` は個別の値の上書き。毎回YAMLを編集する必要はない。たとえば次のコマンドで、その組み合わせの設定を表示できる。

```bash
nnwa validate --cfg job --resolve experiment=paired_d7 runtime=local_cpu runtime.allocated_cpus=1
```

| プリセット | 段階・データ系列 | 距離・ラウンド・窓 |
|---|---|---|
| `experiment=pilot_d5`（既定） | `study_stage=pilot`、`data.family_id=pilot_d5` | 距離5、60ラウンド、コア5、左右5ずつ |
| `experiment=paired_d5` | `main`、`paired_d5` | 同じ幾何。pilotとデータ系列を分ける |
| `experiment=paired_d7` | `main`、`paired_d7` | 距離7、84ラウンド、コア7、左右7ずつ |

生成から解析まで同じ科学条件を指定する。`data.family_id`、`seed`、`data.replicate`、`data.split` はどのデータ系列を使うかに関わる。`data.id` はそのデータを保存・参照する名前。`run_name` は学習・評価・解析などの出力名。学習条件を選ぶときは `validation`、選択を終えて最終精度を報告するときは独立な `test` を使う。

## 回路、時間窓、モデルの既定値

| 設定 | pilot_d5の既定値 | 実験上の意味 |
|---|---|---|
| `code.family`、`code.layout`、`code.memory_basis` | `surface_code`、`rotated`、`Z` | 回転表面符号のZメモリ回路を生成する。 |
| `code.distance`、`code.rounds_per_distance` | `5`、`12` | 総ラウンド数は両者の積。距離や時間を延ばすと観測数・グラフ処理量・データ容量が変わる。 |
| `noise.name`、`noise.probability` | `uniform_circuit`、`0.003` | 一様な回路ノイズ。確率を変えるとイベント数・復号課題・LERが変わる。 |
| `noise.after_clifford_depolarization`、`noise.before_round_data_depolarization`、`noise.before_measure_flip_probability`、`noise.after_reset_flip_probability` | すべて `noise.probability` を参照 | Clifford後、各ラウンド前、測定前、リセット後に適用。共通確率を変えるには `noise.probability` を指定する。 |
| `window.strategy`、`window.core_per_distance` | `independent_core_parity`、`1` | 主実験の比は1に固定し、コア長は距離と等しい。既定では60ラウンドを重ならない12個のコアに分け、コアラベルのXORが各ショットの全体ラベルに一致する。 |
| `window.left_per_distance`、`window.right_per_distance` | 各 `1` | 左右に観測を何ラウンド追加するか。指定できる比は0、0.5、1。各幅は距離との積を切り上げる。既定の内部窓は15ラウンド、端では利用可能な観測に合わせる。バッファはラベルの担当範囲を広げない。 |
| `window.reset_recurrent_state`、`window.crop_before_features`、`window.final_readout_policy` | `true`、`true`、`final_core_only` | 独立窓ごとに状態をリセットし、生の観測を切り出した後にグラフ化する。窓方式では破壊的終端読み出しを最後のコアにだけ供給する。教師ラベルは入力特徴に入らない。 |
| `model.name`、`model.upstream_commit` | `query_gnn_rnn`、`458210ac74ed0047760b8789c169e6b9eff732b8` | Langeバックボーンと参照した実装版。 |
| `model.graph_channels`、`model.feature_group_rounds`、`model.hidden_size`、`model.gru_layers` | `[3,64,256]`、`2`、`256`、`4` | グラフのチャネル数、特徴をまとめるラウンド数、表現次元、GRU層数。主実験では固定し、CPU並列数とは独立に保つ。 |
| `model.spatial_aggregation`、`model.query_projection`、`model.head`、`model.empty_observation_embedding` | `distance_dependent_lange`、`additive`、`shared_scalar`、`learned` | 距離別の空間集約、クエリの加算、全体・コア共通の出力、無イベント観測用の学習埋め込み。 |

## 学習から検証・解析までを試す

以下は新しい一時ディレクトリを使うBashの例。8ショットで4回更新し、学習に使っていない5ショットで検証する。`training.log_every_updates=2` なので、更新2と4の詳細が残る。今回の保存先は最初の `echo` で確認できる。

```bash
set -eu
workdir="$(mktemp -d)"
export NNWA_DATA_ROOT="$workdir/data" NNWA_RUN_ROOT="$workdir/runs"
echo "$workdir"
cpu=(experiment=pilot_d5 runtime=local_cpu runtime.allocated_cpus=1)
nnwa validate "${cpu[@]}" run_name=configuration
nnwa generate "${cpu[@]}" data.split=train data.id=training data.shots=8 data.shard_shots=4
nnwa train "${cpu[@]}" run_name=base data.train_source=stored_shards data.id=training \
  training.max_steps=4 training.batch_size=2 training.microbatch_size=1 \
  training.optimizer=adam training.learning_rate=0.001 training.scheduler=none training.log_every_updates=2
nnwa generate "${cpu[@]}" data.split=validation data.id=held data.shots=5 data.shard_shots=3
nnwa evaluate "${cpu[@]}" run_name=paired data.split=validation data.id=held evaluation=paired \
  "checkpoint.path=$NNWA_RUN_ROOT/base/checkpoint.pt"
nnwa analyze "${cpu[@]}" run_name=summary data.split=validation data.id=held \
  "analysis.predictions=[$NNWA_RUN_ROOT/paired/manifest.json]" \
  analysis.sparse_rule=1 analysis.bootstrap_replicates=99 analysis.bootstrap_seed=17
cat "$NNWA_RUN_ROOT/summary/rates.csv"
```

`configuration/hydra/resolved_config.yaml` はvalidate時点の確定設定。学習済みモデルは `base/checkpoint.pt`、評価結果の入口は `paired/manifest.json`、方式ごとの失敗数とLERは `summary/rates.csv`、対応のある差は `summary/contrasts.csv` に残る。いずれも上の `NNWA_RUN_ROOT` が基準。中断した学習は同じソース、run名、学習条件で `checkpoint.resume_from` に最新のチェックポイントを指定する。

**Shared shots** は、評価方式が同じIDの軌跡と正解を共有するという比較条件。この例では5ショットのそれぞれをG・F・Sの全方式で復号し、各方式のLERの分母は5。`data.shard_shots=3` で作る2シャードが並列評価の仕事の単位で、`evaluation.workers` が並列数を決める。学習用8ショットとは独立している。[コマンド別リファレンス](cli_parameters.md) · [96 CPUでの設定例](cluster_96cpu.md)
