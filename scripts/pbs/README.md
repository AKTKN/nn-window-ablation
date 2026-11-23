# YAMLで設定するPBSジョブ

リポジトリのルートから、実行したい段階を1本ずつ `qsub` する。各ジョブは `PBS_O_WORKDIR` に移動し、ログインノードで準備した `nn_window_decoding` 環境を有効にして `nnwa` を実行する。[CPU環境の導入手順](../../docs/ja/experiment_setup.md)

| 段階 | 投入スクリプト | 編集するYAML | PBS要求と処理 | 既定の出力 |
|---|---|---|---|---|
| 学習データ生成 | [sample_train.sh](sample_train.sh) | [sample_train.yaml](config/sample_train.yaml) | 96 CPU割当、8サンプラ。10,000,128ショット | `data/train10m/` |
| 基本モデル学習 | [train.sh](train.sh) | [train.yaml](config/train.yaml) | 96 CPU、96ランク。192ショット×52,084更新 | `runs/base96/checkpoint.pt` |
| 評価データ生成 | [sample_test.sh](sample_test.sh) | [sample_test.yaml](config/sample_test.yaml) | 96 CPU割当、8サンプラ。独立な10,000,000ショット | `data/test10m/` |
| G/F/S評価 | [evaluate.sh](evaluate.sh) | [evaluate.yaml](config/evaluate.yaml) | 96 CPU、96ワーカー。同じ軌跡を3方式で復号 | `runs/paired96/` |
| 解析 | [analyze.sh](analyze.sh) | [analyze.yaml](config/analyze.yaml) | 1 CPU、1プロセス。ローカルでも実行可 | `runs/summary96/` |

## 初回の準備

1. ログインノードで上記のCPU環境をインストールする。ジョブとローカル解析の両方で同じ版の `nnwa` を使う。
2. リポジトリのルートで `mkdir -p logs` を実行する。各スクリプトの `#PBS -o` と `#PBS -e` は、このディレクトリ内に段階別の標準出力・標準エラーを書き出す。同じ段階のジョブを並行して投入するときは、`qsub -o logs/train-B.o -e logs/train-B.e ...` のように別名にする。
3. `#PBS -l select=1:ncpus=96:host=cpucluster031` のホスト名は提示例に合わせてある。実際に96 CPUを要求できるノード名に差し替える。解析は1 CPU。投入時にノードを変えるなら `qsub -l select=1:ncpus=96:host=対象ノード scripts/pbs/train.sh` でも指定できる。walltime、メモリ、キューの指定は省略している。
4. [common.yaml](config/common.yaml) の `paths.data_root`、`paths.run_root` を各ジョブから見える共有ストレージに設定する。既定は投入元の `./data` と `./runs`。`experiment`、`seed` など、全段階に揃えて適用する科学条件もここで設定する。

各段階のYAMLは、`command` にサブコマンド名、`common` に共通YAMLの位置、`arguments` に実際の `nnwa` の引数を持つ。`arguments` の各行は1引数。`common` の相対パスは**そのYAMLが置かれているディレクトリ**、データ・モデルなどの相対パスは**投入元のリポジトリ**を基準にする。`${paths.run_root}` は共通YAMLの値を参照するHydra式。PBSが読む `#PBS` 行はシェル展開されないので、CPU要求を変えたら対象ジョブの `runtime.allocated_cpus` と並列数もYAMLで合わせる。[パラメータの意味と既定値](../../docs/ja/cli_parameters.md) · [96 CPU設定の根拠](../../docs/ja/cluster_96cpu.md)

## 段階ごとに投入する

下記はリポジトリのルートで実行する例。投入するとそれぞれジョブIDが返る。2種類のデータ生成は互いに独立している。

```bash
mkdir -p logs
qsub scripts/pbs/sample_train.sh
qsub scripts/pbs/sample_test.sh
```

学習用の `data/train10m/manifest.json` が完成してから学習を投入する。

```bash
qsub scripts/pbs/train.sh
```

学習が正常終了し、評価用の `data/test10m/manifest.json` も完成したら評価を投入する。学習中にも同名のチェックポイントが作られるため、学習ジョブの完了を確認する。

```bash
qsub scripts/pbs/evaluate.sh
```

`runs/paired96/manifest.json` が完成したら、クラスタ上で解析できる。

```bash
qsub scripts/pbs/analyze.sh
```

別の実験条件を使う場合は `cp -r scripts/pbs/config my-config` で共通YAMLを含めてコピーし、使う段階で `CONFIG_FILE` を渡す。名前を変更した場合は入力側の `data.id`、`checkpoint.path`、`analysis.predictions` も揃える。

```bash
qsub -v CONFIG_FILE=my-config/train.yaml scripts/pbs/train.sh
```

ジョブから `conda` が見えないサイトでは、ログインノードの有効な環境で `qsub -v "CONDA_BASE=$(conda info --base),CONFIG_FILE=my-config/train.yaml" scripts/pbs/train.sh` とする。別名の環境を使う場合は `NNWA_CONDA_ENV` も `-v` で渡せる。学習再開は元のtrain YAMLの `checkpoint.resume_from` を元の `checkpoint.pt` に設定して同じrun名・ランク数で投入する。評価再開は元のevaluate YAMLを `evaluation.resume=true` に変更して同じrun名で投入する。

### 旧版でステージング削除時に Errno 39 が出た場合

- **データ生成**は `data/<data.id>/manifest.json` を公開してから一時領域を削除する。学習データのmanifestが既にある場合は `python -c 'from pathlib import Path; from nn_window_ablation.data import open_dataset; open_dataset(Path("data/train10m")).verify_payloads()'` で既存シャードを検証できる。検証を通ったデータはそのまま学習・評価で使える。
- **評価**は一時領域を削除してから `runs/<run_name>/manifest.json` を公開する。全シャードがコミット済みで失敗後のステージングが空なら、**更新前の元のソース・環境**で同じrun名の `evaluation.resume=true` を実行すると、シャードを検証して最終manifestを作れる。すでに修正版へ切り替えて別のソースとして扱われる場合は、元のチェックポイントと評価データを使い、新しいrun名で評価できる。

## 解析だけ手元のPCで実行する

評価ジョブが完了したら、`data/test10m/` と `runs/paired96/` をそれぞれ**ディレクトリごと**ローカルに転送する。既定と同じ配置なら、ローカルのリポジトリのルートで次を実行する。別の場所に転送した場合はコピーした共通YAMLの `paths.data_root` と `paths.run_root` を変更する。

```bash
bash scripts/pbs/analyze.sh
# 別の設定コピーを使う例
# CONFIG_FILE=my-config/analyze.yaml bash scripts/pbs/analyze.sh
```

`runs/summary96/rates.csv` が各方式の論理エラー率、`contrasts.csv` が方式間の対応のある差、`report.json` が解析設定を含む全結果。解析時に必要な入力は評価データと予測結果で、学習データとチェックポイントはクラスタ側に保管できる。
