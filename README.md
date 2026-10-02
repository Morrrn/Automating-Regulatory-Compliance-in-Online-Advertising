# 面向在线广告的政策对齐式监管筛查

本仓库保存一项硕士研究项目的核心代码和部分研究资料。项目依据英国 CAP Code，为在线广告生成可供人工复核的潜在合规问题信号；模型输出**不是**广告违规的法律认定。

## 仓库内容

| 路径 | 内容 |
| --- | --- |
| `src/bad_ads_regulation/` | 数据、标注、API、分类模型和解释方法的核心模块 |
| `scripts/` | 数据清点、标注、分析、训练、评估和解释生成脚本 |
| `annotation_prompt_and_schema/` | 最终 v2.3 标注提示词及输出结构定义 |
| `processed_experiment_data/` | 500 条广告的结构化标注与分析数据、50 条人工评估记录、固定数据划分、案例索引和测试指标 |
| `human_evaluation_template/` | 空白人工评估表 |

为避免重新分发原始资料或暴露个人信息，本仓库**不包含**论文 PDF、原始广告截图、参与者级别数据、模型权重、API 原始响应文件和开发阶段测试。原始数据可从[华盛顿大学 Ad Perceptions Dataset](https://badads.cs.washington.edu/ad-perceptions-dataset/table.html)获取。对应论文为 E. Zeng、T. Kohno 和 F. Roesner 的 *What Makes a 'Bad' Ad? User Perceptions of Problematic Online Advertising*（CHI 2021，[doi:10.1145/3411764.3445459](https://doi.org/10.1145/3411764.3445459)）。下载或再分发原始数据前，请核对其使用条款。

公开 CSV 中的截图路径采用相对格式 `data/chi-bad-ads-data-main/screenshots/<ad_id>.webp`。如需运行图像标注、OCR、模型训练或解释生成，请自行获取原始截图并放在该路径下，且从仓库根目录运行命令。仅查看结构化结果不需要截图。

## 环境准备

需要 Python 3.10 或更新版本。多模态依赖沿用原实验记录；安装指定版本的 PyTorch 可能需要与本机平台匹配的安装包及较大的磁盘空间。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install -r requirements-multimodal.txt
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

如需实际调用 API 标注，请在本机环境中设置 `OPENAI_API_KEY`；**不要**把密钥值写入仓库或提交记录。代码默认使用 `gpt-5.4-mini` 和仓库中的最终提示词。重新运行时，模型可用性及 API 价格可能发生变化。

## 复现主要步骤

1. 单独下载原始数据集，并将其 `screenshots/` 目录放在 `data/chi-bad-ads-data-main/` 下。如需重新清点数据，也应在同一目录保留数据集的其他原始文件。
2. 在 `processed_experiment_data/classifier_manifest_with_ocr.csv` 中查看固定划分。公开版本已清空 OCR 原文；划分比例为 60/20/20，即训练集 300 条、验证集 100 条、留出测试集 100 条。
3. 如需检查 API 请求格式，可先生成少量请求并进行不上传的预览：

```powershell
python scripts/build_openai_batch_file.py --sheet processed_experiment_data/full_annotation_sheet.csv --out outputs/batch_preview.jsonl --preview-out outputs/batch_preview_readable.jsonl --limit 5
python scripts/submit_openai_batch.py --input outputs/batch_preview.jsonl --dry-run
```

正式提交 Batch API 任务会产生费用。无需重复标注时，可直接使用 `processed_experiment_data/openai_batch_annotations_all.csv` 中的结构化结果。

4. 从单独获取的截图在本地重新提取 OCR 文本，再基于已记录的弱标签训练三个基线模型。模型下载和训练可能耗时；检查点会生成在 `outputs/runs/`。

```powershell
python scripts/extract_local_ocr.py --manifest processed_experiment_data/classifier_manifest_with_ocr.csv --out outputs/classifier_manifest_with_ocr.csv
python scripts/train_multimodal_baselines.py --model text --manifest outputs/classifier_manifest_with_ocr.csv --output-dir outputs/runs
python scripts/train_multimodal_baselines.py --model image --manifest outputs/classifier_manifest_with_ocr.csv --output-dir outputs/runs
python scripts/train_multimodal_baselines.py --model fusion --manifest outputs/classifier_manifest_with_ocr.csv --output-dir outputs/runs
python scripts/evaluate_multimodal_baselines.py --manifest outputs/classifier_manifest_with_ocr.csv --run-dir outputs/runs --output-dir outputs/evaluation
```

留出测试集的指标见 `processed_experiment_data/test_metrics_comparison.csv`：文本模型 F1 为 0.735，图像模型为 0.574，后期融合模型为 0.699。这些指标衡量模型与 VLM 弱标签的一致性，**不等于**对 CAP Code 违规的验证。仓库中保留了结构化人工评估评分。

## 隐私、复现与适用范围

- 最终提示词和结构定义对应完整标注实验；再次调用 API 时，输出可能有所不同。
- CSV 中原有的本机绝对截图路径已改为相对路径。公开数据已移除 OCR 原文、广告摘要、解释文本、评审评论和评审者元数据；结构化标签、数据划分、评分与指标仍予保留。
- 仓库不包含 API 密钥值或本机绝对路径。代码中的环境变量名只是运行时读取私有凭据的占位标识。
- 原始广告素材和参与者回答未收入公开归档。获取数据时请遵循原始论文及数据集的适用条款。
- 本项目用于支持人工复核和优先级排序，不能证明广告主是否持有佐证材料，也不能代替正式的合规判断。

---

# Policy-aligned triage of online advertisements

This repository archives code and selected research artifacts for an MSc project on using the UK CAP Code to prioritise online advertisements for human review. The system produces *potential concern* signals, not legal findings of non-compliance.

## What is included

| Path | Contents |
| --- | --- |
| `src/bad_ads_regulation/` | Dataset, annotation, API, classifier, and explanation modules |
| `scripts/` | Pipeline entry points for inventory, annotation, analysis, training, evaluation, and explanations |
| `annotation_prompt_and_schema/` | Final v2.3 prompt and output schema |
| `processed_experiment_data/` | 500-ad annotations, analysis inputs, 50-case human evaluation, fixed classifier split, selected cases, and test metrics |
| `human_evaluation_template/` | Blank human-review forms |

The dissertation PDF, original 500 advertising screenshots, participant-level source data, generated model weights, API response archives, and development-only tests are **not** redistributed here. The source dataset is available from the [University of Washington Ad Perceptions Dataset](https://badads.cs.washington.edu/ad-perceptions-dataset/table.html). Its original publication is E. Zeng, T. Kohno, and F. Roesner, "What Makes a 'Bad' Ad? User Perceptions of Problematic Online Advertising," *CHI 2021*, doi: [10.1145/3411764.3445459](https://doi.org/10.1145/3411764.3445459). Check the dataset's terms before downloading or redistributing it.

The published CSVs use relative screenshot paths of the form `data/chi-bad-ads-data-main/screenshots/<ad_id>.webp`. The screenshot files must be placed there before image-based annotation, OCR, training, or explanation generation. Run commands from the repository root so these paths resolve correctly. No original image is needed to inspect the structured CSV results.

## Environment

Python 3.10 or newer is required. The multimodal requirements were recorded for the original experiment; installing the pinned PyTorch stack may require a platform-specific wheel and substantial disk space.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install -r requirements-multimodal.txt
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

For API annotation, set `OPENAI_API_KEY` in your shell or environment before a live submission. Never put a key in a tracked file. The included code defaults to `gpt-5.4-mini` and the archived final prompt; model availability and API pricing may differ at the time of rerun.

## Reproducing the main stages

1. Download the source dataset separately and place its `screenshots/` directory under `data/chi-bad-ads-data-main/`. For a fresh inventory, retain the dataset's other source files in that same directory.
2. Inspect the recorded split in `processed_experiment_data/classifier_manifest_with_ocr.csv`. Its OCR text has been removed from this public copy. The fixed 60/20/20 split has 300 training, 100 validation, and 100 held-out test advertisements.
3. Build a small API request preview, if desired:

```powershell
python scripts/build_openai_batch_file.py --sheet processed_experiment_data/full_annotation_sheet.csv --out outputs/batch_preview.jsonl --preview-out outputs/batch_preview_readable.jsonl --limit 5
python scripts/submit_openai_batch.py --input outputs/batch_preview.jsonl --dry-run
```

Submitting a real Batch API job incurs charges. The archived annotation results in `processed_experiment_data/openai_batch_annotations_all.csv` can be used without repeating that step.

4. Recreate OCR text locally from the separately downloaded screenshots, then train each baseline on the recorded weak labels. Model downloads and training may take time; checkpoints are generated under `outputs/runs/`.

```powershell
python scripts/extract_local_ocr.py --manifest processed_experiment_data/classifier_manifest_with_ocr.csv --out outputs/classifier_manifest_with_ocr.csv
python scripts/train_multimodal_baselines.py --model text --manifest outputs/classifier_manifest_with_ocr.csv --output-dir outputs/runs
python scripts/train_multimodal_baselines.py --model image --manifest outputs/classifier_manifest_with_ocr.csv --output-dir outputs/runs
python scripts/train_multimodal_baselines.py --model fusion --manifest outputs/classifier_manifest_with_ocr.csv --output-dir outputs/runs
python scripts/evaluate_multimodal_baselines.py --manifest outputs/classifier_manifest_with_ocr.csv --run-dir outputs/runs --output-dir outputs/evaluation
```

The recorded held-out metrics are in `processed_experiment_data/test_metrics_comparison.csv`: text F1 0.735, image F1 0.574, and late-fusion F1 0.699. They evaluate agreement with VLM weak labels, not verified CAP Code breaches. Structured human-review ratings remain in the included data.

## Notes on reproducibility and scope

- The final prompt and schema are the versions used for the full annotation run; API outputs may vary on rerun.
- CSV screenshot paths were rebased from machine-specific absolute paths to repository-relative paths. Free-text OCR, ad summaries, rationales, reviewer comments, and assessor metadata were removed from the public data. Structured labels, splits, ratings, and metrics were retained.
- No API credential value or local machine path is included. The environment-variable name in the code is only a placeholder for a credential supplied privately at runtime.
- The raw advertising material and participant responses were intentionally excluded from this public archive. Follow the source dataset citation and any applicable terms for access.
- The project is research software for human-in-the-loop triage. It does not establish whether an advertiser holds substantiation or whether an ad is legally non-compliant.
