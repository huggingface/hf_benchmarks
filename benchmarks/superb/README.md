# SUPERB

## Installation

Clone the repository and install the requirements:

```
git clone git@github.com:huggingface/evaluate.git
cd evaluate
pip install '.[superb]'
```

## Usage

### Generate predictions

To generate the predictions from a model that is fine-tuned on one of SUPERB's downstream tasks, run

```bash
HF_HUB_TOKEN=yourToken python inference.py 'model_id' 'superb' 'downstream_task' 'dataset_split' 'dataset_column'
```

where `HF_HUB_TOKEN` is the API token associated with your Hugging Face Hub account, `model_id` is the name of the fine-tuned model, and `dataset_column` is the name of the column from which to run inference over.

This sends a job to Hugging Face's inference API. If successful, you should received a message like:

```
Launching inference job with ID bulk-e13a0f17 for model lewtun/superb-s3prl-osanseviero__hubert_base-diarization-7f28b8b5 on dataset superb! Dataset repo created with name lewtun/bulk-superb-s3p-superb-8e373
```

and the resulting predictions will be stored under your user account.

## Datasets and metrics

You can load each task of the SUPERB benchmark using the `datasets` library as follows:

```python
from datasets import load_dataset

dset = load_dataset("superb", "downstream_task")
```

Currently the supported set of tasks are `asr` and `sd` (speaker diarization). Similarly you can load the metrics associated with each downstream task:

```python
from datasets import load_metric

metric = load_metric("metric_name")
```

