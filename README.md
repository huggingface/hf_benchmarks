# Evaluate
> A toolkit for evaluating benchmarks on the [Hugging Face Hub](https://huggingface.co)

## Benchmarks

### SUPERB

To get all the candidate submissions in a given time period, first login to the Hugging Face Hub:

```bash
huggingface-cli login
```

Then you can run:

```bash
python scripts/submission_table.py superb --endpoint models --repo-type model --start-date 2021-09-11 --end-date 2021-09-18
```

which will store all the submissions as a CSV file in the `data` directory.

## Developer installation

Clone the repository and install the requirements:

```
git clone git@github.com:huggingface/evaluate.git
cd evaluate
pip install '.[dev]'
```

