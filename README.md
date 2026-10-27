# Automated Defect Detection for Pork Rasher Packaging

MMA3001 Numerical Methods and Machine Learning, individual project.
Author: Paartiv Annanthan (33288496)

This project trains and validates a YOLOv8 nano object detector that flags packaging defects on vacuum packed pork rasher trays from an overhead conveyor camera. It detects five defect types (loose meat, packaging error, twisted meat, unsealed and wrinkle) and flags any tray with a detection above the confidence threshold for a human inspector.

On a leakage controlled test set, the final model (Run B) flags 89% of defective trays and detects unsealed packs reliably (mAP50 of 0.94), but cannot yet detect the rare defect types.

## Repository structure

```
src/                 Python scripts, one per workflow step
  audit_dataset.py     split sizes, class counts, label checks, leakage checks
  blocked_split.py     leakage resistant split by blocks of video frames
  oversample.py        repeat factor sampling for rare classes (Run C)
  train.py             training, then scoring the best epoch on valid and test
  threshold_sweep.py   confidence threshold tuning on validation
  failure_analysis.py  sorts every test set mistake into a failure mode
tests/               pytest tests for every script (63 tests)
results/             saved scores (JSON) and the threshold sweep chart
docs/                HTML documentation generated from the docstrings by pdoc
test_report.txt      output of the full pytest run
requirements.txt     Python dependencies
LICENSE              MIT licence for the code in this repository
```

The `data/` and `runs/` folders are created locally when you run the workflow and are not stored in the repository, because they hold the dataset and large model files.

## Environment

Tested with Python 3.10.5 on Windows, with an NVIDIA GeForce GTX 1650 Ti (4 GB). Training also runs on a CPU, but much more slowly.

1. Install the GPU build of PyTorch first. Go to [pytorch.org](https://pytorch.org/get-started/locally/), choose Stable, your OS, Pip, Python and a CUDA version no higher than the one shown by `nvidia-smi`, and run the command it gives. This project used PyTorch 2.14.1 with CUDA 12.6.
2. Install the remaining dependencies:
   ```
   pip install -r requirements.txt
   pip install pdoc
   ```
3. Check the GPU is detected:
   ```
   python -c "import torch; print(torch.cuda.is_available())"
   ```

Installing PyTorch before Ultralytics matters, because Ultralytics otherwise installs the CPU only version.

## Data

The dataset is the *Pork Rasher Error Packaging* dataset, Version 4, by hello-2aqe0 on Roboflow Universe, licensed CC BY 4.0:
https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4

On that page, choose Download Dataset, format **YOLOv8**, then "Download zip to computer". Unzip it so that `data/` contains `data.yaml` and the `train`, `valid` and `test` folders directly.

## Running the workflow

Run every command from the repository root, in this order. All random steps use seed 0, so the results reproduce exactly.

```
pytest
python src/audit_dataset.py --data data
python src/blocked_split.py --gap 10
python src/audit_dataset.py --data data/blocked --skip-hash
python src/oversample.py
python src/train.py --data data --name A_roboflow_split
python src/train.py --data data/blocked --name B_blocked_split
python src/train.py --data data/blocked_oversampled --name C_oversampled
python src/threshold_sweep.py --data data/blocked --name B_blocked_split
python src/threshold_sweep.py --data data/blocked_oversampled --name C_oversampled
python src/failure_analysis.py --data data/blocked --name B_blocked_split
```

| Step | What it does | Output |
| --- | --- | --- |
| Audit | Checks labels and measures leakage between splits | Printed report |
| Blocked split | Rebuilds the split by blocks of video frames, with a 10 frame gap | `data/blocked` |
| Oversample | Repeats training images that contain rare classes | `data/blocked_oversampled` |
| Train | Trains YOLOv8 nano and scores it on valid and test | `results/<run>.json`, `runs/<run>` |
| Threshold sweep | Picks the confidence threshold on validation using F2 | `results/<run>_threshold_sweep.json` and `.png` |
| Failure analysis | Sorts every test mistake into a failure mode | `results/<run>_failures_test.json`, `runs/failures` |

Each training run takes about 45 to 80 minutes on the GPU above. If a run is interrupted after training finished, score it without retraining:

```
python src/train.py --data data/blocked --name B_blocked_split --evaluate-only
```

## Results

| Run | Data | Test mAP50 | Test mAP50 (4 classes) |
| --- | --- | --- | --- |
| A | Roboflow's original split | 0.37 | 0.37 |
| B (final) | Blocked split | 0.35 | 0.44 |
| C | Blocked split with oversampling | 0.35 | 0.44 |

At the 0.25 threshold, Run B flags 89% of defective trays and 32% of good trays. See the report for the full validation, comparison and failure analysis.

## Tests and documentation

Run the tests and save a report:

```
pytest -v > test_report.txt
```

Regenerate the HTML documentation from the docstrings:

```
cd src
pdoc -o ../docs audit_dataset blocked_split oversample train threshold_sweep failure_analysis
cd ..
```

Open `docs/index.html` in a browser to read it.

## Acknowledgement of AI use

This project was developed with Claude (Anthropic) through claude.ai, as permitted for this assessment. Claude wrote the Python scripts and tests, guided the environment setup and troubleshooting, explained concepts and drafted the report. I made the experimental decisions, ran every experiment, checked the results against the training logs and images, and identified and corrected several errors in the AI's output. Section 7 of the report describes this in detail.

## Licence

The code in this repository is released under the MIT licence (see `LICENSE`). The dataset is licensed CC BY 4.0 by its authors. Ultralytics YOLOv8 is a separate dependency with its own licence (AGPL 3.0).
