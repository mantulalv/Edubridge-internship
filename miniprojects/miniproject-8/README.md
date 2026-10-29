# NeuralVision AI — Handwritten Digit Recognition

A small convolutional network trained on MNIST, wrapped in a Streamlit app that accepts
an uploaded digit image and explains the prediction with class probabilities.

**Developer:** mantulalv · **Repo:** https://github.com/mantulalv/Edubridge-internship/tree/main/miniprojects/miniproject-8

## Features

- 3-conv-block CNN (~99% test accuracy after 6-8 epochs).
- Training curves, test metrics and a confusion matrix saved to `outputs/`.
- Robust image preprocessing: grayscale, auto-invert, crop to the digit, centre on a
  20x20 box, pad to 28x28 — the same framing MNIST uses.
- Streamlit demo with upload **or** a random MNIST test sample.

## Install

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
```

> `tensorflow-cpu` keeps the download smaller. On Apple Silicon use `tensorflow`
> (or `tensorflow-macos`) instead.

## Train / Evaluate / App / Test

```bash
python src/train.py
python src/evaluate.py
streamlit run app.py
pytest -q
```

Training downloads MNIST (~11 MB) on first run and takes a few minutes on CPU.
The model is written to `models/digit_cnn.keras` (git-ignored).

## Project layout

```
miniproject-8/
├── app.py                 Streamlit demo
├── src/
│   ├── data.py            MNIST loading + image preprocessing
│   ├── model.py           CNN definition
│   ├── train.py           training + artefacts
│   └── evaluate.py        test metrics + confusion matrix
├── tests/
├── models/                (git-ignored)
└── outputs/
```

## Results

Fill in `outputs/metrics.json` figures here after training.

MIT © 2026 mantulalv
