# Intelligent Waste Classification System

Computer-vision waste classifier (plastic, paper, metal, glass, organic) built for
a Nigerian-context final-year project. Two models are trained and compared:

1. A baseline CNN trained from scratch.
2. A MobileNetV2 transfer-learning model (lightweight, deployment-friendly).

The better model is served through a Flask web app where a user uploads a
photo and gets back a predicted class + confidence score.

## Project layout

```
waste_classifier/
├── data/                  # your images go here (see structure below)
│   ├── train/
│   ├── val/
│   └── test/
├── src/
│   ├── data_utils.py      # dataset loading + augmentation
│   ├── train_baseline_cnn.py
│   ├── train_mobilenet.py
│   └── evaluate.py        # metrics + confusion matrix on the test set
├── app/
│   ├── app.py             # Flask backend
│   ├── templates/index.html
│   └── static/style.css
├── tests/
│   ├── test_data_utils.py
│   └── test_app.py
├── notebooks/
│   └── Waste_Classification_Colab.ipynb   # run this in Google Colab (GPU)
├── models/                # trained .keras files land here
└── requirements.txt
```

## 1. Required data folder structure

Every image must sit inside a class-named subfolder under `train`, `val`,
and `test`. Class folder names must match exactly across all three splits:

```
data/
├── train/
│   ├── plastic/   img001.jpg  img002.jpg  ...
│   ├── paper/
│   ├── metal/
│   ├── glass/
│   └── organic/
├── val/
│   ├── plastic/
│   ├── paper/
│   ├── metal/
│   ├── glass/
│   └── organic/
└── test/
    ├── plastic/
    ├── paper/
    ├── metal/
    ├── glass/
    └── organic/
```

Recommended split: 80% train, 10% val, 10% test, per class (roughly matches
Table 3.1 in Chapter Three). Aim for at least 400–500 images per class
before augmentation; mix public dataset images (TrashNet, Kaggle Garbage
Classification) with your own locally captured photos in every split, not
just one, so the model is evaluated on the same kind of local variation it's
trained on.

If your public dataset uses different class names (e.g. `cardboard` instead
of `paper`, or `trash`/`e-waste` categories you don't want), just don't copy
those folders in — the code automatically detects classes from whatever
folders exist under `data/train/`.

## 2. Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Train (recommended: Google Colab, for free GPU)

Open `notebooks/Waste_Classification_Colab.ipynb` in Google Colab, upload
your zipped `data/` folder when prompted, and run all cells. It trains both
models and lets you download the resulting `.keras` files.

To train locally instead:
```bash
python src/train_baseline_cnn.py
python src/train_mobilenet.py
```

## 4. Evaluate

```bash
python src/evaluate.py --model models/mobilenet_waste.keras
```
Prints accuracy/precision/recall/F1 and saves a confusion matrix image.

## 5. Run the web app

Two interfaces are provided — pick whichever you prefer, both use the same
trained model:

**Flask** (simple local server):
```bash
cd app
python app.py
```
Then open `http://127.0.0.1:5000` and upload a waste image.

**Streamlit** (built for easy free hosting — recommended if you want a
public demo link):
```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```
See `streamlit_app/DEPLOY.md` for step-by-step instructions to put this
live on Streamlit Community Cloud for free.

## 6. Run tests

```bash
pytest tests/
```
