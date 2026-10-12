# Deploying to Streamlit Community Cloud

This app is fully built and tested (its logic is exercised in
`tests/`-style checks) — the only two things I genuinely cannot do from
here are (1) train on your real dataset, because my code sandbox has no
internet access to install TensorFlow or download image datasets, and
(2) click "Deploy" for you, because that needs your own GitHub and
Streamlit accounts. Everything else is done. Here's the fastest path to a
live, working app.

## Step 1 — Get a real trained model (Google Colab, ~20–40 min)

1. Organize your images into `data/train|val|test/<class>/` (see the main
   README).
2. Zip the whole `waste_classifier/` project folder.
3. Open `notebooks/Waste_Classification_Colab.ipynb` in Google Colab
   (Runtime > Change runtime type > GPU), upload the zip, run all cells.
4. Download `mobilenet_waste.keras` and `class_names.json` from the
   `models/` folder it produces (the last notebook cell does this for
   you).
5. Copy both files into `models/` in your local copy of this project.

## Step 2 — Run it locally first (sanity check)

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```
Open the local URL Streamlit prints, upload a real waste photo, confirm
you get a sensible prediction.

## Step 3 — Push to GitHub

```bash
git init
git add .
git commit -m "Intelligent waste classification app"
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Your trained `.keras` model file (a few MB to ~15 MB for MobileNetV2) is
small enough to commit directly with `git add models/`. If GitHub rejects
it as too large, use [Git LFS](https://git-lfs.com/) — `git lfs track
"*.keras"` before committing.

## Step 4 — Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click "New app".
3. Repository: your repo. Branch: `main`. Main file path:
   `streamlit_app/app.py`.
4. Click "Deploy". Streamlit Cloud installs
   `streamlit_app/requirements.txt` automatically and builds the app —
   first deploy typically takes a few minutes because it installs
   TensorFlow.
5. You'll get a public URL like `https://<your-app>.streamlit.app` — this
   is the link to put in your project report / demo to your supervisor.

## Notes

- `streamlit_app/requirements.txt` uses `tensorflow-cpu`, which is
  smaller and faster to install than full `tensorflow` — Streamlit
  Cloud's free tier doesn't provide a GPU anyway, so this is the right
  choice for the deployed app (training still happens separately, on
  Colab's GPU).
- If your repo is private, Streamlit Cloud can still deploy from it once
  you grant it access during the GitHub OAuth step.
- The app looks for `models/mobilenet_waste.keras` first, falling back to
  `models/baseline_cnn_waste.keras` — make sure whichever one you commit
  matches the one you intend to demo.
