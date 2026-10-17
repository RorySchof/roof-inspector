# Roof Inspector

Roof Inspector is a small web app for a preliminary look at a roof photo. You upload one image. A fast.ai image classifier returns one of two screening results:

- **Potential repair needed**
- **No obvious repair needed**

The page also shows the model’s confidence and the probability of each class. It is a screening demo, not a professional roof inspection.

## Architecture

```text
React frontend  →  FastAPI POST /predict  →  fast.ai / PyTorch model
```

The frontend is a React app built with Vite. It sends the photo as a multipart field named `file`. The API loads `model.pkl` from the project root once, runs the exported learner, and returns JSON:

```json
{
  "predicted_label": "potential_repair_needed",
  "predicted_index": 1,
  "probabilities": {
    "no_obvious_repair_needed": 0.08,
    "potential_repair_needed": 0.92
  }
}
```

`predicted_label` is `potential_repair_needed` or `no_obvious_repair_needed`. The page turns those into the two headings above. `GET /health` returns `{"status": "ok", "model_ready": true}` when `model.pkl` is present.

## Tech stack

- Python 3.11
- fast.ai
- PyTorch
- ResNet-18
- FastAPI
- React
- Vite

## Dataset

Training images were collected with DuckDuckGo image search, then reviewed and cleaned by hand. After cleanup there are 53 images in `potential_repair_needed` and 46 in `no_obvious_repair_needed`, about 50 per class.

Those photos are for local training. Copyright stays with the photographers. Do not publish the scraped image set.

## Training

`collect_and_train.ipynb` fine-tunes an ImageNet ResNet-18:

- Images resized to 192 pixels
- 80/20 train/validation split (`valid_pct=0.2`, `seed=42`)
- 3 fine-tuning epochs (`fine_tune(3)`)
- Metric: `error_rate`

On the validation split the model got **16/19 correct (84.2% accuracy)**. The three misses were roofs labeled as potential repair and predicted as no obvious repair. That is a small validation set, so the percentage is only a first check.

The exported learner is `model.pkl` in the project root. Git tracks that file. `.gitignore` ignores training images, virtual environments, and env files. It does not ignore `model.pkl`. A clone includes the model file. If `model.pkl` is missing, `POST /predict` returns 503.

`model.pkl` is a ResNet-18 model fine-tuned on a small, manually reviewed set of roof photos. The original training photos are not in this repository. Roof Inspector is a screening demo, not a definitive inspection.

## Informal smoke test

After training, 11 fresh roof photos that were not in the training set were tried by hand. **10/11 were correct (90.9%)**. This was an informal smoke test, not a benchmark. It is not a measure of accuracy on roofs in general.

## Limitations

- Roof Inspector is a screening demo. It is not a professional roof inspection, and it should not be used to decide whether a roof is safe.
- The model classifies the whole image. It can miss small or localized damage.
- Lighting, image quality, roof angle, roof type, and obstructions can change the result.
- The confidence score is the model’s class confidence. It is not a calibrated probability that a roof actually needs repair.

## Run it locally

From this directory, with Python 3.11:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This repository includes `model.pkl`. If you remove it, `POST /predict` returns 503 until the file is back in this directory.

Start the API:

```bash
./venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000
```

Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open [http://127.0.0.1:5173/](http://127.0.0.1:5173/). The page calls `http://127.0.0.1:8000/predict` unless `VITE_API_URL` is set.

To point a production frontend build at an API, set that variable when you build. Do not hardcode a host in the source. There is no deployment URL in this repo yet.

```bash
cd frontend
VITE_API_URL=https://example.invalid/predict npm run build
```

Replace `https://example.invalid/predict` with the real `/predict` URL when one exists. Vite reads `VITE_API_URL` at build time and bakes it into the static files. `frontend/.env.example` shows the local value.
