from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Roof Inspector API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_learner = None


def model_ready() -> bool:
    return MODEL_PATH.is_file()


def get_learner():
    """Load the exported fast.ai learner once, after training creates model.pkl."""
    global _learner
    if _learner is None:
        if not model_ready():
            raise HTTPException(
                status_code=503,
                detail=(
                    "model.pkl is not available yet. Train and export it with "
                    "collect_and_train.ipynb before requesting a prediction."
                ),
            )
        from fastai.vision.all import load_learner

        _learner = load_learner(MODEL_PATH)
    return _learner


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": model_ready()}


async def read_upload(file: UploadFile) -> bytes:
    chunks = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=400,
                detail="Please upload an image smaller than 10 MB.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def open_image(data: bytes):
    from fastai.vision.all import PILImage

    try:
        image = PILImage.create(BytesIO(data))
        image.load()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Please upload a readable image file.",
        ) from None
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    data = await read_upload(file)
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    image = open_image(data)
    learn = get_learner()

    try:
        pred, idx, probs = learn.predict(image)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Inference failed.") from None

    class_probs = {
        str(learn.dls.vocab[i]): float(probs[i]) for i in range(len(probs))
    }

    return {
        "predicted_label": str(pred),
        "predicted_index": int(idx),
        "probabilities": class_probs,
    }
