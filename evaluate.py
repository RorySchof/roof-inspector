"""Evaluate the existing Roof Inspector model on a separate labeled image set.

This script does not train, and it does not change model.pkl or the data/ folder.
It only runs when invoked. The web application does not import it.

A result from this script is not a measure of real-world accuracy unless the
manifest is a sufficiently representative dataset, labeled independently of the
training images.

Precision is undefined when a class is never predicted (no positive predictions).
Recall is undefined when a class never appears as a ground-truth label.
Those cases are printed as "undefined" rather than as zero.

Example:
    ./venv/bin/python evaluate.py \\
        --images eval_images \\
        --manifest eval_images/manifest.csv \\
        --output eval_results/predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
TRAINING_DATA = (ROOT / "data").resolve()
RESULTS_DIR = ROOT / "eval_results"

# Verified from the CategoryMap stored in model.pkl.
EXPECTED_VOCAB = (
    "no_obvious_repair_needed",
    "potential_repair_needed",
)
POSITIVE_CLASS = "potential_repair_needed"
NEGATIVE_CLASS = "no_obvious_repair_needed"
IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jfif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
WILSON_Z = 1.959963984540054


class EvaluationError(Exception):
    """A problem with the evaluation inputs. No predictions file is written."""


def wilson_interval(correct: int, total: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Requires total > 0."""
    if total <= 0:
        raise EvaluationError("Cannot compute a Wilson interval without evaluated images.")
    proportion = correct / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (proportion + z2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt((proportion * (1 - proportion) / total) + z2 / (4 * total * total))
        / denominator
    )
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def reject_training_directory(images_dir: Path) -> None:
    """Reject data/, anything inside it, and any directory that contains it."""
    resolved = images_dir.resolve()
    training = TRAINING_DATA.resolve()
    if is_inside(resolved, training) or is_inside(training, resolved):
        raise EvaluationError(
            f"Refusing to evaluate {resolved}. The evaluation directory cannot be "
            f"{training}, sit inside it, or contain it."
        )


def assert_not_training_image(path: Path) -> None:
    resolved = path.resolve()
    training = TRAINING_DATA.resolve()
    if is_inside(resolved, training):
        raise EvaluationError(
            f"Evaluation image resolves into the training dataset: {path}"
        )
    if not training.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(training, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = [
            name for name in dirnames
            if not name.startswith(".") and not (current / name).is_symlink()
        ]
        for name in filenames:
            candidate = current / name
            if candidate.is_symlink() or name.startswith("."):
                continue
            if candidate.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                if resolved.samefile(candidate):
                    raise EvaluationError(
                        f"Evaluation image is a training dataset file: {path}"
                    )
            except OSError:
                continue


def image_files_under(images_dir: Path, manifest_path: Path) -> dict[str, Path]:
    """Map relative POSIX paths to image files.

    Directory symlinks are not followed. A file or directory symlink that resolves
    outside the evaluation directory is an error, not a skipped path.
    """
    found: dict[str, Path] = {}
    manifest_resolved = manifest_path.resolve() if manifest_path.exists() else None
    for dirpath, dirnames, filenames in os.walk(images_dir, followlinks=False):
        current = Path(dirpath)
        for name in dirnames:
            child = current / name
            if not child.is_symlink():
                continue
            target = child.resolve()
            if not target.exists() or not is_inside(target, images_dir):
                raise EvaluationError(
                    f"Symlinked directory escapes the evaluation directory: {child}"
                )
        for name in filenames:
            path = current / name
            if path.name.startswith("."):
                continue
            if manifest_resolved is not None and path.resolve() == manifest_resolved:
                continue
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if path.is_symlink():
                target = path.resolve()
                if not target.is_file() or not is_inside(target, images_dir):
                    raise EvaluationError(
                        f"Symlinked file escapes the evaluation directory: {path}"
                    )
            elif not path.is_file():
                continue
            resolved = path.resolve()
            assert_not_training_image(resolved)
            relative = path.relative_to(images_dir).as_posix()
            if relative in found:
                raise EvaluationError(
                    f"Multiple files in the evaluation directory resolve to {relative}."
                )
            found[relative] = resolved
    return found


def resolve_manifest_path(images_dir: Path, raw_path: str, row_number: int) -> tuple[str, Path]:
    text = raw_path.strip()
    if not text:
        raise EvaluationError(f"Manifest row {row_number} has an empty path.")
    candidate = Path(text)
    if candidate.is_absolute():
        raise EvaluationError(
            f"Manifest row {row_number} uses an absolute path: {text}"
        )
    if candidate.drive or candidate.root:
        raise EvaluationError(
            f"Manifest row {row_number} is not a relative path: {text}"
        )
    resolved = (images_dir / candidate).resolve()
    if not is_inside(resolved, images_dir):
        raise EvaluationError(
            f"Manifest row {row_number} resolves outside the evaluation directory: {text}"
        )
    relative = resolved.relative_to(images_dir.resolve()).as_posix()
    if not resolved.is_file():
        raise EvaluationError(
            f"Manifest row {row_number} does not point to a file: {text}"
        )
    assert_not_training_image(resolved)
    return relative, resolved


def open_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise EvaluationError(f"Cannot open image {path}: {exc.__class__.__name__}") from None


def read_manifest(manifest_path: Path, images_dir: Path) -> list[tuple[str, Path, str]]:
    if not manifest_path.is_file():
        raise EvaluationError(f"Manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise EvaluationError("Manifest is empty.")
        columns = [(name or "").strip() for name in reader.fieldnames]
        missing = [name for name in ("path", "label") if name not in columns]
        if missing:
            raise EvaluationError(
                "Manifest is missing required column(s): " + ", ".join(missing)
            )
        rows: list[tuple[str, Path, str]] = []
        seen: dict[str, int] = {}
        for row_number, row in enumerate(reader, start=2):
            values = {((key or "").strip()): (value or "") for key, value in row.items()}
            if not any(value.strip() for value in values.values()):
                raise EvaluationError(f"Manifest row {row_number} is empty.")
            relative, resolved = resolve_manifest_path(
                images_dir, values.get("path", ""), row_number
            )
            label = values.get("label", "").strip()
            if label not in EXPECTED_VOCAB:
                raise EvaluationError(
                    f"Manifest row {row_number} has an invalid label: {label!r}. "
                    f"Use one of: {', '.join(EXPECTED_VOCAB)}."
                )
            if relative in seen:
                raise EvaluationError(
                    f"Manifest path {relative} is duplicated on rows {seen[relative]} and {row_number}."
                )
            seen[relative] = row_number
            open_image(resolved)
            rows.append((relative, resolved, label))
    if not rows:
        raise EvaluationError("Manifest has no evaluation rows.")
    return rows


def check_directory_coverage(
    images_dir: Path,
    manifest_path: Path,
    rows: list[tuple[str, Path, str]],
) -> None:
    on_disk = image_files_under(images_dir, manifest_path)
    listed = {relative for relative, _, _ in rows}
    missing_from_manifest = sorted(set(on_disk) - listed)
    missing_from_disk = sorted(listed - set(on_disk))
    if missing_from_disk:
        raise EvaluationError(
            "Manifest entries do not match files in the evaluation directory: "
            + ", ".join(missing_from_disk)
        )
    if missing_from_manifest:
        raise EvaluationError(
            "Image files are not listed in the manifest: "
            + ", ".join(missing_from_manifest)
        )


def load_model():
    if not MODEL_PATH.is_file():
        raise EvaluationError(f"model.pkl was not found at {MODEL_PATH}.")
    from fastai.vision.all import load_learner

    learn = load_learner(MODEL_PATH)
    vocab = tuple(str(label) for label in learn.dls.vocab)
    if vocab != EXPECTED_VOCAB:
        raise EvaluationError(
            "model.pkl class order does not match the verified mapping. "
            f"Found {vocab}. Expected {EXPECTED_VOCAB}."
        )
    return learn


def score_image(learn, path: Path) -> tuple[str, dict[str, float]]:
    from fastai.vision.all import PILImage

    image = PILImage.create(path)
    predicted, _index, probabilities = learn.predict(image)
    scores = {
        str(learn.dls.vocab[i]): float(probabilities[i])
        for i in range(len(probabilities))
    }
    missing_scores = [label for label in EXPECTED_VOCAB if label not in scores]
    if missing_scores:
        raise EvaluationError(
            f"Prediction for {path} did not include scores for: {', '.join(missing_scores)}."
        )
    return str(predicted), scores


def precision_recall(rows: list[dict[str, str]]) -> dict[str, tuple[str, str]]:
    """Return precision and recall for each class. Undefined ratios stay the word undefined."""
    summary: dict[str, tuple[str, str]] = {}
    for label in EXPECTED_VOCAB:
        true_positive = sum(
            1
            for row in rows
            if row["label"] == label and row["predicted_label"] == label
        )
        predicted_positive = sum(1 for row in rows if row["predicted_label"] == label)
        actual_positive = sum(1 for row in rows if row["label"] == label)
        precision = (
            "undefined"
            if predicted_positive == 0
            else f"{true_positive / predicted_positive:.6f}"
        )
        recall = (
            "undefined"
            if actual_positive == 0
            else f"{true_positive / actual_positive:.6f}"
        )
        summary[label] = (precision, recall)
    return summary


def confusion_counts(rows: list[dict[str, str]]) -> dict[tuple[str, str], int]:
    counts = {
        (actual, predicted): 0
        for actual in EXPECTED_VOCAB
        for predicted in EXPECTED_VOCAB
    }
    for row in rows:
        key = (row["label"], row["predicted_label"])
        if key not in counts:
            raise EvaluationError(
                f"Unexpected predicted label {row['predicted_label']!r} for {row['path']}."
            )
        counts[key] += 1
    return counts


def resolve_output(output_path: Path, manifest_path: Path) -> Path:
    """Allow only a new CSV directly inside the project's eval_results/ directory."""
    results = RESULTS_DIR
    if results.is_symlink():
        raise EvaluationError("Refusing to write into eval_results because that path is a symlink.")
    if results.exists() and not results.is_dir():
        raise EvaluationError("eval_results exists and is not a directory.")

    candidate = output_path if output_path.is_absolute() else ROOT / output_path
    if candidate.is_symlink():
        raise EvaluationError(f"Refusing to write through a symlink: {candidate}")

    protected = {MODEL_PATH.resolve(), Path(__file__).resolve()}
    for name in (
        "model.pkl",
        "evaluate.py",
        "api.py",
        "README.md",
        "Dockerfile",
        "requirements.txt",
        "collect_and_train.ipynb",
        ".gitignore",
    ):
        candidate_path = ROOT / name
        if candidate_path.exists():
            protected.add(candidate_path.resolve())
    if manifest_path.exists():
        protected.add(manifest_path.resolve())

    if candidate.exists():
        destination = candidate.resolve()
    else:
        parent = candidate.parent
        if parent.is_symlink():
            raise EvaluationError(f"Refusing to write through a symlinked directory: {parent}")
        if parent.resolve() != results.resolve():
            raise EvaluationError(
                "Write the predictions CSV to a new file inside the project's eval_results/ directory."
            )
        destination = results / candidate.name

    if destination in protected:
        raise EvaluationError(f"Refusing to overwrite protected file: {destination}")
    if destination.exists():
        raise EvaluationError(f"Refusing to overwrite existing file: {destination}")
    if results.exists():
        results_root = results.resolve()
    else:
        results_root = results
    if destination == results_root or not is_inside(destination, results_root):
        raise EvaluationError(
            "Write the predictions CSV to a new file inside the project's eval_results/ directory."
        )
    if is_inside(destination, TRAINING_DATA):
        raise EvaluationError("Refusing to write evaluation output inside the training data/ directory.")
    return destination


def write_predictions(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "label",
        "predicted_label",
        "score_no_obvious_repair_needed",
        "score_potential_repair_needed",
        "correct",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_report(rows: list[dict[str, str]], output_path: Path) -> None:
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"] == "true")
    accuracy = correct / total
    low, high = wilson_interval(correct, total)
    counts = confusion_counts(rows)
    class_metrics = precision_recall(rows)
    false_negatives = [
        row["path"]
        for row in rows
        if row["label"] == POSITIVE_CLASS and row["predicted_label"] == NEGATIVE_CLASS
    ]

    print(f"Evaluated images: {total}")
    print(f"Overall accuracy: {correct}/{total} ({accuracy:.6f})")
    print(
        "Wilson 95% confidence interval for accuracy: "
        f"[{low:.6f}, {high:.6f}]"
    )
    print("Confusion matrix (rows are ground truth, columns are predictions):")
    header = "ground_truth".ljust(28) + "".join(label.rjust(28) for label in EXPECTED_VOCAB)
    print(header)
    for actual in EXPECTED_VOCAB:
        line = actual.ljust(28) + "".join(
            str(counts[(actual, predicted)]).rjust(28) for predicted in EXPECTED_VOCAB
        )
        print(line)
    print("Per-class precision and recall:")
    print("Precision is undefined when that class is never predicted.")
    print("Recall is undefined when that class has no ground-truth images.")
    for label in EXPECTED_VOCAB:
        precision, recall = class_metrics[label]
        print(f"  {label}: precision {precision}, recall {recall}")
    print(
        "False negatives (labeled potential_repair_needed, predicted no_obvious_repair_needed): "
        f"{len(false_negatives)}"
    )
    if false_negatives:
        for path in false_negatives:
            print(f"  {path}")
    else:
        print("  none")
    print(f"Predictions written to {output_path}")
    print(
        "This run does not by itself establish real-world accuracy. "
        "That requires a sufficiently representative dataset labeled independently of training."
    )


def evaluate(images_dir: Path, manifest_path: Path, output_path: Path) -> None:
    if not images_dir.is_dir():
        raise EvaluationError(f"Evaluation image directory not found: {images_dir}")
    reject_training_directory(images_dir)
    if manifest_path.exists() and is_inside(manifest_path.resolve(), TRAINING_DATA):
        raise EvaluationError("Refusing to read a manifest from inside the training data/ directory.")
    output_path = resolve_output(output_path, manifest_path)
    rows_in = read_manifest(manifest_path, images_dir)
    check_directory_coverage(images_dir, manifest_path, rows_in)
    learn = load_model()
    predictions: list[dict[str, str]] = []
    for relative, resolved, label in rows_in:
        predicted, scores = score_image(learn, resolved)
        predictions.append(
            {
                "path": relative,
                "label": label,
                "predicted_label": predicted,
                "score_no_obvious_repair_needed": f"{scores[NEGATIVE_CLASS]:.8f}",
                "score_potential_repair_needed": f"{scores[POSITIVE_CLASS]:.8f}",
                "correct": "true" if predicted == label else "false",
            }
        )
    write_predictions(output_path, predictions)
    print_report(predictions, output_path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score the existing Roof Inspector model.pkl on a separate labeled image directory. "
            "Does not train and does not modify the application."
        ),
        epilog=(
            "Manifest CSV columns: path, label. Paths are relative to --images. "
            "Labels must be no_obvious_repair_needed or potential_repair_needed. "
            "Do not point --images at the training data/ directory or at a folder that contains it. "
            "--output must be a new file inside eval_results/. "
            "Results are not real-world accuracy unless the labeled set is representative "
            "and independent of training."
        ),
    )
    parser.add_argument("--images", required=True, type=Path, help="Directory of evaluation images.")
    parser.add_argument("--manifest", required=True, type=Path, help="CSV with path and label columns.")
    parser.add_argument("--output", required=True, type=Path, help="Where to write the predictions CSV.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        evaluate(args.images, args.manifest, args.output)
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
