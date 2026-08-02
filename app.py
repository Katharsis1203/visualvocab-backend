from __future__ import annotations

import json
import os
import re
import shutil
import uuid
import zipfile
import hashlib
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

APP_NAME = "VisualVocab Dataset Receiver"

BASE_DIRECTORY = Path(__file__).resolve().parent
UPLOAD_DIRECTORY = Path(
    os.getenv(
        "VISUALVOCAB_UPLOAD_DIRECTORY",
        str(BASE_DIRECTORY / "uploads"),
    )
)
UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)


TRAINING_DIRECTORY = Path(
    os.getenv(
        "VISUALVOCAB_TRAINING_DIRECTORY",
        str(BASE_DIRECTORY / "training_output"),
    )
)
TRAINING_DIRECTORY.mkdir(parents=True, exist_ok=True)

COMBINED_DATASET_FILE = (
    TRAINING_DIRECTORY / "visualvocab_combined_yolo.zip"
)

COMBINED_METADATA_FILE = (
    TRAINING_DIRECTORY / "combined_metadata.json"
)

MAX_UPLOAD_BYTES = int(
    os.getenv(
        "VISUALVOCAB_MAX_UPLOAD_BYTES",
        str(250 * 1024 * 1024),
    )
)

UPLOAD_API_KEY = os.getenv(
    "VISUALVOCAB_UPLOAD_API_KEY",
    "",
).strip()

templates = Jinja2Templates(
    directory=str(BASE_DIRECTORY / "templates")
)

app = FastAPI(
    title=APP_NAME,
    version="2.0.0",
)


def require_upload_api_key(
    supplied_key: str | None,
) -> None:
    if not UPLOAD_API_KEY:
        return

    if supplied_key != UPLOAD_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid upload API key.",
        )


def safe_component(
    value: str,
    maximum_length: int = 80,
) -> str:
    cleaned = re.sub(
        r"[^A-Za-z0-9._-]",
        "_",
        value,
    )
    return cleaned[:maximum_length] or "unknown"


def parse_classes_json(
    archive: zipfile.ZipFile,
) -> list[str]:
    try:
        parsed = json.loads(
            archive.read("classes.json").decode("utf-8")
        )
    except (
        KeyError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="classes.json is missing or invalid.",
        ) from exc

    names = parsed.get("names")

    if not isinstance(names, dict):
        raise HTTPException(
            status_code=400,
            detail="classes.json must contain a names object.",
        )

    ordered_items: list[tuple[int, str]] = []

    for key, value in names.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="classes.json class IDs must be integers.",
            ) from exc

        if not isinstance(value, str) or not value.strip():
            raise HTTPException(
                status_code=400,
                detail="classes.json contains an invalid class name.",
            )

        ordered_items.append((index, value.strip()))

    ordered_items.sort(key=lambda item: item[0])

    if [item[0] for item in ordered_items] != list(
        range(len(ordered_items))
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "classes.json class IDs must start at 0 "
                "and be contiguous."
            ),
        )

    return [item[1] for item in ordered_items]


def validate_label_text(
    label_text: str,
    class_count: int,
    file_name: str,
) -> int:
    annotation_count = 0

    for line_number, raw_line in enumerate(
        label_text.splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{file_name} line {line_number} "
                    "must contain five values."
                ),
            )

        try:
            class_id = int(parts[0])
            coordinates = [
                float(value)
                for value in parts[1:]
            ]
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{file_name} line {line_number} "
                    "contains non-numeric values."
                ),
            ) from exc

        if class_id < 0 or class_id >= class_count:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{file_name} line {line_number} "
                    "uses an unknown class ID."
                ),
            )

        if any(
            value < 0.0 or value > 1.0
            for value in coordinates
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{file_name} line {line_number} "
                    "contains coordinates outside 0 to 1."
                ),
            )

        _, _, width, height = coordinates

        if width <= 0.0 or height <= 0.0:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{file_name} line {line_number} "
                    "contains a non-positive box."
                ),
            )

        annotation_count += 1

    return annotation_count


def validate_dataset_zip(
    zip_path: Path,
) -> dict:
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            names = {
                name
                for name in archive.namelist()
                if not name.endswith("/")
            }

            required_files = {
                "dataset.yaml",
                "classes.json",
            }

            missing_files = required_files - names

            if missing_files:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Dataset ZIP is missing: "
                        + ", ".join(sorted(missing_files))
                    ),
                )

            image_entries = sorted(
                name
                for name in names
                if name.startswith("images/")
            )

            label_entries = {
                name
                for name in names
                if name.startswith("labels/")
            }

            if not image_entries:
                raise HTTPException(
                    status_code=400,
                    detail="Dataset ZIP contains no images.",
                )

            class_names = parse_classes_json(archive)

            if not class_names:
                raise HTTPException(
                    status_code=400,
                    detail="Dataset contains no class names.",
                )

            total_annotations = 0
            missing_labels: list[str] = []

            for image_entry in image_entries:
                image_stem = Path(image_entry).stem
                label_entry = f"labels/{image_stem}.txt"

                if label_entry not in label_entries:
                    missing_labels.append(label_entry)
                    continue

                try:
                    label_text = archive.read(
                        label_entry
                    ).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{label_entry} is not valid UTF-8.",
                    ) from exc

                total_annotations += validate_label_text(
                    label_text=label_text,
                    class_count=len(class_names),
                    file_name=label_entry,
                )

            if missing_labels:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Missing label files: "
                        + ", ".join(missing_labels[:10])
                    ),
                )

            if total_annotations <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Dataset contains no YOLO annotations.",
                )

            return {
                "datasetFormat": "ultralytics-yolo",
        "trainingManager": "/visualvocab/training",
                "imageCount": len(image_entries),
                "labelFileCount": len(label_entries),
                "annotationCount": total_annotations,
                "categoryCount": len(class_names),
                "classes": class_names,
            }
    except zipfile.BadZipFile as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded dataset is not a valid ZIP file.",
        ) from exc


def stable_split(
    key: str,
    validation_fraction: float = 0.2,
) -> str:
    digest = hashlib.sha256(
        key.encode("utf-8")
    ).digest()

    value = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    ) / float(2**64 - 1)

    return (
        "val"
        if value < validation_fraction
        else "train"
    )


def read_uploaded_package(
    folder: Path,
) -> dict:
    dataset_path = folder / "dataset.zip"
    metadata_path = folder / "metadata.json"

    if (
        not dataset_path.is_file()
        or not metadata_path.is_file()
    ):
        raise ValueError(
            f"Incomplete upload folder: {folder.name}"
        )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    with zipfile.ZipFile(
        dataset_path,
        "r",
    ) as archive:
        classes = parse_classes_json(
            archive
        )

        images: list[dict] = []

        image_entries = sorted(
            name
            for name in archive.namelist()
            if (
                name.startswith("images/")
                and not name.endswith("/")
            )
        )

        for image_entry in image_entries:
            image_stem = Path(
                image_entry
            ).stem

            label_entry = (
                f"labels/{image_stem}.txt"
            )

            label_text = archive.read(
                label_entry
            ).decode("utf-8")

            images.append(
                {
                    "imageEntry": image_entry,
                    "labelEntry": label_entry,
                    "imageBytes": archive.read(
                        image_entry
                    ),
                    "labelText": label_text,
                }
            )

    return {
        "folder": folder,
        "metadata": metadata,
        "classes": classes,
        "images": images,
    }


def remap_label_text(
    label_text: str,
    source_classes: list[str],
    target_class_ids: dict[str, int],
) -> tuple[str, int]:
    remapped_lines: list[str] = []
    annotation_count = 0

    for raw_line in label_text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        parts = line.split()
        source_class_id = int(parts[0])
        class_name = source_classes[
            source_class_id
        ]

        target_class_id = target_class_ids[
            class_name
        ]

        remapped_lines.append(
            " ".join(
                [
                    str(target_class_id),
                    *parts[1:],
                ]
            )
        )
        annotation_count += 1

    text = "\n".join(
        remapped_lines
    )

    if text:
        text += "\n"

    return text, annotation_count


def build_combined_training_dataset() -> dict:
    upload_folders = sorted(
        folder
        for folder in UPLOAD_DIRECTORY.iterdir()
        if folder.is_dir()
    )

    packages: list[dict] = []

    for folder in upload_folders:
        try:
            packages.append(
                read_uploaded_package(
                    folder
                )
            )
        except (
            OSError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            zipfile.BadZipFile,
        ):
            continue

    if not packages:
        raise HTTPException(
            status_code=400,
            detail=(
                "No valid uploaded YOLO datasets "
                "are available."
            ),
        )

    class_names = sorted(
        {
            class_name
            for package in packages
            for class_name in package[
                "classes"
            ]
        }
    )

    class_ids = {
        class_name: index
        for index, class_name in enumerate(
            class_names
        )
    }

    examples: list[dict] = []
    annotation_count = 0

    for package in packages:
        upload_id = package[
            "metadata"
        ].get(
            "uploadId",
            package["folder"].name,
        )

        prefix = safe_component(
            str(upload_id),
            maximum_length=40,
        )

        for index, image in enumerate(
            package["images"]
        ):
            original_suffix = Path(
                image["imageEntry"]
            ).suffix.lower()

            if not original_suffix:
                original_suffix = ".jpg"

            unique_stem = (
                f"{prefix}_{index:06d}"
            )

            split = stable_split(
                unique_stem
            )

            remapped_label, count = (
                remap_label_text(
                    label_text=image[
                        "labelText"
                    ],
                    source_classes=package[
                        "classes"
                    ],
                    target_class_ids=class_ids,
                )
            )

            annotation_count += count

            examples.append(
                {
                    "split": split,
                    "imageName": (
                        unique_stem
                        + original_suffix
                    ),
                    "labelName": (
                        unique_stem
                        + ".txt"
                    ),
                    "imageBytes": image[
                        "imageBytes"
                    ],
                    "labelText": remapped_label,
                    "uploadId": upload_id,
                }
            )

    if not examples:
        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded datasets contain no images."
            ),
        )

    # Ensure both splits exist when at least two images are available.
    train_examples = [
        item
        for item in examples
        if item["split"] == "train"
    ]
    val_examples = [
        item
        for item in examples
        if item["split"] == "val"
    ]

    if len(examples) == 1:
        examples[0]["split"] = "train"
    elif not val_examples:
        examples[-1]["split"] = "val"
    elif not train_examples:
        examples[0]["split"] = "train"

    train_count = sum(
        1
        for item in examples
        if item["split"] == "train"
    )
    val_count = sum(
        1
        for item in examples
        if item["split"] == "val"
    )

    dataset_yaml_lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "names:",
    ]

    for index, class_name in enumerate(
        class_names
    ):
        escaped = class_name.replace(
            "\\",
            "\\\\",
        ).replace(
            '"',
            '\\"',
        )

        dataset_yaml_lines.append(
            f'  {index}: "{escaped}"'
        )

    dataset_yaml = (
        "\n".join(dataset_yaml_lines)
        + "\n"
    )

    classes_json = json.dumps(
        {
            "names": {
                str(index): class_name
                for index, class_name in enumerate(
                    class_names
                )
            },
            "input_size": 640,
            "dataset_format": (
                "ultralytics-yolo"
            ),
            "dataset_version": 1,
        },
        indent=2,
    ) + "\n"

    metadata = {
        "builtAt": datetime.now(
            timezone.utc
        ).isoformat(),
        "sourceUploadCount": len(
            packages
        ),
        "imageCount": len(
            examples
        ),
        "trainImageCount": train_count,
        "validationImageCount": val_count,
        "annotationCount": annotation_count,
        "categoryCount": len(
            class_names
        ),
        "classes": class_names,
    }

    temporary_file = (
        TRAINING_DIRECTORY
        / "combined.tmp.zip"
    )

    with zipfile.ZipFile(
        temporary_file,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for item in examples:
            split = item["split"]

            archive.writestr(
                (
                    f"images/{split}/"
                    f"{item['imageName']}"
                ),
                item["imageBytes"],
            )

            archive.writestr(
                (
                    f"labels/{split}/"
                    f"{item['labelName']}"
                ),
                item["labelText"],
            )

        archive.writestr(
            "dataset.yaml",
            dataset_yaml,
        )
        archive.writestr(
            "classes.json",
            classes_json,
        )
        archive.writestr(
            "build_metadata.json",
            json.dumps(
                metadata,
                indent=2,
            ) + "\n",
        )

    temporary_file.replace(
        COMBINED_DATASET_FILE
    )

    COMBINED_METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    return metadata


def get_combined_training_metadata() -> dict | None:
    if (
        not COMBINED_DATASET_FILE.is_file()
        or not COMBINED_METADATA_FILE.is_file()
    ):
        return None

    try:
        metadata = json.loads(
            COMBINED_METADATA_FILE.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    metadata["downloadBytes"] = (
        COMBINED_DATASET_FILE.stat().st_size
    )

    return metadata


def read_upload_records() -> list[dict]:
    records: list[dict] = []

    for folder in UPLOAD_DIRECTORY.iterdir():
        if not folder.is_dir():
            continue

        metadata_path = folder / "metadata.json"
        dataset_path = folder / "dataset.zip"

        if (
            not metadata_path.is_file()
            or not dataset_path.is_file()
        ):
            continue

        try:
            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ):
            continue

        metadata["datasetBytes"] = (
            dataset_path.stat().st_size
        )
        records.append(metadata)

    records.sort(
        key=lambda item: item.get(
            "receivedAt",
            "",
        ),
        reverse=True,
    )

    return records


def find_upload_folder(
    upload_id: str,
) -> Path:
    for folder in UPLOAD_DIRECTORY.iterdir():
        if not folder.is_dir():
            continue

        metadata_path = folder / "metadata.json"

        if not metadata_path.is_file():
            continue

        try:
            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ):
            continue

        if metadata.get("uploadId") == upload_id:
            return folder

    raise HTTPException(
        status_code=404,
        detail="Upload was not found.",
    )


@app.get(
    "/health",
    response_class=JSONResponse,
)
async def health() -> dict:
    return {
        "status": "ok",
        "service": APP_NAME,
        "datasetFormat": "ultralytics-yolo",
        "trainingManager": "/visualvocab/training",
        "uploadDirectory": str(
            UPLOAD_DIRECTORY
        ),
    }


@app.post(
    "/visualvocab/datasets",
    response_class=JSONResponse,
)
async def upload_dataset(
    dataset: Annotated[
        UploadFile,
        File(...),
    ],
    metadata: Annotated[
        str,
        Form(...),
    ],
    installationId: Annotated[
        str,
        Form(...),
    ],
    modelVersion: Annotated[
        str,
        Form(...),
    ],
    exampleCount: Annotated[
        int,
        Form(...),
    ],
    x_visualvocab_api_key: Annotated[
        str | None,
        Header(),
    ] = None,
) -> JSONResponse:
    require_upload_api_key(
        x_visualvocab_api_key
    )

    if dataset.content_type not in {
        "application/zip",
        "application/octet-stream",
        None,
    }:
        raise HTTPException(
            status_code=415,
            detail=(
                "The uploaded dataset must be a ZIP file."
            ),
        )

    try:
        parsed_metadata = json.loads(
            metadata
        )
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "The metadata field is not valid JSON."
            ),
        ) from exc

    upload_id = str(uuid.uuid4())
    received_at = datetime.now(
        timezone.utc
    )
    timestamp = received_at.strftime(
        "%Y%m%dT%H%M%SZ"
    )

    folder_name = (
        f"{timestamp}_"
        f"{safe_component(installationId)}_"
        f"{upload_id}"
    )

    upload_folder = (
        UPLOAD_DIRECTORY / folder_name
    )
    upload_folder.mkdir(
        parents=True,
        exist_ok=False,
    )

    zip_path = (
        upload_folder / "dataset.zip"
    )

    total_bytes = 0

    try:
        with zip_path.open("wb") as output:
            while True:
                chunk = await dataset.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Dataset upload is too large."
                        ),
                    )

                output.write(chunk)

        validation = validate_dataset_zip(
            zip_path
        )

        record = {
            "uploadId": upload_id,
            "receivedAt": (
                received_at.isoformat()
            ),
            "installationId": (
                installationId
            ),
            "modelVersion": modelVersion,
            "exampleCount": exampleCount,
            "datasetBytes": total_bytes,
            "validation": validation,
            "clientMetadata": (
                parsed_metadata
            ),
        }

        (
            upload_folder
            / "metadata.json"
        ).write_text(
            json.dumps(
                record,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        return JSONResponse(
            status_code=201,
            content={
                "uploadId": upload_id,
                "message": (
                    "YOLO dataset uploaded successfully."
                ),
            },
        )
    except Exception:
        shutil.rmtree(
            upload_folder,
            ignore_errors=True,
        )
        raise
    finally:
        await dataset.close()


@app.get(
    "/visualvocab/datasets",
    response_class=JSONResponse,
)
async def list_datasets() -> dict:
    records = read_upload_records()

    return {
        "count": len(records),
        "datasets": records,
    }


@app.get(
    "/visualvocab/datasets/{upload_id}/download",
)
async def download_dataset(
    upload_id: str,
) -> FileResponse:
    upload_folder = find_upload_folder(
        upload_id
    )
    dataset_path = (
        upload_folder / "dataset.zip"
    )

    return FileResponse(
        path=dataset_path,
        filename=(
            f"visual_vocab_yolo_{upload_id}.zip"
        ),
        media_type="application/zip",
    )


@app.get(
    "/visualvocab/datasets/{upload_id}/metadata",
    response_class=JSONResponse,
)
async def get_dataset_metadata(
    upload_id: str,
) -> JSONResponse:
    upload_folder = find_upload_folder(
        upload_id
    )
    metadata_path = (
        upload_folder / "metadata.json"
    )

    return JSONResponse(
        content=json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )
    )



@app.get(
    "/visualvocab/training",
    response_class=HTMLResponse,
)
async def training_dashboard(
    request: Request,
) -> HTMLResponse:
    records = read_upload_records()
    combined = (
        get_combined_training_metadata()
    )

    return templates.TemplateResponse(
        request=request,
        name="training.html",
        context={
            "records": records,
            "combined": combined,
            "service_name": APP_NAME,
        },
    )


@app.post(
    "/visualvocab/training/build",
)
async def build_training_dataset():
    build_combined_training_dataset()

    return RedirectResponse(
        url="/visualvocab/training",
        status_code=303,
    )


@app.post(
    "/visualvocab/training/build.json",
    response_class=JSONResponse,
)
async def build_training_dataset_json():
    metadata = (
        build_combined_training_dataset()
    )

    return {
        "status": "built",
        "metadata": metadata,
        "downloadUrl": (
            "/visualvocab/training/"
            "combined.zip"
        ),
    }


@app.get(
    "/visualvocab/training/status",
    response_class=JSONResponse,
)
async def training_status():
    combined = (
        get_combined_training_metadata()
    )

    return {
        "available": combined is not None,
        "combined": combined,
    }


@app.get(
    "/visualvocab/training/combined.zip",
)
async def download_combined_training_dataset():
    if not COMBINED_DATASET_FILE.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                "Build the combined training "
                "dataset first."
            ),
        )

    return FileResponse(
        path=COMBINED_DATASET_FILE,
        filename=(
            "visualvocab_combined_yolo.zip"
        ),
        media_type="application/zip",
    )


@app.get(
    "/",
    response_class=HTMLResponse,
)
async def dashboard(
    request: Request,
) -> HTMLResponse:
    records = read_upload_records()

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "records": records,
            "service_name": APP_NAME,
        },
    )
