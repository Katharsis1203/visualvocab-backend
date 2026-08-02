from __future__ import annotations

import json
import os
import re
import shutil
import uuid
import zipfile
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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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
