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
    version="1.0.0",
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


def validate_dataset_zip(
    zip_path: Path,
) -> dict[str, int]:
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            names = set(archive.namelist())

            if "annotations.json" not in names:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Dataset ZIP is missing annotations.json."
                    ),
                )

            image_entries = [
                name
                for name in names
                if name.startswith("images/")
                and not name.endswith("/")
            ]

            if not image_entries:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Dataset ZIP contains no images."
                    ),
                )

            try:
                annotations = json.loads(
                    archive.read(
                        "annotations.json"
                    ).decode("utf-8")
                )
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "annotations.json is not valid UTF-8 JSON."
                    ),
                ) from exc

            required_keys = {
                "images",
                "annotations",
                "categories",
            }

            if not required_keys.issubset(
                annotations.keys()
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "annotations.json is not a valid COCO dataset."
                    ),
                )

            return {
                "zipImageCount": len(
                    image_entries
                ),
                "cocoImageCount": len(
                    annotations["images"]
                ),
                "annotationCount": len(
                    annotations["annotations"]
                ),
                "categoryCount": len(
                    annotations["categories"]
                ),
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
                    "Dataset uploaded successfully."
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
            f"visual_vocab_{upload_id}.zip"
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
