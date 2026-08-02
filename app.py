from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Annotated

import yaml
from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
)
from fastapi.templating import Jinja2Templates

APP_NAME = "VisualVocab Dataset Receiver"

BASE_DIRECTORY = Path(__file__).resolve().parent

UPLOAD_DIRECTORY = Path(
    os.getenv(
        "VISUALVOCAB_UPLOAD_DIRECTORY",
        str(BASE_DIRECTORY / "uploads"),
    )
)
UPLOAD_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
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
    directory=str(
        BASE_DIRECTORY / "templates"
    )
)

app = FastAPI(
    title=APP_NAME,
    version="3.1.0",
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

    return (
        cleaned[:maximum_length]
        or "unknown"
    )


def safe_zip_name(
    value: str,
) -> str:
    normalized = value.replace(
        "\\",
        "/",
    )

    path = PurePosixPath(
        normalized
    )

    if (
        path.is_absolute()
        or ".." in path.parts
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "ZIP contains an unsafe path."
            ),
        )

    return str(path)


def write_upload_stream(
    upload: UploadFile,
    destination: Path,
) -> int:
    total_bytes = 0

    with destination.open("wb") as output:
        while True:
            chunk = upload.file.read(
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

    return total_bytes


def parse_classes_json(
    archive: zipfile.ZipFile,
) -> list[str]:
    try:
        parsed = json.loads(
            archive.read(
                "classes.json"
            ).decode("utf-8")
        )
    except (
        KeyError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "classes.json is missing "
                "or invalid."
            ),
        ) from exc

    names = parsed.get("names")

    if not isinstance(
        names,
        dict,
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "classes.json must contain "
                "a names object."
            ),
        )

    ordered_items = []

    for key, value in names.items():
        try:
            index = int(key)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "classes.json class IDs "
                    "must be integers."
                ),
            ) from exc

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "classes.json contains "
                    "an invalid class name."
                ),
            )

        ordered_items.append(
            (
                index,
                value.strip().lower(),
            )
        )

    ordered_items.sort(
        key=lambda item: item[0]
    )

    expected_ids = list(
        range(
            len(ordered_items)
        )
    )

    actual_ids = [
        item[0]
        for item in ordered_items
    ]

    if actual_ids != expected_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                "classes.json class IDs "
                "must start at 0 and be "
                "contiguous."
            ),
        )

    return [
        item[1]
        for item in ordered_items
    ]


def normalize_import_label_text(
    label_text: str,
    class_count: int,
    file_name: str,
) -> tuple[str, int, int]:
    normalized_lines: list[str] = []
    annotation_count = 0
    polygon_conversion_count = 0

    for line_number, raw_line in enumerate(
        label_text.splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()

        try:
            class_id = int(parts[0])
        except (IndexError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{file_name} line {line_number} "
                    "has an invalid class ID."
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

        try:
            values = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{file_name} line {line_number} "
                    "contains non-numeric coordinates."
                ),
            ) from exc

        if len(values) == 4:
            center_x, center_y, width, height = values
        elif len(values) >= 6 and len(values) % 2 == 0:
            x_values = values[0::2]
            y_values = values[1::2]
            left = min(x_values)
            right = max(x_values)
            top = min(y_values)
            bottom = max(y_values)
            width = right - left
            height = bottom - top
            center_x = left + width / 2.0
            center_y = top + height / 2.0
            polygon_conversion_count += 1
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{file_name} line {line_number} must contain "
                    "either four box values or an even number "
                    "of polygon coordinates."
                ),
            )

        coordinates = [center_x, center_y, width, height]

        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{file_name} line {line_number} contains "
                    "coordinates outside 0 to 1."
                ),
            )

        if width <= 0.0 or height <= 0.0:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{file_name} line {line_number} contains "
                    "a non-positive box."
                ),
            )

        normalized_lines.append(
            f"{class_id} {center_x:.6f} {center_y:.6f} "
            f"{width:.6f} {height:.6f}"
        )
        annotation_count += 1

    normalized_text = "\n".join(normalized_lines)
    if normalized_text:
        normalized_text += "\n"

    return normalized_text, annotation_count, polygon_conversion_count


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
                    f"{file_name} line "
                    f"{line_number} must contain "
                    "five values."
                ),
            )

        try:
            class_id = int(
                parts[0]
            )

            coordinates = [
                float(value)
                for value in parts[1:]
            ]
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{file_name} line "
                    f"{line_number} contains "
                    "non-numeric values."
                ),
            ) from exc

        if (
            class_id < 0
            or class_id >= class_count
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{file_name} line "
                    f"{line_number} uses "
                    "an unknown class ID."
                ),
            )

        if any(
            value < 0.0
            or value > 1.0
            for value in coordinates
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{file_name} line "
                    f"{line_number} contains "
                    "coordinates outside 0 to 1."
                ),
            )

        _, _, width, height = (
            coordinates
        )

        if (
            width <= 0.0
            or height <= 0.0
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{file_name} line "
                    f"{line_number} contains "
                    "a non-positive box."
                ),
            )

        annotation_count += 1

    return annotation_count


def validate_native_dataset_zip(
    zip_path: Path,
) -> dict:
    try:
        with zipfile.ZipFile(
            zip_path,
            "r",
        ) as archive:
            names = {
                safe_zip_name(name)
                for name in archive.namelist()
                if not name.endswith("/")
            }

            required_files = {
                "dataset.yaml",
                "classes.json",
            }

            missing_files = (
                required_files - names
            )

            if missing_files:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Dataset ZIP is missing: "
                        + ", ".join(
                            sorted(
                                missing_files
                            )
                        )
                    ),
                )

            image_entries = sorted(
                name
                for name in names
                if name.startswith(
                    "images/"
                )
            )

            label_entries = {
                name
                for name in names
                if name.startswith(
                    "labels/"
                )
            }

            if not image_entries:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Dataset ZIP contains "
                        "no images."
                    ),
                )

            class_names = (
                parse_classes_json(
                    archive
                )
            )

            if not class_names:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Dataset contains "
                        "no class names."
                    ),
                )

            total_annotations = 0
            missing_labels = []

            for image_entry in image_entries:
                image_stem = Path(
                    image_entry
                ).stem

                label_entry = (
                    "labels/"
                    + image_stem
                    + ".txt"
                )

                if (
                    label_entry
                    not in label_entries
                ):
                    missing_labels.append(
                        label_entry
                    )
                    continue

                try:
                    label_text = (
                        archive.read(
                            label_entry
                        ).decode(
                            "utf-8"
                        )
                    )
                except UnicodeDecodeError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"{label_entry} "
                            "is not valid UTF-8."
                        ),
                    ) from exc

                total_annotations += (
                    validate_label_text(
                        label_text=
                            label_text,
                        class_count=
                            len(
                                class_names
                            ),
                        file_name=
                            label_entry,
                    )
                )

            if missing_labels:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Missing label files: "
                        + ", ".join(
                            missing_labels[:10]
                        )
                    ),
                )

            if total_annotations <= 0:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Dataset contains no "
                        "YOLO annotations."
                    ),
                )

            return {
                "datasetFormat":
                    "ultralytics-yolo",
                "imageCount":
                    len(
                        image_entries
                    ),
                "labelFileCount":
                    len(
                        label_entries
                    ),
                "annotationCount":
                    total_annotations,
                "categoryCount":
                    len(
                        class_names
                    ),
                "classes":
                    class_names,
            }
    except zipfile.BadZipFile as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded dataset is not "
                "a valid ZIP file."
            ),
        ) from exc


def locate_data_yaml(
    archive: zipfile.ZipFile,
) -> str:
    candidates = [
        safe_zip_name(name)
        for name in archive.namelist()
        if (
            not name.endswith("/")
            and Path(name).name.lower()
            in {
                "data.yaml",
                "data.yml",
                "dataset.yaml",
                "dataset.yml",
            }
        )
    ]

    if not candidates:
        raise HTTPException(
            status_code=400,
            detail=(
                "Imported ZIP does not "
                "contain data.yaml."
            ),
        )

    candidates.sort(
        key=lambda value: (
            len(
                PurePosixPath(value).parts
            ),
            value,
        )
    )

    return candidates[0]


def parse_import_class_names(
    data_yaml: dict,
) -> list[str]:
    raw_names = data_yaml.get(
        "names"
    )

    if isinstance(
        raw_names,
        list,
    ):
        class_names = [
            str(value)
                .strip()
                .lower()
            for value in raw_names
        ]
    elif isinstance(
        raw_names,
        dict,
    ):
        ordered = sorted(
            (
                int(key),
                str(value)
                    .strip()
                    .lower(),
            )
            for key, value
            in raw_names.items()
        )

        actual_ids = [
            item[0]
            for item in ordered
        ]

        expected_ids = list(
            range(
                len(ordered)
            )
        )

        if (
            actual_ids !=
            expected_ids
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "data.yaml class IDs "
                    "must start at 0 and "
                    "be contiguous."
                ),
            )

        class_names = [
            item[1]
            for item in ordered
        ]
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "data.yaml must contain "
                "a names list or object."
            ),
        )

    if (
        not class_names
        or any(
            not value
            for value in class_names
        )
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "data.yaml contains "
                "invalid class names."
            ),
        )

    return class_names


def collect_standard_yolo_entries(
    archive: zipfile.ZipFile,
) -> list[tuple[str, str]]:
    names = {
        safe_zip_name(name)
        for name in archive.namelist()
        if not name.endswith("/")
    }

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
    }

    image_entries = sorted(
        name
        for name in names
        if (
            Path(name).suffix.lower()
            in image_extensions
            and "/images/" in (
                "/" + name
            )
        )
    )

    pairs = []

    for image_entry in image_entries:
        parts = list(
            PurePosixPath(
                image_entry
            ).parts
        )

        try:
            image_index = (
                parts.index(
                    "images"
                )
            )
        except ValueError:
            continue

        label_parts = (
            parts[:image_index]
            + ["labels"]
            + parts[
                image_index + 1:
            ]
        )

        label_parts[-1] = (
            Path(
                label_parts[-1]
            ).stem
            + ".txt"
        )

        label_entry = str(
            PurePosixPath(
                *label_parts
            )
        )

        if label_entry not in names:
            continue

        pairs.append(
            (
                image_entry,
                label_entry,
            )
        )

    if not pairs:
        raise HTTPException(
            status_code=400,
            detail=(
                "No matching YOLO "
                "image/label pairs were "
                "found. Expected folders "
                "such as train/images and "
                "train/labels."
            ),
        )

    return pairs


def convert_standard_yolo_zip(
    source_zip: Path,
    destination_zip: Path,
    source_name: str,
) -> dict:
    try:
        with zipfile.ZipFile(
            source_zip,
            "r",
        ) as source:
            data_yaml_entry = (
                locate_data_yaml(
                    source
                )
            )

            try:
                data_yaml = yaml.safe_load(
                    source.read(
                        data_yaml_entry
                    ).decode(
                        "utf-8"
                    )
                )
            except (
                UnicodeDecodeError,
                yaml.YAMLError,
            ) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "data.yaml is invalid."
                    ),
                ) from exc

            if not isinstance(
                data_yaml,
                dict,
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "data.yaml must contain "
                        "a YAML object."
                    ),
                )

            class_names = (
                parse_import_class_names(
                    data_yaml
                )
            )

            pairs = (
                collect_standard_yolo_entries(
                    source
                )
            )

            annotation_count = 0
            polygon_conversion_count = 0

            with zipfile.ZipFile(
                destination_zip,
                "w",
                compression=
                    zipfile.ZIP_DEFLATED,
            ) as destination:
                for index, (
                    image_entry,
                    label_entry,
                ) in enumerate(pairs):
                    image_suffix = (
                        Path(
                            image_entry
                        ).suffix.lower()
                        or ".jpg"
                    )

                    unique_stem = (
                        f"import_{index:06d}"
                    )

                    label_text = (
                        source.read(
                            label_entry
                        ).decode(
                            "utf-8"
                        )
                    )

                    (
                        normalized_label_text,
                        normalized_count,
                        converted_polygon_count,
                    ) = normalize_import_label_text(
                        label_text=label_text,
                        class_count=len(class_names),
                        file_name=label_entry,
                    )

                    annotation_count += normalized_count
                    polygon_conversion_count += converted_polygon_count

                    destination.writestr(
                        (
                            "images/"
                            + unique_stem
                            + image_suffix
                        ),
                        source.read(
                            image_entry
                        ),
                    )

                    destination.writestr(
                        (
                            "labels/"
                            + unique_stem
                            + ".txt"
                        ),
                        normalized_label_text,
                    )

                classes_payload = {
                    "names": {
                        str(index):
                            class_name
                        for index,
                        class_name
                        in enumerate(
                            class_names
                        )
                    },
                    "input_size": 640,
                    "dataset_format":
                        "ultralytics-yolo",
                    "dataset_version": 1,
                    "source":
                        source_name,
                }

                dataset_lines = [
                    "path: .",
                    "train: images",
                    "val: images",
                    "names:",
                ]

                for index, class_name in (
                    enumerate(
                        class_names
                    )
                ):
                    escaped = (
                        class_name
                        .replace(
                            "\\",
                            "\\\\"
                        )
                        .replace(
                            '"',
                            '\\"'
                        )
                    )

                    dataset_lines.append(
                        f'  {index}: '
                        f'"{escaped}"'
                    )

                destination.writestr(
                    "classes.json",
                    json.dumps(
                        classes_payload,
                        indent=2,
                    ) + "\n",
                )

                destination.writestr(
                    "dataset.yaml",
                    "\n".join(
                        dataset_lines
                    ) + "\n",
                )

            return {
                "datasetFormat":
                    "ultralytics-yolo",
                "imageCount":
                    len(pairs),
                "labelFileCount":
                    len(pairs),
                "annotationCount":
                    annotation_count,
                "categoryCount":
                    len(
                        class_names
                    ),
                "classes":
                    class_names,
                "polygonAnnotationsConverted":
                    polygon_conversion_count,
                "importedFrom":
                    source_name,
            }
    except zipfile.BadZipFile as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Imported file is not "
                "a valid ZIP."
            ),
        ) from exc


def read_upload_records() -> list[dict]:
    records = []

    for folder in (
        UPLOAD_DIRECTORY.iterdir()
    ):
        if not folder.is_dir():
            continue

        metadata_path = (
            folder /
            "metadata.json"
        )

        dataset_path = (
            folder /
            "dataset.zip"
        )

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

        records.append(
            metadata
        )

    records.sort(
        key=lambda item:
            item.get(
                "receivedAt",
                "",
            ),
        reverse=True,
    )

    return records


def find_upload_folder(
    upload_id: str,
) -> Path:
    for folder in (
        UPLOAD_DIRECTORY.iterdir()
    ):
        if not folder.is_dir():
            continue

        metadata_path = (
            folder /
            "metadata.json"
        )

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

        if (
            metadata.get(
                "uploadId"
            )
            == upload_id
        ):
            return folder

    raise HTTPException(
        status_code=404,
        detail=(
            "Upload was not found."
        ),
    )


def create_upload_folder(
    installation_id: str,
    upload_id: str,
) -> tuple[Path, datetime]:
    received_at = datetime.now(
        timezone.utc
    )

    timestamp = (
        received_at.strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    folder_name = (
        f"{timestamp}_"
        f"{safe_component(installation_id)}_"
        f"{upload_id}"
    )

    upload_folder = (
        UPLOAD_DIRECTORY /
        folder_name
    )

    upload_folder.mkdir(
        parents=True,
        exist_ok=False,
    )

    return (
        upload_folder,
        received_at,
    )


@app.get(
    "/health",
    response_class=JSONResponse,
)
async def health() -> dict:
    return {
        "status": "ok",
        "service": APP_NAME,
        "datasetFormat":
            "ultralytics-yolo",
        "importPage":
            "/visualvocab/import-yolo",
        "uploadDirectory":
            str(
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
    x_visualvocab_api_key:
        Annotated[
            str | None,
            Header(),
        ] = None,
) -> JSONResponse:
    require_upload_api_key(
        x_visualvocab_api_key
    )

    try:
        parsed_metadata = (
            json.loads(
                metadata
            )
        )
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "The metadata field is "
                "not valid JSON."
            ),
        ) from exc

    upload_id = str(
        uuid.uuid4()
    )

    upload_folder, received_at = (
        create_upload_folder(
            installation_id=
                installationId,
            upload_id=
                upload_id,
        )
    )

    zip_path = (
        upload_folder /
        "dataset.zip"
    )

    try:
        total_bytes = (
            write_upload_stream(
                dataset,
                zip_path,
            )
        )

        validation = (
            validate_native_dataset_zip(
                zip_path
            )
        )

        record = {
            "uploadId":
                upload_id,
            "receivedAt":
                received_at
                    .isoformat(),
            "sourceType":
                "android-teach-mode",
            "installationId":
                installationId,
            "modelVersion":
                modelVersion,
            "exampleCount":
                exampleCount,
            "datasetBytes":
                total_bytes,
            "validation":
                validation,
            "clientMetadata":
                parsed_metadata,
        }

        (
            upload_folder /
            "metadata.json"
        ).write_text(
            json.dumps(
                record,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        return JSONResponse(
            status_code=201,
            content={
                "uploadId":
                    upload_id,
                "message":
                    (
                        "YOLO dataset "
                        "uploaded successfully."
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
    "/visualvocab/import-yolo",
    response_class=HTMLResponse,
)
async def import_yolo_page(
    request: Request,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="import_yolo.html",
        context={
            "service_name":
                APP_NAME,
        },
    )


@app.post(
    "/visualvocab/import-yolo",
)
async def import_yolo_dataset(
    dataset: Annotated[
        UploadFile,
        File(...),
    ],
    datasetName: Annotated[
        str,
        Form(...),
    ],
    x_visualvocab_api_key:
        Annotated[
            str | None,
            Header(),
        ] = None,
):
    require_upload_api_key(
        x_visualvocab_api_key
    )

    upload_id = str(
        uuid.uuid4()
    )

    installation_id = (
        "developer-import"
    )

    upload_folder, received_at = (
        create_upload_folder(
            installation_id=
                installation_id,
            upload_id=
                upload_id,
        )
    )

    source_path = (
        upload_folder /
        "source.zip"
    )

    normalized_path = (
        upload_folder /
        "dataset.zip"
    )

    try:
        source_bytes = (
            write_upload_stream(
                dataset,
                source_path,
            )
        )

        validation = (
            convert_standard_yolo_zip(
                source_zip=
                    source_path,
                destination_zip=
                    normalized_path,
                source_name=
                    dataset.filename
                    or datasetName,
            )
        )

        normalized_validation = (
            validate_native_dataset_zip(
                normalized_path
            )
        )

        record = {
            "uploadId":
                upload_id,
            "receivedAt":
                received_at
                    .isoformat(),
            "sourceType":
                "developer-yolo-import",
            "datasetName":
                datasetName.strip()
                or (
                    dataset.filename
                    or "Imported YOLO dataset"
                ),
            "sourceFileName":
                dataset.filename,
            "installationId":
                installation_id,
            "modelVersion":
                "external-dataset",
            "exampleCount":
                normalized_validation[
                    "imageCount"
                ],
            "sourceArchiveBytes":
                source_bytes,
            "datasetBytes":
                normalized_path
                    .stat()
                    .st_size,
            "validation":
                normalized_validation,
            "importValidation":
                validation,
            "clientMetadata": {
                "importedBy":
                    "backend-admin",
            },
        }

        (
            upload_folder /
            "metadata.json"
        ).write_text(
            json.dumps(
                record,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        source_path.unlink(
            missing_ok=True
        )

        return RedirectResponse(
            url=(
                "/visualvocab/import-yolo"
                f"?success={upload_id}"
            ),
            status_code=303,
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
    records = (
        read_upload_records()
    )

    return {
        "count":
            len(records),
        "datasets":
            records,
    }


@app.get(
    "/visualvocab/datasets/{upload_id}/download",
)
async def download_dataset(
    upload_id: str,
) -> FileResponse:
    upload_folder = (
        find_upload_folder(
            upload_id
        )
    )

    dataset_path = (
        upload_folder /
        "dataset.zip"
    )

    return FileResponse(
        path=dataset_path,
        filename=(
            "visual_vocab_yolo_"
            + upload_id
            + ".zip"
        ),
        media_type=
            "application/zip",
    )


@app.get(
    "/visualvocab/datasets/{upload_id}/metadata",
    response_class=JSONResponse,
)
async def get_dataset_metadata(
    upload_id: str,
) -> JSONResponse:
    upload_folder = (
        find_upload_folder(
            upload_id
        )
    )

    metadata_path = (
        upload_folder /
        "metadata.json"
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
    records = (
        read_upload_records()
    )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "records":
                records,
            "service_name":
                APP_NAME,
        },
    )
