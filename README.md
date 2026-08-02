# VisualVocab Render Backend

This FastAPI service receives dataset ZIP files uploaded from VisualVocab
Teach Mode.

## Included endpoints

- `GET /health`
- `POST /visualvocab/datasets`
- `GET /visualvocab/datasets`
- `GET /visualvocab/datasets/{uploadId}/download`
- `GET /visualvocab/datasets/{uploadId}/metadata`
- `GET /` — simple upload dashboard
- `GET /docs` — FastAPI interactive documentation

## Deploy to Render

### 1. Create a GitHub repository

Create a new repository, for example:

```text
visualvocab-backend
```

Upload every file from this folder to the repository root.

The repository should look like:

```text
visualvocab-backend/
├── app.py
├── requirements.txt
├── Dockerfile
├── render.yaml
├── README.md
├── .gitignore
├── templates/
│   └── index.html
└── uploads/
    └── .gitkeep
```

### 2. Create the Render service

1. Sign in to Render.
2. Choose **New → Blueprint**.
3. Connect the `visualvocab-backend` repository.
4. Select `render.yaml`.
5. Deploy.

Render will create a URL similar to:

```text
https://visualvocab-api.onrender.com
```

### 3. Test it

Open:

```text
https://YOUR-SERVICE.onrender.com/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "VisualVocab Dataset Receiver"
}
```

Open the dashboard:

```text
https://YOUR-SERVICE.onrender.com/
```

### 4. Connect Android

Replace the placeholder in:

```text
app/src/main/java/com/example/visualvocab/data/datasetupload/DatasetUploadConfig.kt
```

with:

```kotlin
const val UPLOAD_URL =
    "https://YOUR-SERVICE.onrender.com/visualvocab/datasets"
```

Rebuild the Android app once.

Then:

1. Open an image.
2. Enter Teach Mode.
3. Confirm and save at least one object.
4. Tap **Upload dataset**.
5. Refresh the Render dashboard.

## Local development

Create an environment:

```bash
python -m venv .venv
```

Activate it:

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Optional upload API key

The server supports an optional header:

```text
X-VisualVocab-Api-Key
```

Set this Render environment variable:

```text
VISUALVOCAB_UPLOAD_API_KEY
```

The current Android upload client does not send this header yet. Leave the
environment variable unset until Android key support is added.

Do not embed a valuable permanent secret in a public Android application.
An app secret can be extracted from the APK.

## Storage warning

Render's ordinary service filesystem may be lost when the service redeploys,
restarts on another instance, or is replaced.

This package is suitable for proving the complete upload pipeline. For durable
production storage, use one of:

- a Render persistent disk on a supported plan;
- Amazon S3;
- Cloudflare R2;
- Google Cloud Storage;
- another object-storage service.

Do not treat free-instance local storage as permanent archival storage.

## Dataset validation

The server rejects uploads when:

- the file is not a valid ZIP;
- `annotations.json` is missing;
- there are no files under `images/`;
- `annotations.json` is not valid COCO-style JSON;
- the upload exceeds the configured size limit.

Each accepted upload is stored as:

```text
uploads/
└── TIMESTAMP_INSTALLATION_UPLOAD-ID/
    ├── dataset.zip
    └── metadata.json
```
