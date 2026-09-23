# DigiGuide

A mobile + AI-powered tourist guide that recognizes historical landmarks through photos and returns detailed information instantly.

This project contains:

* **Flutter Frontend** (mobile app)
* **FastAPI Backend** (AI inference + recognition API)

The goal is to build an app where a user points their camera at a monument, snaps a photo, and instantly gets:

* Landmark name
* Description & history
* Geolocation
* Related images
* Recognition confidence

---

## Project Structure

```
digiguide/
 ├─ frontend/        # Flutter mobile app
 ├─ main.py          # FastAPI backend entry point
 ├─ .gitignore
 ├─ .venv/           # Virtual environment (ignored)
 └─ README.md
```

---

## Features (current & planned)

### Current

* Flutter UI structure created
* FastAPI backend initialized
* Virtual environment setup completed
* Git repository initialized

### Coming Next

* `/recognize` API endpoint (image upload)
* CLIP-based image embedding
* FAISS index search
* Integration with Flutter camera
* Offline/online recognition modes

---

## Setup Instructions

### 1️ Clone or Initialize the Project

If you haven't already:

```
git clone https://github.com/<your-username>/digiguide.git
cd digiguide
```

### 2️ Backend Setup (FastAPI)

#### Create & activate virtual environment

**Windows:**

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Install FastAPI

```bash
pip install fastapi uvicorn
```

#### Run backend

```bash
uvicorn main:app --reload --port 8000
```

Access backend:

* API root: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* Swagger docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### 3️ Frontend Setup (Flutter)

```
cd frontend
flutter pub get
flutter run
```

Make sure you've installed:

* Flutter SDK
* Android Studio / Xcode
* Device emulator or physical device

---

## Planned AI Architecture

The backend will evolve into:

* **CLIP Model** (image → vector)
* **FAISS Index** (vector → nearest landmark)
* **PostgreSQL** (metadata storage)
* **Object Storage (S3/MinIO)** for images

Workflow:

```
Flutter App → FastAPI → CLIP Encoder → FAISS Vector Search → Landmark Data → Flutter UI
```

---

## Recognition API (Planned Example)

```http
POST /recognize
Content-Type: multipart/form-data
file: <image>
```

Response:

```json
{
  "landmark": "Old Town Gate",
  "confidence": 0.97,
  "description": "A medieval gate from the 14th century...",
  "location": {
    "lat": 49.8728,
    "lon": 8.6512
  }
}
```

---

## Testing

After backend starts:

```bash
curl -X GET http://127.0.0.1:8000/
```

Should output:

```json
{"message": "DigiGuide backend running successfully!"}
```

---

## Contribution Guide

* Use feature branches (`feature/<name>`) for enhancements
* Commit messages should be clear and concise
* Follow Python & Dart best practices

---

##  License

This project is currently private and unlicensed.

---

> **Let’s build DigiGuide into a world-class AI tourist guide 🚀**
