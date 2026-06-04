# GradeGluay

Flask app for grading a banana comb photo using a Thai 10-Baht coin as scale.

## Project Structure

```text
app.py                  # WSGI entrypoint
gradegluay/
  __init__.py           # Flask app factory
  config.py             # Environment-based config
  routes.py             # Web routes
  utils/
    image_processing.py # Banana/coin image logic
    storage.py          # Sales CSV initialization
templates/              # Jinja templates
static/                 # Public static assets
uploads/                # Runtime uploads, ignored by git
data/                   # Runtime CSV data, ignored by git
```

## Local Setup

1. Create a virtual environment.

```bash
python -m venv .venv
```

2. Activate it.

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Create local environment config.

```bash
cp .env.example .env
```

Set `SECRET_KEY` in `.env` to a long random value before deploying.

5. Add the guide image.

Place the guide image at:

```text
static/guide.png
```

6. Run locally.

```bash
python app.py
```

Open `http://127.0.0.1:5000/`.

## Deployment

This repo includes a `Procfile` for Railway or Render:

```text
web: gunicorn app:app
```

Set environment variables in the platform dashboard. At minimum set:

```text
SECRET_KEY
FLASK_DEBUG=0
```

For production rate limiting across multiple workers or instances, replace `RATELIMIT_STORAGE_URI=memory://` with a shared backend such as Redis.
