# Deploying SUPPLIER RISK PREDICTION to a public URL

The dashboard is ready to run on any Streamlit-compatible host. The recommended path is **Streamlit Community Cloud** (free, no server to manage, gives you a permanent `https://….streamlit.app` URL that opens from any laptop or phone). A Dockerfile is included for Render / Railway / Hugging Face Spaces if you prefer.

> Everything below has been prepared in this repository: `requirements.txt`, `.streamlit/config.toml`, `.python-version` (3.12), `Dockerfile`, trained models in `models/` and sample data in `data/`. The **only step that needs your own account** is pushing to GitHub and clicking *Deploy*.

---

## Option A – Streamlit Community Cloud (recommended, ~5 minutes)

### 1. Put the project on GitHub

Open a terminal in the project folder (`SUPPLIER RISK PREDICTION`).

```powershell
# one-time: log in to GitHub from the CLI (opens the browser)
gh auth login

# create a repository from this folder and push it
git add .
git commit -m "Supplier risk prediction dashboard"
gh repo create supplier-risk-prediction --public --source=. --remote=origin --push
```

(If you do not use the GitHub CLI: create an empty repo at <https://github.com/new>, then
`git remote add origin https://github.com/<your-user>/supplier-risk-prediction.git` and `git push -u origin main`.)

The repo is about 25 MB (models + datasets), well within GitHub limits.

### 2. Deploy

1. Go to <https://share.streamlit.io> and sign in with the same GitHub account.
2. Click **Create app → Deploy a public app from GitHub**.
3. Fill in:
   * **Repository:** `<your-user>/supplier-risk-prediction`
   * **Branch:** `main`
   * **Main file path:** `app.py`
   * **App URL:** choose something like `supplier-risk-prediction` → the app will live at `https://supplier-risk-prediction.streamlit.app`
4. Open **Advanced settings** and set **Python version = 3.12**.
5. Click **Deploy**. The first build installs the requirements (2–4 minutes). Later pushes to `main` redeploy automatically.

### 3. Share

Send the `https://….streamlit.app` link to anyone – it works from any laptop, no login needed for viewers. The app title in the browser tab is **SUPPLIER RISK PREDICTION**.

Free tier limits: 1 GB RAM per app (this dashboard uses ~300–400 MB with the 10 000-row dataset) and the app sleeps after a few days without visitors (it wakes on the next visit).

---

## Option B – Docker on Render / Railway / Fly.io

1. Push the repo to GitHub as in step A-1.
2. **Render:** New → Web Service → connect the repo → Runtime *Docker* → Instance *Free*. Render injects `$PORT`; the Dockerfile already binds to it. Your URL: `https://<service>.onrender.com`.
3. **Railway:** New Project → Deploy from GitHub → it detects the Dockerfile automatically → Settings → Generate Domain.

Local check of the container:

```powershell
docker build -t supplier-risk .
docker run -p 8501:8501 supplier-risk
```

---

## Option C – Hugging Face Spaces

1. Create a Space at <https://huggingface.co/new-space> with **SDK = Streamlit**, Python 3.12 hardware *CPU basic*.
2. Add these lines at the very top of `README.md` (Spaces reads them):

   ```yaml
   ---
   title: SUPPLIER RISK PREDICTION
   sdk: streamlit
   sdk_version: 1.64.0
   app_file: app.py
   pinned: false
   ---
   ```
3. Push the repository to the Space's git remote. URL: `https://huggingface.co/spaces/<user>/<space>`.

---

## Option D – Temporary public URL from this laptop (demo only)

If you just need to show the running app from your own machine for a presentation:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py          # terminal 1
# terminal 2 – any one of:
npx localtunnel --port 8501                               # prints https://xxxx.loca.lt
cloudflared tunnel --url http://localhost:8501            # prints https://xxxx.trycloudflare.com
ngrok http 8501                                           # prints https://xxxx.ngrok-free.app
```

The link works only while your laptop and the tunnel are running. Use Option A for a permanent URL.

---

## Configuration notes

* `.streamlit/config.toml` sets the light theme, `server.headless=true`, upload limit 200 MB. Community Cloud, Render and Spaces all honour it.
* No secrets are required. If you later add any, use **App settings → Secrets** on Community Cloud (never commit them).
* `pyspark` is intentionally **not** in `requirements.txt` (it needs Java and ~400 MB). The app detects its absence and uses the identical pandas implementation; install it locally if you want to demonstrate the Big-Data engine toggle.
* To retrain the models on a host: `python scripts/train_models.py` (uses `data/supplier_risk_10000rows_19columns.xlsx`).
