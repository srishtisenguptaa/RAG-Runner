# The Architect RAG — Docker Deployment Guide

## Project Structure Expected

```
RAG-Runner/
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── nginx.conf
├── requirements.txt
├── .env                   ← you create this (copy from .env.example)
├── .env.example
├── .dockerignore
├── apiCall/
│   └── api.py
├── model/
│   ├── model.py
│   └── modelLG.py
└── FrontEnd/
    ├── index.html
    ├── main.js            ← use the updated version from this package
    └── styles.css
```

---

## 1. Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Your API keys ready

---

## 2. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxx
```

---

## 3. Build & Run

```bash
docker compose up --build
```

First build takes ~5–8 minutes (downloading ML models).  
Subsequent starts are fast.

Open your browser at: **http://localhost**

---

## 4. Useful Commands

| Command | Description |
|---|---|
| `docker compose up --build` | Build images and start |
| `docker compose up -d` | Start in background |
| `docker compose down` | Stop containers |
| `docker compose logs -f backend` | Stream backend logs |
| `docker compose logs -f frontend` | Stream nginx logs |
| `docker compose restart backend` | Restart backend only |

---

## 5. Deploying to a Cloud VM (AWS EC2 / GCP / Azure / DigitalOcean)

### Step 1 — Provision a VM
Recommended: **2 vCPU, 4 GB RAM** minimum (8 GB preferred for LLM embeddings).
Open ports **80** (HTTP) and optionally **443** (HTTPS) in the firewall/security group.

### Step 2 — Install Docker on the VM

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

### Step 3 — Copy files to VM

```bash
# From your local machine
scp -r . user@YOUR_VM_IP:~/rag-runner/
```

Or clone from Git if you have a repo:
```bash
git clone https://github.com/yourrepo/rag-runner.git
cd rag-runner
```

### Step 4 — Set up .env and run

```bash
cp .env.example .env
nano .env          # fill in your API keys
docker compose up --build -d
```

Your app is now live at **http://YOUR_VM_IP**

### Step 5 — Add HTTPS with Let's Encrypt (optional but recommended)

```bash
sudo apt install certbot
sudo certbot certonly --standalone -d yourdomain.com
```

Then update `nginx.conf` to add SSL (see `nginx.conf` comments).

---

## 6. Render.com / Railway (free-tier cloud)

Both platforms support Docker deployment from a Git repo.

### Render
1. Push your project to GitHub
2. New → Web Service → Connect repo
3. Set **Dockerfile path** to `Dockerfile.backend`
4. Add env vars `GROQ_API_KEY` and `TAVILY_API_KEY` in the dashboard
5. Deploy the frontend separately as a **Static Site** (just the FrontEnd/ folder)
6. Update the `API` constant in `main.js` to point to your Render backend URL

### Railway
1. New Project → Deploy from GitHub
2. Add both services (backend + frontend) pointing to their respective Dockerfiles
3. Set env vars in the Railway dashboard

---

## 7. Notes

- **Vector index is in-memory** — it resets when the container restarts. For persistence,
  add a FAISS save/load to disk in `model.py` and mount a Docker volume.
- **Session history is in-memory** — same caveat.
- The `sentence-transformers` model (`all-MiniLM-L6-v2`) is downloaded at first startup
  and cached inside the container layer after the first build.
