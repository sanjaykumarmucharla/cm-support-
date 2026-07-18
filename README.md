# 🇮🇳 CM Support — AI Customer Support Agent

AI support agent whose brain is the **Constitution of India** (as on 1 May 2024, up to the
106th Amendment). Answers queries with Article citations, creates tickets, tracks orders,
and escalates to human agents — following the workflow:

`Customer asks → AI understands → Searches Knowledge Base (Chroma) → Generates response (Groq LLM) → Escalates if needed`

## Free stack
| Layer | Tool | Cost |
|---|---|---|
| API framework | FastAPI | Free |
| LLM | Groq (Llama 3.3 70B) | Free tier |
| RAG orchestration | LangChain | Free |
| Vector DB | ChromaDB (embedded) | Free |
| Embeddings | FastEmbed (bge-small, ONNX) | Free, local |
| Hosting | Render free tier | Free |
| Mobile app | PWA (Add to Home Screen) | Free |

## Deploy on Render (step by step)

1. **Get a free Groq API key** → https://console.groq.com → API Keys → Create.
2. **Push this folder to GitHub** (public or private repo):
   ```bash
   git init && git add . && git commit -m "CM Support agent"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/cm-support.git
   git push -u origin main
   ```
3. **Render** → https://render.com (sign in with GitHub) → **New → Blueprint** →
   select your repo. Render reads `render.yaml` automatically.
4. When prompted, paste your **GROQ_API_KEY**.
5. Click **Apply**. First build takes ~5–10 min (it builds the vector index from the PDF).
6. Your app is live at `https://cm-support.onrender.com` 🎉

## Make it a mobile app (free)
Open the URL on your phone → browser menu → **Add to Home Screen**.
It installs as a fullscreen tricolor app (PWA) — no app store needed.

## Notes for the free tier
- Free Render services **sleep after 15 min idle**; first request after sleep takes ~50s.
  Keep-warm trick: free cron ping via https://cron-job.org every 10 min.
- Tickets are in-memory (reset on restart). For persistence, plug in **Supabase**
  (free tier) or **Google Sheets** via `gspread` in `make_ticket()`.
- Replace `DEMO_ORDERS` in `main.py` with your real orders API/database.

## Run locally
```bash
pip install -r requirements.txt
python ingest.py                       # builds chroma_db/ from data/constitution.pdf
export GROQ_API_KEY=gsk_...
uvicorn main:app --reload              # open http://localhost:8000
```
