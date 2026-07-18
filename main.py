"""
main.py — CM Support (Common Support) AI Customer Support Agent backend.

Implements the workflow from the architecture diagram:
  Customer asks a question -> AI understands -> Searches Knowledge Base (Chroma Vector DB
  over the Constitution of India PDF) -> Generates response (Groq free-tier LLM) ->
  Escalates to human / creates ticket when needed.

Endpoints:
  POST /chat            -> main agent endpoint (answer | order_status | create_ticket | escalate)
  GET  /orders/{id}     -> demo order lookup (replace with your real orders DB/API)
  GET  /tickets         -> list tickets
  POST /tickets/{id}/resolve
  GET  /health
  /                     -> serves the mobile PWA frontend from /static
"""

import json
import os
import random
import re
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# ----------------------------------------------------------------------------- config
DB_DIR = os.getenv("DB_DIR", "chroma_db")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are the AI agent of "CM Support" (Common Support).
Your knowledge base is the CONSTITUTION OF INDIA (as on 1st May 2024, updated up to the
106th Amendment Act, 2023). Answer using the retrieved knowledge-base context below plus
your knowledge of the Constitution. Always cite Article numbers.

WORKFLOW: understand the question -> use the retrieved context -> generate a clear answer
-> escalate to a human agent when needed.

RULES:
1. Answer Constitution questions accurately and concisely (2-6 sentences). Cite Articles.
2. If the user asks about an ORDER (status, delivery, tracking), set action "order_status".
3. If the user is frustrated, asks for a human, has a personal legal dispute, or you cannot
   answer confidently, set action "escalate".
4. If the user reports an issue needing follow-up (refund, account, complaint) or asks for
   a ticket, set action "create_ticket" and fill the ticket object.
5. You are NOT a lawyer; recommend professional counsel for personal disputes and escalate.
6. Reply in the user's language (English, Hindi, Hinglish, etc.).

RESPOND WITH ONLY VALID JSON, no markdown fences:
{"reply":"...","action":"answer|order_status|create_ticket|escalate",
 "ticket":{"subject":"...","category":"Constitution Query|Refund|Account|Complaint|Legal Escalation|Other","priority":"Low|Medium|High"}}
Include "ticket" only for create_ticket/escalate."""

# ----------------------------------------------------------------------------- app
app = FastAPI(title="CM Support — AI Customer Support Agent")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Lazy singletons (loaded on first request so startup stays fast)
_retriever = None
_llm = None


def get_retriever():
    global _retriever
    if _retriever is None:
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
            collection_name="constitution_of_india",
        )
        _retriever = db.as_retriever(search_kwargs={"k": 4})
    return _retriever


def get_llm():
    global _llm
    if _llm is None:
        if not GROQ_API_KEY:
            raise HTTPException(500, "GROQ_API_KEY env var not set")
        _llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.2)
    return _llm


# ----------------------------------------------------------------------------- demo data
DEMO_ORDERS = {
    "12345": {"item": "Constitution of India — Pocket Diglot Edition 2024",
              "status": "Shipped", "eta": "May 18, 2026", "step": 3},
    "67890": {"item": "Bare Act Bundle (5 books)",
              "status": "Out for delivery", "eta": "Today, by 8 PM", "step": 4},
    "24680": {"item": "Constitutional Law Study Guide",
              "status": "Processing", "eta": "Jul 24, 2026", "step": 1},
}
TICKETS: List[dict] = []  # in-memory; swap for Supabase/Google Sheets in production


# ----------------------------------------------------------------------------- models
class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    text: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    reply: str
    action: str
    ticket: Optional[dict] = None
    order: Optional[dict] = None
    sources: List[str] = []


# ----------------------------------------------------------------------------- helpers
def make_ticket(t: Optional[dict], status: str = "Open") -> dict:
    ticket = {
        "id": f"TK-{random.randint(10000, 99999)}",
        "subject": (t or {}).get("subject", "Support request"),
        "category": (t or {}).get("category", "Other"),
        "priority": (t or {}).get("priority", "Medium"),
        "status": status,
        "time": datetime.now().strftime("%I:%M %p"),
    }
    TICKETS.insert(0, ticket)
    return ticket


def find_order(text: str) -> Optional[dict]:
    m = re.search(r"\d{5}", text)
    if m and m.group() in DEMO_ORDERS:
        return {"id": m.group(), **DEMO_ORDERS[m.group()]}
    return None


# ----------------------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"ok": True, "kb": "Constitution of India (May 2024)", "model": GROQ_MODEL}


@app.get("/orders/{order_id}")
def get_order(order_id: str):
    if order_id not in DEMO_ORDERS:
        raise HTTPException(404, "Order not found")
    return {"id": order_id, **DEMO_ORDERS[order_id]}


@app.get("/tickets")
def list_tickets():
    return TICKETS


@app.post("/tickets/{ticket_id}/resolve")
def resolve_ticket(ticket_id: str):
    for t in TICKETS:
        if t["id"] == ticket_id:
            t["status"] = "Resolved"
            return t
    raise HTTPException(404, "Ticket not found")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(400, "messages required")
    user_text = req.messages[-1].text

    # Fast path: order tracking (the "APIs" branch of the workflow diagram)
    order = None
    if re.search(r"order|track|deliver|shipp|parcel", user_text, re.I):
        order = find_order(user_text)
        if order:
            return ChatResponse(
                reply=(f"Your order #{order['id']} — {order['item']} — is "
                       f"{order['status'].lower()}. Estimated delivery: {order['eta']}."),
                action="order_status", order=order,
            )

    # RAG: search the knowledge base (Chroma over the Constitution PDF)
    docs = get_retriever().invoke(user_text)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    sources = sorted({f"page {d.metadata.get('page', '?')}" for d in docs})

    # Build LLM conversation (last 10 turns for context)
    lc_messages = [("system", SYSTEM_PROMPT +
                    f"\n\nRETRIEVED KNOWLEDGE BASE CONTEXT:\n{context}")]
    for m in req.messages[-10:]:
        lc_messages.append(("human" if m.role == "user" else "ai", m.text))

    raw = get_llm().invoke(lc_messages).content

    try:
        parsed = json.loads(re.sub(r"```json|```", "", raw).strip())
    except Exception:
        parsed = {"reply": raw, "action": "answer"}

    action = parsed.get("action", "answer")
    ticket = None
    if action == "escalate":
        ticket = make_ticket(parsed.get("ticket") or
                             {"subject": "Escalation", "category": "Legal Escalation",
                              "priority": "High"}, status="Escalated")
    elif action == "create_ticket":
        ticket = make_ticket(parsed.get("ticket"))
    elif action == "order_status":
        order = find_order(user_text)
        if not order:
            parsed["reply"] = ("I couldn't find that order. Please share a 5-digit order "
                               "number (try #12345 or #67890 in this demo).")

    return ChatResponse(reply=parsed.get("reply", ""), action=action,
                        ticket=ticket, order=order, sources=sources)


# Serve the mobile PWA frontend (must be mounted last)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
