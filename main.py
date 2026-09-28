import os
import json
import requests
from datetime import datetime
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

load_dotenv()

# ----------------- DATABASE CONFIGURATION -----------------#
DATABASE_URL = "sqlite:///./flashcards.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DeckModel(Base):
    __tablename__ = "decks"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(150), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    cards = relationship("FlashcardModel", back_populates="deck", cascade="all, delete-orphan", lazy="joined")

class FlashcardModel(Base):
    __tablename__ = "flashcards"
    id = Column(Integer, primary_key=True, index=True)
    deck_id = Column(Integer, ForeignKey("decks.id"))
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    deck = relationship("DeckModel", back_populates="cards")

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----------------- GROQ CONFIG -----------------#
groq_api_key = os.getenv("GROQ_API_KEY")

# ----------------- SCHEMAS -----------------#
class GenerateRequest(BaseModel):
    title: str
    content: str
    card_count: int = 4

class CardResponse(BaseModel):
    id: int
    question: str
    answer: str

    class Config:
        from_attributes = True

class DeckResponse(BaseModel):
    id: int
    title: str
    created_at: datetime
    cards: List[CardResponse] = []

    class Config:
        from_attributes = True

# ----------------- FASTAPI APP -----------------#
app = FastAPI(title="AI Flashcards")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#...............FOR DEPLOYMENT..............#

@app.get("/")
def serve_home():
    return FileResponse("index.html")

# DIRECT FRONTEND ROUTE (Live Server auto-reload bypass)
@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/api/generate-deck", response_model=DeckResponse)
def generate_deck(req: GenerateRequest, db: Session = Depends(get_db)):
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing in .env file.")

    prompt = f"""
    Create {req.card_count} concise flashcards from the notes below.
    Return ONLY a JSON object with key 'flashcards', an array of objects with keys 'question' and 'answer'.
    
    Topic: {req.title}
    Notes: {req.content}
    """

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key.strip()}",
        "Content-Type": "application/json"
    }

    candidate_models = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
        "compound-beta"
    ]

    cards_list = None
    session = requests.Session()
    session.trust_env = False

    for model_name in candidate_models:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a flashcard generator that always outputs valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }

        try:
            res = session.post(url, headers=headers, json=payload, timeout=25)
            if res.status_code == 200:
                data = res.json()
                raw_text = data["choices"][0]["message"]["content"]
                parsed = json.loads(raw_text)
                cards_list = parsed.get("flashcards", [])
                if not cards_list and isinstance(parsed, dict):
                    cards_list = list(parsed.values())[0] if parsed else []
                break
        except Exception:
            continue

    if not cards_list:
        raise HTTPException(status_code=500, detail="Flashcards generation failed")

    # DB Save
    new_deck = DeckModel(title=req.title)
    db.add(new_deck)
    db.commit()
    db.refresh(new_deck)

    for item in cards_list:
        if isinstance(item, dict):
            card = FlashcardModel(
                deck_id=new_deck.id,
                question=item.get("question", "N/A"),
                answer=item.get("answer", "N/A")
            )
            db.add(card)
    db.commit()

    return db.query(DeckModel).filter(DeckModel.id == new_deck.id).first()

@app.get("/api/decks", response_model=List[DeckResponse])
def get_all_decks(db: Session = Depends(get_db)):
    return db.query(DeckModel).order_by(DeckModel.created_at.desc()).all()

