# app.py
import os
import math
import json
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any

from fastapi import (
    FastAPI, Depends, HTTPException, Request,
    WebSocket, WebSocketDisconnect, Query,
)
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, EmailStr, field_validator

import jwt
from jwt import InvalidTokenError
from passlib.context import CryptContext

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Boolean, Text,
    ForeignKey, UniqueConstraint, and_, or_,
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# =========================
# 環境設定
# =========================
# =========================
# 環境設定
# =========================
# =========================
# 環境設定
# =========================
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"
ACCESS_TOKEN_TTL_MIN = 60 * 24 * 7  # 7 days

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dating.db")

# Postgres 統一用 psycopg v3
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+psycopg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

engine_kwargs = {"echo": False, "future": True}

# SQLite 與 PostgreSQL 需要不同 connect_args
connect_args: Dict[str, Any] = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

# ✅ 建立資料庫引擎，含連線池參數
engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_size=5,         # 最大常駐連線數
    max_overflow=0,      # 禁止超額連線，避免 Supabase 免費版被拒絕
    pool_timeout=30,     # 30 秒內沒有可用連線就報錯
    pool_recycle=1800,   # 連線最長存活 30 分鐘，自動回收
    **engine_kwargs
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# =========================
# 資料表
# =========================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False, unique=True, index=True)
    email = Column(String(200), nullable=True)
    password_hash = Column(String(200), nullable=False)
    display_name = Column(String(60), nullable=True)
    gender = Column(String(10), nullable=True)
    birthday = Column(DateTime, nullable=True)
    city = Column(String(60), nullable=True)
    bio = Column(Text, nullable=True)
    interests = Column(Text, nullable=True)
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    geo_precise_opt_in = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_profile_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    __table_args__ = (UniqueConstraint("username", name="uq_users_username"),)


class Like(Base):
    __tablename__ = "likes"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    target_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint("user_id", "target_id", name="uq_like_pair"),)


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    sender_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    recipient_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


def create_db():
    Base.metadata.create_all(bind=engine)

# =========================
# Pydantic Schemas
# =========================
def _normalize_interests_to_store(raw: Optional[str | List[str]]) -> str:
    if not raw:
        return ""
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in raw if str(p).strip()]
    norm: List[str] = []
    for p in parts:
        try:
            if p.encode("ascii", errors="ignore"):
                norm.append(p.upper())
            else:
                norm.append(p)
        except Exception:
            norm.append(p)
    seen = set()
    out = []
    for x in norm:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return ",".join(out)


def _interests_to_list_for_api(stored: Optional[str]) -> List[str]:
    if not stored:
        return []
    return [p for p in (s.strip() for s in stored.split(",")) if p]


class SignupIn(BaseModel):
    username: str
    password: str
    email: Optional[EmailStr] = None


class LoginIn(BaseModel):
    username: str
    password: str


def _canon_gender(v: str) -> str:
    v2 = v.strip().lower()
    if v2 in {"male", "男"}:
        return "male"
    if v2 in {"female", "女"}:
        return "female"
    if v2 in {"other", "其他"}:
        return "other"
    raise ValueError("gender must be male/female/other (或 男/女/其他)")


class ProfileIn(BaseModel):
    display_name: Optional[str] = None
    gender: Optional[str] = None
    birthday: Optional[date] = None
    city: Optional[str] = None
    bio: Optional[str] = None
    interests: Optional[List[str] | str] = None
    geo_precise_opt_in: Optional[bool] = None

    @field_validator("gender")
    @classmethod
    def chk_gender(cls, v: Optional[str]):
        if v is None:
            return v
        return _canon_gender(v)


class ProfileOut(BaseModel):
    id: int
    username: str
    display_name: Optional[str]
    gender: Optional[str]
    birthday: Optional[date]
    city: Optional[str]
    bio: Optional[str]
    interests: List[str]
    lat: Optional[float]
    lng: Optional[float]
    updated_at: Optional[datetime]


class LocationIn(BaseModel):
    lat: float
    lng: float


class NearbyIn(BaseModel):
    radius_km: float = 100.0
    gender: Optional[str] = None
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    interest: Optional[str] = None


class PairIn(BaseModel):
    target_ids: List[int]


class ChatMessageOut(BaseModel):
    id: int
    sender_id: int
    recipient_id: int
    content: str
    created_at: datetime

# =========================
# FastAPI App
# =========================
app = FastAPI(title="遊戲配對網後端")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# =========================
# 共用工具
# =========================
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(sub: str, ttl_minutes: int = ACCESS_TOKEN_TTL_MIN) -> str:
    payload = {"sub": sub, "exp": datetime.utcnow() + timedelta(minutes=ttl_minutes)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def decode_access_token(token: str) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload["sub"]
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="token invalid or expired")

def get_bearer_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    raise HTTPException(status_code=401, detail="missing bearer token")

def current_user(db: Session, token: str) -> User:
    username = decode_access_token(token)
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="user not found")
    return user

def calc_age(b: Optional[datetime]) -> Optional[int]:
    if not b:
        return None
    d = b.date()
    today = date.today()
    return today.year - d.year - ((today.month, today.day) < (d.month, d.day))

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def is_matched(db: Session, uid: int, vid: int) -> bool:
    a = db.query(Like).filter(and_(Like.user_id == uid, Like.target_id == vid)).first()
    b = db.query(Like).filter(and_(Like.user_id == vid, Like.target_id == uid)).first()
    return (a is not None) and (b is not None)

# =========================
# 例外處理
# =========================
@app.exception_handler(Exception)
async def all_exception_handler(_req: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"server error: {str(exc)}"})

# =========================
# 健康檢查 (✅ 增強版)
# =========================
@app.get("/health")
def health(db: Session = Depends(get_db)):
    try:
        db.execute("SELECT 1;")
        return {"status": "ok", "time": datetime.utcnow().isoformat()}
    except Exception as e:
        return {"status": "db_error", "detail": str(e)}

# =========================
# Routes
# =========================
@app.post("/auth/signup")
def signup(payload: SignupIn, db: Session = Depends(get_db)):
    if db.query(User.id).filter(User.username == payload.username).first():
        raise HTTPException(status_code=400, detail="username already used")
    user = User(
        username=payload.username,
        email=payload.email,
        password_hash=pwd_context.hash(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"ok": True, "user_id": user.id}


@app.post("/auth/login")
def login(payload: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not pwd_context.verify(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="username or password incorrect")
    token = create_access_token(user.username)
    return {"access_token": token, "token_type": "bearer"}


@app.get("/me", response_model=ProfileOut)
def get_me(request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me = current_user(db, token)
    return ProfileOut(
        id=me.id,
        username=me.username,
        display_name=me.display_name,
        gender=me.gender,
        birthday=me.birthday.date() if me.birthday else None,
        city=me.city,
        bio=me.bio,
        interests=_interests_to_list_for_api(me.interests),
        lat=me.lat,
        lng=me.lng,
        updated_at=me.updated_at,
    )


def _apply_profile_changes(me: User, payload: ProfileIn):
    if payload.display_name is not None:
        me.display_name = payload.display_name
    if payload.gender is not None:
        me.gender = payload.gender
    if payload.birthday is not None:
        me.birthday = datetime(payload.birthday.year, payload.birthday.month, payload.birthday.day)
    if payload.city is not None:
        me.city = payload.city
    if payload.bio is not None:
        me.bio = payload.bio
    if payload.interests is not None:
        me.interests = _normalize_interests_to_store(payload.interests)
    if payload.geo_precise_opt_in is not None:
        me.geo_precise_opt_in = bool(payload.geo_precise_opt_in)


@app.put("/me")
def update_me(payload: ProfileIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me = current_user(db, token)
    _apply_profile_changes(me, payload)
    db.add(me)
    db.commit()
    return {"ok": True}


@app.post("/me")
def update_me_post(payload: ProfileIn, request: Request, db: Session = Depends(get_db)):
    return update_me(payload, request, db)


@app.post("/me/location")
def update_location(payload: LocationIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me = current_user(db, token)
    me.lat = float(payload.lat)
    me.lng = float(payload.lng)
    me.updated_at = datetime.utcnow()
    db.add(me)
    db.commit()
    return {"ok": True}


@app.post("/nearby")
def nearby(payload: NearbyIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me = current_user(db, token)
    if me.lat is None or me.lng is None:
        raise HTTPException(status_code=400, detail="please set your location first")

    q = db.query(User).filter(User.id != me.id)
    if payload.gender:
        q = q.filter(User.gender == _canon_gender(payload.gender))
    users = q.filter(User.lat.isnot(None), User.lng.isnot(None)).all()

    result = []
    interest_upper = payload.interest.upper().strip() if payload.interest else None

    for u in users:
        d_km = haversine_km(me.lat, me.lng, u.lat, u.lng)
        if d_km > (payload.radius_km or 100.0):
            continue
        u_age = calc_age(u.birthday)
        if payload.min_age is not None and (u_age is None or u_age < payload.min_age):
            continue
        if payload.max_age is not None and (u_age is None or u_age > payload.max_age):
            continue
        if interest_upper:
            arr = _interests_to_list_for_api(u.interests)
            if not any(interest_upper in s.upper() for s in arr):
                continue
        result.append({
            "id": u.id,
            "username": u.username,
            "display_name": u.display_name,
            "gender": u.gender,
            "age": u_age,
            "city": u.city,
            "distance_km": round(d_km, 3),
            "interests": _interests_to_list_for_api(u.interests),
            "bio": u.bio or "",
            "liked": db.query(Like.id).filter(and_(Like.user_id == me.id, Like.target_id == u.id)).first() is not None,
            "matched": is_matched(db, me.id, u.id),
        })
    result.sort(key=lambda x: x["distance_km"])
    return {"count": len(result), "users": result}


@app.post("/pair")
def pair_users(payload: PairIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me = current_user(db, token)
    ok = 0
    fail = 0
    for tid in payload.target_ids:
        if tid == me.id:
            fail += 1
            continue
        if not db.query(User.id).filter(User.id == tid).first():
            fail += 1
            continue
        exists = db.query(Like.id).filter(and_(Like.user_id == me.id, Like.target_id == tid)).first()
        if exists:
            fail += 1
            continue
        like = Like(user_id=me.id, target_id=tid)
        db.add(like)
        try:
            db.commit()
            ok += 1
        except Exception:
            db.rollback()
            fail += 1
    return {"ok": ok, "fail": fail}


@app.get("/matches")
def get_matches(request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me = current_user(db, token)
    mine = db.query(Like).filter(Like.user_id == me.id).all()
    target_ids = [lk.target_id for lk in mine]
    theirs = db.query(Like).filter(and_(Like.user_id.in_(target_ids), Like.target_id == me.id)).all()
    matched_ids = {lk.user_id for lk in theirs}
    if not matched_ids:
        return {"matches": []}
    users = db.query(User).filter(User.id.in_(matched_ids)).all()
    out = []
    for u in users:
        out.append({
            "id": u.id,
            "username": u.username,
            "display_name": u.display_name,
            "gender": u.gender,
            "age": calc_age(u.birthday),
            "city": u.city,
            "interests": _interests_to_list_for_api(u.interests),
            "bio": u.bio or "",
            "distance_km": (round(haversine_km(me.lat, me.lng, u.lat, u.lng), 3)
                            if (me.lat and me.lng and u.lat and u.lng) else None),
        })
    out.sort(key=lambda x: (x["distance_km"] is None, x["distance_km"] or 10**9))
    return {"matches": out}

# =========================
# 聊天 WebSocket
# =========================
class WSManager:
    def __init__(self):
        self.active: Dict[int, WebSocket] = {}

    async def connect(self, user_id: int, ws: WebSocket):
        await ws.accept()
        self.active[user_id] = ws

    def disconnect(self, user_id: int):
        self.active.pop(user_id, None)

    async def send_to(self, user_id: int, payload: dict):
        ws = self.active.get(user_id)
        if ws:
            await ws.send_text(json.dumps(payload))


ws_manager = WSManager()


@app.get("/chat/history/{peer_id}", response_model=List[ChatMessageOut])
def chat_history(peer_id: int, request: Request, limit: int = 50, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me = current_user(db, token)
    if not is_matched(db, me.id, peer_id):
        raise HTTPException(status_code=403, detail="not matched")
    qs = db.query(Message).filter(
        or_(
            and_(Message.sender_id == me.id, Message.recipient_id == peer_id),
            and_(Message.sender_id == peer_id, Message.recipient_id == me.id),
        )
    ).order_by(Message.created_at.desc()).limit(max(10, min(limit, 200))).all()
    return [
        ChatMessageOut(
            id=m.id, sender_id=m.sender_id, recipient_id=m.recipient_id,
            content=m.content, created_at=m.created_at,
        )
        for m in reversed(qs)
    ]


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket, token: str = Query(...), peer_id: int = Query(...)):
    try:
        username = decode_access_token(token)
    except HTTPException:
        await websocket.accept()
        await websocket.send_text(json.dumps({"type": "error", "detail": "invalid token"}))
        await websocket.close()
        return
    db = SessionLocal()
    try:
        me = db.query(User).filter(User.username == username).first()
        if not me:
            await websocket.accept()
            await websocket.send_text(json.dumps({"type": "error", "detail": "user not found"}))
            await websocket.close()
            return
        if not is_matched(db, me.id, peer_id):
            await websocket.accept()
            await websocket.send_text(json.dumps({"type": "error", "detail": "not matched"}))
            await websocket.close()
            return
        await ws_manager.connect(me.id, websocket)
        await ws_manager.send_to(peer_id, {"type": "peer_online", "peer_id": me.id})
        while True:
            text = await websocket.receive_text()
            text = text.strip()
            if not text:
                continue
            msg = Message(sender_id=me.id, recipient_id=peer_id, content=text)
            db.add(msg)
            db.commit()
            db.refresh(msg)
            payload = {
                "type": "message",
                "id": msg.id,
                "sender_id": me.id,
                "recipient_id": peer_id,
                "content": msg.content,
                "created_at": msg.created_at.isoformat(),
            }
            await ws_manager.send_to(me.id, payload)
            await ws_manager.send_to(peer_id, payload)
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(me.id if "me" in locals() and me else -1)
        db.close()

# =========================
# 首頁
# =========================
@app.get("/", response_class=HTMLResponse)
def index():
    fp = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(fp):
        return FileResponse(fp)
    return HTMLResponse(
        """
        <!doctype html>
        <html lang="zh-Hant"><meta charset="utf-8"/>
        <title>遊戲配對網 API</title>
        <body style="font-family: system-ui, -apple-system, Segoe UI, Roboto">
          <h1>🚀 遊戲配對網 後端 API</h1>
          <p>服務已啟動。你可以前往 <a href="/docs">/docs</a> 測試 API。</p>
          <ul>
            <li>健康檢查：<a href="/health">/health</a></li>
            <li>Swagger：<a href="/docs">/docs</a></li>
          </ul>
        </body></html>
        """
    )

# 啟動時自動建表
create_db()
