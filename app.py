import os, math, json
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Set

from fastapi import FastAPI, Depends, HTTPException, Request, Body, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, EmailStr, field_validator, ConfigDict
import jwt
from jwt import InvalidTokenError

from passlib.context import CryptContext

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, UniqueConstraint,
    ForeignKey, func, text
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# ===================== 基本設定 =====================
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")  # 正式請改環境變數
JWT_ALG = "HS256"
ACCESS_TOKEN_TTL_MIN = 24 * 60

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dating.db")

engine_kwargs = {"echo": False, "future": True}
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
engine = create_engine(DATABASE_URL, connect_args=connect_args, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# 避免 bcrypt 版本問題：使用 pbkdf2_sha256（無需安裝 bcrypt）
pwd_ctx = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# ===================== 資料表 =====================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False, unique=True)
    email = Column(String(200), nullable=True)
    password_hash = Column(String(200), nullable=False)

    # 個人資料
    nickname = Column(String(80), nullable=True)
    gender = Column(String(10), nullable=True)           # male / female / None
    birthday = Column(String(20), nullable=True)         # ISO date string
    bio = Column(String(2000), nullable=True)
    interests = Column(String(4000), nullable=True)      # JSON array string
    city = Column(String(120), nullable=True)

    # 定位
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    updated_at = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("username", name="uq_users_username"),
    )


class ChatRoom(Base):
    __tablename__ = "chat_rooms"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatMember(Base):
    __tablename__ = "chat_members"
    room_id = Column(Integer, ForeignKey("chat_rooms.id"), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True)
    room_id = Column(Integer, ForeignKey("chat_rooms.id"), index=True, nullable=False)
    sender_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(String(1000), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

# 初次建表
Base.metadata.create_all(engine)

# 輕量遷移：缺少欄位就補（不影響既有資料）
def migrate_sqlite_users_table():
    with engine.begin() as conn:
        if not DATABASE_URL.startswith("sqlite"):
            return
        cols = {r[1] for r in conn.exec_driver_sql("PRAGMA table_info('users')").fetchall()}
        needed = {
            "nickname": "TEXT", "gender": "TEXT", "birthday": "TEXT", "bio": "TEXT",
            "interests": "TEXT", "city": "TEXT", "lat": "REAL", "lng": "REAL", "updated_at": "DATETIME"
        }
        for name, typ in needed.items():
            if name not in cols:
                conn.exec_driver_sql(f"ALTER TABLE users ADD COLUMN {name} {typ}")
        # 聊天表
        existing = {r[0] for r in conn.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "chat_rooms" not in existing:
            conn.exec_driver_sql("CREATE TABLE chat_rooms (id INTEGER PRIMARY KEY, created_at DATETIME)")
        if "chat_members" not in existing:
            conn.exec_driver_sql("CREATE TABLE chat_members (room_id INTEGER NOT NULL, user_id INTEGER NOT NULL, PRIMARY KEY (room_id, user_id))")
        if "chat_messages" not in existing:
            conn.exec_driver_sql("CREATE TABLE chat_messages (id INTEGER PRIMARY KEY, room_id INTEGER NOT NULL, sender_id INTEGER NOT NULL, content TEXT NOT NULL, created_at DATETIME)")

migrate_sqlite_users_table()

# ===================== 工具 =====================
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_pw(pw: str) -> str:
    return pwd_ctx.hash(pw)

def verify_pw(pw: str, h: str) -> bool:
    try:
        return pwd_ctx.verify(pw, h)
    except Exception:
        return False

def issue_token(sub: str, ttl_minutes: int = ACCESS_TOKEN_TTL_MIN) -> str:
    payload = {"sub": sub, "exp": datetime.utcnow() + timedelta(minutes=ttl_minutes)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def subject_from_token(token: str) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload["sub"]
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="無效或過期的授權")

def get_bearer_token(request: Request) -> str:
    auth = request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    raise HTTPException(status_code=401, detail="缺少 Bearer Token")

def current_user(db: Session, token: str) -> User:
    username = subject_from_token(token)
    u = db.query(User).filter(User.username == username).first()
    if not u:
        raise HTTPException(status_code=401, detail="用戶不存在")
    return u

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2.0)**2
    return round(2 * R * math.asin(math.sqrt(a)), 3)

def calc_age(birthday_iso: Optional[str]) -> Optional[int]:
    if not birthday_iso:
        return None
    try:
        y, m, d = [int(x) for x in birthday_iso.split("-")]
        b = date(y, m, d)
        today = date.today()
        age = today.year - b.year - ((today.month, today.day) < (b.month, b.day))
        return max(0, age)
    except Exception:
        return None

def normalize_gender(g: Optional[str]) -> Optional[str]:
    if not g: return None
    g = g.strip().lower()
    if g in ("male", "男", "m"): return "male"
    if g in ("female", "女", "f"): return "female"
    return None

def to_gender_label(g: Optional[str]) -> str:
    return {"male": "男", "female": "女"}.get(g or "", "")

def normalize_interests(value: Optional[str]) -> List[str]:
    """
    前端傳進來可能是「字串（逗號分隔）」或「陣列字串」。
    規則：
      - 英文自動轉大寫
      - 中文保持原樣
      - 去除空白與重複
    """
    if not value:
        return []
    items: List[str] = []
    try:
        # 嘗試把 JSON 陣列轉出來
        tmp = json.loads(value) if isinstance(value, str) and value.strip().startswith("[") else value
        if isinstance(tmp, list):
            items = [str(x) for x in tmp]
        else:
            raise ValueError
    except Exception:
        # 以逗號、頓號切
        raw = [p.strip() for p in value.replace("，", ",").split(",")]
        items = [p for p in raw if p]

    out: List[str] = []
    seen = set()
    for s in items:
        s2 = "".join(ch.upper() if ("a" <= ch <= "z" or "A" <= ch <= "Z") else ch for ch in s.strip())
        if s2 and s2 not in seen:
            seen.add(s2)
            out.append(s2)
    return out

def user_to_public(u: User, me: Optional[User] = None) -> dict:
    age = calc_age(u.birthday)
    out = {
        "id": u.id,
        "username": u.username,
        "nickname": u.nickname or "",
        "gender": u.gender,
        "gender_label": to_gender_label(u.gender),
        "age": age,
        "city": u.city,
        "bio": u.bio or "",
        "interests": json.loads(u.interests) if u.interests else [],
    }
    if me and me.lat is not None and me.lng is not None and u.lat is not None and u.lng is not None:
        out["distance_km"] = haversine_km(me.lat, me.lng, u.lat, u.lng)
    return out

# ===================== Schemas =====================
class SignupIn(BaseModel):
    username: str
    password: str
    email: Optional[EmailStr] = None

class LoginIn(BaseModel):
    username: str
    password: str

class MeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    nickname: Optional[str] = None
    gender: Optional[str] = None
    gender_label: Optional[str] = None
    birthday: Optional[str] = None
    age: Optional[int] = None
    bio: Optional[str] = None
    interests: List[str] = []
    city: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    updated_at: Optional[datetime] = None

class ProfileIn(BaseModel):
    nickname: Optional[str] = None
    gender: Optional[str] = None                 # 'male'/'female' 或 '男'/'女'
    birthday: Optional[str] = None               # YYYY-MM-DD
    bio: Optional[str] = None
    interests: Optional[str] = None              # 逗號字串或 JSON 陣列
    city: Optional[str] = None

    @field_validator("gender")
    @classmethod
    def _g(cls, v):
        return normalize_gender(v)

class LocationIn(BaseModel):
    lat: float
    lng: float

class NearbyIn(BaseModel):
    # 若未提供 lat/lng，後端會用自己的資料
    lat: Optional[float] = None
    lng: Optional[float] = None
    radius_km: float = 100.0
    gender: Optional[str] = None
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    interest: Optional[str] = None

# ===================== App / 中介層 =====================
app = FastAPI(title="遊戲配對網 Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.exception_handler(Exception)
async def all_exception_handler(_c: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"server error: {str(exc)}"})

# 靜態檔
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ===================== Routes：健康 / 首頁 =====================
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.get("/", response_class=HTMLResponse)
def index():
    fp = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(fp):
        return FileResponse(fp)
    return HTMLResponse("""
    <h1>遊戲配對網 Backend</h1>
    <ul>
      <li><a href="/health">健康檢查</a></li>
      <li><a href="/static/index.html">開啟前端頁面</a></li>
      <li><a href="/docs">Swagger API 文件</a></li>
    </ul>
    """)

# ===================== Auth =====================
@app.post("/auth/signup")
def signup(payload: SignupIn, db: Session = Depends(get_db)):
    if not payload.username or not payload.password:
        raise HTTPException(status_code=400, detail="需要 username 與 password")
    exists = db.query(User.id).filter(User.username == payload.username).first()
    if exists:
        raise HTTPException(status_code=400, detail="username 已使用")
    u = User(
        username=payload.username,
        email=payload.email,
        password_hash=hash_pw(payload.password),
        updated_at=datetime.utcnow(),
    )
    db.add(u)
    db.commit()
    return {"ok": True, "user_id": u.id}

@app.post("/auth/login")
def login(payload: LoginIn, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.username == payload.username).first()
    if not u or not verify_pw(payload.password, u.password_hash):
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
    return {"access_token": issue_token(u.username), "token_type": "bearer"}

# ===================== 我 / 個人檔 =====================
@app.get("/me", response_model=MeOut)
def me(request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    u = current_user(db, token)
    me_dict = user_to_public(u)
    me_dict.update({
        "lat": u.lat, "lng": u.lng, "updated_at": u.updated_at,
        "birthday": u.birthday, "age": calc_age(u.birthday),
    })
    return me_dict

@app.post("/me/profile")
def set_profile(payload: ProfileIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    u = current_user(db, token)
    if payload.nickname is not None: u.nickname = payload.nickname.strip() or None
    if payload.gender is not None:   u.gender = normalize_gender(payload.gender)
    if payload.birthday is not None: u.birthday = payload.birthday.strip() or None
    if payload.bio is not None:      u.bio = payload.bio.strip() or None
    if payload.city is not None:     u.city = payload.city.strip() or None
    if payload.interests is not None:
        ints = normalize_interests(payload.interests)
        u.interests = json.dumps(ints, ensure_ascii=False)
    db.add(u)
    db.commit()
    return {"ok": True, "me": me(Request, db)}  # 回傳最新

@app.post("/me/location")
def update_location(payload: LocationIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    u = current_user(db, token)
    u.lat, u.lng = float(payload.lat), float(payload.lng)
    u.updated_at = datetime.utcnow()
    db.add(u)
    db.commit()
    return {"ok": True}

# ===================== 附近 / 配對 =====================
@app.post("/nearby")
def nearby(payload: NearbyIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me = current_user(db, token)

    # 使用自己的定位，除非 body 指定
    lat = payload.lat if payload.lat is not None else me.lat
    lng = payload.lng if payload.lng is not None else me.lng
    if lat is None or lng is None:
        raise HTTPException(status_code=400, detail="尚未回報定位")

    q = db.query(User).filter(User.id != me.id, User.lat.isnot(None), User.lng.isnot(None))
    if payload.gender:
        q = q.filter(User.gender == normalize_gender(payload.gender))

    users = q.all()
    rows = []
    for u in users:
        d = haversine_km(lat, lng, u.lat, u.lng)
        if d <= payload.radius_km:
            row = user_to_public(u, me=me)
            rows.append(row)

    # 年齡/興趣過濾（在計算完 rows 再處理）
    if payload.min_age is not None:
        rows = [r for r in rows if (r.get("age") is not None and r["age"] >= payload.min_age)]
    if payload.max_age is not None:
        rows = [r for r in rows if (r.get("age") is not None and r["age"] <= payload.max_age)]
    if payload.interest:
        term = "".join(ch.upper() if ("a" <= ch <= "z" or "A" <= ch <= "Z") else ch for ch in payload.interest.strip())
        rows = [r for r in rows if term and term in (r.get("interests") or [])]

    rows.sort(key=lambda x: x.get("distance_km", 999999))
    return {"total": len(rows), "users": rows[:200]}

@app.get("/matches")
def matches(request: Request, db: Session = Depends(get_db)):
    """
    簡單版「我的配對」：以自己定位為中心，100 公里內的所有人（不含自己）
    你可以之後改為真正雙向喜歡/滑卡配對。
    """
    token = get_bearer_token(request)
    me = current_user(db, token)
    if me.lat is None or me.lng is None:
        return {"matches": []}
    q = db.query(User).filter(User.id != me.id, User.lat.isnot(None), User.lng.isnot(None)).all()
    out = []
    for u in q:
        d = haversine_km(me.lat, me.lng, u.lat, u.lng)
        if d <= 100.0:
            out.append(user_to_public(u, me=me))
    out.sort(key=lambda x: x.get("distance_km", 999999))
    return {"matches": out[:200]}

# ===================== 聊天（只能從「我的配對」開） =====================
def find_dm_room_id(db: Session, me_id: int, peer_id: int) -> Optional[int]:
    rid_rows = (
        db.query(ChatMember.room_id)
        .filter(ChatMember.user_id.in_([me_id, peer_id]))
        .group_by(ChatMember.room_id)
        .having(func.count(ChatMember.user_id) == 2)
        .all()
    )
    for (rid,) in rid_rows:
        members = db.query(ChatMember.user_id).filter(ChatMember.room_id == rid).all()
        member_ids = {m[0] for m in members}
        if member_ids == {me_id, peer_id}:
            return rid
    return None

@app.post("/chats/open_dm")
def open_dm(request: Request, peer_id: int = Body(..., embed=True), db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me = current_user(db, token)
    peer = db.query(User).filter(User.id == peer_id).first()
    if not peer:
        raise HTTPException(status_code=404, detail="對方不存在")

    rid = find_dm_room_id(db, me.id, peer.id)
    if rid is None:
        room = ChatRoom()
        db.add(room)
        db.flush()
        db.add_all([ChatMember(room_id=room.id, user_id=me.id),
                    ChatMember(room_id=room.id, user_id=peer.id)])
        db.commit()
        rid = room.id

    members = db.query(User).join(ChatMember, ChatMember.user_id == User.id)\
        .filter(ChatMember.room_id == rid).all()
    return {
        "room_id": rid,
        "members": [{"id": u.id, "username": u.username, "nickname": u.nickname} for u in members]
    }

@app.get("/chats/my")
def my_chats(request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me = current_user(db, token)
    room_ids = [r.room_id for r in db.query(ChatMember).filter(ChatMember.user_id == me.id).all()]
    rooms = []
    for rid in room_ids:
        mem_ids = [m.user_id for m in db.query(ChatMember).filter(ChatMember.room_id == rid)]
        members = db.query(User).filter(User.id.in_(mem_ids)).all()
        last = db.query(ChatMessage).filter(ChatMessage.room_id == rid).order_by(ChatMessage.id.desc()).first()
        rooms.append({
            "room_id": rid,
            "members": [{"id": u.id, "username": u.username, "nickname": u.nickname} for u in members],
            "last_message": ({"id": last.id, "sender_id": last.sender_id, "content": last.content,
                              "created_at": last.created_at.isoformat()} if last else None)
        })
    rooms.sort(key=lambda r: r["last_message"]["id"] if r["last_message"] else -1, reverse=True)
    return rooms

@app.get("/chats/{room_id}/history")
def chat_history(room_id: int, request: Request, limit: int = 50, before_id: Optional[int] = None,
                 db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me = current_user(db, token)
    is_member = db.query(ChatMember).filter(ChatMember.room_id == room_id, ChatMember.user_id == me.id).first()
    if not is_member:
        raise HTTPException(status_code=403, detail="無權限")
    q = db.query(ChatMessage).filter(ChatMessage.room_id == room_id)
    if before_id:
        q = q.filter(ChatMessage.id < before_id)
    msgs = q.order_by(ChatMessage.id.desc()).limit(limit).all()
    out = [{"id": m.id, "sender_id": m.sender_id, "content": m.content,
            "created_at": m.created_at.isoformat()} for m in reversed(msgs)]
    return {"messages": out}

# WebSocket
class WSManager:
    def __init__(self):
        self.rooms: Dict[int, Set[WebSocket]] = {}

    async def connect(self, room_id: int, ws: WebSocket):
        await ws.accept()
        self.rooms.setdefault(room_id, set()).add(ws)

    def disconnect(self, room_id: int, ws: WebSocket):
        if room_id in self.rooms:
            self.rooms[room_id].discard(ws)

    async def broadcast(self, room_id: int, data: dict):
        for ws in list(self.rooms.get(room_id, [])):
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(room_id, ws)

ws_manager = WSManager()

@app.websocket("/ws/chat/{room_id}")
async def ws_chat(room_id: int, websocket: WebSocket):
    # 允許 query token
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4401)
        return
    # 效驗
    try:
        username = subject_from_token(token)
    except Exception:
        await websocket.close(code=4401); return

    # 檢查是否為房內成員
    with SessionLocal() as db:
        me = db.query(User).filter(User.username == username).first()
        if not me:
            await websocket.close(code=4401); return
        is_member = db.query(ChatMember).filter(ChatMember.room_id == room_id, ChatMember.user_id == me.id).first()
        if not is_member:
            await websocket.close(code=4403); return

    await ws_manager.connect(room_id, websocket)
    try:
        with SessionLocal() as db:
            me = db.query(User).filter(User.username == username).first()
            while True:
                data = await websocket.receive_json()
                if (data.get("type") or "").lower() != "msg":
                    continue
                content = (data.get("content") or "").strip()
                if not content:
                    continue
                msg = ChatMessage(room_id=room_id, sender_id=me.id, content=content)
                db.add(msg)
                db.commit()
                out = {
                    "type": "msg", "id": msg.id, "room_id": room_id,
                    "sender_id": me.id, "content": msg.content,
                    "created_at": msg.created_at.isoformat()
                }
                await ws_manager.broadcast(room_id, out)
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(room_id, websocket)
