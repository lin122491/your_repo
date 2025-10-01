# app.py
import os, math, json
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Set

from fastapi import FastAPI, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, EmailStr, field_validator
from passlib.context import CryptContext
import jwt
from jwt import InvalidTokenError

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Date, DateTime, Boolean, Text,
    UniqueConstraint
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.exc import IntegrityError


# -------------------- 基本設定 --------------------
APP_NAME = "遊戲配對網"
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")  # 正式請改環境變數
JWT_ALG = "HS256"
ACCESS_TOKEN_TTL_MIN = 24 * 60

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dating.db")
engine_kwargs = {"echo": False, "future": True}
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


# -------------------- 資料表 --------------------
class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("username", name="uq_users_username"),)

    id = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False, unique=True, index=True)
    email = Column(String(200), nullable=True)
    password_hash = Column(String(200), nullable=False)

    display_name = Column(String(60), nullable=True)       # 暱稱
    gender = Column(String(10), nullable=True)             # 男/女/其他
    birthday = Column(Date, nullable=True)                 # 生日
    bio = Column(Text, nullable=True)                      # 自我介紹
    interests_json = Column(Text, nullable=True)           # JSON 陣列字串
    city = Column(String(60), nullable=True)               # 城市

    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)

    consent_agreed_at = Column(DateTime, nullable=True)
    geo_precise_opt_in = Column(Boolean, nullable=False, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Like(Base):
    __tablename__ = "likes"
    __table_args__ = (UniqueConstraint("user_id", "target_id", name="uq_like_pair"),)

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, index=True, nullable=False)
    target_id = Column(Integer, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    room = Column(String(50), index=True, nullable=False)     # "{min_id}:{max_id}"
    sender_id = Column(Integer, index=True, nullable=False)
    recipient_id = Column(Integer, index=True, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


def create_db():
    Base.metadata.create_all(engine)


def _table_has_column(conn, table: str, col: str) -> bool:
    # 使用 exec_driver_sql 避免 ObjectNotExecutableError
    rows = conn.exec_driver_sql(f"PRAGMA table_info('{table}')").fetchall()
    names = {r[1] for r in rows}  # (cid, name, type, notnull, dflt_value, pk)
    return col in names


def migrate_users_table():
    # 輕量遷移：補新欄位（不刪不改型）
    with engine.begin() as conn:
        stmts = []
        def need(c): return not _table_has_column(conn, "users", c)

        if need("display_name"):      stmts.append("ALTER TABLE users ADD COLUMN display_name TEXT")
        if need("gender"):            stmts.append("ALTER TABLE users ADD COLUMN gender TEXT")
        if need("birthday"):          stmts.append("ALTER TABLE users ADD COLUMN birthday DATE")
        if need("bio"):               stmts.append("ALTER TABLE users ADD COLUMN bio TEXT")
        if need("interests_json"):    stmts.append("ALTER TABLE users ADD COLUMN interests_json TEXT")
        if need("city"):              stmts.append("ALTER TABLE users ADD COLUMN city TEXT")
        if need("lat"):               stmts.append("ALTER TABLE users ADD COLUMN lat REAL")
        if need("lng"):               stmts.append("ALTER TABLE users ADD COLUMN lng REAL")
        if need("consent_agreed_at"): stmts.append("ALTER TABLE users ADD COLUMN consent_agreed_at DATETIME")
        if need("geo_precise_opt_in"):stmts.append("ALTER TABLE users ADD COLUMN geo_precise_opt_in INTEGER DEFAULT 0")
        if need("created_at"):        stmts.append("ALTER TABLE users ADD COLUMN created_at DATETIME")
        if need("updated_at"):        stmts.append("ALTER TABLE users ADD COLUMN updated_at DATETIME")

        for s in stmts:
            conn.exec_driver_sql(s)


# -------------------- Schemas --------------------
class SignupIn(BaseModel):
    username: str
    password: str
    email: Optional[EmailStr] = None
    consent_agreed: bool  # 必須為 True

    @field_validator("email", mode="before")
    @classmethod
    def _empty_email_to_none(cls, v):
        if isinstance(v, str) and v.strip() == "":
            return None
        return v


class LoginIn(BaseModel):
    username: str
    password: str


class ProfileIn(BaseModel):
    display_name: Optional[str] = None
    gender: Optional[str] = None           # 男/女/其他（或 m/f）
    birthday: Optional[date] = None
    bio: Optional[str] = None
    interests: Optional[List[str]] = None  # 陣列
    city: Optional[str] = None


class MeOut(BaseModel):
    id: int
    username: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    gender: Optional[str] = None
    birthday: Optional[date] = None
    bio: Optional[str] = None
    interests: List[str] = []
    city: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    updated_at: Optional[datetime] = None
    consent_agreed_at: Optional[datetime] = None
    geo_precise_opt_in: bool = False


class LocationIn(BaseModel):
    lat: float
    lng: float


class NearbyIn(BaseModel):
    lat: float
    lng: float
    radius_km: float = 100.0


class LikeIn(BaseModel):
    target: str  # 對方 username


class SendIn(BaseModel):
    content: str


# -------------------- FastAPI --------------------
app = FastAPI(title=f"{APP_NAME} 後端 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


@app.exception_handler(Exception)
async def all_exception_handler(_req: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"server error: {str(exc)}"})


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------- Auth --------------------
def create_access_token(sub: str, ttl_minutes: int = ACCESS_TOKEN_TTL_MIN) -> str:
    payload = {"sub": sub, "exp": datetime.utcnow() + timedelta(minutes=ttl_minutes)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def get_username_from_token(token: str) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return str(payload["sub"])
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="無效或過期的授權")


def get_bearer_token(request: Request) -> str:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    raise HTTPException(status_code=401, detail="缺少 Bearer Token")


def current_user(db: Session, token: str) -> User:
    username = get_username_from_token(token)
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="用戶不存在")
    return user


# -------------------- Helpers --------------------
def to_interests_json(arr: Optional[List[str]]) -> Optional[str]:
    if not arr:
        return None
    norm = []
    for s in arr:
        s = (s or "").strip()
        if not s:
            continue
        norm.append(s.upper() if s.isascii() else s)  # 英數轉大寫，中文保留
    # 去重保序
    seen, out = set(), []
    for s in norm:
        if s not in seen:
            seen.add(s); out.append(s)
    return json.dumps(out, ensure_ascii=False)


def from_interests_json(s: Optional[str]) -> List[str]:
    if not s:
        return []
    try:
        return [str(x) for x in json.loads(s)]
    except Exception:
        return []


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def room_key_for(a_id: int, b_id: int) -> str:
    return f"{min(a_id, b_id)}:{max(a_id, b_id)}"


# -------------------- Routes：Auth / Profile / Nearby --------------------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/auth/signup")
def signup(payload: SignupIn, db: Session = Depends(get_db)):
    if not payload.consent_agreed:
        raise HTTPException(status_code=400, detail="需勾選同意條款後才能註冊")
    if db.query(User.id).filter(User.username == payload.username).first():
        raise HTTPException(status_code=400, detail="username 已使用")
    try:
        u = User(
            username=payload.username.strip(),
            email=(payload.email or None),
            password_hash=pwd_context.hash(payload.password),
            consent_agreed_at=datetime.utcnow(),
        )
        db.add(u)
        db.commit()
        return {"ok": True}
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="帳號已存在")
    except Exception as e:
        db.rollback()
        return JSONResponse(status_code=500, content={"detail": f"server error: {str(e)}"})


@app.post("/auth/login")
def login(payload: LoginIn, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.username == payload.username).first()
    if not u or not pwd_context.verify(payload.password, u.password_hash):
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
    return {"access_token": create_access_token(u.username), "token_type": "bearer"}


@app.get("/me", response_model=MeOut)
def get_me(request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    u = current_user(db, token)
    return MeOut(
        id=u.id, username=u.username, email=u.email,
        display_name=u.display_name, gender=u.gender, birthday=u.birthday,
        bio=u.bio, interests=from_interests_json(u.interests_json), city=u.city,
        lat=u.lat, lng=u.lng, updated_at=u.updated_at,
        consent_agreed_at=u.consent_agreed_at, geo_precise_opt_in=bool(u.geo_precise_opt_in),
    )


@app.patch("/me")
def patch_me(payload: ProfileIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    u = current_user(db, token)

    if payload.display_name is not None:
        u.display_name = payload.display_name.strip() or None
    if payload.gender is not None:
        g = (payload.gender or "").strip().lower()
        if g in ("男", "male", "m"):      u.gender = "男"
        elif g in ("女", "female", "f"):  u.gender = "女"
        elif g:                           u.gender = "其他"
        else:                             u.gender = None
    if payload.birthday is not None:
        u.birthday = payload.birthday
    if payload.bio is not None:
        u.bio = payload.bio.strip() or None
    if payload.interests is not None:
        u.interests_json = to_interests_json(payload.interests)
    if payload.city is not None:
        u.city = payload.city.strip() or None

    db.add(u)
    db.commit()
    return {"ok": True}


@app.post("/me/location")
def update_location(payload: LocationIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    u = current_user(db, token)
    u.lat = float(payload.lat)
    u.lng = float(payload.lng)
    db.add(u)
    db.commit()
    return {"ok": True}


@app.post("/nearby")
def nearby(payload: NearbyIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me = current_user(db, token)

    users = (
        db.query(User)
        .filter(User.id != me.id)
        .filter(User.lat.isnot(None), User.lng.isnot(None))
        .all()
    )
    out = []
    for u in users:
        d = haversine_km(payload.lat, payload.lng, u.lat, u.lng)
        if d <= max(0.0, payload.radius_km):
            out.append({
                "username": u.username,
                "display_name": u.display_name or u.username,
                "gender": u.gender,
                "birthday": u.birthday.isoformat() if u.birthday else None,
                "bio": u.bio,
                "interests": from_interests_json(u.interests_json),
                "city": u.city,
                "distance_km": round(d, 2),
            })
    out.sort(key=lambda x: x["distance_km"])
    return {"users": out}


# -------------------- 配對（Like / Match） --------------------
@app.post("/like")
def like_user(payload: LikeIn, request: Request, db: Session = Depends(get_db)):
    me = current_user(db, get_bearer_token(request))
    target = db.query(User).filter(User.username == payload.target).first()
    if not target or target.id == me.id:
        raise HTTPException(status_code=400, detail="對象不存在")

    try:
        db.add(Like(user_id=me.id, target_id=target.id))
        db.commit()
    except IntegrityError:
        db.rollback()  # 已點過同一人，忽略即可

    # 是否互讚（成為配對）
    mutual = db.query(Like).filter(Like.user_id == target.id, Like.target_id == me.id).first() is not None
    return {"ok": True, "matched": mutual}


@app.get("/matches")
def get_matches(request: Request, db: Session = Depends(get_db)):
    me = current_user(db, get_bearer_token(request))
    # 我喜歡的人
    liked_ids = {r.target_id for r in db.query(Like.target_id).filter(Like.user_id == me.id).all()}
    # 喜歡我的人
    liked_me_ids = {r.user_id for r in db.query(Like.user_id).filter(Like.target_id == me.id).all()}
    match_ids = liked_ids & liked_me_ids
    if not match_ids:
        return {"matches": []}
    users = db.query(User).filter(User.id.in_(list(match_ids))).all()
    out = []
    for u in users:
        out.append({
            "username": u.username,
            "display_name": u.display_name or u.username,
            "gender": u.gender,
            "birthday": u.birthday.isoformat() if u.birthday else None,
            "bio": u.bio,
            "interests": from_interests_json(u.interests_json),
            "city": u.city,
        })
    # 依 display_name/username 排序
    out.sort(key=lambda x: (x["display_name"] or x["username"]))
    return {"matches": out}


# -------------------- 訊息（HTTP） --------------------
@app.get("/messages/{peer_username}")
def get_messages(peer_username: str, request: Request, db: Session = Depends(get_db)):
    me = current_user(db, get_bearer_token(request))
    peer = db.query(User).filter(User.username == peer_username).first()
    if not peer:
        raise HTTPException(status_code=404, detail="對方不存在")

    room = room_key_for(me.id, peer.id)
    msgs = (
        db.query(Message)
        .filter(Message.room == room)
        .order_by(Message.created_at.asc())
        .limit(200)
        .all()
    )
    return {
        "messages": [
            {
                "sender": (me.username if m.sender_id == me.id else peer.username),
                "recipient": (peer.username if m.sender_id == me.id else me.username),
                "content": m.content,
                "created_at": m.created_at.isoformat() + "Z",
            }
            for m in msgs
        ]
    }


@app.post("/messages/{peer_username}")
def send_message(peer_username: str, payload: SendIn, request: Request, db: Session = Depends(get_db)):
    me = current_user(db, get_bearer_token(request))
    peer = db.query(User).filter(User.username == peer_username).first()
    if not peer:
        raise HTTPException(status_code=404, detail="對方不存在")
    content = (payload.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="訊息不得為空白")

    room = room_key_for(me.id, peer.id)
    msg = Message(room=room, sender_id=me.id, recipient_id=peer.id, content=content)
    db.add(msg); db.commit(); db.refresh(msg)

    # 若 WebSocket 房間有人在線，推播
    _payload = {
        "room": room,
        "sender": me.username,
        "recipient": peer.username,
        "content": content,
        "created_at": msg.created_at.isoformat() + "Z",
    }
    for ws in list(ROOMS.get(room, set())):
        try:
            import anyio  # 保守地避免阻塞
            # FastAPI/Starlette 的 WS 是 async send；這裡在同步路由中，僅嘗試排程（忽略失敗）
            anyio.from_thread.run(ws.send_json, _payload)
        except Exception:
            ROOMS[room].discard(ws)

    return {"ok": True}


# -------------------- WebSocket 聊天 --------------------
ROOMS: Dict[str, Set[WebSocket]] = {}

@app.websocket("/ws")
async def ws_chat(websocket: WebSocket):
    # 以 query string 帶 token & peer（前端已處理）
    token = websocket.query_params.get("token")
    peer_username = websocket.query_params.get("peer")
    if not token or not peer_username:
        await websocket.close(code=1008)
        return

    # 取自己與對方
    try:
        username = get_username_from_token(token)
    except HTTPException:
        await websocket.close(code=1008)
        return

    db = SessionLocal()
    try:
        me = db.query(User).filter(User.username == username).first()
        peer = db.query(User).filter(User.username == peer_username).first()
        if not me or not peer:
            await websocket.close(code=1008)
            return
        room = room_key_for(me.id, peer.id)
    finally:
        db.close()

    await websocket.accept()
    ROOMS.setdefault(room, set()).add(websocket)

    try:
        while True:
            data = await websocket.receive_json()
            content = str(data.get("content", "")).strip()
            if not content:
                continue
            # 寫庫＆廣播
            db = SessionLocal()
            try:
                msg = Message(room=room, sender_id=me.id, recipient_id=peer.id, content=content)
                db.add(msg); db.commit(); db.refresh(msg)
                payload = {
                    "room": room,
                    "sender": me.username,
                    "recipient": peer.username,
                    "content": content,
                    "created_at": msg.created_at.isoformat() + "Z",
                }
            finally:
                db.close()

            for ws in list(ROOMS.get(room, set())):
                try:
                    await ws.send_json(payload)
                except Exception:
                    ROOMS[room].discard(ws)
    except WebSocketDisconnect:
        pass
    finally:
        ROOMS.get(room, set()).discard(websocket)


# ---------- 免責聲明 / 隱私權政策 ----------
DISCLAIMER_HTML = f"""
<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<title>{APP_NAME}｜免責聲明</title>
<body style="font-family:system-ui, -apple-system, 'Noto Sans TC', Arial; line-height:1.7; max-width:900px; margin:40px auto; padding:0 16px;">
<h1>{APP_NAME}｜免責聲明</h1>
<p>本服務提供交友配對與聊天平台，不擔保使用者資料或配對結果之真實性與適用性；線上互動與線下會面風險由您自行承擔。如發現違法或不當內容，請來信 <b>call91122@gmail.com</b> 檢舉。</p>
<p>服務可能因維護或第三方異常而中斷或延遲，本服務不負賠償責任（法律強制規定除外）。</p>
<p>您使用本服務，即表示已閱讀並同意本免責聲明與隱私權政策。</p>
<p>服務提供者：<b>林俊穎</b>　聯絡信箱：<b>call91122@gmail.com</b></p>
</body></html>
"""

PRIVACY_HTML = f"""
<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<title>{APP_NAME}｜隱私權政策</title>
<body style="font-family:system-ui, -apple-system, 'Noto Sans TC', Arial; line-height:1.7; max-width:900px; margin:40px auto; padding:0 16px;">
<h1>{APP_NAME}｜隱私權政策</h1>
<p>我們僅蒐集提供服務所需之最小資料：暱稱、性別、生日、自介、興趣標籤、城市；啟用精準定位時，將另行徵得同意後蒐集經緯度，可隨時關閉。</p>
<p>您得行使查詢、閱覽、複製、補正、刪除及停止使用等權利；請寄信至 <b>call91122@gmail.com</b>，我們將於合理期間內處理。</p>
<p>資料可能儲存於境內/境外雲端，我們採傳輸加密、強雜湊等安全措施保護您的資料。</p>
<p>如有重大變更，將公告於本頁。</p>
</body></html>
"""

@app.get("/disclaimer", response_class=HTMLResponse)
def disclaimer_page():
    return HTMLResponse(content=DISCLAIMER_HTML)

@app.get("/privacy", response_class=HTMLResponse)
def privacy_page():
    return HTMLResponse(content=PRIVACY_HTML)


# ---------- 靜態與首頁 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    fp = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(fp):
        return FileResponse(fp)
    return HTMLResponse(f"""
    <!doctype html><meta charset="utf-8"><title>{APP_NAME}</title>
    <div style="font-family:system-ui;-apple-system,'Noto Sans TC',Arial;max-width:720px;margin:60px auto;line-height:1.7;">
      <h1>{APP_NAME}</h1>
      <p>後端已啟動。請放置 <code>static/index.html</code> 以使用前端。</p>
      <p><a href="/docs">Swagger API 文件</a> ｜ <a href="/disclaimer" target="_blank">免責聲明</a> ｜ <a href="/privacy" target="_blank">隱私權政策</a></p>
    </div>
    """)


# ---------- 啟動 ----------
@app.on_event("startup")
def _on_startup():
    create_db()
    migrate_users_table()
