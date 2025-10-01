import os
import math
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, EmailStr, field_validator
import jwt
from jwt import InvalidTokenError
from passlib.context import CryptContext

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, UniqueConstraint
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.exc import IntegrityError

# ---------------- 基本設定 ----------------
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")  # 正式請改環境變數
JWT_ALG = "HS256"
ACCESS_TOKEN_TTL_MIN = 60 * 24 * 7  # 7 天

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dating.db")

engine_kwargs = {"echo": False, "future": True}
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
engine = create_engine(DATABASE_URL, connect_args=connect_args, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# 使用 PBKDF2-SHA256（避免 bcrypt 相容性問題）
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# ---------------- Models ----------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(200), nullable=True)
    password_hash = Column(String(200), nullable=False)
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    __table_args__ = (UniqueConstraint("username", name="uq_username"),)

# ---------------- DB 建表 & 輕量遷移 ----------------
def create_db():
    Base.metadata.create_all(bind=engine)

def migrate_sqlite_users_table():
    """SQLite 輕量遷移：缺欄位就補（lat/lng/updated_at），不中斷服務、不清資料。"""
    try:
        if engine.dialect.name != "sqlite":
            return
        with engine.begin() as conn:
            rows = conn.exec_driver_sql("PRAGMA table_info('users')").fetchall()
            cols = {r[1] for r in rows}
            stmts = []
            if "lat" not in cols:
                stmts.append("ALTER TABLE users ADD COLUMN lat REAL")
            if "lng" not in cols:
                stmts.append("ALTER TABLE users ADD COLUMN lng REAL")
            if "updated_at" not in cols:
                stmts.append("ALTER TABLE users ADD COLUMN updated_at DATETIME")
            for s in stmts:
                conn.exec_driver_sql(s)
            if stmts:
                print("[migrate] users 補欄位：", ", ".join(s.split()[-1] for s in stmts))
    except Exception as e:
        print("[migrate] 跳過/失敗：", e)

# ---------------- Schemas ----------------
class SignupIn(BaseModel):
    username: str
    password: str
    email: Optional[EmailStr] = None

    @field_validator("email", mode="before")
    @classmethod
    def empty_to_none(cls, v):
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

class LoginIn(BaseModel):
    username: str
    password: str

class MeOut(BaseModel):
    id: int
    username: str
    email: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    updated_at: Optional[datetime] = None

class LocationIn(BaseModel):
    lat: float
    lng: float

class NearbyIn(BaseModel):
    lat: float
    lng: float
    radius_km: float = 5.0

# ---------------- App 本體 ----------------
app = FastAPI(title="Dating Prototype")

# CORS：前端可能不同網域，先全開
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def all_exception_handler(_: Request, exc: Exception):
    # 統一 500 格式，方便前端顯示
    return JSONResponse(status_code=500, content={"detail": f"server error: {str(exc)}"})

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------- Auth (PyJWT) ----------------
def create_access_token(sub: str, ttl_minutes: int = ACCESS_TOKEN_TTL_MIN) -> str:
    payload = {"sub": sub, "exp": datetime.utcnow() + timedelta(minutes=ttl_minutes)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload["sub"]
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="無效或過期的憑證")

def current_user(db: Session, token: str) -> User:
    username = decode_token(token)
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="使用者不存在")
    return user

def get_bearer_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    raise HTTPException(status_code=401, detail="缺少 Bearer Token")

# ---------------- Routes ----------------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.post("/auth/signup")
def signup(payload: SignupIn, db: Session = Depends(get_db)):
    try:
        if db.query(User.id).filter(User.username == payload.username).first():
            raise HTTPException(status_code=400, detail="username 已被使用")
        user = User(
            username=payload.username,
            email=payload.email,
            password_hash=pwd_context.hash(payload.password),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return {"ok": True, "user_id": user.id}
    except HTTPException:
        db.rollback()
        raise
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="帳號或 email 已存在")
    except Exception as e:
        db.rollback()
        return JSONResponse(status_code=500, content={"detail": f"server error: {str(e)}"})

@app.post("/auth/login")
def login(payload: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not pwd_context.verify(payload.password, user.password_hash):
        raise HTTPException(status_code=400, detail="帳號或密碼錯誤")
    token = create_access_token(user.username)
    return {"access_token": token, "token_type": "bearer"}

@app.get("/me", response_model=MeOut)
def me(request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    user = current_user(db, token)
    return MeOut(
        id=user.id,
        username=user.username,
        email=user.email,
        lat=user.lat,
        lng=user.lng,
        updated_at=user.updated_at,
    )

@app.post("/me/location")
def update_location(payload: LocationIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    user = current_user(db, token)
    user.lat = payload.lat
    user.lng = payload.lng
    user.updated_at = datetime.utcnow()
    db.add(user)
    db.commit()
    return {"ok": True}

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

@app.post("/nearby")
def nearby(payload: NearbyIn, request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    me = current_user(db, token)
    users: List[User] = (
        db.query(User)
        .filter(User.id != me.id)
        .filter(User.lat.isnot(None), User.lng.isnot(None))
        .all()
    )
    out: List[Tuple[str, float]] = []
    for u in users:
        d = haversine_km(payload.lat, payload.lng, u.lat, u.lng)
        if d <= payload.radius_km:
            out.append({"username": u.username, "distance_km": round(d, 3)})
    out.sort(key=lambda x: x["distance_km"])
    return {"count": len(out), "users": out}

@app.get("/matches")
def matches(request: Request, db: Session = Depends(get_db)):
    token = get_bearer_token(request)
    _ = current_user(db, token)
    return {"matches": []}

# ---------------- 靜態檔案與首頁 ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

FALLBACK_HTML = """<!doctype html>
<html lang="zh-Hant"><meta charset="utf-8">
<title>服務已啟動</title>
<style>body{font-family:system-ui,-apple-system,'Segoe UI',Roboto,'Noto Sans TC',Arial;margin:40px;line-height:1.6}</style>
<h1>🚀 服務已啟動</h1>
<p>找不到 <code>static/index.html</code>，顯示預設頁。</p>
<ul>
  <li><a href="/docs" target="_blank">Swagger API 文件</a></li>
  <li><a href="/health" target="_blank">健康檢查</a></li>
  <li><a href="/static/index.html" target="_blank">打開前端頁</a></li>
</ul>
</html>"""

@app.get("/", response_class=HTMLResponse)
def index():
    fp = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(fp):
        return FileResponse(fp)
    return HTMLResponse(FALLBACK_HTML)

# ---------------- 啟動前：建表 + 輕量遷移 ----------------
create_db()
migrate_sqlite_users_table()
