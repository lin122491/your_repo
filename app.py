"""
交友 APP 後端（最新）— FastAPI + SQLite（單檔），支援 Render 部署與中文 Swagger
功能：
- 註冊/登入（JWT）
- 個資/定位
- 附近的人（哈弗辛距離）
- 喜歡/配對
- 提供靜態前端（/web），可直接部署在 Render（持久化 SQLite）

使用：
1) pip install fastapi uvicorn "sqlalchemy>=2" passlib[bcrypt] pyjwt pydantic email-validator python-multipart
2) 將前端 index.html 放到 ./static/index.html
3) 本機啟動：uvicorn app:app --reload
4) Render：見 requirements.txt + Procfile 或 render.yaml；會自動讀 JWT_SECRET / DATABASE_URL
"""
from __future__ import annotations
import os
import math
import time
import datetime as dt
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, EmailStr
import sqlalchemy as sa
from sqlalchemy.orm import declarative_base, relationship, Session, sessionmaker
from passlib.hash import bcrypt
import jwt

# -------------------- 設定 --------------------
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")  # 正式請改成環境變數
JWT_ALG = "HS256"
ACCESS_TOKEN_TTL_MIN = 60 * 24 * 7  # 7 天

# SQLite 連線字串：本機預設 ./dating.db；在 Render 會設為 sqlite:////data/dating.db
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dating.db")

# 若是 SQLite，為了讓多執行緒可以使用要加 check_same_thread=False
engine_kwargs = {"echo": False, "future": True}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
engine = sa.create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

# -------------------- 資料表 --------------------
class User(Base):
    __tablename__ = "users"
    id = sa.Column(sa.Integer, primary_key=True)
    username = sa.Column(sa.String(50), unique=True, nullable=False, index=True)
    email = sa.Column(sa.String(120), unique=True, nullable=True)
    password_hash = sa.Column(sa.String(255), nullable=False)

    bio = sa.Column(sa.String(500), default="")
    gender = sa.Column(sa.String(20), default="")
    birthdate = sa.Column(sa.Date, nullable=True)

    last_lat = sa.Column(sa.Float, nullable=True)
    last_lng = sa.Column(sa.Float, nullable=True)
    last_seen = sa.Column(sa.DateTime, default=sa.func.now())

    likes_sent = relationship("Like", back_populates="liker", foreign_keys="Like.liker_id")
    likes_received = relationship("Like", back_populates="liked", foreign_keys="Like.liked_id")

class Like(Base):
    __tablename__ = "likes"
    id = sa.Column(sa.Integer, primary_key=True)
    liker_id = sa.Column(sa.Integer, sa.ForeignKey("users.id"), index=True)
    liked_id = sa.Column(sa.Integer, sa.ForeignKey("users.id"), index=True)
    created_at = sa.Column(sa.DateTime, default=sa.func.now())

    liker = relationship("User", foreign_keys=[liker_id], back_populates="likes_sent")
    liked = relationship("User", foreign_keys=[liked_id], back_populates="likes_received")

    __table_args__ = (
        sa.UniqueConstraint("liker_id", "liked_id", name="uq_like_once"),
    )

Base.metadata.create_all(engine)

# -------------------- Schema（Pydantic v2） --------------------
class SignupIn(BaseModel):
    username: str = Field(min_length=3, max_length=30, title="用戶名", description="登入用的唯一用戶名，3~30 字")
    password: str = Field(min_length=6, max_length=128, title="密碼", description="6~128 字")
    email: Optional[EmailStr] = Field(default=None, title="電子信箱", description="可選；若提供會做格式驗證")

class LoginIn(BaseModel):
    username: str = Field(title="用戶名")
    password: str = Field(title="密碼")

class TokenOut(BaseModel):
    access_token: str = Field(title="存取權杖")
    token_type: str = Field(default="bearer", title="權杖類型")
    expires_in: int = Field(title="秒數有效期")

class ProfileOut(BaseModel):
    id: int
    username: str
    bio: str = Field(default="", title="自我介紹")
    gender: str = Field(default="", title="性別")
    birthdate: Optional[dt.date] = Field(default=None, title="生日")
    last_lat: Optional[float] = Field(default=None, title="最近緯度")
    last_lng: Optional[float] = Field(default=None, title="最近經度")
    last_seen: Optional[dt.datetime] = Field(default=None, title="最後上線時間 (UTC)")

    class Config:
        from_attributes = True  # v2 寫法

class ProfileUpdateIn(BaseModel):
    bio: Optional[str] = Field(default=None, max_length=500, title="自我介紹")
    gender: Optional[str] = Field(default=None, max_length=20, title="性別")
    birthdate: Optional[dt.date] = Field(default=None, title="生日")

class LocationIn(BaseModel):
    lat: float = Field(ge=-90, le=90, title="緯度")
    lng: float = Field(ge=-180, le=180, title="經度")

class NearbyQuery(BaseModel):
    lat: float = Field(ge=-90, le=90, title="查詢中心緯度")
    lng: float = Field(ge=-180, le=180, title="查詢中心經度")
    radius_km: float = Field(5.0, ge=0.1, le=100.0, title="半徑（公里）")
    active_within_min: int = Field(60, ge=1, le=60*24, title="最近上線分鐘內")

class NearbyUserOut(BaseModel):
    id: int
    username: str
    bio: str
    distance_km: float = Field(title="距離（公里）")

class LikeOut(BaseModel):
    you_liked: int = Field(title="你喜歡的用戶 ID")
    they_liked_back: bool = Field(title="對方是否也喜歡你")
    is_match: bool = Field(title="是否配對成功")

# -------------------- 認證 --------------------
security = HTTPBearer()
app = FastAPI(
    title="交友 APP 後端（中文）",
    version="0.3.0",
    description=(
        "這是一個簡易的交友 APP 後端原型。"
        "➡️ 先在 /auth/signup 建立帳號並取得 Token，右上角 Authorize 輸入 Bearer Token。"
        "➡️ 用 /me 與 /me/location 維護個人資料與定位。"
        "➡️ 用 /nearby 搜尋附近的人、/like/{target_id} 送出喜歡、/matches 查看配對。"
    ),
    swagger_ui_parameters={"defaultModelsExpandDepth": 0}
)

# CORS（前端不同來源時可呼叫 API）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 掛載前端（/web）
if os.path.isdir("static"):
    app.mount("/web", StaticFiles(directory="static", html=True), name="web")


def create_token(user_id: int) -> str:
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + ACCESS_TOKEN_TTL_MIN * 60,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)) -> User:
    token = creds.credentials
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        user_id = int(data["sub"])
    except Exception:
        raise HTTPException(status_code=401, detail="權杖無效或已過期")
    with SessionLocal() as db:
        user = db.get(User, user_id)
        if not user:
            raise HTTPException(status_code=401, detail="找不到使用者")
        return user

# -------------------- 工具 --------------------
EARTH_R_KM = 6371.0088

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_R_KM * c

# -------------------- 路由 --------------------
@app.post(
    "/auth/signup",
    response_model=TokenOut,
    summary="註冊並取得權杖",
    description="建立新帳號，回傳 JWT 權杖（預設 7 天有效）。用於後續認證。",
)
def signup(payload: SignupIn):
    with SessionLocal() as db:
        if db.scalar(sa.select(sa.func.count()).select_from(User).where(User.username == payload.username)):
            raise HTTPException(400, "用戶名已被使用")
        user = User(
            username=payload.username,
            email=payload.email,
            password_hash=bcrypt.hash(payload.password),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        token = create_token(user.id)
        return TokenOut(access_token=token, expires_in=ACCESS_TOKEN_TTL_MIN*60)

@app.post(
    "/auth/login",
    response_model=TokenOut,
    summary="登入並取得權杖",
    description="輸入用戶名與密碼，成功後回傳 JWT 權杖。",
)
def login(payload: LoginIn):
    with SessionLocal() as db:
        user = db.scalar(sa.select(User).where(User.username == payload.username))
        if not user or not bcrypt.verify(payload.password, user.password_hash):
            raise HTTPException(401, "帳號或密碼錯誤")
        token = create_token(user.id)
        return TokenOut(access_token=token, expires_in=ACCESS_TOKEN_TTL_MIN*60)

@app.get(
    "/me",
    response_model=ProfileOut,
    summary="取得我的個人資料",
    description="需要 Bearer Token。回傳目前帳號的公開資料與最後定位資訊。",
)
def me(current: User = Depends(get_current_user)):
    return current

@app.patch(
    "/me",
    response_model=ProfileOut,
    summary="更新我的個人資料",
    description="可更新自我介紹、性別、生日等欄位。",
)
def update_me(update: ProfileUpdateIn, current: User = Depends(get_current_user)):
    with SessionLocal() as db:
        user = db.get(User, current.id)
        if update.bio is not None:
            user.bio = update.bio
        if update.gender is not None:
            user.gender = update.gender
        if update.birthdate is not None:
            user.birthdate = update.birthdate
        db.commit()
        db.refresh(user)
        return user

@app.post(
    "/me/location",
    summary="回報我的定位",
    description="上報目前的經緯度，系統會記錄最後上線時間（UTC）。",
)
def update_location(loc: LocationIn, current: User = Depends(get_current_user)):
    with SessionLocal() as db:
        user = db.get(User, current.id)
        user.last_lat = loc.lat
        user.last_lng = loc.lng
        user.last_seen = dt.datetime.utcnow()
        db.commit()
    return {"ok": True, "訊息": "定位已更新"}

@app.post(
    "/nearby",
    response_model=List[NearbyUserOut],
    summary="搜尋附近的人",
    description="以指定的中心點與半徑（公里）搜尋，僅顯示最近 X 分鐘內上線的用戶。結果依距離近到遠排序。",
)
def nearby(q: NearbyQuery, current: User = Depends(get_current_user)):
    cutoff = dt.datetime.utcnow() - dt.timedelta(minutes=q.active_within_min)
    with SessionLocal() as db:
        # 先用外框粗篩，加速查詢
        lat_margin = (q.radius_km / EARTH_R_KM) * (180 / math.pi)
        lng_margin = lat_margin / max(math.cos(math.radians(q.lat)), 0.01)
        candidates = db.scalars(
            sa.select(User).where(
                User.id != current.id,
                User.last_lat.isnot(None),
                User.last_lng.isnot(None),
                User.last_seen >= cutoff,
                User.last_lat.between(q.lat - lat_margin, q.lat + lat_margin),
                User.last_lng.between(q.lng - lng_margin, q.lng + lng_margin),
            )
        ).all()

    results = []
    for u in candidates:
        dist = haversine_km(q.lat, q.lng, u.last_lat, u.last_lng)
        if dist <= q.radius_km:
            results.append(NearbyUserOut(id=u.id, username=u.username, bio=u.bio or "", distance_km=round(dist, 2)))
    results.sort(key=lambda x: x.distance_km)
    return results

@app.post(
    "/like/{target_id}",
    response_model=LikeOut,
    summary="送出喜歡",
    description="對指定用戶送出喜歡（不重複）。若對方也曾喜歡你，is_match 會為 true。",
)
def like_user(target_id: int, current: User = Depends(get_current_user)):
    if target_id == current.id:
        raise HTTPException(400, "不能喜歡自己")
    with SessionLocal() as db:
        me = db.get(User, current.id)
        target = db.get(User, target_id)
        if not target:
            raise HTTPException(404, "找不到目標用戶")
        # 建立 like（若不存在）
        exists = db.scalar(sa.select(Like).where(Like.liker_id == me.id, Like.liked_id == target.id))
        if not exists:
            db.add(Like(liker_id=me.id, liked_id=target.id))
            db.commit()
        # 檢查對方是否也喜歡你
        they_like_back = db.scalar(sa.select(Like).where(Like.liker_id == target.id, Like.liked_id == me.id)) is not None
        return LikeOut(you_liked=target.id, they_liked_back=they_like_back, is_match=bool(they_like_back))

@app.get(
    "/matches",
    response_model=List[ProfileOut],
    summary="查看我的配對",
    description="列出和你互相喜歡的用戶（配對）。",
)
def matches(current: User = Depends(get_current_user)):
    with SessionLocal() as db:
        liked_ids = sa.select(Like.liked_id).where(Like.liker_id == current.id)
        back = db.scalars(sa.select(User).where(
            User.id.in_(liked_ids),
            sa.exists(sa.select(Like.id).where(Like.liker_id == User.id, Like.liked_id == current.id))
        )).all()
        return back

# -------------------- 健康檢查/首頁 --------------------
@app.get("/health", summary="服務狀態")
def health():
    return {"service": "dating-prototype", "ok": True}

@app.get("/", include_in_schema=False)
def root():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"service": "dating-prototype", "ok": True}

# 允許本檔被 python 直接啟動（本機測試用）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
