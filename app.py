<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>遊戲配對網</title>
  <style>
    body{font-family:system-ui,-apple-system,"Noto Sans TC","Segoe UI",Roboto,Arial;background:#f7f7fb;margin:0;color:#222}
    .wrap{max-width:1200px;margin:28px auto;padding:0 16px}
    h1{font-size:28px;margin:0 0 8px}
    .sub{color:#666;margin-bottom:16px}
    .card{background:#fff;border-radius:14px;box-shadow:0 6px 20px rgba(0,0,0,.08);padding:18px 20px;margin:14px 0}
    .row{display:flex;gap:16px;flex-wrap:wrap}
    .col{flex:1 1 420px;min-width:320px}
    label{display:block;font-size:14px;margin:8px 0 6px;color:#444}
    input,select,textarea{width:100%;box-sizing:border-box;border:1px solid #d7d7e0;border-radius:10px;padding:10px 12px;font-size:15px;outline:none}
    input:focus,textarea:focus{border-color:#5b8def;box-shadow:0 0 0 3px rgba(91,141,239,.15)}
    .btn{display:inline-block;border:0;border-radius:10px;padding:10px 16px;background:#2962ff;color:#fff;cursor:pointer;font-weight:600}
    .btn.green{background:#0aa56b}.btn.gray{background:#e5e7eb;color:#111}.btn.red{background:#e63b3b}
    .status{font-weight:700;padding:5px 9px;border-radius:999px}.ok{background:#e6f7ef;color:#117a47}.bad{background:#fde8e8;color:#9b2226}
    .muted{color:#666;font-size:13px}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
    .tag{display:inline-block;background:#eef2ff;color:#1f3a8a;border-radius:999px;padding:2px 8px;margin:2px 4px 0 0;font-size:12px}
    .small{font-size:12px;color:#555}
    /* Chat panel */
    .chat{position:sticky;top:12px;height:520px;display:flex;flex-direction:column}
    .msgs{flex:1;border:1px solid #d7d7e0;border-radius:10px;padding:10px;overflow:auto;background:#fff}
    .msg{margin:6px 0;padding:8px 10px;border-radius:10px;max-width:80%}
    .me{background:#e6f7ef;margin-left:auto}.other{background:#eef2ff;margin-right:auto}
    .chatbar{display:flex;gap:8px;margin-top:8px}
  </style>
</head>
<body>
<div class="wrap">
  <h1>遊戲配對網</h1>
  <div class="sub">註冊需同意 <a href="/disclaimer" target="_blank">免責聲明</a> 與 <a href="/privacy" target="_blank">隱私權政策</a>。</div>

  <div class="row">
    <div class="card col">
      <h3>建立帳號</h3>
      <label>用戶名</label><input id="su_username" autocomplete="username">
      <label>密碼（≥6）</label><input id="su_password" type="password" autocomplete="new-password">
      <label>電子信箱（可空白）</label><input id="su_email" type="email">
      <label><input type="checkbox" id="su_consent"> 我已詳讀並同意 <a href="/disclaimer" target="_blank">免責聲明</a> 與 <a href="/privacy" target="_blank">隱私權政策</a></label>
      <div style="margin-top:10px"><button class="btn" id="btnSignup">註冊</button></div>
      <div class="muted" id="su_msg" style="margin-top:8px"></div>
    </div>

    <div class="card col">
      <h3>登入</h3>
      <label>用戶名</label><input id="li_username" autocomplete="username">
      <label>密碼</label><input id="li_password" type="password" autocomplete="current-password">
      <div style="margin-top:10px"><button class="btn green" id="btnLogin">登入</button></div>
      <div class="muted" id="li_msg" style="margin-top:8px">登入成功後將自動回報定位。</div>
      <div style="margin-top:8px">登入狀態：<span id="authStatus" class="status bad">未登入</span> <button class="btn gray" id="btnLogout">登出</button></div>
    </div>
  </div>

  <div class="card">
    <h3>我的資料</h3>
    <div class="row">
      <div class="col"><label>暱稱</label><input id="pf_name"></div>
      <div class="col">
        <label>性別</label>
        <select id="pf_gender">
          <option value="">（未設定）</option>
          <option value="男">男</option>
          <option value="女">女</option>
          <option value="其他">其他</option>
        </select>
      </div>
      <div class="col"><label>生日</label><input id="pf_bday" type="date"></div>
      <div class="col"><label>城市</label><input id="pf_city" placeholder="例：台北市"></div>
    </div>
    <label>自我介紹</label><textarea id="pf_bio" rows="3"></textarea>
    <label>興趣標籤（以逗號分隔；英數自動大寫、中文維持）</label>
    <input id="pf_interests" placeholder="例如：LOL, 漫畫, APEX">
    <div style="margin-top:10px;display:flex;gap:8px">
      <button class="btn" id="btnSaveProfile">儲存</button>
      <button class="btn gray" id="btnRefreshMe">重新載入</button>
    </div>
    <div class="muted" id="me_msg" style="margin-top:8px"></div>
  </div>

  <div class="row">
    <div class="card col">
      <h3>附近的人（預設 100 公里）</h3>
      <div class="small">已嘗試自動回報定位；若未授權定位，請允許或手動變更城市。</div>
      <div style="margin:10px 0">
        <label>搜尋半徑（公里）</label>
        <input id="radius" value="100">
        <button class="btn" id="btnNearby">搜尋附近</button>
      </div>
      <div id="nearbyList" class="grid"></div>
    </div>

    <div class="card col chat">
      <h3>我的配對 & 聊天</h3>
      <div id="matchList" class="grid" style="margin-bottom:8px"></div>
      <div id="chatWrap" style="display:none">
        <div class="small" id="chatTitle">與誰聊天</div>
        <div id="msgs" class="msgs"></div>
        <div class="chatbar">
          <input id="msgInput" placeholder="輸入訊息..." />
          <button class="btn" id="btnSend">送出</button>
        </div>
      </div>
      <div class="muted" id="chatMsg"></div>
    </div>
  </div>

  <div class="card">
    <div><a href="/disclaimer" target="_blank">免責聲明</a> ｜ <a href="/privacy" target="_blank">隱私權政策</a> ｜ <a href="/docs" target="_blank">API 文件</a></div>
  </div>
</div>

<script>
  // ------------ API base（隱藏，使用同源） ------------
  const API_BASE = window.location.origin;
  const WS_BASE = API_BASE.replace(/^http/, 'ws');

  // ------------ Auth Token ------------
  const tokenKey = 'authToken';
  const setToken = t => { localStorage.setItem(tokenKey, t); refreshAuthStatus(); };
  const getToken = () => localStorage.getItem(tokenKey) || '';
  const clearToken = () => { localStorage.removeItem(tokenKey); refreshAuthStatus(); };
  const authHeader = () => getToken() ? { 'Authorization': `Bearer ${getToken()}` } : {};

  function refreshAuthStatus(){
    const el = document.getElementById('authStatus');
    if(getToken()){
      el.textContent = '已登入'; el.className = 'status ok';
    }else{
      el.textContent = '未登入'; el.className = 'status bad';
    }
  }
  refreshAuthStatus();

  // ------------ 註冊 / 登入 / 登出 ------------
  document.getElementById('btnSignup').onclick = async () => {
    const username = document.getElementById('su_username').value.trim();
    const password = document.getElementById('su_password').value;
    const email = document.getElementById('su_email').value.trim();
    const consent = document.getElementById('su_consent').checked;
    const msg = document.getElementById('su_msg');
    msg.textContent = '送出中…';
    try{
      const r = await fetch(`${API_BASE}/auth/signup`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({username, password, email, consent_agreed: consent})
      });
      const data = await r.json();
      if(!r.ok) throw new Error(data.detail || r.statusText);
      msg.textContent = '註冊成功，請登入。';
    }catch(e){ msg.textContent = `失敗：${e.message || e}`; }
  };

  document.getElementById('btnLogin').onclick = async () => {
    const username = document.getElementById('li_username').value.trim();
    const password = document.getElementById('li_password').value;
    const el = document.getElementById('li_msg');
    el.textContent = '登入中…';
    try{
      const r = await fetch(`${API_BASE}/auth/login`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({username, password})
      });
      const data = await r.json();
      if(!r.ok) throw new Error(data.detail || r.statusText);
      setToken(data.access_token);
      el.textContent = '登入成功！將嘗試自動回報定位…';
      tryAutoGeoReport();
      // 載入我的資料、配對
      loadMe(); loadMatches();
    }catch(e){ el.textContent = `登入失敗：${e.message || e}`; }
  };

  document.getElementById('btnLogout').onclick = () => {
    clearToken();
    document.getElementById('nearbyList').innerHTML = '';
    document.getElementById('matchList').innerHTML = '';
    closeChat();
  };

  // ------------ 我 / 儲存 ------------
  async function loadMe(){
    const msg = document.getElementById('me_msg');
    msg.textContent = '載入中…';
    try{
      const r = await fetch(`${API_BASE}/me`, { headers: {...authHeader()} });
      const data = await r.json();
      if(!r.ok) throw new Error(data.detail || r.statusText);
      document.getElementById('pf_name').value = data.display_name || '';
      document.getElementById('pf_gender').value = data.gender || '';
      document.getElementById('pf_bday').value = data.birthday || '';
      document.getElementById('pf_city').value = data.city || '';
      document.getElementById('pf_bio').value = data.bio || '';
      document.getElementById('pf_interests').value = (data.interests||[]).join(', ');
      msg.textContent = '已載入。';
    }catch(e){ msg.textContent = `錯誤：${e.message || e}`; }
  }
  document.getElementById('btnRefreshMe').onclick = loadMe;

  document.getElementById('btnSaveProfile').onclick = async () => {
    const msg = document.getElementById('me_msg');
    msg.textContent = '儲存中…';
    const payload = {
      display_name: val('pf_name'),
      gender: val('pf_gender'),
      birthday: val('pf_bday') || null,
      city: val('pf_city'),
      bio: val('pf_bio'),
      interests: val('pf_interests').split(',').map(s => s.trim()).filter(Boolean)
    };
    try{
      const r = await fetch(`${API_BASE}/me`, {
        method:'PATCH', headers:{'Content-Type':'application/json', ...authHeader()},
        body: JSON.stringify(payload)
      });
      const data = await r.json();
      if(!r.ok) throw new Error(data.detail || r.statusText);
      msg.textContent = '已儲存。';
      loadMatches(); // 更新卡片資訊
    }catch(e){ msg.textContent = `錯誤：${e.message || e}`; }
  };

  const val = id => document.getElementById(id).value.trim();

  // ------------ 定位 / 附近 ------------
  async function tryAutoGeoReport(){
    if(!getToken()) return;
    if(!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(async (pos)=>{
      try{
        await fetch(`${API_BASE}/me/location`, {
          method:'POST', headers:{'Content-Type':'application/json', ...authHeader()},
          body: JSON.stringify({lat: pos.coords.latitude, lng: pos.coords.longitude})
        });
      }catch(_){}
    }, (_)=>{}, {enableHighAccuracy:false, timeout:5000, maximumAge:60000});
  }

  document.getElementById('btnNearby').onclick = searchNearby;

  async function searchNearby(){
    const list = document.getElementById('nearbyList');
    list.innerHTML = '搜尋中…';
    if(!getToken()){ list.innerHTML = '<div class="muted">請先登入</div>'; return; }

    let lat=null, lng=null;
    try{
      const me = await (await fetch(`${API_BASE}/me`, { headers:{...authHeader()} })).json();
      lat = me.lat; lng = me.lng;
      if((lat==null || lng==null) && navigator.geolocation){
        await new Promise(resolve=>{
          navigator.geolocation.getCurrentPosition((pos)=>{ lat=pos.coords.latitude; lng=pos.coords.longitude; resolve(); }, ()=>resolve(), {timeout:4000});
        });
      }
    }catch(_){}

    if(lat==null || lng==null){ list.innerHTML = '<div class="muted">尚無定位，請允許定位或於個人資料填寫城市。</div>'; return; }

    const radius = parseFloat(document.getElementById('radius').value || '100');
    try{
      const r = await fetch(`${API_BASE}/nearby`, {
        method:'POST', headers:{'Content-Type':'application/json', ...authHeader()},
        body: JSON.stringify({lat, lng, radius_km: radius})
      });
      const data = await r.json();
      if(!r.ok) throw new Error(data.detail || r.statusText);
      renderNearby(data.users || []);
    }catch(e){ list.innerHTML = `<div class="muted">錯誤：${e.message || e}</div>`; }
  }

  function renderNearby(users){
    const list = document.getElementById('nearbyList');
    if(!users.length){ list.innerHTML = '<div class="muted">目前查無附近用戶</div>'; return; }
    list.innerHTML = '';
    for(const u of users){
      const age = u.birthday ? calcAge(u.birthday) : '';
      const g = zhGender(u.gender);
      const tags = (u.interests||[]).map(t=>`<span class="tag">${escapeHtml(t)}</span>`).join(' ');
      const bio = u.bio ? escapeHtml(u.bio) : '（尚無自我介紹）';
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px">
          <div>
            <div style="font-weight:700">${escapeHtml(u.display_name || u.username)}</div>
            <div class="small">${[g, age?`${age} 歲`:null, u.city||null].filter(Boolean).join("・")}　距離：約 ${u.distance_km} km</div>
          </div>
          <button class="btn" data-username="${u.username}">配對（喜歡）</button>
        </div>
        <div style="margin-top:6px">${bio}</div>
        <div style="margin-top:6px">${tags}</div>
      `;
      card.querySelector('button').onclick = () => likeUser(u.username);
      list.appendChild(card);
    }
  }

  async function likeUser(username){
    if(!getToken()) return alert('請先登入');
    try{
      const r = await fetch(`${API_BASE}/like`, {
        method:'POST', headers:{'Content-Type':'application/json', ...authHeader()},
        body: JSON.stringify({target: username})
      });
      const data = await r.json();
      if(!r.ok) throw new Error(data.detail || r.statusText);
      if(data.matched){
        alert('互相喜歡！已成為配對，右側可開始聊天。');
        loadMatches();
      }else{
        alert('已送出喜歡，等待對方也喜歡你。');
      }
    }catch(e){ alert(`失敗：${e.message || e}`); }
  }

  const zhGender = g => {
    if(!g) return '未設定';
    const s = String(g).toLowerCase();
    if(s==='male' || s==='m' || s==='男') return '男';
    if(s==='female' || s==='f' || s==='女') return '女';
    return '其他';
  };

  const calcAge = iso => {
    try{
      const d = new Date(iso);
      const today = new Date();
      let age = today.getFullYear() - d.getFullYear();
      const m = today.getMonth() - d.getMonth();
      if (m < 0 || (m === 0 && today.getDate() < d.getDate())) age--;
      return age;
    }catch(_){ return ''; }
  };

  const escapeHtml = s => s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

  // ------------ 我的配對 & 聊天 ------------
  let currentPeer = null;
  let ws = null;

  async function loadMatches(){
    if(!getToken()) return;
    const box = document.getElementById('matchList');
    box.innerHTML = '載入中…';
    try{
      const r = await fetch(`${API_BASE}/matches`, { headers:{...authHeader()} });
      const data = await r.json();
      if(!r.ok) throw new Error(data.detail || r.statusText);

      const arr = data.matches || [];
      if(!arr.length){ box.innerHTML = '<div class="muted">尚無配對</div>'; closeChat(); return; }
      box.innerHTML = '';
      for(const u of arr){
        const g = zhGender(u.gender);
        const bio = u.bio ? escapeHtml(u.bio) : '（尚無自我介紹）';
        const card = document.createElement('div');
        card.className = 'card';
        card.style.cursor = 'pointer';
        card.innerHTML = `
          <div style="font-weight:700">${escapeHtml(u.display_name || u.username)}</div>
          <div class="small">${[g, u.city||null].filter(Boolean).join("・")}</div>
          <div style="margin-top:6px">${bio}</div>
        `;
        card.onclick = ()=> openChat(u.username, u.display_name || u.username);
        box.appendChild(card);
      }
    }catch(e){ box.innerHTML = `<div class="muted">錯誤：${e.message || e}</div>`; }
  }

  function closeChat(){
    document.getElementById('chatWrap').style.display = 'none';
    document.getElementById('msgs').innerHTML = '';
    if(ws){ try{ ws.close(); }catch(_){ } ws = null; }
    currentPeer = null;
  }

  async function openChat(peerUsername, displayName){
    currentPeer = peerUsername;
    document.getElementById('chatWrap').style.display = '';
    document.getElementById('chatTitle').textContent = `與「${displayName}」聊天`;
    document.getElementById('msgs').innerHTML = '載入訊息…';

    // 歷史訊息
    try{
      const r = await fetch(`${API_BASE}/messages/${encodeURIComponent(peerUsername)}`, { headers:{...authHeader()} });
      const data = await r.json();
      if(!r.ok) throw new Error(data.detail || r.statusText);
      renderMsgs(data.messages || []);
    }catch(e){
      document.getElementById('msgs').innerHTML = `<div class="muted">錯誤：${e.message || e}</div>`;
    }

    // 建立 WebSocket
    if(ws){ try{ ws.close(); }catch(_){ } }
    try{
      ws = new WebSocket(`${WS_BASE}/ws?peer=${encodeURIComponent(peerUsername)}&token=${encodeURIComponent(getToken())}`);
      ws.onmessage = (ev)=>{
        const msg = JSON.parse(ev.data);
        appendMsg(msg.sender, msg.content, msg.created_at);
      };
      ws.onclose = ()=>{};
    }catch(_){
      // 無法建立 WS 就用 HTTP fallback（送出時走 POST）
    }
  }

  function renderMsgs(arr){
    const box = document.getElementById('msgs'); box.innerHTML = '';
    for(const m of arr){ appendMsg(m.sender, m.content, m.created_at); }
    box.scrollTop = box.scrollHeight;
  }

  function appendMsg(sender, content, ts){
    const me = document.getElementById('li_username').value.trim() || '(你)';
    const div = document.createElement('div');
    const isMe = (sender === me) || (sender === getCachedUsername());
    div.className = `msg ${isMe?'me':'other'}`;
    const time = new Date(ts || Date.now()).toLocaleTimeString();
    div.innerHTML = `<div class="small" style="opacity:.7">${sender}・${time}</div><div>${escapeHtml(content)}</div>`;
    const box = document.getElementById('msgs');
    box.appendChild(div); box.scrollTop = box.scrollHeight;
  }

  function getCachedUsername(){
    // 粗略從 /me 取得一次後快取（簡化：從表單取不到時）
    return window.__cachedUser || '';
  }

  document.getElementById('btnSend').onclick = async ()=>{
    const input = document.getElementById('msgInput');
    const text = input.value.trim();
    if(!text || !currentPeer) return;
    input.value = '';

    // 優先走 WebSocket
    if(ws && ws.readyState === 1){
      ws.send(JSON.stringify({content: text}));
      return;
    }
    // Fallback：HTTP
    try{
      const r = await fetch(`${API_BASE}/messages/${encodeURIComponent(currentPeer)}`, {
        method:'POST', headers:{'Content-Type':'application/json', ...authHeader()},
        body: JSON.stringify({content: text})
      });
      const data = await r.json();
      if(!r.ok) throw new Error(data.detail || r.statusText);
      // 立即顯示（時間由後端決定，但這裡先本地顯示）
      appendMsg(getCachedUsername() || '我', text, new Date().toISOString());
    }catch(e){
      document.getElementById('chatMsg').textContent = `送出失敗：${e.message || e}`;
      setTimeout(()=>document.getElementById('chatMsg').textContent='', 2000);
    }
  };

  // ------------ 工具 ------------
  // 初次載入
  refreshAuthStatus();
  if(getToken()){ loadMe(); tryAutoGeoReport(); loadMatches(); }

</script>
</body>
</html>
