#!/usr/bin/env python3
"""diary.py — the diary body's serve (DIARY_BODY.md): two writers, one
page, one symbol per tick. No turns, no end marks, no breath, no hush:
silence is a symbol the body chooses. Reuses the organism's organs
(scripts/organism.py: the model and its memory, the face lesson,
cortisol, mood, doses, nights, save, reset) and replaces the turn
protocol with a clock.

  python3 scripts/diary.py data/organism_diary_0p5b.pt data/tok_char.json --dev mps \\
      --temp 0.05 --store-read-beta 1.0 --store-boost 16 --store-boost-min 0.15 \\
      --live-lr 1e-6 --store-decay 0.9 --save data/organism_diary_0p5b.pt --port 8018

Endpoints: GET / (the page), GET /state?since=N, GET /pulse;
POST /type {"text"}, /face {"expr"}, /sleep, /save, /reset.
"""
import collections
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch  # noqa: E402
import scripts.organism as O  # noqa: E402


class Diary(O.Organism):
    def __init__(self, a):
        super().__init__(a)
        self.queue = collections.deque()          # your symbols waiting for ticks
        self.page = []                            # every shown symbol: (text, who, your face, its face)
        self.level = 0                            # the face level felt now
        self.credit = collections.deque(maxlen=64)  # the mouth's recent symbols: [id, credit]
        self.recent_user = collections.deque(maxlen=240)
        self.last = {}
        self.ticks = 0
        self.awake_ticks = True
        self.lock = threading.RLock()
        self.period = float(getattr(a, "diary_period", 0.5))
        self.who0 = torch.tensor([[0]], device=self.dev)
        self.who1 = torch.tensor([[1]], device=self.dev)   # the mouth's noise
        self.who2 = torch.tensor([[2]], device=self.dev)   # the mouth from memory (part of the thought)
        self._bans = [i for i in range(11) if i != self.sil]   # it may choose silence, never a mark
        self.m.reset_bag(self.st) if hasattr(self.m, "reset_bag") else None
        threading.Thread(target=self._loop, daemon=True).start()

    # ---- the clock ----
    def _loop(self):
        while True:
            t0 = time.time()
            if self.awake_ticks:
                try:
                    with self.lock:
                        self.tick()
                except Exception as e:  # never let one tick stop the clock
                    self.last = {"error": str(e)[:200], "tick": self.ticks}
            time.sleep(max(0.0, self.period - (time.time() - t0)))

    def tick(self):
        """one tick: your symbol (or silence) enters as the ear; its symbol
        (or silence) is chosen and enters as the mouth"""
        self._decay_mood(); self._decay_cort()
        u = self.queue.popleft() if self.queue else self.sil
        face = float(self.face_now)
        lvl = max(-6, min(6, int(face)))
        felt = 0
        if lvl != self.level:                      # only a change of face is felt
            felt, self.level = lvl, lvl
        pl = None
        if felt:
            pl = torch.zeros(1, 1, dtype=torch.long, device=self.dev)
            pl[0, 0] = min(felt, 2) if felt > 0 else 2 - max(felt, -2)   # 1..4 = +1 +2 -1 -2
        aff = self._aff(1)
        with torch.no_grad():
            lg, self.st, _ = self.m(torch.tensor([[u]], device=self.dev), self.st,
                                    press_levels=pl, affect=aff, who=self.who0)
        its_face = self._face_lesson(face)
        # the mouth chooses: its own logits, memory's vote inside them
        v = lg[0, -1].float().clone()
        v[self._bans] = float("-inf")
        _lv = getattr(self.m, "_last_votes", None)
        own_ent = float(getattr(self.m, "_last_own_ent", 0.0) or 0.0)
        mem_max = float(_lv[0][0]) if _lv and _lv[0] else 0.0
        if _lv and _lv[1] and mem_max >= float(getattr(self.a, "store_boost_min", 0.0) or 0.0) and own_ent > 0.9:
            v[int(_lv[1][0])] = v[int(_lv[1][0])] + float(getattr(self.a, "sure_mem", 6.0))
        if self.cortisol > 0:                      # speaking costs: stress favours silence
            v[self.sil] = v[self.sil] + float(getattr(self.a, "cort_k", 0.5)) * self.cortisol
        p1 = torch.softmax(v, -1)
        ent = float(-(p1 * (p1 + 1e-9).log()).sum() / math.log(max(2, p1.numel())))
        pr = torch.softmax(v / max(0.02, float(self.a.temp)), -1).cpu()
        nxt = int(torch.multinomial(pr, 1, generator=self.gen))
        backed = bool(_lv and _lv[1] and int(_lv[1][0]) == nxt and nxt != self.sil
                      and mem_max >= float(getattr(self.a, "store_boost_min", 0.0) or 0.0))
        with torch.no_grad():
            _, self.st, _ = self.m(torch.tensor([[nxt]], device=self.dev), self.st, affect=aff,
                                    who=(self.who2 if backed else self.who1))
        if nxt != self.sil:
            # speaking costs: each symbol adds a little stress (half-life 120 s),
            # stress favours silence and weighs a little on mood
            self.cortisol += float(getattr(self.a, "cort_rate", 0.15)) * 0.1
            self.mood = max(-6.0, min(6.0, self.mood - 0.01 * self.cortisol))
        # the day's record (what was written, both hands; silences are not rehearsed)
        if u != self.sil:
            self._who_now = 0; self.day_buf.append(u); self._rec_face(1); self.recent_user.append(u)
        if nxt != self.sil:
            self._who_now = 1; self.day_buf.append(nxt); self._rec_face(1); self._who_now = 0
            self.credit.append([nxt, 0.0])
        if felt:
            for k_, item in enumerate(reversed(list(self.credit)[-6:])):
                item[1] += felt * (0.7 ** k_)
            self.mood = max(-6.0, min(6.0, self.mood + 0.5 * felt))
            self._dose_if_due()
        self.ticks += 1
        self.page.append(((self.tok.decode([u]) if u != self.sil else ""), 0, round(face, 2),
                          None if its_face is None else round(its_face, 2)))
        self.page.append(((self.tok.decode([nxt]) if nxt != self.sil else ""), 1, round(face, 2),
                          None if its_face is None else round(its_face, 2)))
        self.last = {"tick": self.ticks, "you": round(face, 2),
                     "face": None if its_face is None else round(its_face, 2),
                     "mood": round(self.mood, 2), "cort": round(self.cortisol, 2), "ent": round(ent, 2),
                     "mem": [[self.tok.decode([int(i_)]), round(float(v_), 2)] for v_, i_ in zip(*_lv)] if _lv else None,
                     "felt": felt, "said": self.tok.decode([nxt]) if nxt != self.sil else "", "backed": backed}

    def _dose_if_due(self):
        """rolling doses: the mouth's recent symbols carrying credit above 0.5
        are absorbed as its answer to your recent text; at or below -1.5,
        unlearned. The same thresholds as the word body (LIVE_BODY §10)."""
        items = list(self.credit)
        pos = [it for it in items if it[1] > 0.5]
        neg = [it for it in items if it[1] <= -1.5]
        if not pos and not neg:
            return
        q = self.tok.decode(list(self.recent_user)[-80:])
        ans = self.tok.decode([it[0] for it in items[-16:]])
        if pos and ans.strip():
            try:
                self.absorb(q, ans, 1)
            except Exception as e:
                self.last["dose_error"] = str(e)[:120]
        if neg and ans.strip():
            try:
                self._unlearn_reply(ans, 1)
            except Exception as e:
                self.last["dose_error"] = str(e)[:120]
        for it in items:
            it[1] = 0.0

    # ---- your hand ----
    def type_text(self, s):
        n = 0
        for ch in s:
            i = self.tok.token_to_id(ch)
            if i is not None and i >= 11:
                self.queue.append(i); n += 1
        return {"queued": len(self.queue), "took": n}

    def set_face(self, expr):
        try:
            self.face_now = max(-6.0, min(6.0, float(expr)))
        except Exception:
            self.face_now = 0.0
        return {"face": self.face_now}

    def state(self, since=0):
        with self.lock:
            return {"page": self.page[since:], "n": len(self.page), "last": self.last,
                    "queued": len(self.queue), "awake": self.awake_ticks, "period": self.period,
                    "lived": len(self.day_buf)}

    def night(self):
        self.awake_ticks = False
        try:
            with self.lock:
                if not self.session:
                    self.session = [{"diary": True}]     # the lived-day replay wants a day
                res = self.sleep()
                if hasattr(self.m, "reset_bag"):
                    self.m.reset_bag(self.st)
                self.level = 0
                self.credit.clear()
        finally:
            self.awake_ticks = True
        return res


PAGE = """<!doctype html><meta charset=utf-8><title>the diary</title>
<style>
body{margin:0;background:#f5f1e6;color:#222;font:16px/1.6 Georgia,serif}
#pg{white-space:pre-wrap;padding:32px 40px 120px;min-height:70vh;max-width:820px;margin:0 auto}
.u{color:#1a1a1a}.m{color:#7a2e0e}
#bar{position:fixed;left:0;right:0;bottom:0;background:#eae4d3;border-top:1px solid #cbbfa3;padding:10px 40px;font:13px ui-monospace,monospace;display:flex;gap:18px;flex-wrap:wrap;align-items:center}
#bar b{font-weight:600}
button{font:12px ui-monospace,monospace;padding:4px 10px;background:#f5f1e6;border:1px solid #cbbfa3;cursor:pointer}
#hint{color:#6b6250;font:13px Georgia,serif;padding:12px 40px 0;max-width:820px;margin:0 auto}
</style>
<div id=hint>Type anywhere: your letters go in as you type, one per tick. Its letters appear in brown. Arrow up / down: your face (&minus;6..6). Enter: a new line. Nothing is edited; this is a diary.</div>
<div id=pg></div>
<div id=bar>
 <span>you <b id=you>0.0</b></span><span>its face <b id=face>–</b></span><span>mood <b id=mood>–</b></span>
 <span>stress <b id=cort>–</b></span><span>uncertainty <b id=ent>–</b></span><span>memory <b id=mem>–</b></span>
 <span>tick <b id=tick>0</b></span><span>queued <b id=q>0</b></span>
 <button onclick="post('/sleep',{}).then(r=>{document.getElementById('night').textContent=JSON.stringify(r).slice(0,160)})">sleep</button>
 <button onclick="post('/save',{})">save</button><span id=night></span>
</div>
<script>
let face=0,seen=0;
const pg=document.getElementById('pg');
function post(p,b){return fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}).then(r=>r.json())}
document.addEventListener('keydown',e=>{
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  if(e.key==='ArrowUp'){face=Math.min(6,face+0.5);post('/face',{expr:face});e.preventDefault();return}
  if(e.key==='ArrowDown'){face=Math.max(-6,face-0.5);post('/face',{expr:face});e.preventDefault();return}
  if(e.key==='Enter'){post('/type',{text:'\\n'});e.preventDefault();return}
  if(e.key.length===1){post('/type',{text:e.key});e.preventDefault()}
});
function render(items){
  for(const [t,who] of items){
    if(!t)continue;
    const s=document.createElement('span');s.className=who?'m':'u';s.textContent=t;pg.appendChild(s);
  }
  window.scrollTo(0,document.body.scrollHeight);
}
async function poll(){
  try{
    const r=await fetch('/state?since='+seen).then(r=>r.json());
    render(r.page);seen=r.n;
    const l=r.last||{};const f=(x,d)=>x==null?'–':Number(x).toFixed(d);
    document.getElementById('you').textContent=f(face,1);
    document.getElementById('face').textContent=f(l.face,2);
    document.getElementById('mood').textContent=f(l.mood,2);
    document.getElementById('cort').textContent=f(l.cort,2);
    document.getElementById('ent').textContent=f(l.ent,2);
    document.getElementById('mem').textContent=l.mem?l.mem.slice(0,2).map(x=>JSON.stringify(x[0])+' '+x[1]).join('  '):'–';
    document.getElementById('tick').textContent=l.tick||0;
    document.getElementById('q').textContent=r.queued;
  }catch(e){}
  setTimeout(poll,250);
}
poll();
</script>"""

ORG = None
LOCK = threading.Lock()


class DH(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/state":
            since = int((parse_qs(u.query).get("since") or ["0"])[0])
            self._json(ORG.state(since))
            return
        if u.path == "/pulse":
            self._json({"awake": ORG.awake_ticks, "tick": ORG.ticks, "mood": round(ORG.mood, 2)})
            return
        b = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        try:
            if self.path == "/type":
                self._json(ORG.type_text(str(body.get("text", ""))))
            elif self.path == "/face":
                self._json(ORG.set_face(body.get("expr", 0)))
            elif self.path == "/sleep":
                self._json(ORG.night())
            elif self.path == "/save":
                with ORG.lock:
                    self._json(ORG.save())
            elif self.path == "/reset":
                with ORG.lock:
                    ORG.reset(); ORG.page = []; ORG.queue.clear(); ORG.credit.clear(); ORG.level = 0
                    self._json({"reset": True})
            else:
                self._json({"error": "unknown path"}, 404)
        except Exception as e:
            self._json({"error": str(e)[:300]}, 500)


def main():
    global ORG
    a = O.build_parser().parse_args()
    print("[diary] two writers, one page — no assists exist in this build", file=sys.stderr)
    ORG = Diary(a)
    print(f"[diary] the page is open on http://localhost:{a.port} ({a.ckpt} on {ORG.dev}, tick {ORG.period}s)",
          file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", a.port), DH).serve_forever()


if __name__ == "__main__":
    main()
