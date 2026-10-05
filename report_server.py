#!/usr/bin/env python3
"""
report_server.py - a tiny LOCAL web server that lets the buttons in the _up report run the screener.

    python report_server.py            # start it by hand (Ctrl+C to stop), or let install_report_server.sh
                                       # register it with macOS (launchd) so it starts at login

It listens on 127.0.0.1 only (nothing outside this Mac can reach it) and can do exactly two things:
    run_trend_entry refresh-setups <list>   ("Refresh setups": only the setup tickers of the list the page shows)
    run_trend_entry refresh <list>   ("Refresh all": every ticker of the list the page shows, e.g. Large)
    run_trend_entry full             ("Full scan": all lists)
using the same Python that runs this server (your venv), in this folder. One run at a time.
Runs go to a background WORKER (screener_worker.py) that keeps Python and the screener loaded, so a refresh does not
pay the 2-3 s start-up each time. The worker is restarted automatically when any .py file (or .env) in this folder
changed, so code edits are picked up. If the worker breaks, the run falls back to a fresh `python run_trend_entry.py`.
A secret token (file .report_server_token next to this script, created on first use) must be in every run link,
so another web page you happen to visit cannot start a run.

Pages:  /status   live output of the current / last run (after a button click it then opens the updated report)
        /log      the same output as plain text
        /reports/<file>.html   the reports folder (read only), so the updated report opens after a run
        /ping.gif  "I am running" check used by the report to show its buttons
"""
import html
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("REPORT_SERVER_PORT", "8765"))
REPORTS = os.path.join(HERE, "reports")
SCRIPT = "run_trend_entry.py"
WORKER_SCRIPT = "screener_worker.py"
MODES = ("refresh-setups", "refresh", "full")
TOKEN_FILE = os.path.join(HERE, ".report_server_token")
GIF = (b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,"
       b"\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;")


def get_token() -> str:
    """The shared secret (enrich_html_v2.py reads the same file when it writes the report buttons)."""
    try:
        with open(TOKEN_FILE, encoding="utf-8") as f:
            tok = f.read().strip()
        if tok:
            return tok
    except OSError:
        pass
    tok = secrets.token_urlsafe(18)
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(tok)
    try:
        os.chmod(TOKEN_FILE, 0o600)
    except OSError:
        pass
    return tok


TOKEN = get_token()
RUN = {"active": False, "via": None, "worker": None, "mode": None, "started": None, "ended": None, "code": None,
       "back": None, "target": None}
WORKER = {"proc": None, "started": 0.0, "ready": False, "import_s": None}
LOG = deque(maxlen=3000)
LOCK = threading.Lock()
ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}


def _newest_up(prefix: str):
    """Newest <prefix>_signal_report_*_up.html in reports/ (a full scan writes a new file name)."""
    try:
        files = [f for f in os.listdir(REPORTS)
                 if f.startswith(prefix + "_signal_report_") and f.endswith("_up.html")]
    except OSError:
        return None
    return max(files, key=lambda f: os.path.getmtime(os.path.join(REPORTS, f))) if files else None


def _safe_report(name: str):
    """A plain *.html file name inside reports/, or None (no paths, no other folders)."""
    name = os.path.basename(name or "")
    return name if name.endswith(".html") and os.path.isfile(os.path.join(REPORTS, name)) else None


def _list_of(report_file: str):
    """'Large' from 'Large_signal_report_<time>_up.html' (None if the name does not look like a report)."""
    m = re.match(r"^([A-Za-z0-9-]+)_signal_report_", report_file or "")
    return m.group(1) if m else None


def _code_mtime():
    """(newest change time, file name) over the .py files and .env in this folder (a change = restart the worker)."""
    m, name = 0.0, None
    for f in os.listdir(HERE):
        if f.endswith(".py") or f == ".env":
            try:
                t = os.path.getmtime(os.path.join(HERE, f))
            except OSError:
                continue
            if t > m:
                m, name = t, f
    return m, name


def _finish(code: int) -> None:
    with LOCK:
        if not RUN["active"]:
            return
        RUN.update(active=False, ended=time.time(), code=code)
        back_, mode = RUN["back"], RUN["mode"].split()[0]
        # refresh writes over the same file; a full scan (or a refresh that became one) writes a new one
        # -> then open the newest _up of the same list
        same = (mode.startswith("refresh") and _safe_report(back_)
                and os.path.getmtime(os.path.join(REPORTS, back_)) >= RUN["started"] - 1)
        RUN["target"] = back_ if same else (_newest_up(back_.split("_signal_report_")[0]) if back_ else None)
        took = RUN["ended"] - RUN["started"]
    LOG.append(f"--- finished with exit code {code} after {took:.1f}s ---")


def _pump_worker(proc) -> None:
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.startswith("@@READY"):
            WORKER["ready"], WORKER["import_s"] = True, (line.split() + ["?"])[1]
            continue
        if line.startswith("@@DONE"):
            parts = line.split()
            try:
                code = int(parts[1])
            except (IndexError, ValueError):
                code = 1
            if RUN["worker"] is proc:
                _finish(code)
            continue
        LOG.append(line)
    with LOCK:                                         # the worker ended (killed for a restart, or crashed)
        if WORKER["proc"] is proc:
            WORKER["proc"] = None
        died_mid_run = RUN["active"] and RUN["worker"] is proc
    if died_mid_run:
        LOG.append("worker stopped unexpectedly - the next click starts a new one")
        _finish(-1)


def _spawn_worker():
    """Start (or restart) the background worker. Called with LOCK held."""
    old = WORKER["proc"]
    if old is not None and old.poll() is None:
        old.terminate()
    proc = subprocess.Popen([sys.executable, "-u", WORKER_SCRIPT], cwd=HERE, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=ENV)
    WORKER.update(proc=proc, started=time.time(), ready=False, import_s=None)
    threading.Thread(target=_pump_worker, args=(proc,), daemon=True).start()
    return proc


def _worker():
    """The running worker, restarted first if it is gone or the code changed.
    Returns (proc, why_restarted or None). LOCK held."""
    p = WORKER["proc"]
    if p is None or p.poll() is not None:
        return _spawn_worker(), "no worker was running"
    m, name = _code_mtime()
    if m > WORKER["started"]:
        return _spawn_worker(), f"{name} changed since the worker started"
    return p, None


def _run_oneoff(mode: str, lst):
    """Fallback: a fresh `python run_trend_entry.py <mode> [list]` (the slower, pre-worker way). LOCK held."""
    cmd = [sys.executable, "-u", SCRIPT, mode] + ([lst] if lst else [])
    LOG.append(f"$ {' '.join(cmd)}   (in {HERE})")
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=ENV)
    RUN.update(via="process", worker=None)

    def pump():
        for line in proc.stdout:
            LOG.append(line.rstrip("\n"))
        _finish(proc.wait())

    threading.Thread(target=pump, daemon=True).start()


def _start(mode: str, back: str) -> bool:
    with LOCK:
        if RUN["active"]:
            return False                                   # already running
        LOG.clear()
        lst = _list_of(back) if mode.startswith("refresh") else None      # refresh = only the list this page shows
        RUN.update(active=True, mode=mode + (f" {lst}" if lst else ""), started=time.time(), ended=None, code=None,
                   back=back, target=None)
        if not os.path.exists(os.path.join(HERE, WORKER_SCRIPT)):
            _run_oneoff(mode, lst)
            return True
        try:
            proc, why = _worker()
            LOG.append(f"> {mode} {lst or '(all lists)'} in the background worker"
                       + (f" - (re)started first ({why}): loading Python and the screener ..." if why
                          else " - already loaded"))
            RUN.update(via="worker", worker=proc)
            proc.stdin.write(json.dumps({"mode": mode, "lists": [lst] if lst else []}) + "\n")
            proc.stdin.flush()
        except (OSError, ValueError) as e:
            LOG.append(f"worker not usable ({type(e).__name__}: {e}) - running a fresh process instead")
            _run_oneoff(mode, lst)
    return True


STATUS_CSS = ("body{background:#0f1115;color:#e6e6e6;font:14px -apple-system,Segoe UI,Roboto,sans-serif;margin:24px}"
              "pre{background:#171a21;border:1px solid #262c39;border-radius:8px;padding:12px;white-space:pre-wrap;"
              "font:12px ui-monospace,Menlo,monospace;max-height:70vh;overflow:auto}"
              "a{color:#9cdcfe}.ok{color:#2ecc71}.bad{color:#e74c3c}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):                    # keep the launchd log quiet
        pass

    def _send(self, code, body, ctype="text/html; charset=utf-8", extra=None):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, where):
        self.send_response(303)
        self.send_header("Location", where)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/ping.gif":
            return self._send(200, GIF, "image/gif", {"Access-Control-Allow-Origin": "*"})
        if u.path == "/run":
            if not secrets.compare_digest(q.get("token", ""), TOKEN):
                return self._send(403, "<p>Wrong or missing token. Regenerate the report (the token lives in "
                                       ".report_server_token next to report_server.py).</p>")
            mode = q.get("mode", "")
            if mode not in MODES:
                return self._send(400, f"<p>mode must be one of {MODES}</p>")
            _start(mode, os.path.basename(q.get("back", "")))   # already running -> just show its status
            return self._redirect("/status?open=1")      # open the report when this run is done
        if u.path == "/log":
            with LOCK:
                txt = "\n".join(LOG) or "(no run yet)"
            return self._send(200, txt + "\n", "text/plain; charset=utf-8")
        if u.path == "/status":
            return self._send(200, self._status_page())
        if u.path == "/state":
            return self._send(200, json.dumps(self._state()), "application/json")
        if u.path.startswith("/reports/"):
            name = _safe_report(u.path[len("/reports/"):])
            if not name:
                return self._send(404, "<p>No such report.</p>")
            with open(os.path.join(REPORTS, name), "rb") as f:
                return self._send(200, f.read())
        if u.path == "/":
            return self._redirect("/status")
        return self._send(404, "<p>Not found.</p>")

    def _state(self):
        with LOCK:
            r = dict(RUN)
        w = WORKER["proc"]
        alive = w is not None and w.poll() is None
        return {
            "started": r["started"] is not None, "active": r["active"], "mode": r["mode"], "code": r["code"],
            "elapsed": round((r["ended"] or time.time()) - r["started"], 1) if r["started"] else None,
            "target": ("/reports/" + quote(r["target"])) if (not r["active"] and r["code"] == 0 and r["target"]) else None,
            "worker": ("loaded" + (f" (start-up took {WORKER['import_s']}s)" if WORKER["import_s"] else "")
                       if alive and WORKER["ready"] else "starting ..." if alive else "not running"),
            "log": "\n".join(LOG),
        }

    def _status_page(self):
        st = self._state()
        # The page asks /state 4 times a second and opens the report the moment the run is done.
        return ("<!doctype html><html><head><meta charset='utf-8'><title>Screener run</title>"
                f"<style>{STATUS_CSS}</style></head><body>"
                "<h2 id='head'></h2><p id='sub'></p><pre id='log'></pre>"
                "<p id='wk' style='color:#9aa4b2;font-size:12px'></p>"
                "<noscript><p>JavaScript is off: reload this page to see progress.</p></noscript>"
                "<script>var OPEN=/[?&]open=1/.test(location.search);var first=" + json.dumps(st) + ";"
                "function show(s){var h=document.getElementById('head'),p=document.getElementById('sub'),"
                "l=document.getElementById('log');"
                "document.getElementById('wk').textContent='worker: '+s.worker;"
                "var atEnd=l.scrollTop+l.clientHeight>=l.scrollHeight-20;l.textContent=s.log||'(no run yet)';"
                "if(atEnd)l.scrollTop=l.scrollHeight;"
                "if(!s.started){h.textContent='No run yet';h.className='';p.textContent='Use the buttons in an _up report.';return true;}"
                "if(s.active){h.textContent='Running: '+s.mode+' \u2026 '+s.elapsed+'s';h.className='';"
                "p.textContent='The report opens as soon as the run is done.';return false;}"
                "if(s.target){h.textContent='Done ('+s.mode+', '+s.elapsed+'s)';h.className='ok';"
                "if(OPEN){p.textContent='Opening the report ...';location.replace(s.target);}"
                "else{p.innerHTML='<a href=\"'+s.target+'\">Open the report</a> &middot; "
                "<a href=\"/log\">plain-text output</a>';}return true;}"
                "h.textContent='Finished ('+s.mode+') with exit code '+s.code;h.className=s.code===0?'ok':'bad';"
                "p.textContent=s.code===0?'':'Something went wrong - see the output below.';return true;}"
                "if(!show(first)){var t=setInterval(function(){fetch('/state',{cache:'no-store'})"
                ".then(function(r){return r.json()}).then(function(s){if(show(s))clearInterval(t);})"
                ".catch(function(){});},250);}"
                "</script></body></html>")


def main():
    if not os.path.exists(os.path.join(HERE, SCRIPT)):
        sys.exit(f"{SCRIPT} not found next to {__file__}")
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    if os.path.exists(os.path.join(HERE, WORKER_SCRIPT)):
        with LOCK:
            _spawn_worker()                    # warm up now, so the first click is already fast
    print(f"report server on http://127.0.0.1:{PORT}  (python: {sys.executable}, folder: {HERE})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
