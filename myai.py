#!/usr/bin/env python3
"""
myai v2 - one-file, offline, terminal-only coding assistant.

FIRST RUN (needs internet once):
    python myai.py run         -> installs everything, downloads the model,
                                  creates the trigger command, opens the chat
AFTER THAT (fully offline):
    myai                       -> opens the chat right in your terminal

Other commands:
    python myai.py doctor      -> prints a full diagnostic report
    python myai.py update      -> fetches the newest myai.py from GitHub
    python myai.py repair      -> deletes the model and downloads it again
    python myai.py uninstall   -> removes everything this script created

Optional environment variables:
    MYAI_MODEL     0.5b | 1.5b | 3b   (default: chosen from your RAM)
    MYAI_CONNS     parallel download connections (default 6)
    MYAI_THREADS   CPU threads used for answers (default: auto)
    MYAI_CTX       context length in tokens (default: auto)
    HF_TOKEN       Hugging Face token (optional; the models are public)
    HF_ENDPOINT    custom Hugging Face mirror, e.g. https://hf-mirror.com

Everything lives in ~/.myai  (nothing is installed system-wide).
"""

import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import venv
from collections import deque

# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
VERSION = "2.0"
TRIGGER = "myai"  # the word you type in any terminal to start the chat
SCRIPT_URL = "https://raw.githubusercontent.com/vrr167167-rgb/myai/main/myai.py"

# tier -> (Hugging Face repo, file name, minimum believable size in bytes)
MODELS = {
    "0.5b": ("Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
             "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", 300_000_000),
    "1.5b": ("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
             "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 800_000_000),
    "3b": ("Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
           "qwen2.5-coder-3b-instruct-q4_k_m.gguf", 1_500_000_000),
}
TIER_ORDER = ["0.5b", "1.5b", "3b"]  # small -> big

# Tried in order until one works. HF_ENDPOINT (if set) is tried first.
HF_ENDPOINTS = ["https://huggingface.co", "https://hf-mirror.com"]

# Prebuilt CPU wheels, so no C++ compiler is needed on most machines
WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"

SYSTEM_PROMPT = (
    "You are a concise coding assistant. When asked for code, give complete, "
    "compilable, working examples with a short explanation."
)

CHUNK = 4 * 1024 * 1024      # download piece size
READ_TIMEOUT = 20            # seconds of silence before a connection is dropped
STALL_LIMIT = 60             # seconds with no data at all before switching source
NO_CHUNK_LIMIT = 240         # seconds without ANY finished piece before switching
UA = {"User-Agent": "myai/" + VERSION}

# ----------------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------------
IS_WIN = os.name == "nt"
HOME = os.path.expanduser("~")
APP_DIR = os.path.join(HOME, ".myai")
VENV_DIR = os.path.join(APP_DIR, "venv")
VENV_PY = os.path.join(
    VENV_DIR, "Scripts" if IS_WIN else "bin", "python.exe" if IS_WIN else "python"
)
SCRIPT_COPY = os.path.join(APP_DIR, "myai.py")
ACTIVE_FILE = os.path.join(APP_DIR, "active.txt")
LEGACY_MODEL = os.path.join(APP_DIR, "model.gguf")
UNIX_BIN_DIR = os.path.join(HOME, ".local", "bin")
WIN_BIN_DIR = os.path.join(APP_DIR, "bin")
_LOCALAPPDATA = os.environ.get("LOCALAPPDATA", "")
WIN_APPS_DIR = (
    os.path.join(_LOCALAPPDATA, "Microsoft", "WindowsApps") if _LOCALAPPDATA else ""
)


def say(msg=""):
    print(msg, flush=True)


def human_time(sec):
    sec = int(max(sec, 0))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def mb(n):
    return f"{n / 1e6:,.0f}"


# ----------------------------------------------------------------------------
# HARDWARE
# ----------------------------------------------------------------------------
def total_ram_bytes():
    try:
        if IS_WIN:
            import ctypes

            class MEMSTATUS(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = MEMSTATUS()
            st.dwLength = ctypes.sizeof(MEMSTATUS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullTotalPhys)
        if sys.platform == "darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=5)
            return int(out.strip())
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def pick_tier():
    forced = os.environ.get("MYAI_MODEL", "").strip().lower()
    if forced in MODELS:
        return forced, "chosen by MYAI_MODEL"
    ram = total_ram_bytes()
    if ram is None:
        return "1.5b", "RAM unknown, using the safe default"
    gb = ram / (1 << 30)
    if gb < 3.5:
        return "0.5b", f"{gb:.1f} GB RAM: small model for speed"
    if gb < 12:
        return "1.5b", f"{gb:.1f} GB RAM: balanced model"
    return "3b", f"{gb:.1f} GB RAM: larger, smarter model"


def hw_settings():
    cpu = os.cpu_count() or 4
    ram = total_ram_bytes() or (8 << 30)
    try:
        threads = int(os.environ.get("MYAI_THREADS", "0"))
    except ValueError:
        threads = 0
    if threads <= 0:
        threads = cpu if cpu <= 4 else max(4, cpu // 2)
    try:
        ctx = int(os.environ.get("MYAI_CTX", "0"))
    except ValueError:
        ctx = 0
    if ctx <= 0:
        gb = ram / (1 << 30)
        ctx = 2048 if gb < 4 else 4096 if gb < 8 else 8192
    return max(1, min(threads, 32)), ctx


# ----------------------------------------------------------------------------
# DOWNLOADER: parallel, resumable, stall-proof, with live progress
# ----------------------------------------------------------------------------
class StallError(RuntimeError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def resolve(url, token=""):
    """Follow redirects by hand. Returns (final_url, total_size, range_ok).
    The token is only ever sent to the original host, never to the CDN."""
    origin = urllib.parse.urlparse(url).netloc
    opener = urllib.request.build_opener(_NoRedirect)
    for _ in range(10):
        headers = dict(UA)
        headers["Range"] = "bytes=0-0"
        if token and urllib.parse.urlparse(url).netloc == origin:
            headers["Authorization"] = "Bearer " + token
        try:
            resp = opener.open(urllib.request.Request(url, headers=headers), timeout=30)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                url = urllib.parse.urljoin(url, e.headers["Location"])
                continue
            raise
        with resp:
            status = resp.status
            cr = resp.headers.get("Content-Range", "")
            cl = resp.headers.get("Content-Length", "")
        if status == 206 and "/" in cr and cr.rsplit("/", 1)[1].isdigit():
            return url, int(cr.rsplit("/", 1)[1]), True
        if status == 200 and cl.isdigit():
            return url, int(cl), False
        raise RuntimeError("Unexpected server reply while checking the download.")
    raise RuntimeError("Too many redirects.")


def _load_state(path, total, chunk):
    try:
        with open(path) as f:
            s = json.load(f)
        if s.get("total") == total and s.get("chunk") == chunk:
            return set(int(x) for x in s.get("done", []))
    except (OSError, ValueError):
        pass
    return set()


def _save_state(path, total, chunk, done):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump({"total": total, "chunk": chunk, "done": sorted(done)}, f)
        os.replace(tmp, path)
    except OSError:
        pass


def download_file(url, token, dest, conns):
    """Download url -> dest. Resumable; raises on failure. Never silent:
    progress is printed by the main thread, so a stalled connection still
    produces a visible status line."""
    final_url, total, ranged = resolve(url, token)
    part, state = dest + ".part", dest + ".state"
    chunk = CHUNK if ranged else total
    nchunks = -(-total // chunk)

    if os.path.exists(part) and os.path.getsize(part) == total:
        done = _load_state(state, total, chunk)
    else:
        free = shutil.disk_usage(os.path.dirname(dest) or ".").free
        if free < total + 200 * 1024 * 1024:
            raise RuntimeError(
                f"Not enough free disk space: need about {mb(total + 200 * 1024 * 1024)} MB, "
                f"only {mb(free)} MB free."
            )
        with open(part, "wb") as f:
            f.truncate(total)
        done = set()

    lock = threading.Lock()
    stop = threading.Event()
    fatal = []
    cur = {"url": final_url}
    prog = {
        "bytes": sum(min(chunk, total - i * chunk) for i in done),
        "last_data": time.time(),
        "last_done": time.time(),
    }
    q = queue.Queue()
    for i in range(nchunks):
        if i not in done:
            q.put(i)
    url_lock = threading.Lock()
    last_refresh = [0.0]

    def refresh_url():
        with url_lock:
            if time.time() - last_refresh[0] < 5:
                return
            last_refresh[0] = time.time()
            try:
                cur["url"] = resolve(url, token)[0]
            except Exception:
                pass

    def worker():
        try:
            f = open(part, "r+b")
        except OSError as e:
            fatal.append(e)
            stop.set()
            return
        fails = 0
        try:
            while not stop.is_set():
                try:
                    i = q.get_nowait()
                except queue.Empty:
                    return
                start = i * chunk
                end = min(total - 1, start + chunk - 1)
                got = 0
                try:
                    hdr = dict(UA)
                    hdr["Range"] = f"bytes={start}-{end}"
                    req = urllib.request.Request(cur["url"], headers=hdr)
                    with urllib.request.urlopen(req, timeout=READ_TIMEOUT) as r:
                        if not (r.status == 206 or (r.status == 200 and start == 0)):
                            raise RuntimeError(f"HTTP {r.status}")
                        f.seek(start)
                        remaining = end - start + 1
                        while remaining > 0:
                            if stop.is_set():
                                raise InterruptedError()
                            b = r.read(min(131072, remaining))
                            if not b:
                                break
                            try:
                                f.write(b)
                            except OSError as e:  # disk full / permission
                                fatal.append(e)
                                stop.set()
                                return
                            got += len(b)
                            remaining -= len(b)
                            with lock:
                                prog["bytes"] += len(b)
                                prog["last_data"] = time.time()
                    if remaining:
                        raise IOError("connection closed early")
                    f.flush()
                    with lock:
                        done.add(i)
                        prog["last_done"] = time.time()
                    fails = 0
                except Exception:
                    with lock:
                        prog["bytes"] -= got
                    q.put(i)
                    if stop.is_set():
                        return
                    fails += 1
                    refresh_url()
                    time.sleep(min(1 + fails, 8))
        finally:
            f.close()

    todo = nchunks - len(done)
    threads = [
        threading.Thread(target=worker, daemon=True)
        for _ in range(max(0, min(conns, todo)))
    ]
    for t in threads:
        t.start()

    tty = sys.stdout.isatty()
    samples = deque(maxlen=14)
    t0 = time.time()
    last_print = 0.0
    last_save = time.time()
    problem = None

    def render(now, idle):
        with lock:
            b = prog["bytes"]
        samples.append((now, b))
        t_old, b_old = samples[0]
        speed = (b - b_old) / max(now - t_old, 0.5) / 1e6 if len(samples) > 1 else 0.0
        speed = max(speed, 0.0)
        pct = min(100, int(b * 100 / total)) if total else 100
        eta = human_time((total - b) / (speed * 1e6)) if speed > 0.01 else "--:--"
        bar = ("#" * (pct // 4)).ljust(25, "-")
        line = (f"  [{bar}] {pct:3d}%  {mb(b)}/{mb(total)} MB  "
                f"{speed:5.1f} MB/s  ETA {eta}")
        if idle > 8:
            line += f"  (no data for {int(idle)}s, reconnecting...)"
        return line

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
            now = time.time()
            with lock:
                idle = now - prog["last_data"]
                idle_done = now - prog["last_done"]
            if now - last_print >= (0.5 if tty else 5):
                last_print = now
                line = render(now, idle)
                if tty:
                    print("\r" + line.ljust(110), end="", flush=True)
                else:
                    say(line)
            if now - last_save > 3:
                last_save = now
                with lock:
                    snap = set(done)
                _save_state(state, total, chunk, snap)
            if fatal:
                break
            if idle > STALL_LIMIT:
                problem = StallError(f"no data received for {STALL_LIMIT}s")
                break
            if idle_done > NO_CHUNK_LIMIT:
                problem = StallError("connection keeps dropping, nothing finishes")
                break
    except KeyboardInterrupt:
        stop.set()
        with lock:
            snap = set(done)
        _save_state(state, total, chunk, snap)
        print()
        raise
    stop.set()
    if tty:
        print()
    for t in threads:
        t.join(timeout=READ_TIMEOUT + 5)
    with lock:
        snap = set(done)
    _save_state(state, total, chunk, snap)

    if fatal:
        raise fatal[0]
    if problem:
        raise problem
    if len(snap) < nchunks:
        raise RuntimeError("Download incomplete.")
    if os.path.getsize(part) != total:
        raise RuntimeError("Downloaded file has the wrong size.")
    for attempt in range(10):
        try:
            os.replace(part, dest)
            break
        except OSError:
            if attempt == 9:
                raise
            time.sleep(1)
    try:
        os.remove(state)
    except OSError:
        pass
    say(f"  Done in {human_time(time.time() - t0)}.")


def model_path(tier):
    return os.path.join(APP_DIR, MODELS[tier][1])


def model_valid(tier):
    p = model_path(tier)
    return os.path.exists(p) and os.path.getsize(p) >= MODELS[tier][2]


def download_model(tier):
    repo, fname, _ = MODELS[tier]
    dest = model_path(tier)
    if model_valid(tier):
        return dest
    endpoints = []
    custom = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")
    if custom:
        endpoints.append(custom)
    endpoints += [e for e in HF_ENDPOINTS if e not in endpoints]
    token = os.environ.get("HF_TOKEN", "").strip()
    try:
        conns = max(1, min(int(os.environ.get("MYAI_CONNS", "6")), 16))
    except ValueError:
        conns = 6

    notfound = set()
    failures = 0
    while failures < 12:
        for ep in endpoints:
            if ep in notfound:
                continue
            url = f"{ep}/{repo}/resolve/main/{fname}"
            is_hf = urllib.parse.urlparse(ep).netloc == "huggingface.co"
            say(f"  Source: {ep}")
            try:
                download_file(url, token if is_hf else "", dest, conns)
                return dest
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    say(f"  Not found on {ep}")
                    notfound.add(ep)
                    continue
                say(f"  {ep} answered HTTP {e.code}")
            except StallError as e:
                say(f"  Stalled ({e}); switching source...")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                say(f"  Problem: {e}")
                if "disk space" in str(e).lower():
                    raise
            failures += 1
        if len(notfound) == len(endpoints):
            raise FileNotFoundError(fname)
        wait = min(3 * failures, 15)
        say(f"  Retrying in {wait}s (progress is kept)...")
        time.sleep(wait)
    raise RuntimeError("Could not finish the download.")


def active_model():
    try:
        with open(ACTIVE_FILE) as f:
            name = f.read().strip()
    except OSError:
        name = ""
    if name:
        p = os.path.join(APP_DIR, name)
        if os.path.exists(p):
            return p
    if os.path.exists(LEGACY_MODEL) and os.path.getsize(LEGACY_MODEL) > 500_000_000:
        return LEGACY_MODEL
    for t in reversed(TIER_ORDER):
        if model_valid(t):
            return model_path(t)
    return None


def get_model():
    existing = active_model()
    if existing and not os.environ.get("MYAI_MODEL"):
        say("[3/4] Model already downloaded.")
        return existing
    want, why = pick_tier()
    tiers = [want] + list(reversed(TIER_ORDER[: TIER_ORDER.index(want)]))
    say(f"[3/4] Downloading the model: Qwen2.5-Coder {want.upper()} ({why}).")
    say("      One time only. If it stops, run the same command again to resume.")
    for t in tiers:
        try:
            path = download_model(t)
        except FileNotFoundError:
            say(f"  The {t} model file is not available; trying a smaller one...")
            continue
        except KeyboardInterrupt:
            say("\nStopped. Run the same command again to resume the download.")
            sys.exit(130)
        except Exception as e:
            say(f"\nDownload problem: {e}")
            say("Check your internet/VPN and run the same command again (it resumes).")
            sys.exit(1)
        with open(ACTIVE_FILE, "w") as f:
            f.write(os.path.basename(path))
        return path
    say("\nNo model could be downloaded from any source.")
    sys.exit(1)


# ----------------------------------------------------------------------------
# INSTALL STEPS
# ----------------------------------------------------------------------------
def _pip_env():
    env = dict(os.environ)
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_NO_INPUT"] = "1"
    return env


def run_visible(cmd, label):
    """Run a command with its output visible plus a heartbeat, so a long
    silent step (like compiling) never looks frozen."""
    p = subprocess.Popen(cmd, env=_pip_env())
    t0 = time.time()
    last = t0
    while p.poll() is None:
        time.sleep(1)
        if time.time() - last >= 30:
            last = time.time()
            say(f"  ... {label} still working ({human_time(time.time() - t0)})")
    return p.returncode


def venv_ok():
    if not os.path.exists(VENV_PY):
        return False
    try:
        r = subprocess.run([VENV_PY, "-m", "pip", "--version"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


def make_venv():
    if venv_ok():
        return
    say("[1/4] Creating private Python environment...")
    shutil.rmtree(VENV_DIR, ignore_errors=True)
    try:
        venv.EnvBuilder(with_pip=True, clear=True).create(VENV_DIR)
    except Exception as e:
        say(f"  Standard method failed ({e}); trying fallback...")
        shutil.rmtree(VENV_DIR, ignore_errors=True)
        try:
            venv.EnvBuilder(with_pip=False, clear=True).create(VENV_DIR)
            gp = os.path.join(APP_DIR, "get-pip.py")
            req = urllib.request.Request("https://bootstrap.pypa.io/get-pip.py", headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r, open(gp, "wb") as f:
                shutil.copyfileobj(r, f)
            if subprocess.call([VENV_PY, gp], env=_pip_env()) != 0:
                raise RuntimeError("get-pip failed")
        except Exception as e2:
            say(f"\nCould not create the environment: {e2}")
            say("On Debian/Ubuntu run:  sudo apt install python3-venv python3-pip  and try again.")
            sys.exit(1)
    if not venv_ok():
        say("\nThe private environment was created but pip does not work in it.")
        sys.exit(1)


def engine_check():
    try:
        r = subprocess.run(
            [VENV_PY, "-c", "import llama_cpp; print(llama_cpp.__version__)"],
            capture_output=True, text=True, timeout=120,
        )
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except Exception as e:
        return False, str(e)


def install_vc_redist():
    """Fresh Windows PCs sometimes lack the Visual C++ runtime that the
    engine needs. Install it quietly through winget if we can."""
    if not (IS_WIN and shutil.which("winget")):
        return False
    say("  Installing Microsoft Visual C++ runtime (needed by the engine)...")
    r = subprocess.call([
        "winget", "install", "-e", "--id", "Microsoft.VCRedist.2015+.x64",
        "--silent", "--accept-source-agreements", "--accept-package-agreements",
    ])
    return r == 0


def install_engine():
    ok, _ = engine_check()
    if ok:
        return
    say("[2/4] Installing the AI engine (llama-cpp-python)...")
    base = [VENV_PY, "-m", "pip", "install", "--default-timeout", "60", "--retries", "10"]
    subprocess.call([VENV_PY, "-m", "pip", "install", "-q", "--upgrade", "pip"],
                    env=_pip_env())
    attempts = [
        ("prebuilt engine", base + ["--only-binary=:all:", "--extra-index-url",
                                    WHEEL_INDEX, "llama-cpp-python"]),
        ("prebuilt engine (retry)", base + ["--only-binary=:all:", "--extra-index-url",
                                            WHEEL_INDEX, "llama-cpp-python"]),
        ("engine build from source", base + ["--prefer-binary", "llama-cpp-python"]),
    ]
    installed = False
    for label, cmd in attempts:
        if label.startswith("engine build"):
            say("  No prebuilt engine for this system. Building from source.")
            say("  This needs a C++ compiler and can take 5-20 minutes. Please wait.")
        if run_visible(cmd, label) == 0:
            installed = True
            break
        time.sleep(3)
    if installed:
        ok, out = engine_check()
        if not ok and IS_WIN and install_vc_redist():
            ok, out = engine_check()
        if ok:
            return
        say("\nThe engine installed but cannot start:")
        say(out[-1500:])
        sys.exit(1)
    say("\nEngine install failed. Usually no prebuilt file exists for your system")
    say("and a C++ compiler is needed. Install one and run the command again:")
    say("  Windows: Visual Studio Build Tools   Mac: xcode-select --install")
    say("  Linux:   sudo apt install build-essential cmake")
    sys.exit(1)


def _on_path(directory):
    target = os.path.normcase(os.path.normpath(directory))
    for p in os.environ.get("PATH", "").split(os.pathsep):
        if p and os.path.normcase(os.path.normpath(os.path.expandvars(p))) == target:
            return True
    return False


def add_to_path_unix():
    if _on_path(UNIX_BIN_DIR):
        return False
    line = f'export PATH="{UNIX_BIN_DIR}:$PATH"'
    rcs = [os.path.join(HOME, n) for n in (".bashrc", ".zshrc", ".profile")]
    if "zsh" in os.environ.get("SHELL", "") or sys.platform == "darwin":
        open(rcs[1], "a").close()
    targets = [p for p in rcs if os.path.exists(p)] or [rcs[2]]
    for rc in targets:
        try:
            with open(rc, "a+") as f:
                f.seek(0)
                if line not in f.read():
                    f.write(f"\n# added by myai\n{line}\n")
        except OSError:
            pass
    return True


def add_to_path_windows():
    import winreg

    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER, "Environment", 0,
        winreg.KEY_READ | winreg.KEY_WRITE,
    ) as key:
        try:
            current, _ = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            current = ""
        if WIN_BIN_DIR.lower() in current.lower():
            return False
        new = (current + ";" if current else "") + WIN_BIN_DIR
        winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new)
    try:  # tell running programs (Explorer) that PATH changed
        import ctypes
        ctypes.windll.user32.SendMessageTimeoutW(
            0xFFFF, 0x1A, 0, "Environment", 2, 5000, ctypes.byref(ctypes.c_ulong())
        )
    except Exception:
        pass
    return True


def make_trigger():
    """Creates the command. Returns True if a NEW terminal is needed."""
    say(f"[4/4] Creating the '{TRIGGER}' command...")
    if IS_WIN:
        content = (
            "@echo off\r\n"
            "chcp 65001 >nul\r\n"
            "set PYTHONUTF8=1\r\n"
            f'"{VENV_PY}" "{SCRIPT_COPY}" chat %*\r\n'
        )
        if WIN_APPS_DIR and os.path.isdir(WIN_APPS_DIR) and _on_path(WIN_APPS_DIR):
            try:
                with open(os.path.join(WIN_APPS_DIR, TRIGGER + ".cmd"), "w", newline="") as f:
                    f.write(content)
                return False
            except OSError:
                pass
        os.makedirs(WIN_BIN_DIR, exist_ok=True)
        with open(os.path.join(WIN_BIN_DIR, TRIGGER + ".cmd"), "w", newline="") as f:
            f.write(content)
        try:
            add_to_path_windows()
        except Exception:
            pass
        return True
    os.makedirs(UNIX_BIN_DIR, exist_ok=True)
    path = os.path.join(UNIX_BIN_DIR, TRIGGER)
    with open(path, "w") as f:
        f.write(f'#!/bin/sh\nexec "{VENV_PY}" "{SCRIPT_COPY}" chat "$@"\n')
    os.chmod(path, 0o755)
    return add_to_path_unix()


def clean_legacy():
    """Remove leftovers of older versions and broken earlier attempts."""
    try:
        for name in os.listdir(APP_DIR):
            if name.startswith("model.gguf.seg") or name == "model.gguf.part":
                try:
                    os.remove(os.path.join(APP_DIR, name))
                except OSError:
                    pass
    except OSError:
        pass


def install():
    say(f"\n=== Setting up '{TRIGGER}' (v{VERSION}) ===\n")
    if sys.version_info < (3, 8):
        say("Python 3.8 or newer is required.")
        sys.exit(1)
    if sys.version_info >= (3, 13):
        say("Note: Python 3.13+ may have no prebuilt engine yet; Python 3.10-3.12 is safest.\n")
    os.makedirs(APP_DIR, exist_ok=True)
    try:
        if os.path.abspath(__file__) != os.path.abspath(SCRIPT_COPY):
            shutil.copyfile(__file__, SCRIPT_COPY)
    except NameError:
        say("Save this script to a file first, then run it.")
        sys.exit(1)
    clean_legacy()
    make_venv()
    install_engine()
    get_model()
    needs_new_terminal = make_trigger()

    say("\n=====================================================")
    say(" DONE. Everything is downloaded and ready.")
    if needs_new_terminal:
        say(f" In a NEW terminal window, type:  {TRIGGER}")
    else:
        say(f" Just type:  {TRIGGER}   (works right now, in this window)")
    say(" From now on it runs fully offline.")
    say("=====================================================\n")


def run():
    install()
    say("Starting the chat...\n")
    sys.exit(subprocess.call([VENV_PY, SCRIPT_COPY, "chat"]))


# ----------------------------------------------------------------------------
# CHAT
# ----------------------------------------------------------------------------
def _fix_console():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if IS_WIN:
        os.system("chcp 65001 >nul")


def _read_input():
    line = input("\nyou > ")
    if line.strip() == '"""':  # multi-line paste mode
        lines = []
        while True:
            more = input("  ... ")
            if more.strip() == '"""':
                break
            lines.append(more)
        return "\n".join(lines).strip()
    return line.strip()


def chat():
    model = active_model()
    if not model:
        say(f"Model not found. Run:  python \"{SCRIPT_COPY}\" run   to set up first.")
        sys.exit(1)

    inside = os.path.normcase(os.path.abspath(sys.prefix)) == os.path.normcase(
        os.path.abspath(VENV_DIR)
    )
    if not inside and os.path.exists(VENV_PY) and not os.environ.get("MYAI_INNER"):
        os.environ["MYAI_INNER"] = "1"
        args = [VENV_PY, os.path.abspath(__file__), "chat"] + sys.argv[2:]
        if IS_WIN:
            sys.exit(subprocess.call(args))
        os.execv(VENV_PY, args)

    _fix_console()
    try:
        from llama_cpp import Llama
    except Exception as e:
        say(f"The AI engine is not working ({e}).")
        say(f"Run:  python \"{SCRIPT_COPY}\" repair")
        sys.exit(1)

    threads, ctx = hw_settings()
    say("Loading model, please wait...")
    try:
        llm = Llama(model_path=model, n_ctx=ctx, n_threads=threads, verbose=False)
    except Exception as e:
        say(f"\nThe model could not be loaded: {e}")
        say(f"If the file is damaged, run:  python \"{SCRIPT_COPY}\" repair")
        sys.exit(1)

    say("\n" + "=" * 56)
    say(f" {TRIGGER} - offline coding assistant")
    say(f" model: {os.path.basename(model)}")
    say(f" threads: {threads}   context: {ctx} tokens")
    say(' Type your question and press Enter.  Paste code: type """ then')
    say(' the code, then """ on its own line.')
    say(" /clear = new conversation   /exit = quit   Ctrl+C = stop answer")
    say("=" * 56)

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            q = _read_input()
        except (EOFError, KeyboardInterrupt):
            say()
            break
        if not q:
            continue
        if q in ("/exit", "/quit", "exit", "quit"):
            break
        if q == "/clear":
            history = history[:1]
            say("Conversation cleared.")
            continue

        history.append({"role": "user", "content": q})
        if len(history) > 11:
            history = history[:1] + history[-10:]
            while len(history) > 1 and history[1]["role"] != "user":
                del history[1]

        reply, ntok = "", 0
        t0 = time.time()
        print("\nai  > ", end="", flush=True)
        try:
            for chunk in llm.create_chat_completion(
                messages=history, max_tokens=1024, temperature=0.2, stream=True
            ):
                tok = chunk["choices"][0]["delta"].get("content")
                if tok:
                    print(tok, end="", flush=True)
                    reply += tok
                    ntok += 1
        except KeyboardInterrupt:
            print("\n[stopped]", end="")
        except ValueError:
            history = history[:1]
            print("\n[too long - conversation reset, please ask again]", end="")
            continue
        except Exception as e:
            history.pop()
            print(f"\n[error: {e}]", end="")
            continue
        dt = max(time.time() - t0, 0.01)
        print(f"\n[{ntok} tokens, {ntok / dt:.1f} tokens/s]")
        history.append({"role": "assistant", "content": reply})


# ----------------------------------------------------------------------------
# DOCTOR / UPDATE / REPAIR / UNINSTALL
# ----------------------------------------------------------------------------
def _reach(url):
    try:
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=8) as r:
            return f"OK ({r.status})"
    except urllib.error.HTTPError as e:
        return f"reachable (HTTP {e.code})"
    except Exception as e:
        return f"FAILED ({type(e).__name__})"


def doctor():
    ram = total_ram_bytes()
    threads, ctx = hw_settings()
    say(f"myai v{VERSION}")
    say(f"System:   {platform.platform()}  ({platform.machine()})")
    say(f"Python:   {sys.version.split()[0]}  at {sys.executable}")
    say(f"CPU:      {os.cpu_count()} logical cores -> using {threads} threads")
    say(f"RAM:      {'unknown' if ram is None else f'{ram / (1 << 30):.1f} GB'} -> context {ctx}")
    try:
        free = shutil.disk_usage(HOME).free
        say(f"Disk:     {free / (1 << 30):.1f} GB free")
    except OSError:
        pass
    say(f"Tier:     {pick_tier()[0]} ({pick_tier()[1]})")
    say(f"Folder:   {APP_DIR}  ({'exists' if os.path.isdir(APP_DIR) else 'missing'})")
    say(f"Venv:     {'OK' if venv_ok() else 'missing/broken'}")
    if os.path.exists(VENV_PY):
        ok, out = engine_check()
        say(f"Engine:   {'OK, llama-cpp-python ' + out if ok else 'NOT WORKING: ' + out[-300:]}")
    m = active_model()
    say(f"Model:    {m + ' (' + mb(os.path.getsize(m)) + ' MB)' if m else 'not downloaded'}")
    trig = [
        os.path.join(UNIX_BIN_DIR, TRIGGER),
        os.path.join(WIN_BIN_DIR, TRIGGER + ".cmd"),
        os.path.join(WIN_APPS_DIR, TRIGGER + ".cmd") if WIN_APPS_DIR else "",
    ]
    found = [p for p in trig if p and os.path.exists(p)]
    say(f"Command:  {found[0] if found else 'not created'}")
    say("Network:")
    for name, url in (
        ("huggingface.co", "https://huggingface.co"),
        ("hf-mirror.com", "https://hf-mirror.com"),
        ("pypi.org", "https://pypi.org"),
        ("wheel index", WHEEL_INDEX + "/llama-cpp-python/"),
    ):
        say(f"  {name:16s} {_reach(url)}")


def update():
    say("Downloading the newest myai.py ...")
    tmp = SCRIPT_COPY + ".new"
    try:
        req = urllib.request.Request(SCRIPT_URL, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        compile(open(tmp, encoding="utf-8").read(), tmp, "exec")  # sanity check
        os.replace(tmp, SCRIPT_COPY)
    except Exception as e:
        say(f"Update failed: {e}")
        sys.exit(1)
    say("Updated.")


def repair():
    say("Removing the downloaded model so it can be fetched again...")
    for name in os.listdir(APP_DIR) if os.path.isdir(APP_DIR) else []:
        if name.endswith((".gguf", ".part", ".state")) or name == "active.txt":
            try:
                os.remove(os.path.join(APP_DIR, name))
            except OSError:
                pass
    run()


def uninstall():
    shutil.rmtree(APP_DIR, ignore_errors=True)
    for p in (
        os.path.join(UNIX_BIN_DIR, TRIGGER),
        os.path.join(WIN_BIN_DIR, TRIGGER + ".cmd"),
        os.path.join(WIN_APPS_DIR, TRIGGER + ".cmd") if WIN_APPS_DIR else "",
    ):
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    say("Removed everything myai created.")


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    try:
        if cmd == "chat":
            chat()
        elif cmd == "install":
            install()
        elif cmd == "run":
            run()
        elif cmd == "doctor":
            doctor()
        elif cmd == "update":
            update()
        elif cmd == "repair":
            repair()
        elif cmd == "uninstall":
            uninstall()
        else:
            say(__doc__)
    except KeyboardInterrupt:
        say("\nStopped. Run the same command again to continue.")
        sys.exit(130)


if __name__ == "__main__":
    main()
