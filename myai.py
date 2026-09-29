#!/usr/bin/env python3
"""
myai - one-file, offline, terminal-only coding assistant.

FIRST RUN (needs internet once):
    python3 myai.py            -> installs everything, downloads the model,
                                  creates the trigger command
AFTER THAT (fully offline):
    myai                       -> opens the chat right in your terminal

Other commands:
    python3 myai.py uninstall  -> removes everything this script created

Optional environment variables:
    HF_TOKEN      Hugging Face token (optional; the model is public)
    HF_ENDPOINT   custom Hugging Face mirror, e.g. https://hf-mirror.com
    MYAI_CONNS    number of parallel download connections (default 8)

Everything lives in ~/.myai  (nothing is installed system-wide).
"""

import os
import sys
import time
import shutil
import threading
import subprocess
import urllib.request
import urllib.error
import urllib.parse
import venv
from concurrent.futures import ThreadPoolExecutor

# ----------------------------------------------------------------------------
# SETTINGS - change these if you like
# ----------------------------------------------------------------------------
TRIGGER = "myai"  # the word you type in any terminal to start the chat

MODEL_REPO = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
MODEL_FILENAME = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"

# Optional built-in token. Leave empty: the model is public. If a token is
# set and Hugging Face rejects it, the script retries without it.
HF_TOKEN_DEFAULT = ""

# Tried in order until one works. HF_ENDPOINT (if set) is tried first.
HF_ENDPOINTS = ["https://huggingface.co", "https://hf-mirror.com"]

# Prebuilt CPU wheels, so no C++ compiler is needed on most machines
WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"

CONNECTIONS = int(os.environ.get("MYAI_CONNS", "8"))

SYSTEM_PROMPT = (
    "You are a concise coding assistant. When asked for code, give complete, "
    "compilable, working examples with a short explanation."
)

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
MODEL_FILE = os.path.join(APP_DIR, "model.gguf")
SCRIPT_COPY = os.path.join(APP_DIR, "myai.py")
TOKEN_FILE = os.path.join(APP_DIR, "hf_token")
UNIX_BIN_DIR = os.path.join(HOME, ".local", "bin")
WIN_BIN_DIR = os.path.join(APP_DIR, "bin")
_LOCALAPPDATA = os.environ.get("LOCALAPPDATA", "")
# Already on PATH by default on Windows 10/11, so the command works instantly.
WIN_APPS_DIR = (
    os.path.join(_LOCALAPPDATA, "Microsoft", "WindowsApps") if _LOCALAPPDATA else ""
)


def say(msg=""):
    print(msg, flush=True)


def get_token():
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        return tok
    try:
        with open(TOKEN_FILE) as f:
            tok = f.read().strip()
            if tok:
                return tok
    except OSError:
        pass
    return HF_TOKEN_DEFAULT.strip()


# ----------------------------------------------------------------------------
# FAST, RESUMABLE, PARALLEL DOWNLOAD
# ----------------------------------------------------------------------------
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def resolve(url, token):
    """Follow redirects manually. Returns (final_url, total_size).
    The token is only sent to the original host, never to the CDN."""
    origin = urllib.parse.urlparse(url).netloc
    opener = urllib.request.build_opener(_NoRedirect)
    for _ in range(8):
        headers = {"User-Agent": "myai", "Range": "bytes=0-0"}
        if token and urllib.parse.urlparse(url).netloc == origin:
            headers["Authorization"] = f"Bearer {token}"
        try:
            resp = opener.open(urllib.request.Request(url, headers=headers), timeout=30)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                url = urllib.parse.urljoin(url, e.headers["Location"])
                continue
            raise
        with resp:
            cr = resp.headers.get("Content-Range", "")
        if "/" in cr and cr.split("/")[-1].isdigit():
            return url, int(cr.split("/")[-1])
        raise RuntimeError("Server does not support ranged downloads.")
    raise RuntimeError("Too many redirects.")


class Progress:
    def __init__(self, total, done):
        self.total = total
        self.done = done
        self.lock = threading.Lock()

    def add(self, n):
        with self.lock:
            self.done += n


def fetch_segment(idx, start, end, seg_path, get_url, prog):
    length = end - start + 1
    failures = 0
    refresh = False
    while True:
        have = os.path.getsize(seg_path) if os.path.exists(seg_path) else 0
        if have >= length:
            return
        try:
            url = get_url(refresh)
            refresh = False
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "myai", "Range": f"bytes={start + have}-{end}"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp, open(seg_path, "ab") as f:
                if resp.status != 206:
                    raise RuntimeError("Server ignored the range request.")
                remaining = length - have
                while remaining > 0:
                    block = resp.read(min(256 * 1024, remaining))
                    if not block:
                        break
                    f.write(block)
                    remaining -= len(block)
                    prog.add(len(block))
            failures = 0
        except Exception:
            failures += 1
            refresh = True  # signed CDN links can expire; get a fresh one
            if failures > 12:
                raise
            time.sleep(min(2 * failures, 15))


def parallel_download(url, token, dest, conns):
    final_url, total = resolve(url, token)
    lock = threading.Lock()
    state = {"url": final_url}

    def get_url(refresh):
        if refresh:
            with lock:
                try:
                    state["url"], _ = resolve(url, token)
                except Exception:
                    pass
        return state["url"]

    conns = max(1, min(conns, 32))
    size = -(-total // conns)  # ceil division
    segs = []
    for i in range(conns):
        s = i * size
        e = min(total - 1, s + size - 1)
        if s <= e:
            segs.append((i, s, e, f"{dest}.seg{conns}_{i}"))

    already = sum(os.path.getsize(p) for _, _, _, p in segs if os.path.exists(p))
    prog = Progress(total, already)

    with ThreadPoolExecutor(max_workers=len(segs)) as pool:
        futures = [
            pool.submit(fetch_segment, i, s, e, p, get_url, prog) for i, s, e, p in segs
        ]
        t0, base = time.time(), already
        while not all(f.done() for f in futures):
            time.sleep(0.5)
            speed = (prog.done - base) / max(time.time() - t0, 0.1) / 1e6
            print(
                f"\r  {prog.done / 1e6:,.0f} / {total / 1e6:,.0f} MB "
                f"({prog.done * 100 // total}%)  {speed:.1f} MB/s   ",
                end="", flush=True,
            )
        for f in futures:
            f.result()  # raises if a segment failed
    print()

    say("  Assembling file...")
    tmp = dest + ".tmp"
    with open(tmp, "wb") as out:
        for _, _, _, p in segs:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out, 1 << 20)
    if os.path.getsize(tmp) != total:
        os.remove(tmp)
        raise RuntimeError("Assembled file has the wrong size.")
    os.replace(tmp, dest)
    for _, _, _, p in segs:
        try:
            os.remove(p)
        except OSError:
            pass


def get_model():
    if os.path.exists(MODEL_FILE) and os.path.getsize(MODEL_FILE) > 500_000_000:
        return
    say("[3/4] Downloading the model (about 1 GB, one time only)...")
    token = get_token()
    endpoints = []
    custom = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")
    if custom:
        endpoints.append(custom)
    endpoints += [e for e in HF_ENDPOINTS if e not in endpoints]

    last_err = None
    for ep in endpoints:
        url = f"{ep}/{MODEL_REPO}/resolve/main/{MODEL_FILENAME}"
        say(f"  Trying {ep} ...")
        # Only send the token to the official Hugging Face host.
        is_hf = urllib.parse.urlparse(ep).netloc == "huggingface.co"
        attempts = [token, ""] if (token and is_hf) else [""]
        for tk in attempts:
            try:
                parallel_download(url, tk, MODEL_FILE, CONNECTIONS)
                return
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (401, 403) and tk:
                    say("  Token was rejected, retrying without it...")
                    continue
                say(f"\n  Failed on {ep}: {e}")
                break
            except Exception as e:
                last_err = e
                say(f"\n  Failed on {ep}: {e}")
                break
    say(f"\nDownload problem: {last_err}")
    say("Check your internet/VPN and run the same command again (it resumes).")
    sys.exit(1)


# ----------------------------------------------------------------------------
# INSTALL STEPS
# ----------------------------------------------------------------------------
def make_venv():
    if os.path.exists(VENV_PY):
        return
    say("[1/4] Creating private Python environment...")
    try:
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    except Exception as e:
        say(f"\nCould not create the environment: {e}")
        say("On Debian/Ubuntu run:  sudo apt install python3-venv  and try again.")
        sys.exit(1)


def install_engine():
    check = subprocess.run(
        [VENV_PY, "-c", "import llama_cpp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if check.returncode == 0:
        return
    say("[2/4] Installing the model engine (llama-cpp-python)...")
    common = [
        VENV_PY, "-m", "pip", "install", "--quiet", "--prefer-binary",
        "--default-timeout", "60", "--retries", "10", "llama-cpp-python",
    ]
    attempts = [
        common + ["--extra-index-url", WHEEL_INDEX],
        common + ["--extra-index-url", WHEEL_INDEX],  # retry once more
        common,  # plain PyPI (may need a compiler)
    ]
    for cmd in attempts:
        if subprocess.call(cmd) == 0:
            return
        time.sleep(3)
    say("\nEngine install failed. Usually this means no prebuilt file exists")
    say("for your system and a C++ compiler is needed. Try again after")
    say("installing one (Xcode tools on Mac, build-essential on Linux,")
    say("Visual Studio Build Tools on Windows), or check your internet.")
    sys.exit(1)


def _on_path(directory):
    target = os.path.normcase(os.path.normpath(directory))
    for p in os.environ.get("PATH", "").split(os.pathsep):
        if p and os.path.normcase(os.path.normpath(os.path.expandvars(p))) == target:
            return True
    return False


def add_to_path_unix():
    if UNIX_BIN_DIR in os.environ.get("PATH", "").split(os.pathsep):
        return False
    line = f'export PATH="{UNIX_BIN_DIR}:$PATH"'
    rcs = [os.path.join(HOME, n) for n in (".bashrc", ".zshrc", ".profile")]
    if "zsh" in os.environ.get("SHELL", ""):
        open(rcs[1], "a").close()  # make sure .zshrc exists on macOS
    targets = [p for p in rcs if os.path.exists(p)] or [rcs[2]]
    for rc in targets:
        with open(rc, "a+") as f:
            f.seek(0)
            if line not in f.read():
                f.write(f"\n# added by myai\n{line}\n")
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
    return True


def make_trigger():
    """Creates the command. Returns True if a NEW terminal is needed."""
    say(f"[4/4] Creating the '{TRIGGER}' command...")
    if IS_WIN:
        content = f'@echo off\r\n"{VENV_PY}" "{SCRIPT_COPY}" chat %*\r\n'
        # Best: a folder that is already on PATH -> works instantly, same window.
        if WIN_APPS_DIR and os.path.isdir(WIN_APPS_DIR) and _on_path(WIN_APPS_DIR):
            try:
                with open(
                    os.path.join(WIN_APPS_DIR, TRIGGER + ".cmd"), "w", newline=""
                ) as f:
                    f.write(content)
                return False
            except OSError:
                pass
        # Fallback: own folder + user PATH (needs a new terminal window).
        os.makedirs(WIN_BIN_DIR, exist_ok=True)
        with open(os.path.join(WIN_BIN_DIR, TRIGGER + ".cmd"), "w", newline="") as f:
            f.write(content)
        add_to_path_windows()
        return True
    os.makedirs(UNIX_BIN_DIR, exist_ok=True)
    path = os.path.join(UNIX_BIN_DIR, TRIGGER)
    with open(path, "w") as f:
        f.write(f'#!/bin/sh\nexec "{VENV_PY}" "{SCRIPT_COPY}" chat "$@"\n')
    os.chmod(path, 0o755)
    return add_to_path_unix()


def install():
    say(f"\n=== Setting up '{TRIGGER}' ===\n")
    os.makedirs(APP_DIR, exist_ok=True)
    try:
        if os.path.abspath(__file__) != os.path.abspath(SCRIPT_COPY):
            shutil.copyfile(__file__, SCRIPT_COPY)
    except NameError:
        say("Save this script to a file first, then run it.")
        sys.exit(1)
    make_venv()
    install_engine()
    get_model()
    needs_new_terminal = make_trigger()

    say("\n=====================================================")
    say(" DONE. Everything is downloaded and ready.")
    if needs_new_terminal:
        say(f" Open a NEW terminal (or restart VS Code) and type:  {TRIGGER}")
    else:
        say(f" Just type:  {TRIGGER}   (works right now, in this window)")
    say(" From now on it runs fully offline.")
    say("=====================================================\n")


# ----------------------------------------------------------------------------
# CHAT
# ----------------------------------------------------------------------------
def chat():
    if not os.path.exists(MODEL_FILE):
        say(f"Model not found. Run:  python3 {SCRIPT_COPY}  to set up first.")
        sys.exit(1)

    # Make sure we run inside the private environment
    if os.path.abspath(sys.prefix) != os.path.abspath(VENV_DIR) and os.path.exists(VENV_PY):
        sys.exit(subprocess.call([VENV_PY, os.path.abspath(__file__), "chat"]))

    from llama_cpp import Llama

    say("Loading model, please wait...")
    llm = Llama(
        model_path=MODEL_FILE,
        n_ctx=4096,
        n_threads=os.cpu_count() or 4,
        verbose=False,
    )

    say("\n" + "=" * 52)
    say(f" {TRIGGER} - offline coding assistant")
    say(" Type your question and press Enter.")
    say(" /clear = new conversation   /exit = quit   Ctrl+C = stop answer")
    say("=" * 52)

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            q = input("\nyou > ").strip()
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
        # Keep memory short; always keep the system prompt and start on a
        # user turn so roles keep alternating correctly.
        if len(history) > 12:
            history = history[:1] + history[-11:]

        reply = ""
        print("\nai  > ", end="", flush=True)
        try:
            for chunk in llm.create_chat_completion(
                messages=history, max_tokens=1024, temperature=0.2, stream=True
            ):
                tok = chunk["choices"][0]["delta"].get("content")
                if tok:
                    print(tok, end="", flush=True)
                    reply += tok
        except KeyboardInterrupt:
            print("\n[stopped]", end="")
        except ValueError:
            history = history[:1]
            print("\n[too long - conversation reset, please ask again]", end="")
            continue
        print()
        history.append({"role": "assistant", "content": reply})


# ----------------------------------------------------------------------------
# UNINSTALL
# ----------------------------------------------------------------------------
def uninstall():
    shutil.rmtree(APP_DIR, ignore_errors=True)
    for p in (
        os.path.join(UNIX_BIN_DIR, TRIGGER),
        os.path.join(WIN_BIN_DIR, TRIGGER + ".cmd"),
        os.path.join(WIN_APPS_DIR, TRIGGER + ".cmd") if WIN_APPS_DIR else "",
    ):
        if p and os.path.exists(p):
            os.remove(p)
    say("Removed everything myai created.")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "install"
    if cmd == "chat":
        chat()
    elif cmd == "uninstall":
        uninstall()
    else:
        install()


if __name__ == "__main__":
    main()