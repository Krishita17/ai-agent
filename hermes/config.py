"""Paths, settings and the local secret vault.

Everything Hermes owns lives under ~/.hermes so uninstalling is `rm -rf ~/.hermes`.
API keys are encrypted at rest with a keystream derived from a 0600 machine
secret. This protects keys from casual disclosure (backups, sync folders, other
users); it is not protection against an attacker who already has your login.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path

HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
DB_PATH = HOME / "hermes.db"
SECRET_PATH = HOME / "secret.key"
WORKSPACE = HOME / "workspace"
LOGS = HOME / "logs"
WEB_DIR = Path(__file__).parent / "web"

DEFAULT_PORT = 4317


def ensure_dirs() -> None:
    for d in (HOME, WORKSPACE, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    HOME.chmod(0o700)


def _secret() -> bytes:
    ensure_dirs()
    if not SECRET_PATH.exists():
        SECRET_PATH.write_bytes(secrets.token_bytes(32))
        SECRET_PATH.chmod(0o600)
    return SECRET_PATH.read_bytes()


# Written on every new secret. Blobs without it use the original derivation,
# which shared one key between the keystream and the tag — see _mac below.
V2 = "v2:"

# Domain separation. Without it the keystream and the tag are the same function
# of the same key, and for one ciphertext length the tag input is byte-for-byte
# a keystream input. Not reachable here — the nonce is ours and decrypt is not
# an oracle — but two different jobs should never share a derivation.
_STREAM_LABEL = b"hermes/vault/stream\x00"
_MAC_LABEL = b"hermes/vault/mac\x00"


def _keystream(nonce: bytes, length: int, label: bytes = b"") -> bytes:
    """HMAC-SHA256 in counter mode. Stdlib-only stream cipher."""
    key = _secret()
    out = bytearray()
    counter = 0
    while len(out) < length:
        out += hmac.new(key, label + nonce + counter.to_bytes(8, "big"),
                        hashlib.sha256).digest()
        counter += 1
    return bytes(out[:length])


def _mac(nonce: bytes, cipher: bytes, label: bytes = b"") -> bytes:
    return hmac.new(_secret(), label + nonce + cipher, hashlib.sha256).digest()[:16]


class VaultError(RuntimeError):
    """The blob is there but this machine's secret cannot open it."""


def encrypt(plaintext: str) -> str:
    if not plaintext:
        return ""
    raw = plaintext.encode()
    nonce = secrets.token_bytes(16)
    cipher = bytes(a ^ b for a, b in zip(raw, _keystream(nonce, len(raw), _STREAM_LABEL)))
    return V2 + base64.b64encode(nonce + _mac(nonce, cipher, _MAC_LABEL) + cipher).decode()


def _open(blob: str) -> str:
    """Decrypt or raise. Reads both the current and the original format."""
    labelled = blob.startswith(V2)
    stream_label = _STREAM_LABEL if labelled else b""
    mac_label = _MAC_LABEL if labelled else b""
    body = blob[len(V2):] if labelled else blob
    try:
        data = base64.b64decode(body, validate=True)
    except Exception:
        raise VaultError("the stored value is not valid base64") from None
    if len(data) < 32:
        raise VaultError("the stored value is too short to be a vault blob")
    nonce, tag, cipher = data[:16], data[16:32], data[32:]
    if not hmac.compare_digest(tag, _mac(nonce, cipher, mac_label)):
        raise VaultError(
            "it does not match this machine's secret — secret.key has been "
            "replaced, or ~/.hermes was copied from another machine")
    try:
        return bytes(a ^ b for a, b in
                     zip(cipher, _keystream(nonce, len(cipher), stream_label))).decode()
    except UnicodeDecodeError:
        raise VaultError("it decrypted to something that is not text") from None


def decrypt(blob: str) -> str:
    """Best-effort read. Empty for anything unreadable — use vault_state() to
    tell 'nothing stored' from 'stored but this machine cannot open it'."""
    if not blob:
        return ""
    try:
        return _open(blob)
    except VaultError:
        return ""


def vault_state(blob: str) -> tuple:
    """(state, detail) where state is 'empty', 'ok' or 'unreadable'.

    These used to be one empty string, so a replaced secret.key looked exactly
    like a key that had never been set: the console said "No API key set", the
    operator re-entered it, and silently overwrote a key that was never lost.
    """
    if not blob:
        return "empty", ""
    try:
        _open(blob)
        return "ok", ""
    except VaultError as e:
        return "unreadable", str(e)


def mask(value: str) -> str:
    """Never show a full key back to the browser."""
    if not value:
        return ""
    if len(value) <= 10:
        return "*" * len(value)
    return f"{value[:4]}{'*' * 8}{value[-4:]}"


DEFAULTS = {
    "brand.name": "HERMES",
    "brand.engine": "OpenClaw",
    "default.provider": "ollama",
    "default.model": "qwen2.5:latest",
    "judge.provider": "",
    "judge.model": "",
    "custom.base_url": "",
    "safety.shell_timeout": "60",
    "safety.max_steps": "18",
    "safety.max_cost_per_run": "1.00",
    "safety.max_cost_per_day": "10.00",
    "verify.enabled": "1",
    "workforce.enabled": "1",
    "email.address": "",
    "email.imap_host": "",
    "email.imap_port": "993",
    "email.smtp_host": "",
    "email.smtp_port": "465",
    "email.allowed_recipients": "",
    "server.allowed_hosts": "",
    "server.trusted_proxy": "",
    "retention.event_days": "14",
}


def bootstrap_env() -> dict:
    """API keys already exported in the shell are picked up as a fallback."""
    return {
        "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
        "openai": os.environ.get("OPENAI_API_KEY", ""),
        "groq": os.environ.get("GROQ_API_KEY", ""),
        "gemini": os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", ""),
    }
