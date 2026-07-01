#!/usr/bin/env python3
"""
pwa_patch.py - inject PWA tags into Streamlit's served index.html.

Streamlit has no supported way to add <link rel="manifest">, theme-color
meta, or a service-worker <script> to the real document <head> — HTML
injected via st.markdown(unsafe_allow_html=True) only lands in a body div,
and injected <script> tags don't execute (browsers block script execution
from innerHTML). The only document that's actually the top-level page is
Streamlit's own built index.html inside the installed package.

This patches that file in place, idempotently (safe to call every run —
checks for a marker before touching anything). Needed again only if the
venv is recreated or Streamlit is reinstalled/upgraded, which is why
dashboard.py calls ensure_pwa_patch() on every startup instead of relying
on a one-time manual step.
"""

from pathlib import Path

MARKER = "<!-- PWA_PATCH -->"

PWA_HEAD_SNIPPET = """<!-- PWA_PATCH -->
<link rel="manifest" href="/app/static/manifest.json">
<meta name="theme-color" content="#0e1117">
<link rel="apple-touch-icon" href="/app/static/icon-192.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<script>
  if ("serviceWorker" in navigator) {
    window.addEventListener("load", function () {
      navigator.serviceWorker.register("/app/static/sw.js").catch(function () {});
    });
  }
</script>
"""


def find_streamlit_index_html() -> Path | None:
    try:
        import streamlit
    except ImportError:
        return None
    candidate = Path(streamlit.__file__).parent / "static" / "index.html"
    return candidate if candidate.exists() else None


def ensure_pwa_patch() -> bool:
    """Patch Streamlit's index.html if not already patched. Returns True if patched or already OK."""
    index_html = find_streamlit_index_html()
    if index_html is None:
        return False

    content = index_html.read_text(encoding="utf-8")
    if MARKER in content:
        return True  # already patched

    if "</head>" not in content:
        return False

    patched = content.replace("</head>", PWA_HEAD_SNIPPET + "</head>")
    index_html.write_text(patched, encoding="utf-8")
    return True


if __name__ == "__main__":
    ok = ensure_pwa_patch()
    print("PWA patch applied." if ok else "PWA patch FAILED — check streamlit install path.")
