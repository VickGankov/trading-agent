// Minimal service worker — exists to satisfy PWA installability.
// Deliberately does NOT cache live pages/data: this dashboard shows
// account balances and positions, so serving anything stale from cache
// would be actively misleading. Only the static icons/manifest (which
// never change) are cached; everything else always goes to network.

const SHELL_CACHE = "ta-shell-v1";
const SHELL_ASSETS = [
  "/app/static/icon-192.png",
  "/app/static/icon-512.png",
  "/app/static/manifest.json",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE).then((cache) => cache.addAll(SHELL_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== SHELL_CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  const isShellAsset = SHELL_ASSETS.some((path) => url.pathname === path);

  if (isShellAsset) {
    event.respondWith(
      caches.match(event.request).then((cached) => cached || fetch(event.request))
    );
    return;
  }
  // Everything else (the app itself, websocket, live data) — always network.
  event.respondWith(fetch(event.request));
});
