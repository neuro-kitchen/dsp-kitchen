// Makes the page cross-origin isolated where the host cannot send the headers (GitHub Pages):
// every response of this origin gets Cross-Origin-Opener-Policy and -Embedder-Policy, which a
// page needs for threads (shared memory). Registered by the loader in index.html, which reloads
// the page once so the page itself comes through this worker. Only this page's own files pass
// through it: the app loads nothing from other origins.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.cache === "only-if-cached" && request.mode !== "same-origin") return;
  event.respondWith(
    fetch(request).then((response) => {
      if (response.status === 0) return response;
      const headers = new Headers(response.headers);
      headers.set("Cross-Origin-Embedder-Policy", "require-corp");
      headers.set("Cross-Origin-Opener-Policy", "same-origin");
      return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
    }),
  );
});
