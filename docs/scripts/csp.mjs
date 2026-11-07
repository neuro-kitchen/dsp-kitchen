// Adds a Content-Security-Policy <meta> to every built page (run after `astro build`).
//
// The website's policy, with what Starlight needs: each page's inline scripts are allowed by
// their hash, so only those run; 'wasm-unsafe-eval' lets the search (Pagefind, WebAssembly)
// start; inline styles are allowed because Starlight sets style attributes (sidebar depth, icon
// sizes), which hashes cannot cover. Astro's own CSP option misses Starlight's head script and
// those attributes, hence this step.
import { createHash } from 'node:crypto';
import { readdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const DIST = new URL('../dist/', import.meta.url).pathname;
const INLINE_SCRIPT = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g;

const policy = (scriptHashes) =>
  [
    "default-src 'self'",
    `script-src 'self' 'wasm-unsafe-eval' ${scriptHashes.join(' ')}`.trim(),
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data:",
    "font-src 'self'",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'none'",
    'upgrade-insecure-requests',
  ].join('; ');

async function* htmlFiles(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) yield* htmlFiles(path);
    else if (entry.name.endsWith('.html')) yield path;
  }
}

let pages = 0;
for await (const file of htmlFiles(DIST)) {
  const html = await readFile(file, 'utf8');
  const hashes = new Set();
  for (const [, body] of html.matchAll(INLINE_SCRIPT)) {
    if (body.length) hashes.add(`'sha256-${createHash('sha256').update(body).digest('base64')}'`);
  }
  const meta = `<meta http-equiv="content-security-policy" content="${policy([...hashes])}">`;
  // First in <head>: a policy only covers what comes after it.
  const out = html.replace(/<head>/, `<head>${meta}`);
  if (out === html) throw new Error(`${file}: no <head> to put the policy in`);
  await writeFile(file, out);
  pages += 1;
}
console.log(`csp: policy added to ${pages} page(s)`);
