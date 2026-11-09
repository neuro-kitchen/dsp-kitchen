// Copies the playground's committed figures into public/course/ before a build or dev run, so the
// Course pages show exactly what the scripts drew (public/course/ is git-ignored).
import { cp, mkdir, readdir } from 'node:fs/promises';

const playground = new URL('../../playground/', import.meta.url);
const target = new URL('../public/course/', import.meta.url);
let copied = 0;
for (const entry of await readdir(playground, { withFileTypes: true })) {
  if (!entry.isDirectory() || !/^\d\d_/.test(entry.name)) continue;
  const figures = new URL(`${entry.name}/figures/`, playground);
  let files;
  try {
    files = await readdir(figures);
  } catch {
    continue;
  }
  const out = new URL(`${entry.name}/`, target);
  await mkdir(out, { recursive: true });
  for (const file of files.filter((f) => f.endsWith('.png'))) {
    await cp(new URL(file, figures), new URL(file, out));
    copied += 1;
  }
}
// The app's screenshot, also shown in the repository's README
const img = new URL('../public/img/', import.meta.url);
await mkdir(img, { recursive: true });
await cp(new URL('../../assets/img/dsp-app_main-darkmode.png', import.meta.url), new URL('dsp-app_main-darkmode.png', img));
// The committed benchmark figures, for the Validation pages
const results = new URL('../../benchmarks/signal-operations/results/', import.meta.url);
const benchmarks = new URL('../public/benchmarks/', import.meta.url);
await mkdir(benchmarks, { recursive: true });
for (const file of (await readdir(results)).filter((f) => f.endsWith('.png'))) await cp(new URL(file, results), new URL(file, benchmarks));
console.log(`figures: ${copied} copied from the playground`);
