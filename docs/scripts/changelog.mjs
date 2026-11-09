// The Changelog page, written from the repository's CHANGELOG.md before every build and dev run:
// CHANGELOG.md stays the one source, and this page (git-ignored) can't drift from it.
import { mkdir, readFile, writeFile } from 'node:fs/promises';

const source = new URL('../../CHANGELOG.md', import.meta.url);
const folder = new URL('../src/content/docs/changelog/', import.meta.url);

const text = await readFile(source, 'utf8');
// The first line is the file's own title; the page has its title in the frontmatter.
const body = text.replace(/^# Changelog\s*\n/, '').trim();
const releases = 'https://github.com/neuro-kitchen/dsp-kitchen/releases';
// `<` outside inline code would be read as HTML
const escaped = body.split(/(`[^`\n]*`)/).map((part) => (part.startsWith('`') ? part : part.replace(/</g, '&lt;'))).join('');
const page = `---
title: Changelog
description: "What changed in each release of dsp-kitchen."
editUrl: https://github.com/neuro-kitchen/dsp-kitchen/edit/main/CHANGELOG.md
tableOfContents:
  minHeadingLevel: 2
  maxHeadingLevel: 2
---

Each version's downloads are on [GitHub Releases](${releases}).

${escaped}
`;
await mkdir(folder, { recursive: true });
await writeFile(new URL('index.md', folder), page);
console.log('changelog: page written from CHANGELOG.md');
