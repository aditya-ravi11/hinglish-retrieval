// Run: node --test tests/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chooseNext, doneCount, ASSIGN } = require('../form_gas/Assign.js');

const make = (n) => Array.from({ length: n }, (_, i) => ({ id: `q${i}`, lang: 'hi', text: `t${i}`, priority: i }));

// Simulates the server: every choice is logged as a "shown" row.
function serve(sources, rows, anon, now) {
  const s = chooseNext(sources, rows, anon, now);
  if (s) rows.push({ type: 'shown', anon_id: anon, source_id: s.id, at: now });
  return s;
}
const answer = (rows, anon, sid, now, skipped = false) =>
  rows.push({ type: 'rewrite', anon_id: anon, source_id: sid, skipped, at: now });

test('never repeats for the same annotator', () => {
  const src = make(3), rows = [], seen = [];
  let s;
  while ((s = serve(src, rows, 'alice', 0))) { seen.push(s.id); answer(rows, 'alice', s.id, 0); }
  assert.equal(new Set(seen).size, 3);
  assert.equal(doneCount(rows, 'alice'), 3);
});

test('completes pairs first', () => {
  const src = make(4), rows = [];
  const s = serve(src, rows, 'alice', 0);
  answer(rows, 'alice', s.id, 0);
  assert.equal(serve(src, rows, 'bob', 1).id, s.id);
});

test('cap rises to 3 only after every query has 2', () => {
  const src = make(2), rows = [];
  for (const who of ['a', 'b', 'c', 'd']) {
    let s;
    while ((s = serve(src, rows, who, 0))) answer(rows, who, s.id, 0);
  }
  for (const id of ['q0', 'q1']) {
    const n = new Set(rows.filter((r) => r.type === 'rewrite' && r.source_id === id).map((r) => r.anon_id)).size;
    assert.equal(n, 3);
  }
});

test('reservation blocks overfill and expires', () => {
  const src = make(1), rows = [];
  answer(rows, 'x', 'q0', 0);
  assert.equal(serve(src, rows, 'alice', 0).id, 'q0');
  assert.equal(serve(src, rows, 'bob', 1), null);
  assert.equal(serve(src, rows, 'bob', ASSIGN.RESERVE_MS + 1).id, 'q0');
});

test('reload returns the same query', () => {
  const src = make(3), rows = [];
  const first = serve(src, rows, 'alice', 0);
  assert.equal(serve(src, rows, 'alice', 5).id, first.id);
});

test('retired after three skips', () => {
  const src = make(1), rows = [];
  for (const who of ['a', 'b', 'c']) { const s = serve(src, rows, who, 0); answer(rows, who, s.id, 0, true); }
  assert.equal(serve(src, rows, 'd', 0), null);
});
