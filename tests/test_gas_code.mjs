// Runs form_gas/Code.js against an in-memory fake of the Sheets and Lock APIs.
// Run: node --test tests/test_gas_code.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const dir = new URL('../form_gas/', import.meta.url);
const read = (f) => readFileSync(new URL(f, dir), 'utf8');

function fakeSheet() {
  const cells = []; // array of row arrays (row 1 = headers)
  const range = (r0, c0, nr, nc) => ({
    setNumberFormat() { return this; },
    setValues(v) { v.forEach((row, i) => { cells[r0 - 1 + i] = row.slice(); }); return this; },
    getValues() { return cells.slice(r0 - 1, r0 - 1 + nr).map((r) => r.slice(c0 - 1, c0 - 1 + nc)); },
  });
  return {
    cells,
    getLastRow: () => cells.length,
    getMaxRows: () => 1000,
    getRange: range,
    setFrozenRows() {},
  };
}

function load(excluded) {
  const sheets = {};
  const ctx = {
    SpreadsheetApp: { getActive: () => ({
      getSheetByName: (n) => sheets[n] || null,
      insertSheet: (n) => (sheets[n] = fakeSheet()),
    }) },
    LockService: { getScriptLock: () => ({ waitLock() {}, releaseLock() {} }) },
    module: undefined,
  };
  vm.createContext(ctx);
  // a small source list instead of the 1,300 real ones
  vm.runInContext('var SOURCES = [["q0","hi","a",0],["q1","hi","b",1],["q2","hi","c",2],["q3","hi","d",3]];', ctx);
  vm.runInContext(read('Assign.js'), ctx);
  vm.runInContext(read('Code.js'), ctx);
  if (excluded) vm.runInContext(`EXCLUDED_ANON = ${JSON.stringify(excluded)};`, ctx);
  return { ctx, sheets };
}

const A = 'a'.repeat(32), B = 'b'.repeat(32), X = 'f'.repeat(32);

test('skip keeps a normal visitor on the task screen', () => {
  const { ctx } = load([X]);
  const first = ctx.startSession(A, 'Daily', '');
  const after = ctx.skipQuery(A, first.source.id, first.shownAt);
  assert.equal(after.needConsent, undefined);
  assert.ok(after.source);
  assert.notEqual(after.source.id, first.source.id);
});

test('excluded session can submit and skip without being sent back to consent', () => {
  const { ctx } = load([X]);
  let r = ctx.startSession(X, '', '');
  r = ctx.skipQuery(X, r.source.id, r.shownAt);
  assert.equal(r.needConsent, undefined);
  r = ctx.submitRewrite(X, r.source.id, r.shownAt, 'kya haal hai');
  assert.equal(r.needConsent, undefined);
  assert.equal(r.done, 2);
});

test('excluded session never fills or reserves a slot for real visitors', () => {
  const { ctx } = load([X]);
  let x = ctx.startSession(X, '', '');
  ctx.submitRewrite(X, x.source.id, x.shownAt, 'test');
  // real visitor A still starts at q0, the question X answered
  const a = ctx.startSession(A, '', '');
  assert.equal(a.source.id, 'q0');
  ctx.submitRewrite(A, a.source.id, a.shownAt, 'kya');
  // B completes the pair on q0: X's answer did not count toward the two
  const b = ctx.startSession(B, '', '');
  assert.equal(b.source.id, 'q0');
});

test('validation rejects empty and Devanagari answers', () => {
  const { ctx } = load([]);
  const r = ctx.startSession(A, '', '');
  assert.ok(ctx.submitRewrite(A, r.source.id, r.shownAt, '   ').error);
  assert.ok(ctx.submitRewrite(A, r.source.id, r.shownAt, 'क्या').error);
});
