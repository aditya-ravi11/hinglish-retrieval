/**
 * Hinglish query collection form, served as an Apps Script web app.
 * Rows are appended to the "rows" tab of the Google Sheet this script is
 * attached to. Nothing personal is stored: a random id from the visitor's
 * browser, what they typed, and timestamps.
 */
var APP_VERSION = 'gas-0.1.0';
var CONSENT_VERSION = '2026-11-13';
var HEADERS = [
  'type', 'anon_id', 'source_id', 'rewrite_json', 'skipped', 'shown_at', 'at',
  'app_version', 'consent_version', 'typing_freq', 'devanagari_reading',
];
var DEVANAGARI = /[ऀ-ॿ]/;
var ANON_RE = /^[0-9a-f]{32}$/;
var MAX_LEN = 300;
// Sessions that must not count toward assignment (kept in sync with
// data/excluded_anon_ids.txt): deployment smoke test and the project owner's
// two browsers.
var EXCLUDED_ANON = [
  'ceb72dfbfe0bfecdcd257c9bd0bddd02',
  'f8a96a38c5afd7e9ecef2bb26675be37',
  'ed252411a2149a8e529988341265fd64',
];
var _sources = null;

function doGet() {
  return HtmlService.createHtmlOutputFromFile('Index')
    .setTitle('Hinglish queries')
    .addMetaTag('viewport', 'width=device-width, initial-scale=1');
}

function sources_() {
  if (!_sources) {
    _sources = SOURCES.map(function (r) {
      return { id: r[0], lang: r[1], text: r[2], priority: r[3] };
    });
  }
  return _sources;
}

function sheet_() {
  var ss = SpreadsheetApp.getActive();
  var sh = ss.getSheetByName('rows');
  if (!sh) {
    sh = ss.insertSheet('rows');
    // plain text everywhere, so Sheets never turns input into dates or formulas
    sh.getRange(1, 1, sh.getMaxRows(), HEADERS.length).setNumberFormat('@');
    sh.getRange(1, 1, 1, HEADERS.length).setValues([HEADERS]);
    sh.setFrozenRows(1);
  }
  return sh;
}

function readRows_(sh) {
  var n = sh.getLastRow();
  if (n < 2) return [];
  return sh.getRange(2, 1, n - 1, HEADERS.length).getValues().map(function (r) {
    return { type: r[0], anon_id: r[1], source_id: r[2], skipped: r[4] === 'true', at: Date.parse(r[6]) };
  });
}

// Rows that count toward assignment. Excluded sessions still work normally
// (their consent is honoured), they just never fill or reserve a slot.
function counted_(rows) {
  return rows.filter(function (r) { return EXCLUDED_ANON.indexOf(r.anon_id) < 0; });
}

function append_(sh, rows, obj) {
  var row = HEADERS.map(function (h) {
    return obj[h] === undefined || obj[h] === null ? '' : String(obj[h]);
  });
  sh.getRange(sh.getLastRow() + 1, 1, 1, HEADERS.length).setNumberFormat('@').setValues([row]);
  rows.push({ type: obj.type, anon_id: obj.anon_id, source_id: obj.source_id,
              skipped: obj.skipped === true, at: Date.parse(obj.at) });
}

function locked_(fn) {
  var lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    return fn();
  } finally {
    lock.releaseLock();
  }
}

function checkAnon_(anon) {
  if (typeof anon !== 'string' || !ANON_RE.test(anon)) throw new Error('bad id');
}

function next_(sh, rows, anon) {
  var now = Date.now();
  // An excluded session sees the queue as everyone else does; its own rows
  // are added back so it does not get the same question twice.
  var view = counted_(rows);
  if (EXCLUDED_ANON.indexOf(anon) >= 0) {
    view = view.concat(rows.filter(function (r) { return r.anon_id === anon; }));
  }
  var s = chooseNext(sources_(), view, anon, now);
  var done = doneCount(rows, anon);
  if (!s) return { done: done, finished: true };
  var at = new Date(now).toISOString();
  append_(sh, rows, { type: 'shown', anon_id: anon, source_id: s.id, at: at, app_version: APP_VERSION });
  return { done: done, source: { id: s.id, lang: s.lang, text: s.text }, shownAt: at };
}

function hasConsent_(rows, anon) {
  return rows.some(function (r) { return r.type === 'consent' && r.anon_id === anon; });
}

/** Called once, when the visitor presses "I agree, start". */
function startSession(anon, freq, reading) {
  checkAnon_(anon);
  var allowedFreq = ['Daily', 'Sometimes', 'Rarely', 'Never', ''];
  var allowedRead = ['Easily', 'Slowly', 'Not really', ''];
  freq = allowedFreq.indexOf(freq) >= 0 ? freq : '';
  reading = allowedRead.indexOf(reading) >= 0 ? reading : '';
  return locked_(function () {
    var sh = sheet_();
    var rows = readRows_(sh);
    if (!hasConsent_(rows, anon)) {
      append_(sh, rows, {
        type: 'consent', anon_id: anon, at: new Date().toISOString(), app_version: APP_VERSION,
        consent_version: CONSENT_VERSION, typing_freq: freq, devanagari_reading: reading,
      });
    }
    return next_(sh, rows, anon);
  });
}

function getNext(anon) {
  checkAnon_(anon);
  return locked_(function () {
    var sh = sheet_();
    var rows = readRows_(sh);
    if (!hasConsent_(rows, anon)) return { needConsent: true };
    return next_(sh, rows, anon);
  });
}

function record_(anon, sourceId, shownAt, text, skipped) {
  checkAnon_(anon);
  var known = sources_().some(function (s) { return s.id === sourceId; });
  if (!known) throw new Error('unknown query');
  return locked_(function () {
    var sh = sheet_();
    var rows = readRows_(sh);
    if (!hasConsent_(rows, anon)) return { needConsent: true };
    var already = rows.some(function (r) {
      return r.type === 'rewrite' && r.anon_id === anon && r.source_id === sourceId;
    });
    if (!already) {
      append_(sh, rows, {
        type: 'rewrite', anon_id: anon, source_id: sourceId,
        // JSON keeps the text exactly as typed and stops Sheets reading it as a formula
        rewrite_json: skipped ? '' : JSON.stringify(text),
        skipped: skipped, shown_at: String(shownAt || ''), at: new Date().toISOString(),
        app_version: APP_VERSION,
      });
    }
    return next_(sh, rows, anon);
  });
}

function submitRewrite(anon, sourceId, shownAt, text) {
  if (typeof text !== 'string' || !text.trim()) return { error: 'Type something, or press Skip.' };
  if (DEVANAGARI.test(text)) return { error: 'Please use English letters (a to z) only.' };
  if (text.length > MAX_LEN) return { error: 'That is too long. Keep it to one question.' };
  return record_(anon, sourceId, shownAt, text, false);
}

function skipQuery(anon, sourceId, shownAt) {
  return record_(anon, sourceId, shownAt, '', true);
}
