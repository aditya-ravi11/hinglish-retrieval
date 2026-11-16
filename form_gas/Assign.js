/**
 * Which source query an annotator sees next. Pure functions, no Apps Script
 * APIs, so tests/test_gas_assign.mjs can run them in Node.
 *
 * Same rules as form/assign.py:
 *  - nobody sees a query twice (rewritten or skipped)
 *  - fuller queries first, so pairs of rewrites complete early; ties go to
 *    the fixed random priority
 *  - 2 rewrites per query from 2 different people; once every live query
 *    has 2, the cap rises to 3
 *  - a query skipped by 3 different people is retired
 *  - a query on someone's screen is reserved for 15 minutes ("shown" rows)
 *
 * rows: [{type: 'consent'|'shown'|'rewrite', anon_id, source_id, skipped, at (ms)}]
 * sources: [{id, lang, text, priority}]
 */
var ASSIGN = {
  TARGET: 2,
  MAX_CAP: 3,
  RETIRE_AFTER_SKIPS: 3,
  RESERVE_MS: 15 * 60 * 1000,
};

function assignState_(rows) {
  var doneBy = {}, rewrites = {}, skips = {}, shown = {};
  rows.forEach(function (r) {
    if (r.type === 'rewrite') {
      (doneBy[r.anon_id] = doneBy[r.anon_id] || {})[r.source_id] = true;
      var bucket = r.skipped ? skips : rewrites;
      (bucket[r.source_id] = bucket[r.source_id] || {})[r.anon_id] = true;
    } else if (r.type === 'shown') {
      var s = (shown[r.source_id] = shown[r.source_id] || {});
      if (!(s[r.anon_id] >= r.at)) s[r.anon_id] = r.at;
    }
  });
  return { doneBy: doneBy, rewrites: rewrites, skips: skips, shown: shown };
}

function size_(o) {
  return o ? Object.keys(o).length : 0;
}

function chooseNext(sources, rows, anon, now) {
  var st = assignState_(rows);
  var mine = st.doneBy[anon] || {};

  function liveReservers(sid) {
    var out = [];
    var res = st.shown[sid] || {};
    Object.keys(res).forEach(function (a) {
      var done = st.doneBy[a] && st.doneBy[a][sid];
      if (!done && now - res[a] <= ASSIGN.RESERVE_MS) out.push(a);
    });
    return out;
  }

  // A query already on this person's screen comes back (page reload).
  var latest = null;
  sources.forEach(function (s) {
    var t = (st.shown[s.id] || {})[anon];
    if (t !== undefined && !mine[s.id] && now - t <= ASSIGN.RESERVE_MS && (!latest || t > latest.t)) {
      latest = { s: s, t: t };
    }
  });
  if (latest) return latest.s;

  var live = sources.filter(function (s) {
    return size_(st.skips[s.id]) < ASSIGN.RETIRE_AFTER_SKIPS;
  });
  var cap = ASSIGN.TARGET;
  if (live.length && live.every(function (s) { return size_(st.rewrites[s.id]) >= ASSIGN.TARGET; })) {
    cap = ASSIGN.MAX_CAP;
  }

  var best = null, bestLoad = -1;
  live.forEach(function (s) {
    if (mine[s.id]) return;
    var load = size_(st.rewrites[s.id]) + liveReservers(s.id).length;
    if (load >= cap) return;
    if (load > bestLoad || (load === bestLoad && s.priority < best.priority)) {
      best = s;
      bestLoad = load;
    }
  });
  return best;
}

function doneCount(rows, anon) {
  var st = assignState_(rows);
  return size_(st.doneBy[anon]);
}

if (typeof module !== 'undefined') {
  module.exports = { chooseNext: chooseNext, doneCount: doneCount, ASSIGN: ASSIGN };
}
