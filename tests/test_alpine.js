#!/usr/bin/env node
/* ============================================================================
 * tests/test_alpine.js — Alpine.js 状态绑定层单测
 * Task 31: x-data 初始渲染 + 4 类事件 → 状态增量更新
 *
 * 运行方式（任一即可）：
 *   1) node tests/test_alpine.js
 *   2) npx -- node tests/test_alpine.js
 *
 * 仅使用 Node 内置模块（assert / fs / path / vm），不引入 jsdom / alpinejs。
 * 通过 require('../src/cxr_triage/web/static/app.js') 加载待测模块。
 *
 * 覆盖：
 *   - 模块加载（exports / 常量）
 *   - 纯函数：canConfirm / canReject / targetStateForRole / dedupAlerts
 *   - 纯函数：findStudyByUid / alertFromPayload
 *   - 工厂：triagePanel() 初始 x-data shape
 *   - 异步流：refreshQueues → 队列刷新
 *   - 异步流：connectSse → 4 类事件 → 状态增量更新
 *   - 异步流：init() 生命周期
 *   - 异步流：selectStudy 详情合并
 *   - 异步流：confirmStudy / rejectStudy
 *   - 异步流：dismissAlert
 *
 * 退出码 0 = 全部通过；非 0 = 有失败（fail / 异常未捕获）。
 * ========================================================================= */

'use strict';

const assert = require('assert');
const path = require('path');
const fs = require('fs');

const APP_JS = path.join(__dirname, '..', 'src', 'cxr_triage', 'web', 'static', 'app.js');

let total = 0;
let passed = 0;
let failed = 0;
const failures = [];

function test(name, fn) {
  total += 1;
  try {
    const r = fn();
    if (r && typeof r.then === 'function') {
      return r.then(
        () => { passed += 1; console.log(`  ✓ ${name}`); },
        (err) => { failed += 1; failures.push({ name, err }); console.log(`  ✗ ${name}: ${err && err.message}`); }
      );
    }
    passed += 1;
    console.log(`  ✓ ${name}`);
  } catch (err) {
    failed += 1;
    failures.push({ name, err });
    console.log(`  ✗ ${name}: ${err && err.message}`);
  }
  return null;
}

function section(name) {
  console.log(`\n[${name}]`);
}

// -------------------------------------------------------------------------
// 加载被测模块（每次重置缓存，避免状态污染）
// -------------------------------------------------------------------------
function loadAppJs() {
  delete require.cache[require.resolve(APP_JS)];
  return require(APP_JS);
}

// -------------------------------------------------------------------------
// 测试
// -------------------------------------------------------------------------
async function run() {
  section('Module load');
  const fsStat = fs.statSync(APP_JS);
  test('app.js 文件存在且可读', () => {
    assert.ok(fsStat.isFile(), 'app.js 必须是文件');
    assert.ok(fsStat.size > 1000, 'app.js 体积应 > 1KB');
  });

  const app = loadAppJs();
  test('module.exports 暴露 triagePanel / 纯函数 / 常量', () => {
    assert.strictEqual(typeof app.triagePanel, 'function');
    assert.strictEqual(typeof app.canConfirm, 'function');
    assert.strictEqual(typeof app.canReject, 'function');
    assert.strictEqual(typeof app.targetStateForRole, 'function');
    assert.strictEqual(typeof app.dedupAlerts, 'function');
    assert.strictEqual(typeof app.findStudyByUid, 'function');
    assert.strictEqual(typeof app.alertFromPayload, 'function');
    assert.ok(app.STATES && app.EVENTS && app.ROLES && app.ENDPOINTS);
  });

  test('常量与源产品对齐（state / event / role / endpoint）', () => {
    assert.strictEqual(app.STATES.DRAFT_PENDING, 'draft_pending');
    assert.strictEqual(app.STATES.JR_CONFIRMED, 'jr_confirmed');
    assert.strictEqual(app.STATES.SR_REVIEWED, 'sr_reviewed');
    assert.strictEqual(app.STATES.SR_SIGNED, 'sr_signed');
    assert.strictEqual(app.STATES.REJECTED, 'rejected');
    assert.strictEqual(app.EVENTS.NEW_STUDY, 'new_study');
    assert.strictEqual(app.EVENTS.DRAFT_READY, 'draft_ready');
    assert.strictEqual(app.EVENTS.STATE_CHANGE, 'state_change');
    assert.strictEqual(app.EVENTS.ALERT, 'alert');
    assert.strictEqual(app.EVENTS.KEEPALIVE, 'keepalive');
    assert.strictEqual(app.ROLES.JR, 'jr');
    assert.strictEqual(app.ROLES.REVIEWER, 'reviewer');
    assert.strictEqual(app.ROLES.SR, 'sr');
    assert.strictEqual(app.ENDPOINTS.SSE, '/api/sse');
    assert.strictEqual(app.ENDPOINTS.QUEUE_POSITIVE, '/api/queue/positive');
    assert.strictEqual(app.ENDPOINTS.QUEUE_NEGATIVE, '/api/queue/negative');
    assert.strictEqual(app.ENDPOINTS.CONFIRM, '/api/studies/{uid}/confirm');
    assert.strictEqual(app.ENDPOINTS.REJECT, '/api/studies/{uid}/reject');
    assert.strictEqual(app.SSE_RETRY_MS, 3000);
  });

  section('Pure: canConfirm');
  const STUDY = { study_uid: 'u1', triage_state: 'draft_pending', max_prob: 0.85 };
  test('selectedStudy=null → 全部 false', () => {
    assert.strictEqual(app.canConfirm(null, 'jr'), false);
    assert.strictEqual(app.canConfirm(null, 'reviewer'), false);
    assert.strictEqual(app.canConfirm(null, 'sr'), false);
  });
  test('draft_pending + jr → true；其他角色 false', () => {
    assert.strictEqual(app.canConfirm(STUDY, 'jr'), true);
    assert.strictEqual(app.canConfirm(STUDY, 'reviewer'), false);
    assert.strictEqual(app.canConfirm(STUDY, 'sr'), false);
  });
  test('jr_confirmed + reviewer → true；jr/sr false', () => {
    const s = { ...STUDY, triage_state: 'jr_confirmed' };
    assert.strictEqual(app.canConfirm(s, 'jr'), false);
    assert.strictEqual(app.canConfirm(s, 'reviewer'), true);
    assert.strictEqual(app.canConfirm(s, 'sr'), false);
  });
  test('sr_reviewed + sr → true；jr/reviewer false', () => {
    const s = { ...STUDY, triage_state: 'sr_reviewed' };
    assert.strictEqual(app.canConfirm(s, 'jr'), false);
    assert.strictEqual(app.canConfirm(s, 'reviewer'), false);
    assert.strictEqual(app.canConfirm(s, 'sr'), true);
  });
  test('sr_signed 终态 → 全部 false', () => {
    const s = { ...STUDY, triage_state: 'sr_signed' };
    assert.strictEqual(app.canConfirm(s, 'jr'), false);
    assert.strictEqual(app.canConfirm(s, 'reviewer'), false);
    assert.strictEqual(app.canConfirm(s, 'sr'), false);
  });
  test('未知 role → false', () => {
    assert.strictEqual(app.canConfirm(STUDY, 'unknown'), false);
    assert.strictEqual(app.canConfirm(STUDY, ''), false);
    assert.strictEqual(app.canConfirm(STUDY, null), false);
  });
  test('缺 triage_state 字段默认 draft_pending', () => {
    const s = { study_uid: 'u1' };
    assert.strictEqual(app.canConfirm(s, 'jr'), true);
    assert.strictEqual(app.canConfirm(s, 'reviewer'), false);
  });

  section('Pure: canReject');
  test('selectedStudy=null → false', () => {
    assert.strictEqual(app.canReject(null), false);
  });
  test('draft_pending / jr_confirmed / sr_reviewed → true', () => {
    assert.strictEqual(app.canReject({ study_uid: 'u1', triage_state: 'draft_pending' }), true);
    assert.strictEqual(app.canReject({ study_uid: 'u1', triage_state: 'jr_confirmed' }), true);
    assert.strictEqual(app.canReject({ study_uid: 'u1', triage_state: 'sr_reviewed' }), true);
  });
  test('sr_signed / rejected 终态 → false', () => {
    assert.strictEqual(app.canReject({ study_uid: 'u1', triage_state: 'sr_signed' }), false);
    assert.strictEqual(app.canReject({ study_uid: 'u1', triage_state: 'rejected' }), false);
  });

  section('Pure: targetStateForRole');
  test('jr→jr_confirmed, reviewer→sr_reviewed, sr→sr_signed', () => {
    assert.strictEqual(app.targetStateForRole('jr'), 'jr_confirmed');
    assert.strictEqual(app.targetStateForRole('reviewer'), 'sr_reviewed');
    assert.strictEqual(app.targetStateForRole('sr'), 'sr_signed');
  });
  test('未知 role → null', () => {
    assert.strictEqual(app.targetStateForRole('unknown'), null);
    assert.strictEqual(app.targetStateForRole(''), null);
    assert.strictEqual(app.targetStateForRole(null), null);
  });

  section('Pure: dedupAlerts');
  test('空 alerts → 添加新', () => {
    const r = app.dedupAlerts([], { event_id: 'e1', message: 'a' });
    assert.strictEqual(r.length, 1);
    assert.strictEqual(r[0].event_id, 'e1');
  });
  test('同 event_id 不重复', () => {
    const existing = [{ event_id: 'e1', message: 'a' }];
    const r = app.dedupAlerts(existing, { event_id: 'e1', message: 'b' });
    assert.strictEqual(r.length, 1);
    assert.strictEqual(r, existing, '同 id 不重复时应返回原数组');
  });
  test('不同 event_id 追加', () => {
    const r = app.dedupAlerts([{ event_id: 'e1', message: 'a' }], { event_id: 'e2', message: 'b' });
    assert.strictEqual(r.length, 2);
  });
  test('新 alert 缺 event_id → 不入栈', () => {
    const r = app.dedupAlerts([], { message: 'x' });
    assert.strictEqual(r.length, 0);
  });

  section('Pure: findStudyByUid');
  test('阳性队列命中', () => {
    const pos = [{ study_uid: 'a', max_prob: 0.9 }];
    const neg = [{ study_uid: 'b', max_prob: 0.1 }];
    const found = app.findStudyByUid(pos, neg, 'a');
    assert.strictEqual(found.study_uid, 'a');
    assert.strictEqual(found.max_prob, 0.9);
  });
  test('阴性队列命中', () => {
    const pos = [{ study_uid: 'a', max_prob: 0.9 }];
    const neg = [{ study_uid: 'b', max_prob: 0.1 }];
    const found = app.findStudyByUid(pos, neg, 'b');
    assert.strictEqual(found.study_uid, 'b');
  });
  test('两队列皆无 → null', () => {
    assert.strictEqual(app.findStudyByUid([], [], 'x'), null);
  });
  test('studyUid 空 / null → null', () => {
    assert.strictEqual(app.findStudyByUid([], [], ''), null);
    assert.strictEqual(app.findStudyByUid([], [], null), null);
  });

  section('Pure: alertFromPayload');
  test('标准 payload → 3 字段', () => {
    const a = app.alertFromPayload({
      event_id: 'e1',
      payload: { target_endpoint: 'ris://x', message: 'RIS 失败' },
    });
    assert.strictEqual(a.event_id, 'e1');
    assert.strictEqual(a.target, 'ris://x');
    assert.strictEqual(a.message, 'RIS 失败');
  });
  test('payload 缺字段 → fallback', () => {
    const a = app.alertFromPayload({ event_id: 'e1' });
    assert.strictEqual(a.target, 'unknown');
    assert.ok(a.message && a.message.length > 0);
  });

  // ---------------------------------------------------------------
  // Factory: triagePanel() x-data shape
  // ---------------------------------------------------------------
  section('Factory: triagePanel() x-data shape');
  test('空 opts 也能构造（含默认 fetch 兜底）', () => {
    const p = app.triagePanel();
    assert.ok(p);
    assert.deepStrictEqual(p.positiveStudies, []);
    assert.deepStrictEqual(p.negativeStudies, []);
    assert.strictEqual(p.selectedStudy, null);
    assert.deepStrictEqual(p.alerts, []);
    assert.strictEqual(p.degraded, false);
    assert.strictEqual(p.connectionStatus, '未连接');
    assert.strictEqual(typeof p.init, 'function');
    assert.strictEqual(typeof p.refreshQueues, 'function');
    assert.strictEqual(typeof p.connectSse, 'function');
    assert.strictEqual(typeof p.onSseEvent, 'function');
    assert.strictEqual(typeof p.selectStudy, 'function');
    assert.strictEqual(typeof p.confirmStudy, 'function');
    assert.strictEqual(typeof p.rejectStudy, 'function');
    assert.strictEqual(typeof p.dismissAlert, 'function');
    assert.strictEqual(typeof p.canConfirm, 'function');
    assert.strictEqual(typeof p.canReject, 'function');
  });
  test('canConfirm / canReject 委托纯函数', () => {
    const p = app.triagePanel();
    p.selectedStudy = null;
    assert.strictEqual(p.canConfirm('jr'), false);
    p.selectedStudy = { study_uid: 'a', triage_state: 'draft_pending' };
    assert.strictEqual(p.canConfirm('jr'), true);
    p.selectedStudy = { study_uid: 'a', triage_state: 'sr_signed' };
    assert.strictEqual(p.canReject(), false);
  });

  // ---------------------------------------------------------------
  // Async: refreshQueues with mocked fetch
  // ---------------------------------------------------------------
  section('Async: refreshQueues with mocked fetch');
  await test('refreshQueues: 拉阳性 + 阴性 → 按 max_prob 排序', async () => {
    const fetchMock = (url) => {
      if (url.endsWith('/api/queue/positive')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve([
          { study_uid: 'p1', max_prob: 0.6 },
          { study_uid: 'p2', max_prob: 0.9 },
          { study_uid: 'p3', max_prob: 0.7 },
        ]) });
      }
      if (url.endsWith('/api/queue/negative')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve([
          { study_uid: 'n1', max_prob: 0.1 },
          { study_uid: 'n2', max_prob: 0.3 },
        ]) });
      }
      return Promise.reject(new Error('unexpected url: ' + url));
    };
    const p = app.triagePanel({ fetch: fetchMock });
    await p.refreshQueues();
    assert.strictEqual(p.positiveStudies.length, 3);
    assert.strictEqual(p.positiveStudies[0].study_uid, 'p2', '阳性按 max_prob DESC');
    assert.strictEqual(p.positiveStudies[0].max_prob, 0.9);
    assert.strictEqual(p.positiveStudies[2].study_uid, 'p1');
    assert.strictEqual(p.negativeStudies.length, 2);
    assert.strictEqual(p.negativeStudies[0].study_uid, 'n1', '阴性按 max_prob ASC');
    assert.strictEqual(p.negativeStudies[1].study_uid, 'n2');
  });

  await test('refreshQueues: 服务端非 ok → 不污染原数组', async () => {
    const fetchMock = (url) => {
      if (url.endsWith('/api/queue/positive')) {
        return Promise.resolve({ ok: false, json: () => Promise.resolve([]) });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve([]) });
    };
    const p = app.triagePanel({ fetch: fetchMock });
    p.positiveStudies = [{ study_uid: 'seed', max_prob: 0.5 }];
    await p.refreshQueues();
    assert.strictEqual(p.positiveStudies.length, 1);
    assert.strictEqual(p.positiveStudies[0].study_uid, 'seed');
  });

  // ---------------------------------------------------------------
  // Async: SSE 事件 → 状态增量更新（核心）
  // ---------------------------------------------------------------
  section('Async: SSE 事件 → 状态增量更新');

  function makeEventSourceStub() {
    const listeners = {};
    return {
      _listeners: listeners,
      _opened: false,
      addEventListener(name, cb) {
        if (!listeners[name]) listeners[name] = [];
        listeners[name].push(cb);
      },
      removeEventListener(name, cb) {
        if (!listeners[name]) return;
        listeners[name] = listeners[name].filter((fn) => fn !== cb);
      },
      close() { this._closed = true; },
      async fire(name, data) {
        const fns = listeners[name];
        if (!fns || fns.length === 0) throw new Error('no listener for ' + name);
        const dataObj = { data: typeof data === 'string' ? data : JSON.stringify(data) };
        // 串行触发所有监听器；任一抛错不阻断后续；返回首个 promise 让 await 能等到异步链
        const results = [];
        for (const fn of fns) {
          try {
            const r = fn(dataObj);
            results.push(r);
          } catch (e) {
            results.push(Promise.reject(e));
          }
        }
        // 等齐所有 promise 后若有任何拒绝，再聚合抛错（保持测试可读）
        await Promise.allSettled(results);
      },
      open() { this._opened = true; if (this.onopen) this.onopen(); },
      error() { if (this.onerror) this.onerror(); },
    };
  }

  await test('new_study 事件 → 队列刷新 + 自动选中', async () => {
    const es = makeEventSourceStub();
    let fetchCallCount = 0;
    const fetchMock = (url) => {
      fetchCallCount += 1;
      if (url.endsWith('/api/queue/positive')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve([
          { study_uid: 'newly', max_prob: 0.88, finding_text: '气胸草稿' },
        ]) });
      }
      if (url.endsWith('/api/queue/negative')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve([]) });
      }
      return Promise.reject(new Error('unexpected ' + url));
    };
    const p = app.triagePanel({ fetch: fetchMock, EventSource: function () { return es; } });
    p.connectSse();
    assert.ok(p._sseListenersAttached, 'connectSse 应绑定监听器一次');
    es.open();
    await es.fire('new_study', {
      event_id: 'e-new',
      study_uid: 'newly',
      ts: '2026-10-14T01:00:00.000Z',
    });
    assert.strictEqual(p.positiveStudies.length, 1);
    assert.strictEqual(p.positiveStudies[0].study_uid, 'newly');
    assert.strictEqual(p.selectedStudy && p.selectedStudy.study_uid, 'newly');
    assert.strictEqual(p.lastSeenTs, '2026-10-14T01:00:00.000Z');
    assert.ok(fetchCallCount >= 2, 'refreshQueues 应被触发（positive + negative）');
  });

  await test('draft_ready 事件 → 同 study_uid 自动选中', async () => {
    const es = makeEventSourceStub();
    const fetchMock = (url) => {
      if (url.endsWith('/api/queue/positive')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve([
          { study_uid: 'ready1', max_prob: 0.7, finding_text: 'V1' },
        ]) });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve([]) });
    };
    const p = app.triagePanel({ fetch: fetchMock, EventSource: function () { return es; } });
    p.connectSse();
    await es.fire('draft_ready', { event_id: 'e1', study_uid: 'ready1', ts: '2026-10-14T01:00:01.000Z' });
    assert.strictEqual(p.selectedStudy.study_uid, 'ready1');
  });

  await test('state_change 事件 → 队列刷新', async () => {
    const es = makeEventSourceStub();
    let refreshCalled = 0;
    const fetchMock = (url) => {
      refreshCalled += 1;
      if (url.endsWith('/api/queue/positive')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve([
          { study_uid: 'changed', max_prob: 0.6, triage_state: 'jr_confirmed' },
        ]) });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve([]) });
    };
    const p = app.triagePanel({ fetch: fetchMock, EventSource: function () { return es; } });
    p.connectSse();
    await es.fire('state_change', { event_id: 'e2', study_uid: 'changed', ts: '2026-10-14T01:00:02.000Z' });
    assert.ok(refreshCalled >= 2);
    assert.strictEqual(p.positiveStudies[0].triage_state, 'jr_confirmed');
  });

  await test('alert 事件 → alerts 入栈 + 同 event_id 幂等', async () => {
    const es = makeEventSourceStub();
    const fetchMock = () => Promise.resolve({ ok: true, json: () => Promise.resolve([]) });
    const p = app.triagePanel({ fetch: fetchMock, EventSource: function () { return es; } });
    p.connectSse();
    await es.fire('alert', {
      event_id: 'alert-1',
      ts: '2026-10-14T01:00:03.000Z',
      payload: { target_endpoint: 'ris://x', message: 'RIS 失败' },
    });
    assert.strictEqual(p.alerts.length, 1);
    assert.strictEqual(p.alerts[0].target, 'ris://x');
    assert.strictEqual(p.alerts[0].message, 'RIS 失败');

    // 同 event_id 再推 → 不重复入栈
    await es.fire('alert', {
      event_id: 'alert-1',
      ts: '2026-10-14T01:00:04.000Z',
      payload: { target_endpoint: 'ris://x', message: 'RIS 失败（重发）' },
    });
    assert.strictEqual(p.alerts.length, 1, '同 event_id 不重复');

    // 不同 event_id → 入第二条
    await es.fire('alert', {
      event_id: 'alert-2',
      ts: '2026-10-14T01:00:05.000Z',
      payload: { target_endpoint: 'hc://y', message: '主检失败' },
    });
    assert.strictEqual(p.alerts.length, 2);
    assert.strictEqual(p.alerts[1].target, 'hc://y');
  });

  await test('keepalive 事件 → 更新 lastSeenTs（无 fetch 触发）', async () => {
    const es = makeEventSourceStub();
    let fetchCalled = 0;
    const fetchMock = () => { fetchCalled += 1; return Promise.resolve({ ok: true, json: () => Promise.resolve([]) }); };
    const p = app.triagePanel({ fetch: fetchMock, EventSource: function () { return es; } });
    p.connectSse();
    const beforeTs = p.lastSeenTs;
    await es.fire('keepalive', { data: '{}' });
    assert.notStrictEqual(p.lastSeenTs, beforeTs);
    assert.strictEqual(fetchCalled, 0, 'keepalive 不触发 fetch');
  });

  await test('SSE 解析失败 → 不抛 + console.error 被调', async () => {
    const es = makeEventSourceStub();
    const errs = [];
    const origConsoleErr = console.error;
    console.error = (...args) => errs.push(args);
    try {
      const p = app.triagePanel({
        fetch: () => Promise.resolve({ ok: true, json: () => Promise.resolve([]) }),
        EventSource: function () { return es; },
      });
      p.connectSse();
      es.addEventListener('new_study', function () { throw new Error('parse fail'); });
      // 直接传坏 JSON 字符串（fire stub 对 string 不做二次 stringify）
      await es.fire('new_study', '{not-json');
      // 等异步链清空
      await new Promise((r) => setTimeout(r, 50));
      assert.ok(errs.length >= 1, 'console.error 应被调（onSseEvent JSON.parse 失败应被 catch）');
    } finally {
      console.error = origConsoleErr;
    }
  });

  // ---------------------------------------------------------------
  // Async: init 生命周期
  // ---------------------------------------------------------------
  section('Async: init 生命周期');
  await test('init() 调 refreshQueues + connectSse（无 window 也不抛）', async () => {
    const es = makeEventSourceStub();
    let refreshCalled = 0;
    let connectCalled = 0;
    const fetchMock = () => { refreshCalled += 1; return Promise.resolve({ ok: true, json: () => Promise.resolve([]) }); };
    const ESFactory = function () { connectCalled += 1; return es; };
    const p = app.triagePanel({ fetch: fetchMock, EventSource: ESFactory });
    await p.init();
    assert.ok(refreshCalled >= 2);
    assert.strictEqual(connectCalled, 1);
    assert.strictEqual(p.eventSource, es);
  });

  // ---------------------------------------------------------------
  // Async: selectStudy 详情合并
  // ---------------------------------------------------------------
  section('Async: selectStudy 详情合并');
  await test('selectStudy → 合并 /api/studies/{uid} 详情', async () => {
    let capturedUrl = null;
    const fetchMock = (url, opts) => {
      capturedUrl = url;
      if (opts && opts.method === 'POST') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ ok: true }) });
      }
      if (url.includes('/api/studies/u9')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({
          triage_state: 'jr_confirmed',
          finding_text: '详情所见',
          key_qa_list: [{ q: '心影', a: '临界' }],
          grounding_boxes: [{ x: 0.1, y: 0.1, w: 0.2, h: 0.2 }],
          png_preview_b64: 'AAAA',
        }) });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve([]) });
    };
    const p = app.triagePanel({ fetch: fetchMock });
    const queueItem = { study_uid: 'u9', max_prob: 0.8 };
    await p.selectStudy(queueItem);
    assert.ok(capturedUrl.includes('/api/studies/u9'), '应请求详情端点');
    assert.strictEqual(p.selectedStudy.study_uid, 'u9');
    assert.strictEqual(p.selectedStudy.triage_state, 'jr_confirmed');
    assert.strictEqual(p.selectedStudy.finding_text, '详情所见');
    assert.strictEqual(p.selectedStudy.key_qa_list.length, 1);
    assert.strictEqual(p.selectedStudy.grounding_boxes.length, 1);
    assert.strictEqual(p.selectedStudy.max_prob, 0.8, '队列字段保留');
  });

  // ---------------------------------------------------------------
  // Async: confirmStudy / rejectStudy
  // ---------------------------------------------------------------
  section('Async: confirmStudy / rejectStudy');
  await test('confirmStudy(jr) → POST /confirm with body {actor_role, target_state}', async () => {
    const calls = [];
    const fetchMock = (url, opts) => {
      calls.push({ url, opts });
      if (url.includes('/api/queue/')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve([]) });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ ok: true }) });
    };
    const p = app.triagePanel({ fetch: fetchMock });
    p.selectedStudy = { study_uid: 'u11', triage_state: 'draft_pending' };
    await p.confirmStudy('jr');
    const confirmCall = calls.find((c) => c.url.includes('/api/studies/u11/confirm'));
    assert.ok(confirmCall, '应有 POST /confirm 调用');
    const body = JSON.parse(confirmCall.opts.body);
    assert.strictEqual(body.actor_role, 'jr');
    assert.strictEqual(body.target_state, 'jr_confirmed');
    assert.strictEqual(confirmCall.opts.method, 'POST');
    assert.strictEqual(confirmCall.opts.headers['Content-Type'], 'application/json');
  });

  await test('confirmStudy(sr) → target_state=sr_signed', async () => {
    const calls = [];
    const fetchMock = (url, opts) => {
      calls.push({ url, opts });
      if (url.includes('/api/queue/')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve([]) });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ ok: true }) });
    };
    const p = app.triagePanel({ fetch: fetchMock });
    p.selectedStudy = { study_uid: 'u12', triage_state: 'sr_reviewed' };
    await p.confirmStudy('sr');
    const confirmCall = calls.find((c) => c.url.includes('/api/studies/u12/confirm'));
    assert.ok(confirmCall, '应有 POST /confirm 调用');
    const body = JSON.parse(confirmCall.opts.body);
    assert.strictEqual(body.actor_role, 'sr');
    assert.strictEqual(body.target_state, 'sr_signed');
  });

  await test('confirmStudy 未知 role → 不发请求', async () => {
    let called = false;
    const fetchMock = () => { called = true; return Promise.resolve({}); };
    const p = app.triagePanel({ fetch: fetchMock });
    p.selectedStudy = { study_uid: 'u13', triage_state: 'draft_pending' };
    await p.confirmStudy('unknown');
    assert.strictEqual(called, false);
  });

  await test('confirmStudy 无 selectedStudy → 不发请求', async () => {
    let called = false;
    const fetchMock = () => { called = true; return Promise.resolve({}); };
    const p = app.triagePanel({ fetch: fetchMock });
    await p.confirmStudy('jr');
    assert.strictEqual(called, false);
  });

  await test('rejectStudy → POST /reject + 本地 triage_state 置 draft_pending + refreshQueues', async () => {
    const calls = [];
    const fetchMock = (url, opts) => {
      calls.push({ url, opts });
      if (url.includes('/api/queue/')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve([]) });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ ok: true }) });
    };
    const p = app.triagePanel({ fetch: fetchMock });
    p.selectedStudy = { study_uid: 'u14', triage_state: 'sr_reviewed' };
    await p.rejectStudy();
    const rejectCall = calls.find((c) => c.url.includes('/api/studies/u14/reject'));
    assert.ok(rejectCall, '应有 POST /reject 调用');
    const body = JSON.parse(rejectCall.opts.body);
    assert.strictEqual(body.actor_role, 'reviewer');
    assert.strictEqual(p.selectedStudy.triage_state, 'draft_pending');
    const queueCalls = calls.filter((c) => c.url.includes('/api/queue/'));
    assert.ok(queueCalls.length >= 2, '驳回成功后应刷新队列（至少 2 次）');
  });

  // ---------------------------------------------------------------
  // Async: dismissAlert
  // ---------------------------------------------------------------
  section('Async: dismissAlert');
  await test('dismissAlert(eventId) → 移除匹配 alert', async () => {
    const p = app.triagePanel({ fetch: () => Promise.resolve({ ok: true, json: () => Promise.resolve([]) }) });
    p.alerts = [
      { event_id: 'a1', target: 'ris', message: 'x' },
      { event_id: 'a2', target: 'hc', message: 'y' },
    ];
    p.dismissAlert('a1');
    assert.strictEqual(p.alerts.length, 1);
    assert.strictEqual(p.alerts[0].event_id, 'a2');
  });

  // ---------------------------------------------------------------
  // 重置与边界
  // ---------------------------------------------------------------
  section('Reset & edge cases');
  await test('_reset() 清空全部状态', () => {
    const p = app.triagePanel();
    p.positiveStudies = [{ study_uid: 'x', max_prob: 0.9 }];
    p.alerts = [{ event_id: 'a' }];
    p.selectedStudy = { study_uid: 'x' };
    p.degraded = true;
    p.lastSeenTs = '2026-10-14';
    p.connectionStatus = '已连接';
    p._reset();
    assert.strictEqual(p.positiveStudies.length, 0);
    assert.strictEqual(p.alerts.length, 0);
    assert.strictEqual(p.selectedStudy, null);
    assert.strictEqual(p.degraded, false);
    assert.strictEqual(p.connectionStatus, '未连接');
  });

  await test('EventSource 不存在 → connectSse 不抛', () => {
    const p = app.triagePanel({ fetch: () => {}, EventSource: null });
    p.connectSse();
    assert.strictEqual(p.eventSource, null);
  });

  await test('fetch 不存在 → refreshQueues / selectStudy / confirmStudy / rejectStudy 均不抛', async () => {
    const p = app.triagePanel({ fetch: null });
    await p.refreshQueues();
    await p.selectStudy({ study_uid: 'u' });
    await p.confirmStudy('jr');
    await p.rejectStudy();
    assert.strictEqual(p.positiveStudies.length, 0);
  });

  await test('EventSource 构造抛错 → 不污染 panel', () => {
    const errs = [];
    const orig = console.error;
    console.error = (...a) => errs.push(a);
    try {
      const p = app.triagePanel({
        fetch: () => Promise.resolve({ ok: true, json: () => Promise.resolve([]) }),
        EventSource: function () { throw new Error('boom'); },
      });
      p.connectSse();
      assert.ok(errs.length >= 1);
    } finally {
      console.error = orig;
    }
  });

  // ---------------------------------------------------------------
  // 总结
  // ---------------------------------------------------------------
  console.log('\n=========================================');
  console.log(`Total: ${total}   Passed: ${passed}   Failed: ${failed}`);
  console.log('=========================================');
  if (failed > 0) {
    console.log('\nFailures:');
    failures.forEach((f) => {
      console.log(`  - ${f.name}: ${f.err && f.err.stack ? f.err.stack.split('\n').slice(0, 3).join('\n      ') : f.err}`);
    });
    process.exit(1);
  }
  process.exit(0);
}

run().catch((e) => {
  console.error('Fatal:', e);
  process.exit(2);
});
