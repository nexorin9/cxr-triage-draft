/* ============================================================================
 * app.js — Alpine.js 状态绑定层
 * Task 31: 把 inline <script> 中的 triagePanel() 抽出，便于测试与维护
 *
 * 浏览器端被 index.html 通过 <script src="/static/app.js"></script> 加载；
 * 也可被 Node.js 通过 vm 模块加载做单测（tests/test_alpine.js）。
 *
 * SSE 协议（与 src/cxr_triage/web/sse.py SsePublisher 对齐）：
 *   payload.event ∈ {new_study, draft_ready, state_change, alert, keepalive}
 *   payload.study_uid : string|undefined
 *   payload.ts        : ISO8601 UTC
 *   payload.event_id  : string
 *   payload.payload   : object（alert 事件含 message / target_endpoint 等）
 *
 * 浏览器端 Alpine.js 启动契约：
 *   <body x-data="triagePanel()" x-init="init()">
 *
 * 设计原则：
 *   - 函数式 state，**不**写入 window 之外的全局副作用；
 *   - 所有 fetch / EventSource 操作都通过 this.fetch / this.EventSource
 *     包装，测试可注入 mock；
 *   - canConfirm / canReject 是纯函数，便于直接断言。
 * ========================================================================= */

(function (global) {
  'use strict';

  // -------------------------------------------------------------------------
  // 常量（与 src/cxr_triage/web/sse.py / state/state_machine.py 对齐）
  // -------------------------------------------------------------------------
  const SSE_ENDPOINT = '/api/sse';
  const ENDPOINT_QUEUE_POSITIVE = '/api/queue/positive';
  const ENDPOINT_QUEUE_NEGATIVE = '/api/queue/negative';
  const ENDPOINT_STUDY = '/api/studies/{uid}';
  const ENDPOINT_CONFIRM = '/api/studies/{uid}/confirm';
  const ENDPOINT_REJECT = '/api/studies/{uid}/reject';

  // 状态机 5 态（与 state/state_machine.py CxrState 对齐）
  const STATE_DRAFT_PENDING = 'draft_pending';
  const STATE_JR_CONFIRMED = 'jr_confirmed';
  const STATE_SR_REVIEWED = 'sr_reviewed';
  const STATE_SR_SIGNED = 'sr_signed';
  const STATE_REJECTED = 'rejected';

  // 4 类 SSE 事件
  const EVENT_NEW_STUDY = 'new_study';
  const EVENT_DRAFT_READY = 'draft_ready';
  const EVENT_STATE_CHANGE = 'state_change';
  const EVENT_ALERT = 'alert';
  const EVENT_KEEPALIVE = 'keepalive';

  // 3 个角色
  const ROLE_JR = 'jr';
  const ROLE_REVIEWER = 'reviewer';
  const ROLE_SR = 'sr';

  // SSE 自动重连间隔
  const SSE_RETRY_MS = 3000;

  // -------------------------------------------------------------------------
  // 纯函数（可独立测试）
  // -------------------------------------------------------------------------

  /**
   * 给定 study 当前状态 + 目标角色，决定「确认」按钮是否可点。
   * 与 state/state_machine.py VALID_TRANSITIONS 对齐。
   */
  function canConfirm(selectedStudy, targetRole) {
    if (!selectedStudy) return false;
    const state = selectedStudy.triage_state || STATE_DRAFT_PENDING;
    if (targetRole === ROLE_JR) return state === STATE_DRAFT_PENDING;
    if (targetRole === ROLE_REVIEWER) return state === STATE_JR_CONFIRMED;
    if (targetRole === ROLE_SR) return state === STATE_SR_REVIEWED;
    return false;
  }

  /**
   * 给定 study 当前状态，决定「驳回」按钮是否可点。
   * sr_signed 与 rejected 终态不可驳回。
   */
  function canReject(selectedStudy) {
    if (!selectedStudy) return false;
    const state = selectedStudy.triage_state || STATE_DRAFT_PENDING;
    return [
      STATE_DRAFT_PENDING,
      STATE_JR_CONFIRMED,
      STATE_SR_REVIEWED,
    ].indexOf(state) !== -1;
  }

  /**
   * 给定 role 名 → 目标 state 字符串。
   * 与 web/app.py confirm body target_state 对齐。
   */
  function targetStateForRole(role) {
    if (role === ROLE_JR) return STATE_JR_CONFIRMED;
    if (role === ROLE_REVIEWER) return STATE_SR_REVIEWED;
    if (role === ROLE_SR) return STATE_SR_SIGNED;
    return null;
  }

  /**
   * SSE alert 事件去重：同 event_id 不重复入栈。
   */
  function dedupAlerts(alerts, newAlert) {
    if (!newAlert || !newAlert.event_id) return alerts;
    for (let i = 0; i < alerts.length; i++) {
      if (alerts[i].event_id === newAlert.event_id) return alerts;
    }
    return alerts.concat([newAlert]);
  }

  /**
   * 在两个队列（阳性 + 阴性）中按 study_uid 找一条 study。
   * 用于 SSE 事件 → 选中态自动滚动。
   */
  function findStudyByUid(positiveStudies, negativeStudies, studyUid) {
    if (!studyUid) return null;
    const all = positiveStudies.concat(negativeStudies);
    for (let i = 0; i < all.length; i++) {
      if (all[i].study_uid === studyUid) return all[i];
    }
    return null;
  }

  /**
   * 把 SSE payload 转为 Alpine alerts 数组元素。
   */
  function alertFromPayload(data) {
    const payload = data.payload || {};
    return {
      event_id: data.event_id,
      target: payload.target_endpoint || 'unknown',
      message:
        payload.message ||
        'RIS / 体检主检写回连续失败，请联系信息科。',
    };
  }

  // -------------------------------------------------------------------------
  // 主组件：triagePanel()（Alpine.js x-data factory）
  // -------------------------------------------------------------------------

  /**
   * Alpine.js x-data 工厂。浏览器中由 <body x-data="triagePanel()"> 触发；
   * 测试可独立调用：const panel = triagePanel({ fetch, EventSource });
   */
  function triagePanel(opts) {
    opts = opts || {};
    const fetchImpl = opts.fetch || (typeof fetch !== 'undefined' ? fetch : null);
    const ESImpl = opts.EventSource || (typeof EventSource !== 'undefined' ? EventSource : null);
    const setTimeoutImpl = opts.setTimeout || setTimeout;
    const clearTimeoutImpl = opts.clearTimeout || clearTimeout;

    return {
      // ---- 状态 ----
      serviceName: '',
      positiveStudies: [],
      negativeStudies: [],
      selectedStudy: null,
      alerts: [],
      degraded: false,
      degradationBannerText: '',
      lastSeenTs: null,
      connectionStatus: '未连接',
      eventSource: null,

      // ---- 端点 ----
      endpointSse: SSE_ENDPOINT,
      endpointQueuePositive: ENDPOINT_QUEUE_POSITIVE,
      endpointQueueNegative: ENDPOINT_QUEUE_NEGATIVE,
      endpointStudy: ENDPOINT_STUDY,
      endpointConfirm: ENDPOINT_CONFIRM,
      endpointReject: ENDPOINT_REJECT,

      // ---- 内部 ----
      _retryTimer: null,
      _sseListenersAttached: false,

      // ---- 生命周期 ----
      init: function () {
        const self = this;
        return Promise.resolve()
          .then(function () { return self.refreshQueues(); })
          .then(function () { self.connectSse(); })
          .then(function () {
            if (typeof window !== 'undefined') {
              window.addEventListener('beforeunload', function () {
                if (self.eventSource && self.eventSource.close) self.eventSource.close();
                if (self._retryTimer) clearTimeoutImpl(self._retryTimer);
              });
            }
          });
      },

      // ---- 队列刷新 ----
      refreshQueues: function () {
        const self = this;
        if (!fetchImpl) return Promise.resolve();
        return Promise.all([
          fetchImpl(self.endpointQueuePositive),
          fetchImpl(self.endpointQueueNegative),
        ])
          .then(function (results) {
            const posRes = results[0];
            const negRes = results[1];
            const tasks = [];
            if (posRes && posRes.ok) {
              tasks.push(
                posRes.json().then(function (data) {
                  self.positiveStudies = (Array.isArray(data) ? data : []).slice().sort(function (a, b) {
                    return b.max_prob - a.max_prob;
                  });
                })
              );
            }
            if (negRes && negRes.ok) {
              tasks.push(
                negRes.json().then(function (data) {
                  self.negativeStudies = (Array.isArray(data) ? data : []).slice().sort(function (a, b) {
                    return a.max_prob - b.max_prob;
                  });
                })
              );
            }
            return Promise.all(tasks);
          })
          .catch(function (err) {
            if (typeof console !== 'undefined') console.error('队列刷新失败', err);
          });
      },

      // ---- SSE 连接 ----
      connectSse: function () {
        const self = this;
        if (!ESImpl) return;
        const url = self.lastSeenTs
          ? self.endpointSse + '?last_seen_ts=' + encodeURIComponent(self.lastSeenTs)
          : self.endpointSse;
        try {
          self.eventSource = new ESImpl(url);
        } catch (e) {
          if (typeof console !== 'undefined') console.error('SSE 构造失败', e);
          return;
        }
        self.connectionStatus = '连接中…';
        self.eventSource.onopen = function () { self.connectionStatus = '已连接'; };
        self.eventSource.onerror = function () {
          self.connectionStatus = '连接断开，3s 后重试';
          if (self._retryTimer) clearTimeoutImpl(self._retryTimer);
          self._retryTimer = setTimeoutImpl(function () { self.connectSse(); }, SSE_RETRY_MS);
        };
        if (!self._sseListenersAttached) {
          self._sseListenersAttached = true;
          self.eventSource.addEventListener(EVENT_NEW_STUDY, function (e) { return self.onSseEvent(e, EVENT_NEW_STUDY); });
          self.eventSource.addEventListener(EVENT_DRAFT_READY, function (e) { return self.onSseEvent(e, EVENT_DRAFT_READY); });
          self.eventSource.addEventListener(EVENT_STATE_CHANGE, function (e) { return self.onSseEvent(e, EVENT_STATE_CHANGE); });
          self.eventSource.addEventListener(EVENT_ALERT, function (e) { return self.onSseEvent(e, EVENT_ALERT); });
          self.eventSource.addEventListener(EVENT_KEEPALIVE, function () {
            self.lastSeenTs = new Date().toISOString();
          });
        }
      },

      // ---- SSE 事件处理 ----
      onSseEvent: function (event, kind) {
        const self = this;
        let data;
        try {
          data = JSON.parse(event.data);
        } catch (err) {
          if (typeof console !== 'undefined') console.error('SSE 事件 ' + kind + ' 解析失败', err);
          return Promise.resolve();
        }
        self.lastSeenTs = data.ts || new Date().toISOString();
        if (kind === EVENT_ALERT) {
          self.alerts = dedupAlerts(self.alerts, alertFromPayload(data));
          return Promise.resolve();
        }
        return self.refreshQueues().then(function () {
          if (data.study_uid) {
            const found = findStudyByUid(self.positiveStudies, self.negativeStudies, data.study_uid);
            if (found) self.selectedStudy = found;
          }
        });
      },

      // ---- 选中 ----
      selectStudy: function (study) {
        const self = this;
        if (!study || !study.study_uid) return Promise.resolve();
        self.selectedStudy = study;
        if (!fetchImpl) return Promise.resolve();
        const url = self.endpointStudy.replace('{uid}', encodeURIComponent(study.study_uid));
        return fetchImpl(url)
          .then(function (res) {
            if (res && res.ok) {
              return res.json().then(function (detail) {
                self.selectedStudy = Object.assign({}, study, detail || {});
              });
            }
            return null;
          })
          .catch(function (err) {
            if (typeof console !== 'undefined') console.error('study 详情加载失败', err);
          });
      },

      // ---- 角色流转 ----
      canConfirm: function (target) { return canConfirm(this.selectedStudy, target); },
      canReject: function () { return canReject(this.selectedStudy); },

      confirmStudy: function (actorRole) {
        const self = this;
        if (!self.selectedStudy) return Promise.resolve();
        const targetState = targetStateForRole(actorRole);
        if (!targetState) return Promise.resolve();
        const url = self.endpointConfirm.replace('{uid}', encodeURIComponent(self.selectedStudy.study_uid));
        if (!fetchImpl) return Promise.resolve();
        return fetchImpl(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ actor_role: actorRole, target_state: targetState }),
        })
          .then(function (res) {
            if (res && res.ok) return self.refreshQueues();
            if (typeof console !== 'undefined') {
              console.warn('确认失败', res && res.status, res && res.text && res.text());
            }
            return null;
          })
          .catch(function (err) {
            if (typeof console !== 'undefined') console.error('确认请求异常', err);
          });
      },

      rejectStudy: function () {
        const self = this;
        if (!self.selectedStudy) return Promise.resolve();
        const url = self.endpointReject.replace('{uid}', encodeURIComponent(self.selectedStudy.study_uid));
        if (!fetchImpl) return Promise.resolve();
        return fetchImpl(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ actor_role: ROLE_REVIEWER, reason: '需要人工复核 / 修订' }),
        })
          .then(function (res) {
            if (res && res.ok) {
              self.selectedStudy.triage_state = STATE_DRAFT_PENDING;
              return self.refreshQueues();
            }
            if (typeof console !== 'undefined') {
              console.warn('驳回失败', res && res.status, res && res.text && res.text());
            }
            return null;
          })
          .catch(function (err) {
            if (typeof console !== 'undefined') console.error('驳回请求异常', err);
          });
      },

      // ---- 告警 dismiss ----
      dismissAlert: function (eventId) {
        const self = this;
        self.alerts = self.alerts.filter(function (a) { return a.event_id !== eventId; });
      },

      // ---- 工具 ----
      _reset: function () {
        this.positiveStudies = [];
        this.negativeStudies = [];
        this.selectedStudy = null;
        this.alerts = [];
        this.degraded = false;
        this.lastSeenTs = null;
        this.connectionStatus = '未连接';
      },
    };
  }

  // -------------------------------------------------------------------------
  // 暴露
  // -------------------------------------------------------------------------
  const api = {
    triagePanel: triagePanel,
    canConfirm: canConfirm,
    canReject: canReject,
    targetStateForRole: targetStateForRole,
    dedupAlerts: dedupAlerts,
    findStudyByUid: findStudyByUid,
    alertFromPayload: alertFromPayload,
    STATES: {
      DRAFT_PENDING: STATE_DRAFT_PENDING,
      JR_CONFIRMED: STATE_JR_CONFIRMED,
      SR_REVIEWED: STATE_SR_REVIEWED,
      SR_SIGNED: STATE_SR_SIGNED,
      REJECTED: STATE_REJECTED,
    },
    EVENTS: {
      NEW_STUDY: EVENT_NEW_STUDY,
      DRAFT_READY: EVENT_DRAFT_READY,
      STATE_CHANGE: EVENT_STATE_CHANGE,
      ALERT: EVENT_ALERT,
      KEEPALIVE: EVENT_KEEPALIVE,
    },
    ROLES: { JR: ROLE_JR, REVIEWER: ROLE_REVIEWER, SR: ROLE_SR },
    SSE_RETRY_MS: SSE_RETRY_MS,
    ENDPOINTS: {
      SSE: SSE_ENDPOINT,
      QUEUE_POSITIVE: ENDPOINT_QUEUE_POSITIVE,
      QUEUE_NEGATIVE: ENDPOINT_QUEUE_NEGATIVE,
      STUDY: ENDPOINT_STUDY,
      CONFIRM: ENDPOINT_CONFIRM,
      REJECT: ENDPOINT_REJECT,
    },
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
  if (typeof global !== 'undefined') {
    global.triagePanelApp = api;
    // 也暴露在 window 供 Alpine x-data 表达式使用
    if (typeof window !== 'undefined') {
      window.triagePanel = triagePanel;
      window.triagePanelApp = api;
    }
  }
})(typeof window !== 'undefined' ? window : (typeof global !== 'undefined' ? global : this));
