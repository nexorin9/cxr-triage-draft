/* ============================================================================
 * alert_banner.js
 * Task 27: 监听 SSE 'alert' 事件 → 渲染浏览器顶部红色 banner
 *
 * 用法（Task 31 接入 Alpine.js 后由 x-data 包裹）：
 *   import { AlertBannerHandler } from './alert_banner.js';
 *   const handler = new AlertBannerHandler(eventSource);
 *   handler.attach();
 *
 * SSE 协议约定（与 writers/alert_banner.py build_alert_event_payload 对齐）：
 *   payload.event    === 'alert'
 *   payload.kind     in {'alert.raised','alert.dismissed','alert.summary'}
 *   payload.severity in {'error','warning'}
 *   payload.banner_html   string（前端可直接 innerHTML）
 *   payload.message_zh    string（banner 文本，可读 fallback）
 *   payload.total_alert_count int
 *
 * 行为契约：
 *   1) SSE 收到 'alert' event 且 payload.kind ∈ {raised, summary} →
 *      把 banner_html 注入 #cxr-alert-banner-root；移除 hidden class。
 *   2) SSE 收到 'alert.dismissed' → 把 banner 替换为隐藏态占位。
 *   3) 启动时调 /api/health 一次拿到 outbox_alerts 数值，若 >= 1
 *      主动渲染 banner（避免浏览器初次连接拿不到推送、丢失现场、、
 *      看到告警的场景）。
 *   4) 失败模式：SSE 断线 → EventSource 自动重连；不主动 push 错误
 *      弹窗，避免污染医师工作流。
 *
 * 设计原则：纯 DOM + 事件回调，**不**依赖 Alpine.js 启动；后续
 * Task 31 可挂到 Alpine x-data 的 init() 里。
 * ========================================================================= */

(function (global) {
  'use strict';

  // banner 元素 id（与 writers/alert_banner.py ALERT_BANNER_ELEMENT_ID 一致）
  const BANNER_ID = 'cxr-alert-banner-root';

  // banner 隐藏态 CSS class
  const HIDDEN_CSS = 'cxr-alert-banner cxr-alert-banner--hidden';

  // 触发 banner 渲染的事件 kind
  const VISIBLE_KINDS = new Set(['alert.raised', 'alert.summary']);

  /**
   * 渲染 banner：把 HTML 注入 #cxr-alert-banner-root；缺元素则自动建。
   * @param {Object} payload — SSE alert event payload
   */
  function renderBanner(payload) {
    if (!payload || typeof payload !== 'object') return;
    const root = ensureBannerRoot();
    const html = payload.banner_html;
    if (typeof html === 'string' && html.length > 0) {
      root.innerHTML = stripOuterWrapper(html, root);
    } else {
      // fallback：直接写 message_zh
      const msg = String(payload.message_zh || '');
      const count = String(payload.total_alert_count || 0);
      root.innerHTML =
        '<span class="cxr-alert-banner__icon" aria-hidden="true">⚠</span>' +
        '<span class="cxr-alert-banner__message">' + escapeHtml(msg) + '</span>';
    }
    root.removeAttribute('hidden');
    root.dataset.cxrAlertCount = count;
    root.dataset.cxrAlertBanner = '1';
  }

  /**
   * 把 banner 切到隐藏态（信息科已处理 / dismissed）。
   */
  function hideBanner() {
    const root = ensureBannerRoot();
    root.className = HIDDEN_CSS;
    root.dataset.cxrAlertBanner = '0';
    root.dataset.cxrAlertCount = '0';
  }

  /**
   * 确保 #cxr-alert-banner-root 存在；不存在则建一个并挂在 <body> 头。
   */
  function ensureBannerRoot() {
    let root = document.getElementById(BANNER_ID);
    if (root) return root;
    root = document.createElement('div');
    root.id = BANNER_ID;
    root.className = HIDDEN_CSS;
    root.setAttribute('data-cxr-alert-banner', '0');
    root.setAttribute('data-cxr-alert-count', '0');
    // 插到顶部以匹配 CSS position:fixed
    if (document.body.firstChild) {
      document.body.insertBefore(root, document.body.firstChild);
    } else {
      document.body.appendChild(root);
    }
    return root;
  }

  /**
   * banner_html 是完整 <div ...>...</div>，innerHTML 直接写入会替换整个
   * root；为了保留 root 的 id/属性，把内层 children 抠出来用。
   */
  function stripOuterWrapper(html, root) {
    try {
      const tmp = document.createElement('template');
      tmp.innerHTML = html;
      const first = tmp.content.firstElementChild;
      if (!first) return html;
      // 用 first 的 children 替代（保留 id 在 root 上）
      while (root.firstChild) root.removeChild(root.firstChild);
      // 把外层 wrapper 的 class / data attr 同步到 root（保留
      // 后续 layout 一致）
      if (first.className) root.className = first.className;
      if (first.dataset) {
        Object.keys(first.dataset).forEach((k) => {
          root.dataset[k] = first.dataset[k];
        });
      }
      // 把 wrapper 内部子节点搬过来
      while (first.firstChild) root.appendChild(first.firstChild);
      return '<' + first.tagName + '></' + first.tagName + '>';
    } catch (e) {
      return html;
    }
  }

  /**
   * 简易 HTML escape（fallback 用）。
   */
  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  /**
   * 主类：SSE → DOM banner 桥接。
   * @param {EventSource} eventSource — 已连接的 SSE 客户端（指向 /api/sse）
   * @param {Object} [opts] — 配置项
   * @param {string} [opts.healthUrl='/api/health'] — 启动时查询健康检查端点
   * @param {number} [opts.healthRetryMs=2000] — 健康查询失败重试间隔
   */
  function AlertBannerHandler(eventSource, opts) {
    if (!eventSource || typeof eventSource.addEventListener !== 'function') {
      throw new TypeError(
        'AlertBannerHandler 需要 EventSource 实例（addEventListener 方法）'
      );
    }
    this.eventSource = eventSource;
    this.healthUrl = (opts && opts.healthUrl) || '/api/health';
    this.healthRetryMs = (opts && opts.healthRetryMs) || 2000;
    this._attached = false;
  }

  AlertBannerHandler.prototype.attach = function () {
    if (this._attached) return;
    this._attached = true;
    const self = this;
    this.eventSource.addEventListener('alert', function (ev) {
      try {
        const data = JSON.parse(ev.data || '{}');
        self._handle(data);
      } catch (e) {
        // 忽略：单条事件损坏不阻断 SSE 链路
      }
    });
    // 启动时主动查 /api/health 兜底（避免错过推送）
    this._probeHealth();
  };

  AlertBannerHandler.prototype._handle = function (payload) {
    if (!payload || typeof payload !== 'object') return;
    const kind = payload.kind;
    if (kind === 'alert.dismissed') {
      hideBanner();
      return;
    }
    if (VISIBLE_KINDS.has(kind)) {
      renderBanner(payload);
      return;
    }
    // 未知 kind 不动（防协议扩展噪声）
  };

  AlertBannerHandler.prototype._probeHealth = function () {
    const self = this;
    let aborted = false;
    const tryFetch = function () {
      if (aborted) return;
      fetch(self.healthUrl, { credentials: 'same-origin' })
        .then(function (r) {
          if (!r.ok) throw new Error('HTTP ' + r.status);
          return r.json();
        })
        .then(function (body) {
          const n = (body && body.outbox_alerts) || 0;
          if (n >= 1) {
            // 用同样的 schema 渲染 banner（不依赖 banner_html）
            renderBanner({
              kind: 'alert.summary',
              severity: 'error',
              total_alert_count: n,
              message_zh:
                'RIS / 体检主检写回连续失败 ' +
                n +
                ' 例，请联系信息科。',
              banner_html: '',
            });
          } else {
            hideBanner();
          }
        })
        .catch(function () {
          // 失败：1.5x 后重试一次（最多一次，避免无限重试）
          if (!aborted) setTimeout(tryFetch, self.healthRetryMs);
        });
    };
    tryFetch();
    // 提供一个 abort 方法给测试用
    this._abortProbe = function () {
      aborted = true;
    };
  };

  AlertBannerHandler.prototype.detach = function () {
    this._attached = false;
    if (this._abortProbe) this._abortProbe();
  };

  // 暴露
  global.AlertBannerHandler = AlertBannerHandler;
  global.cxrAlertBanner = {
    render: renderBanner,
    hide: hideBanner,
    ensureRoot: ensureBannerRoot,
  };
})(typeof window !== 'undefined' ? window : globalThis);