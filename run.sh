#!/usr/bin/env bash
# run.sh — 一键启动 体检胸片所见草稿系统
#
# 用法:
#   ./run.sh                # 后台启动 FastAPI 服务 (默认 :8586)
#   ./run.sh stop           # 停止已启动的服务
#   ./run.sh restart        # 重启
#   ./run.sh status         # 查看状态 (端口 / PID / 最近日志)
#   ./run.sh tail           # 实时跟踪 server.log
#
# 前置 (首次启动):
#   1. cp .env.example .env
#   2. 编辑 .env 填写真实凭据 (PACS_* / RIS_* / HEALTHCHECK_* / OPENAI_*)
#   3. ./run.sh
#
# 凭据一律从 .env 读；脚本不存任何密钥。

set -euo pipefail

# ============ 路径 / 常量 ============
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# 让 `python -m cxr_triage` 找到 src/cxr_triage 包（未 pip install 时必需）
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="$LOG_DIR/server.pid"
SERVER_LOG="$LOG_DIR/server.log"

DEFAULT_HOST="${CXR_TRIAGE_HOST:-0.0.0.0}"
DEFAULT_PORT="${CXR_TRIAGE_PORT:-8586}"

# ============ 工具函数 ============
log_info() { printf "[run.sh] %s\n" "$*" >&2; }
log_warn() { printf "[run.sh][WARN] %s\n" "$*" >&2; }
log_err()  { printf "[run.sh][ERROR] %s\n" "$*" >&2; }

ensure_log_dir() {
  if [[ ! -d "$LOG_DIR" ]]; then
    mkdir -p "$LOG_DIR"
  fi
}

ensure_env() {
  if [[ ! -f "$PROJECT_DIR/.env" ]]; then
    log_warn ".env 不存在 — 拷贝 .env.example 后填真实凭据"
    log_warn "cp .env.example .env && 编辑 .env"
    # 允许用占位符启动 (CXR_TRIAGE_ALLOW_PLACEHOLDER=1)，仅供 CI / 本地联调
    if [[ "${CXR_TRIAGE_ALLOW_PLACEHOLDER:-0}" != "1" ]]; then
      log_err "未设置 CXR_TRIAGE_ALLOW_PLACEHOLDER=1 — 拒绝以占位凭据启动"
      exit 78  # EX_CONFIG
    fi
  fi
}

is_running() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "$pid"
      return 0
    fi
  fi
  return 1
}

wait_for_health() {
  local url="$1"
  local max_wait="${2:-15}"
  local i=0
  while (( i < max_wait )); do
    if curl -sfS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    i=$((i+1))
  done
  return 1
}

# ============ 子命令 ============
cmd_serve() {
  ensure_log_dir
  ensure_env

  if pid="$(is_running)"; then
    log_warn "服务已在运行 (PID=$pid)"
    exit 0
  fi

  log_info "启动 cxr-triage FastAPI 服务 @ ${DEFAULT_HOST}:${DEFAULT_PORT}"
  log_info "日志: $SERVER_LOG  PID 文件: $PID_FILE"

  # 后台启动 — uvicorn 由 python -m cxr_triage serve 拉起
  nohup python -m cxr_triage serve \
      --host "$DEFAULT_HOST" \
      --port "$DEFAULT_PORT" \
      --log-dir "$LOG_DIR" \
      >"$SERVER_LOG" 2>&1 &

  local pid=$!
  echo "$pid" >"$PID_FILE"
  log_info "已起进程 PID=$pid"

  # 等待 /api/health 上线
  local health_url="http://127.0.0.1:${DEFAULT_PORT}/api/health"
  if wait_for_health "$health_url" 15; then
    log_info "服务就绪 — $health_url 返回 200"
    log_info "浏览器面板: http://127.0.0.1:${DEFAULT_PORT}/"
  else
    log_err "服务 15s 内未就绪 — 查看 $SERVER_LOG 末尾"
    tail -40 "$SERVER_LOG" >&2 || true
    exit 1
  fi
}

cmd_stop() {
  if ! pid="$(is_running)"; then
    log_info "服务未运行"
    rm -f "$PID_FILE"
    exit 0
  fi
  log_info "停止服务 PID=$pid"
  kill "$pid" 2>/dev/null || true
  # 等待优雅退出
  for _ in 1 2 3 4 5; do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    log_warn "未优雅退出 — SIGKILL"
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  log_info "已停止"
}

cmd_status() {
  if pid="$(is_running)"; then
    log_info "运行中 — PID=$pid  端口 ${DEFAULT_PORT}"
    curl=$(ps -o stat= -p "$pid" 2>/dev/null || echo "?")
    log_info "进程状态: $sf"
  else
    log_info "未运行"
  fi
  if [[ -f "$SERVER_LOG" ]]; then
    log_info "server.log 大小: $(wc -c <"$SERVER_LOG" | tr -d ' ') 字节"
    log_info "server.log 末尾 5 行:"
    tail -5 "$SERVER_LOG" >&2 || true
  fi
}

cmd_tail() {
  if [[ ! -f "$SERVER_LOG" ]]; then
    log_warn "server.log 不存在 — 先 ./run.sh"
    exit 1
  fi
  exec tail -F "$SERVER_LOG"
}

# ============ 入口 ============
case "${1:-serve}" in
  serve)    shift; cmd_serve "$@" ;;
  stop)     shift; cmd_stop "$@" ;;
  restart)  cmd_stop; cmd_serve ;;
  status)   shift; cmd_status "$@" ;;
  tail)     shift; cmd_tail "$@" ;;
  -h|--help|help)
    sed -n '2,/^# 前置/!d; /^# 前置/q' "$0" | sed 's/^# \{0,1\}//'
    ;;
  *)
    log_err "未知子命令: $1"
    log_info "用法: $0 {serve|stop|restart|status|tail}"
    exit 64  # EX_USAGE
    ;;
esac
