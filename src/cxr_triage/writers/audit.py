"""审计日志双层落盘 — ``audit_log``(SQLite) + ``tool_calls_<YYYY-MM-DD>.jsonl``。

设计原则（spec.md §5 维度 6 审计留痕 + 飞检可追溯 + AI 治理审查）：

1. **双层结构**::

       audit_log (SQLite, 飞检/对象查询)
           ├── actor_role + ts + study_uid + event_type + payload_json
           └── 业务状态机、外发回写、PACS 入站、模型降级等统一入口

       tool_calls_<YYYY-MM-DD>.jsonl (JSONL, 每日轮换)
           ├── ts + study_uid + tool + model + prompt_tokens + completion_tokens
           │   + latency_ms + status
           └── 每次 LLM 调用一行；status ∈ {ok, fail, degraded}

2. **职责清晰**：
   - ``audit_log_insert`` 记录**业务事件**（状态跃迁、外发成功/失败、
     PACS 入站、模型降级、对账节点），飞检/管理层查询用；
   - ``tool_call_record`` 记录**模型调用元数据**（prompt 量、latency、
     status），AI 治理 / 单条调试用。两者不互相替代。
3. **并发安全**：
   - SQLite 由 ``aiosqlite`` 单连接串行化；
   - JSONL 由 ``asyncio.Lock`` + ``open("a")`` O_APPEND 行原子追加；
     Python 文本 I/O 单行写入由 GIL 守护，asyncio.Lock 保护多任务交
     错顺序即可。
4. **不静默丢数据**：
   - ``audit_log_insert`` 失败抛 ``AuditLoggerError``，由 caller 决策；
   - ``tool_call_record`` 失败同样抛错；不"返回 None 把责任推给调用方"
5. **payload 灵活**：
   - ``audit_log_insert`` 接受 dict / str / None；dict 由 ``json.dumps``
     序列化（ensure_ascii=False 保留中文所见）；
6. **JSONL 文件轮换**：文件名带 UTC 日期 ``YYYY-MM-DD``；同一 UTC 日多
   次调用落在同一文件。便于审计侧按日切片拉取与归档。
7. **不与业务层耦合**：
   - 上游（state_machine / pacs_listener / engine / writer）只调本类两
     个方法；本类不直接反向耦合任何业务模块；
   - 测试可独立 mock Database 实例 + tmpdir tool_calls_dir。

参考：源仓库 ``medrax/agent/agent.py`` 工具调用日志思想；本任务是把
"模型调用元数据"显式落盘便于院方 AI 治理 / 飞检审查。
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from ..persistence.sqlite import Database

logger = logging.getLogger(__name__)


# =====================================================================
# 常量
# =====================================================================
DEFAULT_TOOL_CALLS_DIR = "/tmp/cxr-triage/tool_calls"

# tool_call_record status 枚举（spec.md §5 维度 4 与全局 Phase 安全边界）
VALID_TOOL_CALL_STATUSES = frozenset({"ok", "fail", "degraded"})

# audit_log_insert 接受的角色集合（与 state.ActorRole 同源；这里独立枚举
# 是为了 writers 不反向依赖 state 模块，避免循环）
VALID_AUDIT_ACTOR_ROLES = frozenset({"jr", "reviewer", "sr", "system"})

# 双写对账事件（spec.md §5 维度 6 飞检双写追溯）
#   - ok / failed 拆两个 event_type，便于飞检 `WHERE event_type LIKE
#     '%.dual_write%'` 一键拉全 + `event_type='ris.dual_write'` 单独筛
#     成功对账
DUAL_WRITE_VALID_STATUSES = frozenset({"ok", "fail"})

#: RIS 双写成功事件
DUAL_WRITE_EVENT_RIS_OK: str = "ris.dual_write"
#: RIS 双写失败事件
DUAL_WRITE_EVENT_RIS_FAILED: str = "ris.dual_write_failed"
#: 体检主检双写成功事件
DUAL_WRITE_EVENT_HEALTHCHECK_OK: str = "healthcheck.dual_write"
#: 体检主检双写失败事件
DUAL_WRITE_EVENT_HEALTHCHECK_FAILED: str = "healthcheck.dual_write_failed"

# 默认双写行为主体（终审签发）
DUAL_WRITE_DEFAULT_ACTOR_ROLE: str = "sr"


# =====================================================================
# 异常
# =====================================================================
class AuditLoggerError(Exception):
    """audit_log/tool_calls 写失败时抛。"""


# =====================================================================
# 时间工具
# =====================================================================
def utc_now_iso() -> str:
    """UTC ISO8601 + 毫秒 + Z 后缀。例 ``2026-10-13T07:23:11.487Z``。"""
    now = datetime.now(timezone.utc)
    # millisecond 精度；保留 3 位
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def utc_today_iso() -> str:
    """UTC 日期 ``YYYY-MM-DD``。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# =====================================================================
# AuditLogger
# =====================================================================
class AuditLogger:
    """双层审计日志：``audit_log``(SQLite) + ``tool_calls_<date>.jsonl``。

    Parameters
    ----------
    db:
        ``Database`` 实例；允许 ``None`` 表示懒加载（首次 ``ensure_db()``
        时按 ``db_path`` 创建）。
    db_path:
        懒加载 ``Database`` 时的路径；缺省 ``/tmp/cxr-triage.db``。
    tool_calls_dir:
        ``tool_calls_<date>.jsonl`` 文件目录；缺省
        ``/tmp/cxr-triage/tool_calls``。
    """

    def __init__(
        self,
        db: Optional[Database] = None,
        db_path: Optional[Union[str, Path]] = None,
        tool_calls_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self._db: Optional[Database] = db
        self._db_path: Optional[Path] = (
            Path(db_path) if db_path is not None else None
        )
        self._owns_db = db is None
        self._tool_calls_dir = Path(
            tool_calls_dir if tool_calls_dir is not None else DEFAULT_TOOL_CALLS_DIR
        )
        self._tool_calls_dir.mkdir(parents=True, exist_ok=True)
        # JSONL 写盘锁：多任务并发追加避免内容交错
        self._jsonl_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # 数据库生命周期
    # ------------------------------------------------------------------
    async def ensure_db(self, db: Optional[Database] = None) -> Database:
        """保证 db 已连接；返回可用的 Database 实例。

        - 显式传入 db 时优先使用（schema 由调用方负责初始化）；
        - 否则使用构造时的 ``db_path`` 懒加载；连接后自动 ``init_db()``
          一次，使 audit_log 表存在（仅在 owned 路径触发）；
        - ``Database.connect()`` 幂等（重复调用不会重连）。
        """
        target = db or self._db
        is_lazy = target is None
        if is_lazy:
            target_path = self._db_path
            if target_path is None:
                target_path = Path("/tmp/cxr-triage.db")
            target = Database(target_path)
            self._db = target
            self._owns_db = True
        await target.connect()
        if is_lazy:
            # 仅在懒加载路径下初始化 schema（避免重复 CREATE TABLE IF NOT
            # EXISTS 开销；幂等可重入）。
            from ..persistence.migrations import init_db  # 局部 import 防循环
            await init_db(target)
        return target

    async def close(self) -> None:
        """仅当 self 持有 db 所有权（懒加载场景）时关闭 db。"""
        if self._owns_db and self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def tool_calls_dir(self) -> Path:
        return self._tool_calls_dir

    # ------------------------------------------------------------------
    # 写 audit_log（SQLite，飞检业务事件流）
    # ------------------------------------------------------------------
    async def audit_log_insert(
        self,
        actor_role: str,
        study_uid: str,
        event_type: str,
        payload: Union[dict, str, None] = None,
        ts: Optional[str] = None,
    ) -> int:
        """写一条审计行到 ``audit_log`` 表，返回 row id（自增主键）。

        Parameters
        ----------
        actor_role:
            行为主体；枚举 ``jr`` / ``reviewer`` / ``sr`` / ``system``。
        study_uid:
            DICOM ``StudyInstanceUID`` 或 ``""``（系统事件）。
        event_type:
            事件类型名（点分 snake_case），如 ``state.transition``、
            ``model.degraded``、``ris.written`` 等。
        payload:
            事件载荷；dict 由 ``json.dumps(ensure_ascii=False)`` 序列化；
            str 原样写；None 转 ``"{}"``。
        ts:
            可选 UTC ISO8601；缺省 ``utc_now_iso()``。

        Returns
        -------
        int
            新插入行的 ``id``。
        """
        if actor_role not in VALID_AUDIT_ACTOR_ROLES:
            raise AuditLoggerError(
                f"无效 actor_role: {actor_role!r}; "
                f"允许 {sorted(VALID_AUDIT_ACTOR_ROLES)}"
            )
        if not event_type:
            raise AuditLoggerError("event_type 不能为空")

        if payload is None:
            payload_json = "{}"
        elif isinstance(payload, str):
            payload_json = payload if payload else "{}"
        elif isinstance(payload, dict):
            try:
                payload_json = json.dumps(payload, ensure_ascii=False)
            except (TypeError, ValueError) as exc:
                raise AuditLoggerError(
                    f"payload 序列化失败: {exc}"
                ) from exc
        else:
            raise AuditLoggerError(
                f"payload 类型不支持: {type(payload).__name__}"
            )

        ts_value = ts or utc_now_iso()
        db = await self.ensure_db()
        async with db.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO audit_log "
                "(actor_role, ts, study_uid, event_type, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (actor_role, ts_value, study_uid, event_type, payload_json),
            )
            row_id = cur.lastrowid
            await cur.close()
        logger.debug(
            "audit_log row id=%s study=%s event=%s role=%s",
            row_id,
            study_uid,
            event_type,
            actor_role,
        )
        return int(row_id or 0)

    # ------------------------------------------------------------------
    # 写 tool_calls_<YYYY-MM-DD>.jsonl（JSONL，AI 治理 / 模型调用元数据）
    # ------------------------------------------------------------------
    async def tool_call_record(
        self,
        tool: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        status: str,
        study_uid: str = "",
        ts: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> Path:
        """追加一行 ``tool_calls_<YYYY-MM-DD>.jsonl``；返回写入文件路径。

        Parameters
        ----------
        tool:
            工具名（如 ``cxr_keyqa`` / ``finding_draft`` / ``classifier`` 等）。
        model:
            模型 id（OpenAI-compatible 返回的 ``model`` 字段）；模型不可达
            时亦可写 ``"unknown"``，方便审计侧筛"无模型标识"行。
        prompt_tokens / completion_tokens:
            整数；模型返回的 usage；缺失写 0。
        latency_ms:
            从调起到接收到响应/失败的耗时 ms。
        status:
            ``ok`` / ``fail`` / ``degraded``；fail 通常对应网络或超
            时，degraded 对应 retry 失败后走确定性降级。
        study_uid:
            关联的 DICOM ``StudyInstanceUID``；可空。
        ts:
            可选 UTC ISO8601；缺省 ``utc_now_iso()``。
        extra:
            可选 dict；附加字段（如 ``http_status`` / ``error_class``），
            最终落到 JSONL ``extra`` 子键。

        Returns
        -------
        pathlib.Path
            写入的 JSONL 文件路径（按 UTC 日期切片）。
        """
        if status not in VALID_TOOL_CALL_STATUSES:
            raise AuditLoggerError(
                f"无效 status: {status!r}; 允许 {sorted(VALID_TOOL_CALL_STATUSES)}"
            )
        if not tool:
            raise AuditLoggerError("tool 不能为空")
        if not isinstance(prompt_tokens, int) or prompt_tokens < 0:
            raise AuditLoggerError(
                f"prompt_tokens 必须是非负整数: {prompt_tokens!r}"
            )
        if not isinstance(completion_tokens, int) or completion_tokens < 0:
            raise AuditLoggerError(
                f"completion_tokens 必须是非负整数: {completion_tokens!r}"
            )
        if not isinstance(latency_ms, int) or latency_ms < 0:
            raise AuditLoggerError(
                f"latency_ms 必须是非负整数: {latency_ms!r}"
            )

        ts_value = ts or utc_now_iso()
        date_str = ts_value[:10]  # 截 YYYY-MM-DD
        record: dict[str, Any] = {
            "ts": ts_value,
            "study_uid": study_uid,
            "tool": tool,
            "model": model,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "latency_ms": int(latency_ms),
            "status": status,
        }
        if extra:
            record["extra"] = extra

        line = json.dumps(record, ensure_ascii=False) + "\n"
        path = self._tool_calls_dir / f"tool_calls_{date_str}.jsonl"

        async with self._jsonl_lock:
            # O_APPEND + 行大小 < PIPE_BUF 保证每次 write 原子追加单行
            try:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(line)
            except OSError as exc:
                raise AuditLoggerError(
                    f"tool_calls JSONL 写盘失败 {path}: {exc}"
                ) from exc
        return path

    # ------------------------------------------------------------------
    # 查询 / 调试入口
    # ------------------------------------------------------------------
    async def list_audit_for_study(self, study_uid: str) -> list[dict]:
        """拉某 ``study_uid`` 全部 ``audit_log`` 行，按 ts ASC。"""
        db = await self.ensure_db()
        async with db.connection() as conn:
            cur = await conn.execute(
                "SELECT actor_role, ts, study_uid, event_type, payload_json "
                "FROM audit_log WHERE study_uid = ? ORDER BY ts ASC",
                (study_uid,),
            )
            rows = await cur.fetchall()
            await cur.close()
        results: list[dict] = []
        for row in rows:
            try:
                payload_obj = json.loads(row["payload_json"])
            except (TypeError, ValueError):
                payload_obj = row["payload_json"]
            results.append(
                {
                    "actor_role": row["actor_role"],
                    "ts": row["ts"],
                    "study_uid": row["study_uid"],
                    "event_type": row["event_type"],
                    "payload": payload_obj,
                }
            )
        return results

    async def count_audit_for_study(self, study_uid: str) -> int:
        db = await self.ensure_db()
        async with db.connection() as conn:
            cur = await conn.execute(
                "SELECT COUNT(*) AS n FROM audit_log WHERE study_uid = ?",
                (study_uid,),
            )
            row = await cur.fetchone()
            await cur.close()
        return int(row["n"]) if row is not None else 0

    def list_tool_call_files(self) -> list[Path]:
        """列 ``tool_calls`` 目录下全部 JSONL 文件（按文件名升序）。"""
        return sorted(self._tool_calls_dir.glob("tool_calls_*.jsonl"))

    @staticmethod
    def read_tool_call_records(path: Union[str, Path]) -> list[dict]:
        """同步读 JSONL 文件全部记录为 dict 列表；空行/损坏行跳过。"""
        records: list[dict] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError:
                    logger.warning(
                        "tool_calls JSONL 损坏行 %s:%s 已跳过",
                        path,
                        line_no,
                    )
                    continue
        return records

    # ------------------------------------------------------------------
    # 双写审计（RIS + 体检主检 同步对账）
    # ------------------------------------------------------------------
    async def write_dual_write(
        self,
        study_uid: str,
        ris_status: str,
        healthcheck_status: str,
        actor_role: str = "sr",
        ts: Optional[str] = None,
        ris_extra: Optional[dict] = None,
        healthcheck_extra: Optional[dict] = None,
    ) -> tuple[int, int]:
        """RIS + 体检主检双写审计落盘（spec.md §5 维度 6 飞检双写对账）。

        同一 ``study_uid`` 由终审签发医师（默认 actor_role=sr）触发双
        写时，对账要求每条 study 落 **两条** ``audit_log`` 行：一条
        ``ris.dual_write`` + 一条 ``healthcheck.dual_write``。两条 ts
        **统一**（同一 UTC ISO8601 + 毫秒），便于飞检按 ``(study_uid,
        ts)`` 一键还原某次双写尝试。

        每条 payload 显式带 ``status``（"ok" / "fail"）+ ``kind`` =
        ``"dual_write"``；可选 ``extra`` 子键把 HTTP status_code /
        payload_size / outbox_id 等附加信息一并落库。

        Parameters
        ----------
        study_uid:
            DICOM ``StudyInstanceUID``；非空字符串。
        ris_status:
            RIS 写回结果，约定 ``"ok"`` / ``"fail"``；非空。
        healthcheck_status:
            体检主检写回结果，约定 ``"ok"`` / ``"fail"``；非空。
        actor_role:
            行为主体；默认 ``"sr"``（终审签发）；白名单见
            ``VALID_AUDIT_ACTOR_ROLES``。
        ts:
            可选 UTC ISO8601；缺省 ``utc_now_iso()``。两条 audit 行共用
            同一 ``ts``，便于按时间切片对齐飞检回放。
        ris_extra / healthcheck_extra:
            可选 dict 附加字段（如 ``http_status_code`` /
            ``payload_size_bytes`` / ``outbox_id``）；最终落到对应行
            ``payload_json`` 里。

        Returns
        -------
        tuple[int, int]
            ``(ris_audit_id, healthcheck_audit_id)``：audit_log
            ``id`` 自增主键对。
        """
        if not isinstance(study_uid, str):
            raise AuditLoggerError(
                f"study_uid 必须是字符串: {type(study_uid).__name__}"
            )
        # 拒绝纯空白 / 含 NUL 字符 / 全空白情形
        if not study_uid or not study_uid.strip() or "\x00" in study_uid:
            raise AuditLoggerError(
                f"study_uid 必须是非空字符串: {study_uid!r}"
            )
        if actor_role not in VALID_AUDIT_ACTOR_ROLES:
            raise AuditLoggerError(
                f"无效 actor_role: {actor_role!r}; "
                f"允许 {sorted(VALID_AUDIT_ACTOR_ROLES)}"
            )
        if ris_status not in DUAL_WRITE_VALID_STATUSES:
            raise AuditLoggerError(
                f"无效 ris_status: {ris_status!r}; "
                f"允许 {sorted(DUAL_WRITE_VALID_STATUSES)}"
            )
        if healthcheck_status not in DUAL_WRITE_VALID_STATUSES:
            raise AuditLoggerError(
                f"无效 healthcheck_status: {healthcheck_status!r}; "
                f"允许 {sorted(DUAL_WRITE_VALID_STATUSES)}"
            )

        # 统一 ts：飞检"哪条 study 在什么时间由谁触发了双写"的关联键
        ts_value = ts or utc_now_iso()

        ris_payload: dict[str, Any] = {
            "kind": "dual_write",
            "status": ris_status,
            "target": "ris",
            "pair_ts": ts_value,
        }
        if ris_extra:
            ris_payload.update(ris_extra)

        healthcheck_payload: dict[str, Any] = {
            "kind": "dual_write",
            "status": healthcheck_status,
            "target": "healthcheck",
            "pair_ts": ts_value,
        }
        if healthcheck_extra:
            healthcheck_payload.update(healthcheck_extra)

        ris_event = (
            DUAL_WRITE_EVENT_RIS_OK
            if ris_status == "ok"
            else DUAL_WRITE_EVENT_RIS_FAILED
        )
        healthcheck_event = (
            DUAL_WRITE_EVENT_HEALTHCHECK_OK
            if healthcheck_status == "ok"
            else DUAL_WRITE_EVENT_HEALTHCHECK_FAILED
        )

        ris_id = await self.audit_log_insert(
            actor_role=actor_role,
            study_uid=study_uid,
            event_type=ris_event,
            payload=ris_payload,
            ts=ts_value,
        )
        healthcheck_id = await self.audit_log_insert(
            actor_role=actor_role,
            study_uid=study_uid,
            event_type=healthcheck_event,
            payload=healthcheck_payload,
            ts=ts_value,
        )
        logger.debug(
            "dual_write rows study=%s ts=%s ris_id=%s (%s) hc_id=%s (%s) role=%s",
            study_uid,
            ts_value,
            ris_id,
            ris_status,
            healthcheck_id,
            healthcheck_status,
            actor_role,
        )
        return (ris_id, healthcheck_id)


# =====================================================================
# 工厂
# =====================================================================
def build_audit_logger(
    db: Optional[Database] = None,
    db_path: Optional[Union[str, Path]] = None,
    tool_calls_dir: Optional[Union[str, Path]] = None,
) -> AuditLogger:
    """工厂：创建 ``AuditLogger`` 实例。

    用于 FastAPI lifespan 注入 / 测试 fixture 构造。
    """
    return AuditLogger(
        db=db,
        db_path=db_path,
        tool_calls_dir=tool_calls_dir,
    )


__all__ = [
    "AuditLogger",
    "AuditLoggerError",
    "DEFAULT_TOOL_CALLS_DIR",
    "VALID_TOOL_CALL_STATUSES",
    "VALID_AUDIT_ACTOR_ROLES",
    "DUAL_WRITE_VALID_STATUSES",
    "DUAL_WRITE_EVENT_RIS_OK",
    "DUAL_WRITE_EVENT_RIS_FAILED",
    "DUAL_WRITE_EVENT_HEALTHCHECK_OK",
    "DUAL_WRITE_EVENT_HEALTHCHECK_FAILED",
    "DUAL_WRITE_DEFAULT_ACTOR_ROLE",
    "utc_now_iso",
    "utc_today_iso",
    "build_audit_logger",
]
