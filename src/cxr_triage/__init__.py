"""体检胸片所见草稿系统 — 把 PACS 落片转成 RIS + 体检主检可见的所见草稿。

子系统: 医技系统（RIS/PACS） / 体检中心
方法族: 多工具协同推理与结构化输出（StateGraph 编排 5 件工具 + 阳性分诊）
交付形态: 常驻 FastAPI 服务 + 浏览器实时面板（SSE 推送）
集成面: PACS DICOM C-STORE / WADO-RS（读）+ RIS HTTP finding-draft（写）
       + 体检主检 HTTP summary-ref（写）+ ModelGateway（OpenAI-compatible 出站）

所有外发凭据一律从环境变量读；任何代码路径都不得硬编码 key/token/host。
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
