from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from api.auth import router as auth_router
from api.routes import router
from config.settings import get_settings
from core.database import init_db
from core.registry import get_registry
from core.security import get_current_user
import memory.long_term as _ltm_module
from memory.long_term import get_long_term_memory, close_long_term_memory
from services.task_runner import get_task_runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Worker 异常回调
# ──────────────────────────────────────────────────────────
def _on_worker_done(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error("Task worker crashed unexpectedly: %s", exc, exc_info=exc)


# ──────────────────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    logger.info("Starting %s v%s", cfg.app_title, cfg.app_version)

    Path("data").mkdir(exist_ok=True)

    # DB 初始化（同步 IO，放进 executor 避免阻塞 event loop）
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, init_db)
        logger.info("Database initialized.")
    except Exception:
        logger.exception("Failed to initialize database.")
        raise  # 致命错误，中止启动

    # 工具扫描
    registry = get_registry()
    try:
        count = await loop.run_in_executor(None, registry.scan_directory, cfg.tools_dir)
        logger.info("Loaded %d tools from '%s'.", count, cfg.tools_dir)
    except Exception:
        logger.exception("Failed to load tools.")

    # 启动 worker
    stop_event = asyncio.Event()
    worker_task = asyncio.create_task(get_task_runner().run_forever(stop_event))
    worker_task.add_done_callback(_on_worker_done)
    logger.info("Task worker started.")

    # 短期记忆：初始化 + warmup
    try:
        from memory.short_term import get_short_term_memory
        short_term = await get_short_term_memory()
        await short_term.warmup()
        logger.info("Short-term memory warmed up.")
    except Exception:
        logger.warning("Failed to warmup short-term memory.", exc_info=True)

    # 长期记忆：初始化 + warmup（对称）
    try:
        long_term = await get_long_term_memory()
        await long_term.warmup()
        logger.info("Long-term memory warmed up.")
    except Exception:
        logger.warning("Failed to warmup long-term memory.", exc_info=True)

    try:
        yield
    finally:
        logger.info("Shutting down...")

        # 优雅停止 worker（30s 超时后强行 cancel）
        stop_event.set()
        try:
            await asyncio.wait_for(worker_task, timeout=30)
        except asyncio.TimeoutError:
            logger.warning("Task worker did not stop in 30s, cancelling.")
            worker_task.cancel()
            try:
                await worker_task
            except (asyncio.CancelledError, Exception):
                pass
        except asyncio.CancelledError:
            pass

        # 关闭短期记忆
        try:
            from memory.short_term import close_short_term_memory
            await close_short_term_memory()
            logger.info("Short-term memory closed.")
        except Exception:
            logger.warning("Failed to close short-term memory.", exc_info=True)

        # 关闭长期记忆
        try:
            await close_long_term_memory()
            logger.info("Long-term memory closed.")
        except Exception:
            logger.warning("Failed to close long-term memory.", exc_info=True)


# ──────────────────────────────────────────────────────────
# App 创建
# ──────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title=cfg.app_title,
        version=cfg.app_version,
        description=cfg.app_description,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS（统一从配置读，不在代码里硬判断环境）
    cors_origins = cfg.cors_origins or []
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        logger.warning(
            "No CORS origins configured (CORS_ORIGINS is empty). "
            "Cross-origin requests will be blocked. "
            "Set CORS_ORIGINS in your .env if needed."
        )

    # 路由
    app.include_router(auth_router, prefix="/api")
    app.include_router(
        router,
        prefix="/api",
        tags=["Agent"],
        dependencies=[Depends(get_current_user)],
    )

    # ── 健康检查 ──────────────────────────────────────────

    @app.get("/api/health", tags=["System"])
    async def health_check():
        registry = get_registry()
        # 实时读模块变量，不受导入时绑定影响
        memory = _ltm_module._global_long_term_memory

        return {
            "status": "ok",
            "version": cfg.app_version,
            "tools_loaded": len(registry),
            "long_term_memory": {
                "initialized": memory is not None,
                "enabled": memory.is_enabled if memory is not None else False,
                "model": cfg.embedding_model if (memory is not None and memory.is_enabled) else None,
            },
        }

    # ── favicon ───────────────────────────────────────────

    @app.get("/favicon.ico", tags=["System"], include_in_schema=False)
    async def favicon():
        favicon_path = Path(__file__).parent / "static" / "favicon.ico"
        if not favicon_path.exists():
            raise HTTPException(status_code=404)
        return FileResponse(str(favicon_path))

    return app


app = create_app()


# ──────────────────────────────────────────────────────────
# 本地启动
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )