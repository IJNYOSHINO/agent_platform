from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
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
from memory.long_term import get_long_term_memory
from services.task_runner import get_task_runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    logger.info("Starting %s v%s", cfg.app_title, cfg.app_version)

    Path("data").mkdir(exist_ok=True)
    init_db()

    registry = get_registry()
    tools_dir = cfg.tools_dir
    count = registry.scan_directory(tools_dir)
    logger.info("Loaded %d tools from '%s'", count, tools_dir)

    stop_event = asyncio.Event()
    worker_task = asyncio.create_task(get_task_runner().run_forever(stop_event))
    logger.info("Task worker started.")

    try:
        yield
    finally:
        stop_event.set()
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        logger.info("Shutting down.")


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

    cors_origins = cfg.cors_origins or (
        [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]
        if cfg.env != "prod"
        else []
    )
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        logger.info("CORS middleware disabled until explicit origins are configured.")

    app.include_router(auth_router, prefix="/api")
    app.include_router(router, prefix="/api", tags=["Agent"], dependencies=[Depends(get_current_user)])

    @app.get("/api/health", tags=["System"])
    async def health_check():
        registry = get_registry()
        memory = get_long_term_memory()
        return {
            "status": "ok",
            "version": cfg.app_version,
            "tools_loaded": len(registry),
            "long_term_memory": {
                "enabled": memory.is_enabled,
                "model": cfg.embedding_model if memory.is_enabled else None,
            },
        }

    @app.get("/favicon.ico", tags=["System"])
    async def favicon():
        favicon_path = Path(__file__).parent / "static" / "favicon.ico"
        if not favicon_path.exists():
            return None
        return FileResponse(str(favicon_path))

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
