from fastapi import FastAPI
from app.routers import inference, result

def create_app() -> FastAPI:
    app = FastAPI(title="Products sold in every shop Prediction")

    app.include_router(inference.router)
    app.include_router(result.router)

    @app.get("/health")
    async def health() -> str:
        return "ok"
    
    return app