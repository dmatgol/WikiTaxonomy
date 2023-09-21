"""Api entrypoint."""
import socket

import uvicorn
from fastapi import FastAPI

from api import cache_results_router, inference_router
from api.api_models import APIState

app = FastAPI(
    title="Wikipedia Taxonomy Classification",
    description="""Predict the taxonomic category of wikipedia articles from their text.
    This project aims to predict the L1 classes. L1 classes are 9 category classes.""",
    version="0.1.0",
)


app.include_router(inference_router.router, prefix="/predict")
app.include_router(cache_results_router.router, prefix="/cache")


@app.get("/", response_model=APIState)
def heartbeat():
    """Return state of API."""
    print("heartbeat queried")
    return APIState(machine_name=socket.gethostname(), version=app.version)


if __name__ == "__main__":
    uvicorn.run("api_entrypoint_main:app", host="0.0.0.0", port=8001)
