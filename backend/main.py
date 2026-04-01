from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from data import get_meteogram_data
import uvicorn

app = FastAPI(title="Meteograma API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Meteograma API operativa", "models": ["ecmwf", "gefs"]}


@app.get("/meteogram")
def meteogram(
    lat: float = Query(..., ge=-90, le=90, description="Latitud en grados"),
    lon: float = Query(..., ge=-180, le=180, description="Longitud en grados"),
    model: str = Query("ecmwf", pattern="^(ecmwf|gefs)$", description="Modelo: ecmwf o gefs"),
):
    """
    Devuelve los datos de meteograma para un punto (lat, lon) y modelo elegido.
    """
    try:
        data = get_meteogram_data(lat, lon, model)
        return data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al acceder a los datos: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
