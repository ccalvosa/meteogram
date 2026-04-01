from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from data import get_meteogram_data
import uvicorn

app = FastAPI(title="Meteograma API", version="1.0.0")

# CORS: permite peticiones desde el frontend (GitHub Pages, Netlify, localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, cambiar por la URL real del frontend
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Meteograma API operativa"}


@app.get("/meteogram")
def meteogram(
    lat: float = Query(..., ge=-90, le=90, description="Latitud en grados"),
    lon: float = Query(..., ge=-180, le=180, description="Longitud en grados"),
):
    """
    Devuelve los datos de meteograma para un punto (lat, lon).
    Extrae la última pasada disponible del ECMWF IFS ENS.
    """
    try:
        data = get_meteogram_data(lat, lon)
        return data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al acceder a los datos: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
