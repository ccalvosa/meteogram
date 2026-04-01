import xarray as xr
import numpy as np
import pandas as pd
from functools import lru_cache

ZARR_URL = "https://data.dynamical.org/ecmwf/ifs-ens/forecast-15-day-0-25-degree/latest.zarr"


def _clean(arr) -> list:
    """Convierte un array numpy a lista, reemplazando NaN/Inf por None (JSON null)."""
    return [None if (x is None or np.isnan(x) or np.isinf(x)) else float(x) for x in arr]

# Variables que necesitamos — solo cargamos estas para no bajar todo el zarr
VARIABLES = [
    "temperature_2m",
    "precipitation_surface",
    "wind_gust_10m",
]




@lru_cache(maxsize=1)
def _open_dataset() -> xr.Dataset:
    """
    Abre el zarr remoto una sola vez y cachea el objeto Dataset.
    Solo descarga metadatos y coordenadas, no los datos en sí.
    """
    print("Abriendo dataset zarr remoto...")
    ds = xr.open_zarr(
        ZARR_URL,
        consolidated=True,
    )
    return ds


def get_meteogram_data(lat: float, lon: float) -> dict:
    """
    Extrae los datos de meteograma para el punto (lat, lon) más cercano.
    Devuelve un diccionario JSON-serializable.
    """
    ds = _open_dataset()

    # Seleccionamos la última pasada disponible
    latest_init = pd.Timestamp(ds.init_time.values[-1])

    # Seleccionamos el punto más cercano y la última pasada
    # method="nearest" evita errores si el punto no está en la rejilla exacta
    point = ds[VARIABLES].sel(
        init_time=latest_init,
        latitude=lat,
        longitude=lon,
        method="nearest",
    ).compute()  # Aquí es donde se descarga realmente (solo ese punto)

    # Coordenadas reales del punto en la rejilla (pueden diferir ligeramente)
    real_lat = float(point.latitude)
    real_lon = float(point.longitude)

    # Tiempos válidos como strings ISO
    # valid_time no existe como coordenada directa: lead_time + init_time
    lead_times = point.lead_time.values  # timedelta64
    valid_times = [
        (latest_init + pd.Timedelta(lt)).isoformat()
        for lt in lead_times
    ]

    # --- Temperatura 2m ---
    # Shape: (lead_time, ensemble_member) → lista de listas
    t2m = point["temperature_2m"].values  # (n_lead, n_member)
    temperature = {
        "valid_time": valid_times,
        # Cada miembro: lista de longitud n_lead
        "members": {
            str(int(m)): _clean(t2m[:, i])
            for i, m in enumerate(point.ensemble_member.values)
        },
        "units": "°C",
    }

    # --- Precipitación ---
    # La variable es tasa media desde el paso anterior (kg m-2 s-1 = mm/s)
    # Para convertir a mm/h: * 3600
    # Para acumular: multiplicar por el intervalo en segundos entre pasos
    prate = point["precipitation_surface"].values  # (n_lead, n_member)

    # Tasa en mm/h por miembro
    precip_mmh = prate * 3600  # (n_lead, n_member)
    precip_mmh = np.maximum(precip_mmh, 0)

    n_members = precip_mmh.shape[1]

    precipitation = {
        "valid_time": valid_times,
        "members": {
            str(int(m)): _clean(precip_mmh[:, i])
            for i, m in enumerate(point.ensemble_member.values)
        },
        "units": "mm/h",
    }

    # --- Rachas de viento ---
    # wind_gust_10m en m/s → convertimos a km/h
    gust_ms = point["wind_gust_10m"].values  # (n_lead, n_member)
    gust_kmh = gust_ms * 3.6

    wind_gust = {
        "valid_time": valid_times,
        "members": {
            str(int(m)): _clean(gust_kmh[:, i])
            for i, m in enumerate(point.ensemble_member.values)
        },
        "units": "km/h",
    }

    # --- Respuesta final ---
    return {
        "metadata": {
            "init_time": latest_init.isoformat(),
            "requested_lat": lat,
            "requested_lon": lon,
            "nearest_lat": real_lat,
            "nearest_lon": real_lon,
            "n_members": n_members,
            "n_lead_times": len(valid_times),
            "source": "ECMWF IFS ENS via dynamical.org",
        },
        "temperature_2m": temperature,
        "precipitation": precipitation,
        "wind_gust_10m": wind_gust,
    }
