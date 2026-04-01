import xarray as xr
import numpy as np
import pandas as pd
from functools import lru_cache

# ─── URLs ────────────────────────────────────────────────────────────────────
ZARR_ECMWF = "https://data.dynamical.org/ecmwf/ifs-ens/forecast-15-day-0-25-degree/latest.zarr"
ZARR_GEFS  = "https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr"

# Leadtime máximo compartido: 15 días = 360 horas
MAX_LEAD_HOURS = 360

# Variables por modelo
VARS_ECMWF = ["temperature_2m", "precipitation_surface", "wind_gust_10m"]
VARS_GEFS  = ["temperature_2m", "precipitation_surface", "wind_u_10m", "wind_v_10m"]


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _clean(arr) -> list:
    """Convierte un array numpy a lista, reemplazando NaN/Inf por None (JSON null)."""
    return [None if (x is None or np.isnan(x) or np.isinf(x)) else float(x) for x in arr]


def _members_dict(data2d, member_coords) -> dict:
    """Convierte array (n_lead, n_member) en dict {member_id: [valores]}."""
    return {
        str(int(m)): _clean(data2d[:, i])
        for i, m in enumerate(member_coords)
    }


def _valid_times(latest_init: pd.Timestamp, lead_times) -> list:
    return [(latest_init + pd.Timedelta(lt)).isoformat() for lt in lead_times]


def _precip_mmh(prate: np.ndarray) -> np.ndarray:
    """Convierte tasa de precipitación (kg m-2 s-1 = mm/s) a mm/h, sin negativos."""
    return np.maximum(prate * 3600, 0)


# ─── Apertura de datasets (cacheados) ────────────────────────────────────────
@lru_cache(maxsize=1)
def _open_ecmwf() -> xr.Dataset:
    print("Abriendo dataset ECMWF IFS ENS...")
    return xr.open_zarr(ZARR_ECMWF, consolidated=True)


@lru_cache(maxsize=1)
def _open_gefs() -> xr.Dataset:
    print("Abriendo dataset NOAA GEFS...")
    return xr.open_zarr(ZARR_GEFS, consolidated=True)


# ─── Función principal ───────────────────────────────────────────────────────
def get_meteogram_data(lat: float, lon: float, model: str) -> dict:
    """
    Extrae datos de meteograma para el punto (lat, lon) más cercano.
    model: 'ecmwf' | 'gefs'
    """
    if model == "ecmwf":
        return _get_ecmwf(lat, lon)
    elif model == "gefs":
        return _get_gefs(lat, lon)
    else:
        raise ValueError(f"Modelo no soportado: {model}. Usa 'ecmwf' o 'gefs'.")


# ─── ECMWF IFS ENS ───────────────────────────────────────────────────────────
def _get_ecmwf(lat: float, lon: float) -> dict:
    ds = _open_ecmwf()
    latest_init = pd.Timestamp(ds.init_time.values[-1])

    point = ds[VARS_ECMWF].sel(
        init_time=latest_init,
        latitude=lat,
        longitude=lon,
        method="nearest",
    ).compute()

    real_lat = float(point.latitude)
    real_lon = float(point.longitude)
    lead_times = point.lead_time.values

    # Recortar a MAX_LEAD_HOURS
    mask = np.array([pd.Timedelta(lt).total_seconds() / 3600 <= MAX_LEAD_HOURS
                     for lt in lead_times])
    lead_times = lead_times[mask]
    valid_times = _valid_times(latest_init, lead_times)
    members = point.ensemble_member.values
    n_members = len(members)

    # Temperatura
    t2m = point["temperature_2m"].values[mask]
    temperature = {"valid_time": valid_times, "members": _members_dict(t2m, members), "units": "°C"}

    # Precipitación
    pcp = _precip_mmh(point["precipitation_surface"].values[mask])
    precipitation = {"valid_time": valid_times, "members": _members_dict(pcp, members), "units": "mm/h"}

    # Racha de viento (directa)
    gust = point["wind_gust_10m"].values[mask] * 3.6  # m/s → km/h
    wind = {
        "valid_time": valid_times,
        "members": _members_dict(gust, members),
        "units": "km/h",
        "label": "Racha máx. 10m",
    }

    return {
        "metadata": {
            "model": "ECMWF IFS ENS",
            "init_time": latest_init.isoformat(),
            "requested_lat": lat, "requested_lon": lon,
            "nearest_lat": real_lat, "nearest_lon": real_lon,
            "n_members": n_members,
            "n_lead_times": len(valid_times),
            "source": "ECMWF IFS ENS via dynamical.org",
        },
        "temperature_2m": temperature,
        "precipitation": precipitation,
        "wind_10m": wind,
    }


# ─── NOAA GEFS ───────────────────────────────────────────────────────────────
def _get_gefs(lat: float, lon: float) -> dict:
    ds = _open_gefs()
    latest_init = pd.Timestamp(ds.init_time.values[-1])

    # Filtramos lead_time a MAX_LEAD_HOURS directamente en xarray antes de compute()
    max_lead = pd.Timedelta(hours=MAX_LEAD_HOURS)
    lead_mask = ds.lead_time <= max_lead

    point = ds[VARS_GEFS].sel(
        init_time=latest_init,
        latitude=lat,
        longitude=lon,
        method="nearest",
    ).isel(lead_time=lead_mask).compute()

    # Transponemos a (lead_time, ensemble_member) para consistencia con ECMWF
    point = point.transpose("lead_time", "ensemble_member")

    real_lat = float(point.latitude)
    real_lon = float(point.longitude)
    lead_times = point.lead_time.values
    valid_times = _valid_times(latest_init, lead_times)
    members = point.ensemble_member.values
    n_members = len(members)

    # Temperatura
    t2m = point["temperature_2m"].values  # (n_lead, n_member)
    temperature = {"valid_time": valid_times, "members": _members_dict(t2m, members), "units": "°C"}

    # Precipitación
    pcp = _precip_mmh(point["precipitation_surface"].values)
    precipitation = {"valid_time": valid_times, "members": _members_dict(pcp, members), "units": "mm/h"}

    # Viento: GEFS no tiene racha → calculamos módulo del viento medio
    u = point["wind_u_10m"].values
    v = point["wind_v_10m"].values
    wspd = np.sqrt(u**2 + v**2) * 3.6  # m/s → km/h
    wind = {
        "valid_time": valid_times,
        "members": _members_dict(wspd, members),
        "units": "km/h",
        "label": "Viento medio 10m",
    }

    return {
        "metadata": {
            "model": "NOAA GEFS",
            "init_time": latest_init.isoformat(),
            "requested_lat": lat, "requested_lon": lon,
            "nearest_lat": real_lat, "nearest_lon": real_lon,
            "n_members": n_members,
            "n_lead_times": len(valid_times),
            "source": "NOAA GEFS via dynamical.org",
        },
        "temperature_2m": temperature,
        "precipitation": precipitation,
        "wind_10m": wind,
    }
