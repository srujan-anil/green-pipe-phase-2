import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ELECTRICITY_MAPS_KEY")


def fetch_live_carbon(zone="DE"):
    """
    Calls the real Electricity Maps API for the given zone.
    Returns: (carbon_intensity: float | None, source: str)
    source is one of: "live", "no_key", "http_<code>", "error"
    """
    if not API_KEY:
        return None, "no_key"

    try:
        url = "https://api.electricitymap.org/v3/carbon-intensity/latest"
        headers = {"auth-token": API_KEY}
        params = {"zone": zone}
        r = requests.get(url, headers=headers, params=params, timeout=10)

        if r.status_code == 200:
            carbon = r.json().get("carbonIntensity")
            return carbon, "live"

        return None, f"http_{r.status_code}"

    except Exception:
        return None, "error"


if __name__ == "__main__":
    for zone in ["DE", "IN", "US-CAL-CISO"]:
        carbon, source = fetch_live_carbon(zone)
        print(f"{zone}: {carbon} gCO2/kWh (source={source})")