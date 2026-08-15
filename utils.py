import numpy as np

def haversine_km(lat1, lng1, lat2, lng2, R=6371.0088):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    d = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(d, 0, 1)))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def latlng_to_xyz(lat, lng):
    lat_rad= np.radians(lat)
    lng_rad = np.radians(lng)

    x = np.cos(lat_rad) * np.cos(lng_rad)
    y = np.cos(lat_rad) * np.sin(lng_rad)
    z = np.sin(lat_rad)
    return x, y, z

def xyz_to_latlng(x, y, z):
    lat_rad = np.arcsin(np.clip(z, -1.0, 1.0))
    lng_rad = np.arctan2(y, x)
    return np.degrees(lat_rad), np.degrees(lng_rad)


