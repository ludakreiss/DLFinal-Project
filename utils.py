import numpy as np

def haversine_km(lat1, lng1, lat2, lng2, R=6371.0088):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    d = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(d, 0, 1)))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
