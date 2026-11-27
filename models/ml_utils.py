
import re

def normalize_column_name(name):
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")

    aliases = {
        "zero_crossing_rate": "zcr",
        "zerocrossingrate": "zcr",
        "spectral_centroid": "centroid",
        "spectral_bandwidth": "bandwidth",
        "spectral_rolloff": "rolloff",
    }

    return aliases.get(name, name)


def normalize_dataframe_columns(df):
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    return df
