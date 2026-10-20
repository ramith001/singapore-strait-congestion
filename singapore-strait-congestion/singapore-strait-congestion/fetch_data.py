"""
Pull raw daily data from the IMF PortWatch public ArcGIS FeatureServer for:
  - the Malacca Strait chokepoint (chokepoint5)
  - the Port of Singapore (port1201)
  - the documented disruption-events layer (filtered to the handful of
    non-natural-disaster "OT" events used as ground truth in analysis.py,
    plus COVID-19 added manually since PortWatch's own events layer does not
    carry a COVID entry)

No API key is required -- this is IMF/Oxford's own public Esri Hub service.
Run this once to (re)populate the data/ folder before running analysis.py.
"""
import time
import requests
import pandas as pd

BASE = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"


def paginate(url, where, order_by=None, extra_params=None, page_size=1000):
    """Generic paginated FeatureServer query -> list of attribute dicts."""
    features = []
    offset = 0
    params = {
        "where": where,
        "outFields": "*",
        "outSR": 4326,
        "f": "json",
        "resultRecordCount": page_size,
    }
    if order_by:
        params["orderByFields"] = order_by
    if extra_params:
        params.update(extra_params)
    while True:
        params["resultOffset"] = offset
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(data["error"])
        batch = [f["attributes"] for f in data.get("features", [])]
        features.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
        time.sleep(0.2)  # be polite to a public service
    return features


def fetch_chokepoint(portid="chokepoint5"):
    url = f"{BASE}/Daily_Chokepoints_Data/FeatureServer/0/query"
    feats = paginate(url, where=f"portid='{portid}'", order_by="date ASC")
    df = pd.DataFrame(feats).sort_values("date")
    return df


def fetch_port(portid="port1201"):
    url = f"{BASE}/Daily_Ports_Data/FeatureServer/0/query"
    feats = paginate(url, where=f"portid='{portid}'", order_by="date ASC")
    df = pd.DataFrame(feats).sort_values("date")
    return df


def fetch_chokepoints_reference():
    url = f"{BASE}/PortWatch_chokepoints_database/FeatureServer/0/query"
    feats = paginate(url, where="1=1")
    return pd.DataFrame(feats)


def fetch_ports_reference(name_filter="Singapore"):
    url = f"{BASE}/PortWatch_ports_database/FeatureServer/0/query"
    feats = paginate(
        url,
        where=f"portname LIKE '%{name_filter}%'",
        extra_params={"outFields": "portid,portname,country,fullname,LOCODE"},
    )
    return pd.DataFrame(feats)


def fetch_disruption_events():
    url = f"{BASE}/portwatch_disruptions_database/FeatureServer/0/query"
    feats = paginate(
        url,
        where="1=1",
        extra_params={
            "outFields": "eventid,eventtype,eventname,alertlevel,country,fromdate,todate",
            "returnGeometry": "false",
        },
    )
    df = pd.DataFrame(feats)
    df["fromdate"] = pd.to_datetime(df["fromdate"], unit="ms")
    df["todate"] = pd.to_datetime(df["todate"], unit="ms")
    # keep only broad operational/geopolitical disruptions ("OT"), not the
    # much larger set of individual cyclones/earthquakes/floods PortWatch also
    # tracks -- those matter for the ports they hit but aren't relevant here
    df = df[df["eventtype"] == "OT"].copy()
    covid = pd.DataFrame([{
        "eventid": 900000, "eventtype": "OT", "eventname": "COVID-19 GLOBAL SHIPPING DISRUPTION",
        "alertlevel": "RED", "country": None,
        "fromdate": pd.Timestamp("2020-01-23"), "todate": pd.Timestamp("2021-12-31"),
    }])
    return pd.concat([df, covid], ignore_index=True).sort_values("fromdate")


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    print("Fetching Malacca Strait chokepoint reference row...")
    print(fetch_chokepoints_reference().query("portname == 'Malacca Strait'"))

    print("Fetching Singapore port reference row...")
    print(fetch_ports_reference("Singapore"))

    print("Fetching Malacca Strait daily vessel transit data (chokepoint5)...")
    mal = fetch_chokepoint("chokepoint5")
    print(f"  -> {len(mal)} rows, {mal['date'].min()} to {mal['date'].max()}")

    print("Fetching Singapore port daily trade/port-call data (port1201)...")
    sg_feats = paginate(f"{BASE}/Daily_Ports_Data/FeatureServer/0/query", where="portid='port1201'", order_by="date ASC")
    if not sg_feats:
        # As of 2026-10-20 the API returns no rows for port1201. Nothing is written, so a rerun cannot replace the
        # committed Malacca file with a longer series that no longer lines up with the Singapore one.
        raise SystemExit("The PortWatch API returned no rows for port1201 (Singapore); data/ was left unchanged. "
                         "See the 'Live data refresh' section of the README.")
    sg = pd.DataFrame(sg_feats).sort_values("date")
    print(f"  -> {len(sg)} rows, {sg['date'].min()} to {sg['date'].max()}")
    mal.to_csv("data/malacca_strait_daily.csv", index=False)
    sg.to_csv("data/singapore_port_daily.csv", index=False)

    print("Fetching documented disruption events...")
    events = fetch_disruption_events()
    events.to_csv("data/disruption_events.csv", index=False)
    print(events[["eventname", "fromdate", "todate"]])

    print("\nDone. Run analysis.py next.")
