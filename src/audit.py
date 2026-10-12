"""Data audit for the MMA3001 project (Dataset 2, Monash smart building data).

Reads the raw environment and occupancy CSV files in chunks (the occupancy
file is very large) and writes a short text summary: row counts, date ranges,
sensors, variables, and data quality flags. Nothing is modified or saved
except the summary text file.

Usage:
    python src/audit.py                      (uses the repo's data/ folder)
    python src/audit.py "C:/path/to/data"    (uses any other folder)

The data folder must contain:
    5EnvSensor_MayToDec2024_180kRows.csv
    5occupancySensor_MayToDec2024_9MRows.csv
    Sensor ID and Locations.xlsx
"""

import argparse
import collections
import json
import os
import time

import pandas as pd

ENV_FILE = "5EnvSensor_MayToDec2024_180kRows.csv"
OCC_FILE = "5occupancySensor_MayToDec2024_9MRows.csv"
SHEET_FILE = "Sensor ID and Locations.xlsx"


def fmt_date(unix_time):
    """Convert a Unix timestamp (seconds) to a UTC date string.

    Args:
        unix_time (int or float): seconds since 1970-01-01 UTC.

    Returns:
        str: date formatted as YYYY-MM-DD.
    """
    return time.strftime("%Y-%m-%d", time.gmtime(unix_time))


def load_sensor_sheet(sheet_path):
    """Load the ENV tab of the sensor location sheet.

    The sensor ID column is found by its content (13 digit IDs starting 6012)
    so the code does not depend on the exact header text.

    Args:
        sheet_path (str): path to Sensor ID and Locations.xlsx.

    Returns:
        pandas.DataFrame: columns 'room' (text description) and 'sensor_id'.
    """
    sheet = pd.read_excel(sheet_path, sheet_name="ENV", dtype=str)
    id_col = next(
        c for c in sheet.columns if sheet[c].astype(str).str.startswith("6012").any()
    )
    sensor_id = sheet[id_col].astype(str).str.replace(r"\.0$", "", regex=True)
    # first three columns are building, floor/room, area
    room = sheet.iloc[:, 0].astype(str) + " " + sheet.iloc[:, 1].astype(str) \
        + " " + sheet.iloc[:, 2].astype(str)
    return pd.DataFrame({"room": room, "sensor_id": sensor_id})


def audit_environment(env_path, sheet_path):
    """Summarise the environment file, which stores readings as JSON text.

    Each row holds a JSON list of variables, each with (time, value) pairs.
    Rows are parsed one at a time; rows that fail to parse are counted.

    Args:
        env_path (str): path to the environment CSV.
        sheet_path (str): path to the sensor location workbook.

    Returns:
        list of str: report lines.
    """
    counts = collections.Counter()      # (sensor, variable) -> readings
    units = {}                          # variable name -> unit
    months = collections.Counter()      # reading month (UTC) -> readings
    first, last = {}, {}                # sensor -> earliest / latest reading
    statuses = collections.Counter()
    created_min = created_max = None
    n_rows = bad_json = 0

    for chunk in pd.read_csv(env_path, chunksize=20000, dtype={"sensorid": str}):
        n_rows += len(chunk)
        statuses.update(chunk["status"])
        c_min, c_max = chunk["createdate"].min(), chunk["createdate"].max()
        created_min = c_min if created_min is None else min(created_min, c_min)
        created_max = c_max if created_max is None else max(created_max, c_max)

        for sid, js in zip(chunk["sensorid"], chunk["jsondata"]):
            try:
                for item in json.loads(js):
                    name = item["variable"]["name"]
                    units[name] = item["variable"].get("unit")
                    for reading in item["values"]:
                        t = reading["time"]
                        counts[(sid, name)] += 1
                        months[time.strftime("%Y-%m", time.gmtime(t))] += 1
                        first[sid] = min(t, first.get(sid, t))
                        last[sid] = max(t, last.get(sid, t))
            except Exception:
                bad_json += 1   # malformed row: count it and move on

    lines = ["===== ENVIRONMENT FILE =====",
             f"rows: {n_rows}, rows with unreadable JSON: {bad_json}",
             f"status values: {dict(statuses)}",
             f"createdate range (string min/max): {created_min} to {created_max}",
             f"variables and units: {units}",
             "readings per sensor and variable:"]
    lines += [f"  {k}: {v}" for k, v in sorted(counts.items())]
    lines.append("first and last reading per sensor (UTC):")
    lines += [f"  {s}: {fmt_date(first[s])} to {fmt_date(last[s])}" for s in first]
    lines.append(f"readings per month (UTC): {dict(sorted(months.items()))}")

    # match sensors to rooms; a duplicated ID in the sheet shows up as 2 rooms
    sheet = load_sensor_sheet(sheet_path)
    lines.append("sensors matched to the location sheet:")
    for s in first:
        rooms = sheet.loc[sheet["sensor_id"] == s, "room"].tolist()
        lines.append(f"  {s} -> " + ("; ".join(rooms) if rooms else "NOT IN SHEET"))
    return lines


def audit_occupancy(occ_path):
    """Summarise the occupancy event log (very large, read in chunks).

    Args:
        occ_path (str): path to the occupancy CSV.

    Returns:
        list of str: report lines.
    """
    n_rows = backwards = 0
    devices, spaces = collections.Counter(), collections.Counter()
    status, headcount = collections.Counter(), collections.Counter()
    pairs, months = collections.Counter(), collections.Counter()
    d_min = d_max = None

    for chunk in pd.read_csv(occ_path, chunksize=500000):
        n_rows += len(chunk)
        devices.update(chunk["deviceid"])
        spaces.update(chunk["floorspaceid"])
        status.update(chunk["occupancystatus"])
        headcount.update(chunk["headcount"])
        pairs.update(zip(chunk["deviceid"], chunk["floorspaceid"]))

        dates = chunk["collecteddate"]
        d_min = dates.min() if d_min is None else min(d_min, dates.min())
        d_max = dates.max() if d_max is None else max(d_max, dates.max())
        months.update(dates.str[:7])
        # rows whose timestamp is earlier than the row before (file order)
        backwards += int((dates.iloc[1:].values < dates.iloc[:-1].values).sum())

    return ["", "===== OCCUPANCY FILE =====",
            f"rows: {n_rows}",
            f"date range (string min/max): {d_min} to {d_max}",
            f"rows per month: {dict(sorted(months.items()))}",
            f"rows where time goes backwards in file order: {backwards}",
            f"device ids: {len(devices)}, floorspace ids: {len(spaces)}",
            f"rows per device: {dict(devices)}",
            f"device and floorspace pairs: {dict(pairs)}",
            f"status counts: {dict(status)}",
            f"headcount counts: {dict(sorted(headcount.items()))}"]


def main():
    """Run both audits and save the summary next to the data files."""
    parser = argparse.ArgumentParser(description="Audit the Dataset 2 files.")
    # default: the data/ folder at the top of the repo (one level above src/)
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    parser.add_argument("data_dir", nargs="?", default=default_dir,
                        help="folder containing the raw data files (default: data/ in the repo)")
    args = parser.parse_args()

    lines = audit_environment(os.path.join(args.data_dir, ENV_FILE),
                              os.path.join(args.data_dir, SHEET_FILE))
    lines += audit_occupancy(os.path.join(args.data_dir, OCC_FILE))

    report = "\n".join(lines)
    with open(os.path.join(args.data_dir, "audit_summary.txt"), "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()