"""Litchi Mission Hub CSV writer.

Format: 46 columns, as exported/imported by Litchi Mission Hub (metric units):
latitude, longitude, altitude(m), heading(deg), curvesize(m), rotationdir,
gimbalmode, gimbalpitchangle, 15 x (actiontypeN, actionparamN), altitudemode,
speed(m/s), poi_latitude, poi_longitude, poi_altitude(m), poi_altitudemode,
photo_timeinterval, photo_distinterval.

Codes: rotationdir 0=clockwise; gimbalmode 0=disabled 1=focus POI 2=interpolate;
action -1=none 0=stay(ms) 1=photo 2=start rec 3=stop rec 4=rotate 5=tilt;
altitudemode 0=relative to take-off 1=above ground; intervals -1=off.
"""
from __future__ import annotations

from .flightplan import Mission

HEADER = (
    "latitude,longitude,altitude(m),heading(deg),curvesize(m),rotationdir,gimbalmode,gimbalpitchangle,"
    + ",".join(f"actiontype{i},actionparam{i}" for i in range(1, 16))
    + ",altitudemode,speed(m/s),poi_latitude,poi_longitude,poi_altitude(m),poi_altitudemode,"
    "photo_timeinterval,photo_distinterval"
)

N_COLUMNS = 46


def _f(v: float, nd: int) -> str:
    s = f"{v:.{nd}f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    return s


def waypoint_row(wp, altitude_mode: str = "relative") -> str:
    heading = round(wp.heading, 1) % 360.0
    pitch = max(-90.0, min(30.0, wp.pitch))
    cols = [
        _f(wp.lat, 8),
        _f(wp.lon, 8),
        _f(max(wp.alt, 0.0), 1),
        _f(heading, 1),
        "0",  # curvesize: 0 = stop exactly at the waypoint
        "0",  # rotationdir: clockwise
        "2",  # gimbalmode: interpolate (holds the pitch below)
        _f(pitch, 1),
    ]
    actions = list(wp.actions)[:15]
    for code, param in actions:
        cols += [str(int(code)), str(int(param))]
    for _ in range(15 - len(actions)):
        cols += ["-1", "0"]
    cols += [
        "1" if altitude_mode == "agl" else "0",
        _f(wp.speed, 1),
        "0", "0", "0", "0",  # no POI
        "-1",  # no time interval
        _f(wp.dist_interval, 1) if wp.dist_interval and wp.dist_interval > 0 else "-1",
    ]
    assert len(cols) == N_COLUMNS
    return ",".join(cols)


def write_mission_csv(path: str, mission: Mission, altitude_mode: str = "relative") -> str:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(HEADER + "\n")
        for wp in mission.waypoints:
            f.write(waypoint_row(wp, altitude_mode) + "\n")
    return path
