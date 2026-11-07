"""Minimal KML writer for previewing plans and results in Google Earth."""
from __future__ import annotations

from xml.sax.saxutils import escape


def _coords(latlons, alt=None) -> str:
    if alt is None:
        return " ".join(f"{lon:.8f},{lat:.8f},0" for lat, lon in latlons)
    return " ".join(f"{lon:.8f},{lat:.8f},{alt:.1f}" for lat, lon in latlons)


# KML colours are aabbggrr
PALETTE = ["ff00a5ff", "ffffff00", "ff00ff00", "ffff00ff", "ff0000ff", "ffffffff", "ff00ffff", "ffff8000"]


class KmlDoc:
    def __init__(self, name: str):
        self.name = name
        self._styles: dict[str, str] = {}
        self._body: list[str] = []
        self._folder_depth = 0

    def style_line(self, sid: str, color: str, width: float = 2.0) -> str:
        self._styles[sid] = (
            f'<Style id="{sid}"><LineStyle><color>{color}</color><width>{width}</width></LineStyle>'
            f"<PolyStyle><fill>0</fill></PolyStyle></Style>"
        )
        return sid

    def style_poly(self, sid: str, line_color: str, fill_color: str, width: float = 2.0) -> str:
        self._styles[sid] = (
            f'<Style id="{sid}"><LineStyle><color>{line_color}</color><width>{width}</width></LineStyle>'
            f"<PolyStyle><color>{fill_color}</color></PolyStyle></Style>"
        )
        return sid

    def style_point(self, sid: str, color: str = "ffffffff", scale: float = 0.8,
                    icon: str = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png") -> str:
        self._styles[sid] = (
            f'<Style id="{sid}"><IconStyle><color>{color}</color><scale>{scale}</scale>'
            f"<Icon><href>{icon}</href></Icon></IconStyle><LabelStyle><scale>0.8</scale></LabelStyle></Style>"
        )
        return sid

    def folder(self, name: str, visible: bool = True) -> None:
        self._body.append(f"<Folder><name>{escape(name)}</name><visibility>{int(visible)}</visibility>")
        self._folder_depth += 1

    def end_folder(self) -> None:
        self._body.append("</Folder>")
        self._folder_depth -= 1

    def polygon(self, name: str, ring_latlon, style: str, description: str = "") -> None:
        ring = list(ring_latlon)
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        self._body.append(
            f"<Placemark><name>{escape(name)}</name><description>{escape(description)}</description>"
            f"<styleUrl>#{style}</styleUrl><Polygon><tessellate>1</tessellate><outerBoundaryIs><LinearRing>"
            f"<coordinates>{_coords(ring)}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>"
        )

    def line(self, name: str, latlons, style: str, description: str = "", alt: float | None = None,
             visible: bool = True) -> None:
        mode = "<altitudeMode>relativeToGround</altitudeMode>" if alt is not None else "<tessellate>1</tessellate>"
        self._body.append(
            f"<Placemark><name>{escape(name)}</name><visibility>{int(visible)}</visibility>"
            f"<description>{escape(description)}</description><styleUrl>#{style}</styleUrl>"
            f"<LineString>{mode}<coordinates>{_coords(latlons, alt)}</coordinates></LineString></Placemark>"
        )

    def point(self, name: str, lat: float, lon: float, style: str, description: str = "",
              visible: bool = True) -> None:
        self._body.append(
            f"<Placemark><name>{escape(name)}</name><visibility>{int(visible)}</visibility>"
            f"<description>{escape(description)}</description><styleUrl>#{style}</styleUrl>"
            f"<Point><coordinates>{lon:.8f},{lat:.8f},0</coordinates></Point></Placemark>"
        )

    def ground_overlay(self, name: str, href: str, north: float, south: float, east: float, west: float,
                       visible: bool = True, opacity_hex: str = "ff") -> None:
        self._body.append(
            f"<GroundOverlay><name>{escape(name)}</name><visibility>{int(visible)}</visibility>"
            f"<color>{opacity_hex}ffffff</color><Icon><href>{escape(href)}</href></Icon>"
            f"<LatLonBox><north>{north:.8f}</north><south>{south:.8f}</south>"
            f"<east>{east:.8f}</east><west>{west:.8f}</west></LatLonBox></GroundOverlay>"
        )

    def raw(self, xml: str) -> None:
        self._body.append(xml)

    def to_string(self) -> str:
        while self._folder_depth > 0:
            self.end_folder()
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<kml xmlns="http://www.opengis.net/kml/2.2">\n<Document>'
            f"<name>{escape(self.name)}</name>\n" + "\n".join(self._styles.values()) + "\n"
            + "\n".join(self._body) + "\n</Document>\n</kml>\n"
        )

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_string())
        return path
