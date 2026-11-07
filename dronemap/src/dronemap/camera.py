"""Camera models used for flight planning (ground sample distance, footprints)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Camera:
    key: str
    name: str
    sensor_w_mm: float
    sensor_h_mm: float
    focal_mm: float
    img_w: int
    img_h: int
    # Fastest reliable photo-to-photo interval when shooting JPEG during a mission.
    min_photo_interval_s: float
    # Sensor readout time (rolling shutter). Used only for warnings; ODM has its own database.
    rolling_shutter_ms: float

    def gsd_m(self, altitude_m: float) -> float:
        """Ground size of one pixel (metres) at a given height above ground, camera pointing straight down."""
        return self.sensor_w_mm * altitude_m / (self.focal_mm * self.img_w)

    def footprint_m(self, altitude_m: float) -> tuple[float, float]:
        """(across-track, along-track) ground footprint in metres for a nadir photo.

        The long side of the image sits across the flight direction, which is how
        DJI drones hold the camera when flying forward.
        """
        g = self.gsd_m(altitude_m)
        return g * self.img_w, g * self.img_h


CAMERAS = {
    # DJI Mavic 2 Pro: Hasselblad L1D-20c, 1" sensor, 20 MP, 28 mm equivalent (10.26 mm real).
    # 3:2 photos are 5472 x 3648. Rolling-shutter readout 56 ms at 3:2 (value from ODM's database).
    "mavic2pro": Camera(
        key="mavic2pro",
        name="DJI Mavic 2 Pro (Hasselblad L1D-20c)",
        sensor_w_mm=13.2,
        sensor_h_mm=8.8,
        focal_mm=10.26,
        img_w=5472,
        img_h=3648,
        min_photo_interval_s=2.5,
        rolling_shutter_ms=56.0,
    ),
}

DEFAULT_CAMERA = "mavic2pro"


def get_camera(key: str) -> Camera:
    try:
        return CAMERAS[key]
    except KeyError:
        raise KeyError(f"Unknown camera '{key}'. Known: {', '.join(CAMERAS)}") from None
