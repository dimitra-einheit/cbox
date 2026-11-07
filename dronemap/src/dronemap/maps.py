"""Static map figures (PNG) in local metres with scale bar and north arrow."""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


class MapFig:
    def __init__(self, grid, title: str, figsize=(11, 9)):
        self.grid = grid
        left, bottom, right, top = grid.bounds
        self.left, self.bottom = left, bottom
        self.W, self.H = right - left, top - bottom
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=150)
        self.ax.set_title(title, fontsize=13, loc="left")
        self.ax.set_aspect("equal")
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(0, self.H)
        self.ax.set_xlabel("metres east")
        self.ax.set_ylabel("metres north")
        self.ax.set_facecolor("#e8e8e8")

    # map coords (CRS metres) -> local figure coords
    def lx(self, x):
        return np.asarray(x) - self.left

    def ly(self, y):
        return np.asarray(y) - self.bottom

    def extent_of(self, transform, shape):
        h, w = shape[:2]
        x0 = transform.c - self.left
        y1 = transform.f - self.bottom
        return (x0, x0 + w * transform.a, y1 + h * transform.e, y1)

    def image(self, arr, transform=None, **kw):
        tr = transform or self.grid.transform
        return self.ax.imshow(arr, extent=self.extent_of(tr, arr.shape), origin="upper",
                              interpolation=kw.pop("interpolation", "nearest"), **kw)

    def background(self, hillshade=None, rgba=None, rgba_transform=None, dim: float = 0.0):
        if rgba is not None:
            img = rgba.astype(float) / 255.0
            if dim:
                img[..., :3] = img[..., :3] * (1 - dim) + dim
            self.image(img, rgba_transform, interpolation="bilinear")
        elif hillshade is not None:
            self.image(hillshade, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")

    def contours(self, contours, major_every: float = 1.0, color="k", label=True):
        for level, lines in contours:
            major = abs(level / major_every - round(level / major_every)) < 1e-6
            for ln in lines:
                self.ax.plot(self.lx(ln[:, 0]), self.ly(ln[:, 1]), color=color, lw=0.9 if major else 0.35,
                             alpha=0.8 if major else 0.55)
            if label and major and lines:
                ln = max(lines, key=len)
                m = len(ln) // 2
                self.ax.text(self.lx(ln[m, 0]), self.ly(ln[m, 1]), f"{level:g}", fontsize=6, color=color,
                             ha="center", va="center",
                             bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

    def polygon(self, pts_xy, color="yellow", lw=2.0, ls="-", label=None):
        xs = [p[0] for p in pts_xy] + [pts_xy[0][0]]
        ys = [p[1] for p in pts_xy] + [pts_xy[0][1]]
        self.ax.plot(self.lx(xs), self.ly(ys), color=color, lw=lw, ls=ls, label=label)

    def finish(self, path: str, legend: bool = False, note: str | None = None):
        self._scalebar()
        self._north()
        if legend:
            self.ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
        if note:
            self.fig.text(0.01, 0.005, note, fontsize=7, color="#444")
        self.fig.tight_layout()
        self.fig.savefig(path)
        plt.close(self.fig)
        return path

    def _scalebar(self):
        target = self.W / 5
        mag = 10 ** math.floor(math.log10(target))
        length = max(m * mag for m in (1, 2, 5) if m * mag <= target)
        x0, y0 = self.W * 0.04, self.H * 0.04
        self.ax.plot([x0, x0 + length], [y0, y0], color="k", lw=4, solid_capstyle="butt")
        self.ax.plot([x0, x0 + length], [y0, y0], color="w", lw=2, solid_capstyle="butt")
        self.ax.text(x0 + length / 2, y0 + self.H * 0.015, f"{length:g} m", ha="center", fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    def _north(self):
        x, y = self.W * 0.95, self.H * 0.90
        self.ax.annotate("N", xy=(x, y), xytext=(x, y - self.H * 0.07), ha="center", fontsize=11,
                         fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))


def colorbar(mappable, ax, label):
    cb = plt.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(label)
    return cb
