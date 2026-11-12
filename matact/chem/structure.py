"""Crystal-structure utilities: prototypes, CIF/POSCAR/XYZ input-output and geometry.

The module is dependency-free so that the environment can run anywhere. CIF
support covers P1 files (all sites listed), which is what the environment writes.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from functools import reduce

from .elements import ELEMENTS

AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660
_FCC = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]
_IMAGES = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]


class StructureError(ValueError):
    """Raised for unreadable or physically invalid structure input."""


def _shift(sites, d):
    return [tuple((s + t) % 1.0 for s, t in zip(site, d)) for site in sites]


# prototype: (space-group symbol, number, fractional sites per element in formula order)
PROTOTYPES = {
    "fcc": ("Fm-3m", 225, [_FCC]),
    "bcc": ("Im-3m", 229, [[(0, 0, 0), (.5, .5, .5)]]),
    "diamond": ("Fd-3m", 227, [_FCC + _shift(_FCC, (.25, .25, .25))]),
    "rocksalt": ("Fm-3m", 225, [_FCC, _shift(_FCC, (.5, .5, .5))]),
    "zincblende": ("F-43m", 216, [_FCC, _shift(_FCC, (.25, .25, .25))]),
    "cscl": ("Pm-3m", 221, [[(0, 0, 0)], [(.5, .5, .5)]]),
    "fluorite": ("Fm-3m", 225, [_FCC, _shift(_FCC, (.25,) * 3) + _shift(_FCC, (.75,) * 3)]),
    "perovskite": ("Pm-3m", 221, [[(0, 0, 0)], [(.5, .5, .5)], [(.5, .5, 0), (.5, 0, .5), (0, .5, .5)]]),
    "wurtzite": ("P6_3mc", 186, [[(1 / 3, 2 / 3, 0), (2 / 3, 1 / 3, .5)],
                                 [(1 / 3, 2 / 3, .375), (2 / 3, 1 / 3, .875)]]),
}


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _norm(a):
    return math.sqrt(_dot(a, a))


def _inverse(m):
    det = _dot(m[0], _cross(m[1], m[2]))
    if abs(det) < 1e-12:
        raise StructureError("singular lattice")
    cols = [_cross(m[1], m[2]), _cross(m[2], m[0]), _cross(m[0], m[1])]
    return [[cols[j][i] / det for j in range(3)] for i in range(3)]


def lattice_from_parameters(a, b, c, alpha, beta, gamma):
    al, be, ga = (math.radians(x) for x in (alpha, beta, gamma))
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return [[a, 0.0, 0.0], [b * math.cos(ga), b * math.sin(ga), 0.0], [cx, cy, cz]]


def reduced_formula(composition) -> str:
    counts = {e: int(n) for e, n in composition.items() if n}
    g = reduce(math.gcd, counts.values())
    order = sorted(counts, key=lambda e: (ELEMENTS[e].chi, e))
    return "".join(e + (str(counts[e] // g) if counts[e] // g != 1 else "") for e in order)


def parse_formula(formula: str):
    """Split a formula into elements in written order, e.g. SrTiO3 -> [Sr, Ti, O]."""
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if not parts or "".join(e + n for e, n in parts) != formula:
        raise StructureError(f"cannot parse formula {formula!r}")
    return [(e, int(n or 1)) for e, n in parts]


class Structure:
    def __init__(self, lattice, species, frac_coords):
        self.lattice = [[float(x) for x in row] for row in lattice]
        self.species = [str(s) for s in species]
        self.frac_coords = [[float(x) % 1.0 for x in c] for c in frac_coords]
        if not self.species or len(self.species) != len(self.frac_coords):
            raise StructureError("species and coordinates do not match")
        unknown = sorted(set(self.species) - set(ELEMENTS))
        if unknown:
            raise StructureError(f"unknown elements: {unknown}")
        if self.volume < 1e-6:
            raise StructureError("degenerate lattice")

    @property
    def volume(self):
        a, b, c = self.lattice
        return abs(_dot(a, _cross(b, c)))

    @property
    def abc(self):
        return tuple(_norm(v) for v in self.lattice)

    @property
    def angles(self):
        a, b, c = self.lattice
        return tuple(math.degrees(math.acos(_dot(x, y) / (_norm(x) * _norm(y))))
                     for x, y in ((b, c), (a, c), (a, b)))

    @property
    def natoms(self):
        return len(self.species)

    @property
    def composition(self):
        return Counter(self.species)

    @property
    def formula(self):
        return reduced_formula(self.composition)

    @property
    def density(self):
        mass = sum(ELEMENTS[s].mass for s in self.species)
        return mass * AMU_PER_A3_TO_G_PER_CM3 / self.volume

    def cart(self, f):
        return [sum(f[j] * self.lattice[j][k] for j in range(3)) for k in range(3)]

    def closest_pair(self):
        """Return (distance, i, j) of the closest pair of sites, periodic images included."""
        n, lat = self.natoms, self.lattice
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - numpy is optional
            np = None
        if np is not None and n > 12:
            frac = np.array(self.frac_coords)
            diff = frac[None, :, :] - frac[:, None, :]
            diff -= np.round(diff)
            best = (float("inf"), 0, 0)
            for im in _IMAGES:
                r2 = (((diff + np.array(im, dtype=float)) @ np.array(lat)) ** 2).sum(-1)
                if im == (0, 0, 0):
                    np.fill_diagonal(r2, np.inf)
                i, j = np.unravel_index(int(np.argmin(r2)), r2.shape)
                if r2[i, j] < best[0] ** 2:
                    best = (math.sqrt(float(r2[i, j])), int(min(i, j)), int(max(i, j)))
            return best
        offsets = [(tuple(im[0] * lat[0][k] + im[1] * lat[1][k] + im[2] * lat[2][k] for k in range(3)), im == (0, 0, 0))
                   for im in _IMAGES]
        best2, bi, bj = float("inf"), 0, 0
        f = self.frac_coords
        for i in range(n):
            for j in range(i, n):
                a, b, c = (f[j][k] - f[i][k] for k in range(3))
                a, b, c = a - round(a), b - round(b), c - round(c)
                x = a * lat[0][0] + b * lat[1][0] + c * lat[2][0]
                y = a * lat[0][1] + b * lat[1][1] + c * lat[2][1]
                z = a * lat[0][2] + b * lat[1][2] + c * lat[2][2]
                for (ox, oy, oz), zero in offsets:
                    if zero and i == j:
                        continue
                    r2 = (x + ox) ** 2 + (y + oy) ** 2 + (z + oz) ** 2
                    if r2 < best2:
                        best2, bi, bj = r2, i, j
        return math.sqrt(best2), bi, bj

    def supercell(self, n):
        lattice = [[x * m for x in row] for row, m in zip(self.lattice, n)]
        species, coords = [], []
        for s, f in zip(self.species, self.frac_coords):
            for i in range(n[0]):
                for j in range(n[1]):
                    for k in range(n[2]):
                        species.append(s)
                        coords.append([(f[0] + i) / n[0], (f[1] + j) / n[1], (f[2] + k) / n[2]])
        return Structure(lattice, species, coords)

    def merge_sites(self, tol):
        """Remove sites of the same element that lie within ``tol`` Angstrom of a kept site."""
        kept = []
        for s, f in zip(self.species, self.frac_coords):
            close = False
            for s2, f2 in kept:
                d = [f[k] - f2[k] - round(f[k] - f2[k]) for k in range(3)]
                if s == s2 and _norm(self.cart(d)) < tol:
                    close = True
                    break
            if not close:
                kept.append((s, f))
        return Structure(self.lattice, [s for s, _ in kept], [f for _, f in kept])

    def to_dict(self):
        return {"lattice": [[round(x, 6) for x in r] for r in self.lattice], "species": self.species,
                "frac_coords": [[round(x, 6) for x in c] for c in self.frac_coords]}

    @classmethod
    def from_dict(cls, d):
        return cls(d["lattice"], d["species"], d["frac_coords"])

    def checksum(self):
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def build_prototype(prototype, elements, a, c=None):
    _, _, groups = PROTOTYPES[prototype]
    if len(groups) != len(elements):
        raise StructureError(f"{prototype} needs {len(groups)} elements")
    if prototype == "wurtzite":
        c = c or a * 1.633
        lattice = [[a, 0, 0], [-a / 2, a * math.sqrt(3) / 2, 0], [0, 0, c]]
    else:
        lattice = [[a, 0, 0], [0, a, 0], [0, 0, a]]
    species, coords = [], []
    for el, sites in zip(elements, groups):
        for site in sites:
            species.append(el)
            coords.append(list(site))
    return Structure(lattice, species, coords)
