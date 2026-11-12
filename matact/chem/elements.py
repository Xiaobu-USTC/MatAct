"""Element data used by the structure utilities and the scientific environment.

Masses are standard atomic weights, radii are covalent radii in Angstrom
(Cordero et al., 2008) and electronegativities are Pauling values. ``metal``
marks elements that are metallic in their standard state.
"""
from __future__ import annotations

from collections import namedtuple

Element = namedtuple("Element", "symbol Z mass radius chi metal period hardness")

_TABLE = """
Li 3 6.94 1.28 0.98 1
B 5 10.81 0.84 2.04 0
C 6 12.011 0.76 2.55 0
N 7 14.007 0.71 3.04 0
O 8 15.999 0.66 3.44 0
F 9 18.998 0.57 3.98 0
Na 11 22.990 1.66 0.93 1
Mg 12 24.305 1.41 1.31 1
Al 13 26.982 1.21 1.61 1
Si 14 28.085 1.11 1.90 0
P 15 30.974 1.07 2.19 0
S 16 32.06 1.05 2.58 0
Cl 17 35.45 1.02 3.16 0
K 19 39.098 2.03 0.82 1
Ca 20 40.078 1.76 1.00 1
Ti 22 47.867 1.60 1.54 1
V 23 50.942 1.53 1.63 1
Cr 24 51.996 1.39 1.66 1
Fe 26 55.845 1.32 1.83 1
Co 27 58.933 1.26 1.88 1
Ni 28 58.693 1.24 1.91 1
Cu 29 63.546 1.32 1.90 1
Zn 30 65.38 1.22 1.65 1
Ga 31 69.723 1.22 1.81 1
Ge 32 72.630 1.20 2.01 0
As 33 74.922 1.19 2.18 0
Se 34 78.971 1.20 2.55 0
Br 35 79.904 1.20 2.96 0
Rb 37 85.468 2.20 0.82 1
Sr 38 87.62 1.95 0.95 1
Zr 40 91.224 1.75 1.33 1
Nb 41 92.906 1.64 1.60 1
Mo 42 95.95 1.54 2.16 1
Pd 46 106.42 1.39 2.20 1
Ag 47 107.868 1.45 1.93 1
Cd 48 112.414 1.44 1.69 1
In 49 114.818 1.42 1.78 1
Sn 50 118.710 1.39 1.96 1
Sb 51 121.760 1.39 2.05 0
Te 52 127.60 1.38 2.10 0
I 53 126.904 1.39 2.66 0
Cs 55 132.905 2.44 0.79 1
Ba 56 137.327 2.15 0.89 1
Ce 58 140.116 2.04 1.12 1
Hf 72 178.49 1.75 1.30 1
Ta 73 180.948 1.70 1.50 1
W 74 183.84 1.62 2.36 1
Pt 78 195.084 1.36 2.28 1
Au 79 196.967 1.36 2.54 1
Pb 82 207.2 1.46 2.33 1
"""

_PERIOD_ENDS = (2, 10, 18, 36, 54, 86, 118)
# Elements whose pseudopotentials need a higher plane-wave cutoff.
_HARDNESS = {"O": 2.0, "F": 2.0, "N": 2.0, "B": 1.5, "C": 1.5}


def _period(z: int) -> int:
    for i, end in enumerate(_PERIOD_ENDS, start=1):
        if z <= end:
            return i
    raise ValueError(z)


def _load():
    table = {}
    for line in _TABLE.strip().splitlines():
        sym, z, mass, radius, chi, metal = line.split()
        z = int(z)
        table[sym] = Element(sym, z, float(mass), float(radius), float(chi),
                             metal == "1", _period(z), _HARDNESS.get(sym, 1.0))
    return table


ELEMENTS = _load()


def is_metallic(elements) -> bool:
    """A compound is treated as metallic when all of its elements are metals."""
    return all(ELEMENTS[e].metal for e in elements)
