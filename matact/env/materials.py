"""Reference materials library of the MatAct environment.

Each entry is a crystal prototype with a literature lattice constant. Reference
("Tier-B protocol") properties are produced by the Tier-B executor so that database records, DFT
runs and verifiers are mutually consistent.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from ..chem.elements import ELEMENTS, is_metallic
from ..chem.structure import PROTOTYPES, build_prototype, parse_formula, reduced_formula

# formula:prototype:a[:c] in Angstrom (formula written in prototype site order)
_LIBRARY = """
Al:fcc:4.050 Ni:fcc:3.524 Cu:fcc:3.615 Pd:fcc:3.891 Ag:fcc:4.086 Pt:fcc:3.924 Au:fcc:4.078 Pb:fcc:4.950
Li:bcc:3.510 Na:bcc:4.291 K:bcc:5.328 V:bcc:3.024 Cr:bcc:2.885 Fe:bcc:2.867 Nb:bcc:3.301 Mo:bcc:3.147
Ta:bcc:3.303 W:bcc:3.165 Ba:bcc:5.028 C:diamond:3.567 Si:diamond:5.431 Ge:diamond:5.658 Sn:diamond:6.489
LiF:rocksalt:4.026 LiCl:rocksalt:5.130 LiBr:rocksalt:5.501 LiI:rocksalt:6.000 NaF:rocksalt:4.634
NaCl:rocksalt:5.640 NaBr:rocksalt:5.977 NaI:rocksalt:6.473 KF:rocksalt:5.347 KCl:rocksalt:6.293
KBr:rocksalt:6.600 KI:rocksalt:7.066 RbF:rocksalt:5.640 RbCl:rocksalt:6.590 RbBr:rocksalt:6.890
RbI:rocksalt:7.342 MgO:rocksalt:4.212 CaO:rocksalt:4.811 SrO:rocksalt:5.160 BaO:rocksalt:5.523
MgS:rocksalt:5.200 CaS:rocksalt:5.689 SrS:rocksalt:6.019 BaS:rocksalt:6.389 MgSe:rocksalt:5.462
CaSe:rocksalt:5.916 SrSe:rocksalt:6.234 BaSe:rocksalt:6.600 CaTe:rocksalt:6.348 SrTe:rocksalt:6.660
BaTe:rocksalt:7.000 BN:zincblende:3.615 BP:zincblende:4.538 BAs:zincblende:4.777 SiC:zincblende:4.358
AlP:zincblende:5.463 AlAs:zincblende:5.661 AlSb:zincblende:6.136 GaP:zincblende:5.451
GaAs:zincblende:5.653 GaSb:zincblende:6.096 InP:zincblende:5.869 InAs:zincblende:6.058
InSb:zincblende:6.479 ZnS:zincblende:5.409 ZnSe:zincblende:5.668 ZnTe:zincblende:6.101
CdS:zincblende:5.818 CdSe:zincblende:6.052 CdTe:zincblende:6.482 GaN:zincblende:4.500
AlN:zincblende:4.380 InN:zincblende:4.980 GaN:wurtzite:3.189:5.185 AlN:wurtzite:3.112:4.982
InN:wurtzite:3.545:5.703 ZnO:wurtzite:3.250:5.207 ZnS:wurtzite:3.820:6.260 CdS:wurtzite:4.136:6.714
CsCl:cscl:4.123 CsBr:cscl:4.286 CsI:cscl:4.567 NiAl:cscl:2.887 CoAl:cscl:2.862 FeAl:cscl:2.909
CaF2:fluorite:5.463 SrF2:fluorite:5.800 BaF2:fluorite:6.200 CdF2:fluorite:5.388 CeO2:fluorite:5.411
ZrO2:fluorite:5.070 HfO2:fluorite:5.080 CaTiO3:perovskite:3.840 SrTiO3:perovskite:3.905
BaTiO3:perovskite:4.000 CaZrO3:perovskite:4.010 SrZrO3:perovskite:4.100 BaZrO3:perovskite:4.193
SrHfO3:perovskite:4.070 BaHfO3:perovskite:4.171 CaSnO3:perovskite:3.950 SrSnO3:perovskite:4.034
BaSnO3:perovskite:4.116 KNbO3:perovskite:4.020 NaNbO3:perovskite:3.950 KTaO3:perovskite:3.989
NaTaO3:perovskite:3.930
"""

# Chemical systems containing any of these elements form the OOD-chem split and
# never occur in training, validation or ID test tasks.
OOD_ELEMENTS = frozenset({"Rb", "Te", "Hf", "Sb"})
_TM = {"Ti", "Zr", "Hf", "Nb", "Ta", "V", "Cr", "Mo", "W", "Fe", "Co", "Ni", "Ce"}
DB_V1_COVERAGE = 0.80


def unit_hash(*keys) -> float:
    """Deterministic pseudo-random number in [0, 1) derived from ``keys``."""
    digest = hashlib.sha256("|".join(map(str, keys)).encode()).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def elemental_energy(el: str) -> float:
    """Reference energy per atom of the elemental phase (eV/atom)."""
    return -round(1.2 + 7.8 * unit_hash(el, "mu"), 4)


@dataclass(frozen=True)
class Material:
    material_id: str
    formula: str
    prototype: str
    space_group: str
    sites: tuple
    a: float
    c: Optional[float]
    split: str
    metallic: bool
    in_db_v1: bool
    a_ref: float
    c_ref: Optional[float]
    energy_per_atom: float
    formation_energy: float
    band_gap: float

    @property
    def elements(self):
        return sorted(set(self.sites))

    def structure(self, relaxed: bool = False):
        if relaxed:
            return build_prototype(self.prototype, list(self.sites), self.a_ref, self.c_ref)
        return build_prototype(self.prototype, list(self.sites), self.a, self.c)

    def record(self) -> dict:
        s = self.structure(relaxed=True)
        rec = {"material_id": self.material_id, "formula": self.formula, "prototype": self.prototype,
               "space_group": self.space_group, "elements": self.elements, "natoms": s.natoms,
               "lattice_a_A": round(self.a_ref, 4), "energy_per_atom_eV": self.energy_per_atom,
               "formation_energy_eV_per_atom": self.formation_energy, "band_gap_eV": self.band_gap,
               "density_g_cm3": round(s.density, 4), "is_metal": self.metallic,
               "calc_type": "PBE/relaxed (MatAct Tier-B protocol)",
               "record_uri": f"workspace://database/matact_refdb/{self.material_id}"}
        if self.c_ref:
            rec["lattice_c_A"] = round(self.c_ref, 4)
        return rec


def _assign_splits(formulas) -> dict:
    """Composition-level split: OOD element systems, then 20% test / 12% validation / rest train."""
    ids = sorted({f for f, els in formulas if not OOD_ELEMENTS.intersection(els)}, key=lambda f: unit_hash(f, "split"))
    n_test, n_valid = round(0.20 * len(ids)), round(0.12 * len(ids))
    split = {f: ("test" if i < n_test else "valid" if i < n_test + n_valid else "train") for i, f in enumerate(ids)}
    split.update({f: "ood" for f, els in formulas if OOD_ELEMENTS.intersection(els)})
    return split


def _band_gap(formula, elements, metallic) -> float:
    if metallic:
        return 0.0
    fixed = {"C": 4.11, "Si": 0.61, "Ge": 0.05}
    if formula in fixed:
        return fixed[formula]
    chi = [ELEMENTS[e].chi for e in elements]
    period = sum(ELEMENTS[e].period for e in elements) / len(elements)
    gap = (-0.4 + 2.3 * (max(chi) - min(chi)) - 0.35 * (period - 3)) * (0.9 + 0.2 * unit_hash(formula, "gap"))
    if _TM.intersection(elements):
        gap *= 0.5
    return round(max(gap, 0.0), 3)


def _formation_energy(formula, prototype, elements, metallic) -> float:
    if len(elements) == 1:
        return 0.0
    chi = [ELEMENTS[e].chi for e in elements]
    dchi = max(chi) - min(chi)
    base = 0.15 + 0.6 * dchi if metallic else 0.1 + 0.45 * dchi ** 1.5
    if {"O", "F"}.intersection(elements):
        base *= 1.5
    return -round(base * (0.85 + 0.3 * unit_hash(formula, prototype, "dhf")), 4)


@lru_cache(maxsize=None)
def library():
    entries = sorted((tuple(tok.split(":")) for tok in _LIBRARY.split()), key=lambda e: (e[1], e[0]))
    rows = []
    for written, prototype, *lat in entries:
        counts = {}
        for e, n in parse_formula(written):
            counts[e] = counts.get(e, 0) + n
        rows.append([written, prototype, lat, counts, reduced_formula(counts), None])
    splits = _assign_splits({(r[4], tuple(sorted(r[3]))) for r in rows})
    for r in rows:
        r[5] = splits[r[4]]
    # Database coverage gaps (missing records) are stratified so that every split has some.
    missing = set()
    for split in {r[5] for r in rows}:
        group = sorted((r for r in rows if r[5] == split), key=lambda r: unit_hash(r[4], r[1], "db"))
        missing.update((r[4], r[1]) for r in group[:round(len(group) * (1 - DB_V1_COVERAGE))])
    materials = []
    for index, (written, prototype, lat, counts, formula, split) in enumerate(rows, start=1):
        a, c = float(lat[0]), (float(lat[1]) if len(lat) > 1 else None)
        sites = tuple(e for e, _ in parse_formula(written))
        elements = sorted(counts)
        metallic = is_metallic(elements)
        delta = 0.008 + 0.012 * unit_hash(formula, prototype, "pbe")
        natoms = sum(counts.values())
        e_form = _formation_energy(formula, prototype, elements, metallic)
        e_atom = round(sum(elemental_energy(e) * n for e, n in counts.items()) / natoms + e_form, 4)
        materials.append(Material(
            material_id=f"mx-{index:04d}", formula=formula, prototype=prototype,
            space_group=PROTOTYPES[prototype][0], sites=sites, a=a, c=c, split=split, metallic=metallic,
            in_db_v1=(formula, prototype) not in missing,
            a_ref=round(a * (1 + delta), 4), c_ref=round(c * (1 + delta), 4) if c else None,
            energy_per_atom=e_atom, formation_energy=e_form,
            band_gap=_band_gap(formula, elements, metallic)))
    return tuple(materials)


def by_id(material_id: str) -> Material:
    for m in library():
        if m.material_id == material_id:
            return m
    raise KeyError(material_id)


def find(formula=None, prototype=None, elements=None, splits=None):
    out = []
    for m in library():
        if formula and m.formula != formula:
            continue
        if prototype and m.prototype != prototype:
            continue
        if elements and not set(elements).issubset(m.elements):
            continue
        if splits and m.split not in splits:
            continue
        out.append(m)
    return out
