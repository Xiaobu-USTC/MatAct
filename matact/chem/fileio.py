"""Readers and writers for CIF (P1), POSCAR (VASP 5) and extended XYZ."""
from __future__ import annotations

import re
from collections import Counter

from .structure import Structure, StructureError, _inverse, lattice_from_parameters

FORMATS = ("cif", "poscar", "xyz")
EXTENSIONS = {"cif": "cif", "poscar": "vasp", "xyz": "xyz"}


def format_from_uri(uri: str) -> str:
    ext = uri.rsplit(".", 1)[-1].lower()
    return {"cif": "cif", "vasp": "poscar", "poscar": "poscar", "xyz": "xyz", "extxyz": "xyz"}.get(ext, "")


def to_cif(s: Structure, name: str = "matact") -> str:
    a, b, c = s.abc
    al, be, ga = s.angles
    lines = [f"data_{name}", "_symmetry_space_group_name_H-M   'P 1'",
             f"_cell_length_a   {a:.6f}", f"_cell_length_b   {b:.6f}", f"_cell_length_c   {c:.6f}",
             f"_cell_angle_alpha   {al:.6f}", f"_cell_angle_beta   {be:.6f}", f"_cell_angle_gamma   {ga:.6f}",
             "loop_", "_atom_site_type_symbol", "_atom_site_label",
             "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]
    seen = Counter()
    for sp, f in zip(s.species, s.frac_coords):
        seen[sp] += 1
        lines.append(f"{sp} {sp}{seen[sp]} {f[0]:.6f} {f[1]:.6f} {f[2]:.6f}")
    return "\n".join(lines) + "\n"


def _num(token: str) -> float:
    return float(re.sub(r"\(\d+\)$", "", token))


def from_cif(text: str) -> Structure:
    cell = {}
    for key in ("a", "b", "c"):
        m = re.search(rf"_cell_length_{key}\s+(\S+)", text)
        if not m:
            raise StructureError(f"missing _cell_length_{key}")
        cell[key] = _num(m.group(1))
    for key in ("alpha", "beta", "gamma"):
        m = re.search(rf"_cell_angle_{key}\s+(\S+)", text)
        cell[key] = _num(m.group(1)) if m else 90.0
    lines = [ln.strip() for ln in text.splitlines()]
    species, coords = [], []
    for i, ln in enumerate(lines):
        if ln != "loop_":
            continue
        cols, j = [], i + 1
        while j < len(lines) and lines[j].startswith("_"):
            cols.append(lines[j].split()[0])
            j += 1
        if "_atom_site_fract_x" not in cols:
            continue
        ix, iy, iz = (cols.index(f"_atom_site_fract_{k}") for k in "xyz")
        isym = cols.index("_atom_site_type_symbol") if "_atom_site_type_symbol" in cols else None
        ilab = cols.index("_atom_site_label") if "_atom_site_label" in cols else None
        while j < len(lines) and lines[j] and not lines[j].startswith(("_", "loop_", "data_")):
            row = lines[j].split()
            if len(row) < len(cols):
                raise StructureError(f"short atom_site row: {lines[j]!r}")
            label = row[isym] if isym is not None else re.sub(r"[^A-Za-z]", "", row[ilab])
            species.append(re.match(r"[A-Z][a-z]?", label).group(0))
            coords.append([_num(row[ix]), _num(row[iy]), _num(row[iz])])
            j += 1
    if not species:
        raise StructureError("no _atom_site loop with fractional coordinates")
    lattice = lattice_from_parameters(cell["a"], cell["b"], cell["c"], cell["alpha"], cell["beta"], cell["gamma"])
    return Structure(lattice, species, coords)


def to_poscar(s: Structure, comment: str = "") -> str:
    order = list(dict.fromkeys(s.species))
    lines = [comment or s.formula, "1.0"]
    lines += ["  " + " ".join(f"{x:.8f}" for x in row) for row in s.lattice]
    lines += ["  " + " ".join(order), "  " + " ".join(str(s.species.count(e)) for e in order), "Direct"]
    for e in order:
        for sp, f in zip(s.species, s.frac_coords):
            if sp == e:
                lines.append("  " + " ".join(f"{x:.8f}" for x in f))
    return "\n".join(lines) + "\n"


def from_poscar(text: str) -> Structure:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    try:
        scale = float(lines[1].split()[0])
        lattice = [[float(x) * scale for x in lines[i].split()[:3]] for i in (2, 3, 4)]
        symbols = lines[5].split()
        if all(t.isdigit() for t in symbols):
            raise StructureError("POSCAR without a species line (VASP 4) is not supported")
        counts = [int(t) for t in lines[6].split()]
        k = 7
        if lines[k][0] in "sS":
            k += 1
        cartesian = lines[k][0] in "cCkK"
        rows = [[float(x) for x in lines[k + 1 + i].split()[:3]] for i in range(sum(counts))]
    except (IndexError, ValueError) as exc:
        raise StructureError(f"malformed POSCAR: {exc}") from exc
    if scale <= 0:
        raise StructureError("negative/zero POSCAR scale factors are not supported")
    species = [e for e, n in zip(symbols, counts) for _ in range(n)]
    if cartesian:
        inv = _inverse(lattice)
        rows = [[sum(r[k] * scale * inv[k][j] for k in range(3)) for j in range(3)] for r in rows]
    return Structure(lattice, species, rows)


def to_xyz(s: Structure, comment: str = "") -> str:
    flat = " ".join(f"{x:.8f}" for row in s.lattice for x in row)
    lines = [str(s.natoms), f'Lattice="{flat}" Properties=species:S:1:pos:R:3 pbc="T T T" {comment}'.strip()]
    for sp, f in zip(s.species, s.frac_coords):
        lines.append(sp + " " + " ".join(f"{x:.8f}" for x in s.cart(f)))
    return "\n".join(lines) + "\n"


def from_xyz(text: str) -> Structure:
    lines = text.splitlines()
    try:
        n = int(lines[0].strip())
        m = re.search(r'Lattice="([^"]+)"', lines[1])
        if not m:
            raise StructureError("extended XYZ needs a Lattice=\"...\" header")
        v = [float(x) for x in m.group(1).split()]
        lattice = [v[0:3], v[3:6], v[6:9]]
        rows = [ln.split() for ln in lines[2:2 + n]]
        species = [r[0] for r in rows]
        cart = [[float(x) for x in r[1:4]] for r in rows]
    except (IndexError, ValueError) as exc:
        raise StructureError(f"malformed XYZ: {exc}") from exc
    if len(species) != n:
        raise StructureError("atom count does not match header")
    inv = _inverse(lattice)
    frac = [[sum(c[k] * inv[k][j] for k in range(3)) for j in range(3)] for c in cart]
    return Structure(lattice, species, frac)


def write(s: Structure, fmt: str, name: str = "matact") -> str:
    if fmt == "cif":
        return to_cif(s, name)
    if fmt == "poscar":
        return to_poscar(s, name)
    if fmt == "xyz":
        return to_xyz(s, name)
    raise StructureError(f"unsupported format {fmt!r}")


def read(text: str, fmt: str) -> Structure:
    readers = {"cif": from_cif, "poscar": from_poscar, "xyz": from_xyz}
    if fmt not in readers:
        raise StructureError(f"unsupported format {fmt!r}")
    return readers[fmt](text)
