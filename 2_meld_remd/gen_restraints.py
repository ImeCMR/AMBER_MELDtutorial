#!/usr/bin/env python3
"""
gen_restraints.py -- build the restraint files for MELD in AMBER pmemd.

    ./gen_restraints.py ss.dat -p ../1_system_setup/system.prmtop

Two output files are written:

  <prefix>.disang   the DISANG file.  The *geometry* of every restraint, one
                    &rst namelist each, in the order MELD claims them.
  <prefix>.indxf    the MELD index file.  The group/collection hierarchy,
                    the *force constants*, and the alpha ladder (temperature
                    scaler, force scalers, time ramp).  Named from mdin as
                    indxf='<prefix>.indxf'.

Their restraint counts match by construction -- pmemd aborts if they disagree.

To change anything, either edit the PARAMS table and DEFAULT_LADDER at the top
of this file, or keep this file alone and work from a parameter file:

    ./gen_restraints.py --write-params meld_params.in    # annotated template
    ./gen_restraints.py ss.dat -p system.prmtop -i meld_params.in

Three selectively active collections are written, in this DISANG order:

  SS  secondary structure.
      This extracts the helices, then the strands, appends both to a single
      list of groups and hands that one list to a single collection
      Both kinds are the same shape: a 5-residue window
      whose 3 interior residues give 2*3 = 6 phi/psi torsions, plus 3 CA-CA
      distances, with RestraintGroup(rests, len(rests)) -- hence KEEP all.

  SP  strand pairing.  For every (residue in strand A, residue in strand B)
      pair across two different extended segments, one group holding the N-O
      and the O-N distance -- groups of two, KEEP 1.

  HY  hydrophobic contacts.  For every pair of hydrophobic residues far enough
      apart in sequence, one group holding every side-chain atom pair -- KEEP 1.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import namedtuple

# ==========================================================================
#  UNITS
#
#  Force constants here are in MELD's own units, exactly as the python MELD
#  quotes them, because that is where these numbers come from:
#
#      distance          kJ/mol/nm^2
#      torsion, angle    kJ/mol/deg^2
#
#  The index file carries a 'UNITS meld' record and pmemd converts on the way
#  in (meld_wrapper.F90), so nothing here has to be pre-converted:
#
#      rk2 [kcal/mol/A^2  ] = k [kJ/mol/nm^2 ] * 0.5/4.184/100        = k * 1.1950287e-3
#      rk2 [kcal/mol/rad^2] = k [kJ/mol/deg^2] * 0.5/4.184*(180/pi)^2 = k * 392.30477
#
#  The two factors differ by five orders of magnitude, which is exactly why a
#  collection holding torsions AND distances -- SS is one -- has to state them
#  separately.  The DISANG rk2/rk3 are written already converted, so the two
#  files agree even though only the index file is read for force constants.
#
#  Distances below are in nm, as in MELD; DISANG gets them in Angstrom.
# ==========================================================================

NM_TO_ANG = 10.0
K_DIST_TO_AMBER = 0.5 / 4.184 / 100.0                    # kJ/mol/nm^2  -> kcal/mol/A^2
K_TORS_TO_AMBER = 0.5 / 4.184 * (180.0 / math.pi) ** 2   # kJ/mol/deg^2 -> kcal/mol/rad^2

# Amber turns a torsion restraint linear beyond r1/r4 while MELD stays
# quadratic all the way round, so r1 = r2 - 180 and r4 = r3 + 180 keeps both
# wings out of reach.  pmemd warns if a DISANG torsion is written any tighter,
# because the energies -- and therefore the MELD group selection -- would not
# match the reference.
TORSION_WING = 180.0


# ==========================================================================
#  PARAMETERS -- edit here, or --write-params and edit the file instead.
#
#  Every entry becomes one 'key = value' line of the parameter template, with
#  its comment, so the template and these defaults cannot drift apart.
#
#  kind:  int | float | name | count | auto
#    count  'all', an integer, a fraction (0.45) or a percentage (45%)
#    auto   'auto', or whatever the comment says instead
# ==========================================================================

Param = namedtuple("Param", "section key default kind comment")

PARAMS = [
    Param("system", "first_residue", "auto", "auto",
          "prmtop residue holding position 1 of the ss string; auto = first non-cap"),

    Param("ss", "ss_active", "85%", "count",
          "groups the collection keeps: reference int(len(groups)*0.85)"),
    Param("ss", "ss_run_length", 5, "int",
          "residues per secondary structure window (parse.py uses 5)"),
    Param("ss", "ss_min_match", 4, "int",
          "how many of those must carry the type (parse.py min_secondary_match)"),
    Param("ss", "ss_k_distance", 2500.0, "float",
          "CA-CA force constant, kJ/mol/nm^2   -> 2.98757 kcal/mol/A^2"),
    Param("ss", "ss_k_torsion", 2.5e-2, "float",
          "phi/psi force constant, kJ/mol/deg^2 -> 9.80762 kcal/mol/rad^2"),
    Param("ss", "ss_scaler", "ss", "name", "a SCALER named in the ladder, or none"),
    Param("ss", "ss_ramp", "warmup", "name", "a RAMP named in the ladder, or none"),

    Param("sp", "sp_active", "auto", "auto",
          "auto = sp_fraction x the number of residues marked E"),
    Param("sp", "sp_fraction", 0.45, "float",
          "the reference's int(active*0.45)"),
    Param("sp", "sp_min_strand_length", 1, "int",
          "ignore runs of E shorter than this"),
    Param("sp", "sp_r2", 0.0, "float", "N-O flat bottom start, nm"),
    Param("sp", "sp_r3", 0.35, "float", "N-O flat bottom end, nm"),
    Param("sp", "sp_k", 250.0, "float",
          "force constant, kJ/mol/nm^2 -> 0.298757 kcal/mol/A^2"),
    Param("sp", "sp_scaler", "prot", "name", "a SCALER named in the ladder, or none"),
    Param("sp", "sp_ramp", "warmup", "name", "a RAMP named in the ladder, or none"),

    Param("hy", "hy_active", "auto", "auto",
          "auto = hy_contacts_per_residue x the hydrophobic residue count"),
    Param("hy", "hy_contacts_per_residue", 1.2, "float",
          "the reference's int(1.2*n_hydrophobic)"),
    Param("hy", "hy_min_sep", 7, "int",
          "minimum separation in sequence between paired residues"),
    Param("hy", "hy_r2", 0.0, "float", "contact flat bottom start, nm"),
    Param("hy", "hy_r3", 0.5, "float", "contact flat bottom end, nm"),
    Param("hy", "hy_k", 250.0, "float",
          "force constant, kJ/mol/nm^2 -> 0.298757 kcal/mol/A^2"),
    Param("hy", "hy_scaler", "prot", "name", "a SCALER named in the ladder, or none"),
    Param("hy", "hy_ramp", "warmup", "name", "a RAMP named in the ladder, or none"),

    Param("shared", "quadratic_cut", 0.2, "float",
          "distances turn linear this far past r3, nm (parse.py quadratic_cut)"),
]

DEFAULTS = {p.key: p.default for p in PARAMS}

SECTION_TITLES = {
    "system": ("system", [
        "How the H/E/. string lines up with the topology.",
        "",
        "The ss string covers the PROTEIN residues only -- one character per",
        "residue of the sequence you folded, and NOTHING for a terminal cap.",
        "'auto' skips leading caps (ACE, FOR, NME, NHE, NH2, OHE), so a capped",
        "build lines itself up:",
        "",
        "   capped    prmtop:  1 ACE   2 MET   3 THR  ...  57 GLU   58 NHE",
        "             ss.dat:          .       E      ...  .",
        "                              ^ ss position 1 = prmtop 2, so auto -> 2",
        "",
        "   uncapped  prmtop:  1 MET   2 THR  ...            auto -> 1",
        "",
        "Override with a number only when auto guesses wrong: a non-standard",
        "residue at the N terminus, or an ss string describing part of a chain.",
        "The run prints the mapping it used and the residues either side of it,",
        "so check that line rather than trusting the guess.",
    ]),
    "ss": ("SS -- secondary structure (helix AND extended, one collection)", [
        "One group per 5-residue window: 6 phi/psi torsions + 3 CA-CA",
        "distances, all of them kept (KEEP all), as parse.py builds them.",
    ]),
    "sp": ("SP -- strand pairing", [
        "One group per residue pair across two strands, holding the N-O and",
        "the O-N distance; either one satisfies the pair (KEEP 1).",
    ]),
    "hy": ("HY -- hydrophobic contacts", [
        "One group per hydrophobic residue pair, holding every side-chain",
        "atom pair; any one contact satisfies the pair (KEEP 1).",
    ]),
    "shared": ("shared restraint shape", []),
}

# --------------------------------------------------------------------------
#  The alpha ladder.  These lines are copied into the .indxf verbatim, so they
#  are plain MELD index syntax and anything the reader accepts can go here:
#  several TSCALE records for a piecewise temperature ladder, plateau scalers,
#  a ramp_switcher, and so on.  Only INFO, ADAPT, TSCALE, TEMPERATURE, SCALER
#  and RAMP belong here -- the collections themselves are generated.
#
#  ADAPT is on if the record is present -- commenting it out is how it is
#  turned off, and that is pmemd's own default.
#
#  alpha = (run - 1)/(nrep - 1), from this replica's -ng group position.
#  Force constant  = base_k x SCALER(alpha) x RAMP(exchange step).
#  Bath temperature = TSCALE(alpha), which replaces temp0 from mdin.
#  A ladder clamps outside the alpha range it covers, so the single TSCALE
#  below is exactly the reference's one GeometricTemperatureScaler.
# --------------------------------------------------------------------------

DEFAULT_LADDER = """\
INFO    off                              # on -> each replica writes meld.info.<rank>

# ADAPT is a switch by PRESENCE. Comment the next line out (or delete it) and the
# alpha ladder never moves, which is pmemd's own default. There is no "ADAPT off":
# the keyword always switches adaptation on and always wants all five numbers.
# ADAPT   2.0 50 50 -1 0.02                # growth burn_in every stop_after min_acc -- [work in progress]

TSCALE  0.0 0.4  300.0 450.0  "geometric"   # GeometricTemperatureScaler(0, 0.4, 300, 450)

SCALER  ss                       "constant"       # SS holds full strength all the way up
SCALER  prot  0.4 1.0 4.0        "nonlinear"      # nonlinear, alpha_min 0.4, alpha_max 1.0, factor 4
RAMP    warmup 1 200 1e-3 1 4.0  "nonlinear_ramp" # restraints fade in over 200 exchange steps
"""

LADDER_KEYWORDS = {"INFO", "ADAPT", "TSCALE", "TEMPERATURE", "SCALER", "RAMP"}


# ==========================================================================
#  Reference data: MELD's own tables.
# ==========================================================================

# The reference's hydrophobic side-chain atoms, keyed by one-letter code.
HP_ATOMS = {
    "A": ["CA", "CB"],
    "V": ["CA", "CB", "CG1", "CG2"],
    "L": ["CA", "CB", "CG", "CD1", "CD2"],
    "I": ["CA", "CB", "CG1", "CG2", "CD1"],
    "F": ["CA", "CB", "CG", "CD1", "CE1", "CZ", "CE2", "CD2"],
    "W": ["CA", "CB", "CG", "CD1", "NE1", "CE2", "CZ2", "CH2", "CZ3", "CE3", "CD2"],
    "M": ["CA", "CB", "CG", "SD", "CE"],
    "P": ["CD", "CG", "CB", "CA"],
}

# Two secondary structure kinds.  d13 is the (i, i+3) and (i+1, i+4)
# CA-CA pair, d14 is the (i, i+4) one; both are (r2, r3) in nm, with r1 = 0 and
# r4 = r3 + quadratic_cut.
SS_KINDS = {
    "H": {
        "name": "helix",
        "phi": (-62.5, 17.5),
        "psi": (-42.5, 17.5),
        "d13": (0.485, 0.561),
        "d14": (0.581, 0.684),
    },
    "E": {
        "name": "extended",
        "phi": (-117.5, 27.5),
        "psi": (145.0, 25.0),
        "d13": (0.785, 1.063),
        "d14": (1.086, 1.394),
    },
}

AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "ASH": "D",
    "CYS": "C", "CYX": "C", "CYM": "C", "GLN": "Q", "GLU": "E",
    "GLH": "E", "GLY": "G", "HIS": "H", "HIE": "H", "HID": "H",
    "HIP": "H", "ILE": "I", "LEU": "L", "LYS": "K", "LYN": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T",
    "TRP": "W", "TYR": "Y", "VAL": "V",
}

CAP_RESIDUES = {"ACE", "FOR", "NME", "NHE", "NH2", "OHE"}

# --------------------------------------------------------------------------
# prmtop
# --------------------------------------------------------------------------

class Prmtop:
    """Read an AMBER prmtop file and provide access to residue and atom information.

    Residue boundaries come from RESIDUE_POINTER, so caps, non-standard residues and anything else the
    topology holds are handled without special cases.
    """

    def __init__(self, filename):
        self.filename = filename
        sections, formats = self._read_sections(filename)
        for flag in ("ATOM_NAME", "RESIDUE_LABEL", "RESIDUE_POINTER"):
            if flag not in sections:
                raise ValueError(f"{filename}: no %FLAG {flag} section")

        self.atom_names = self._parse_fixed(sections["ATOM_NAME"],
                                            self._a_width(formats.get("ATOM_NAME"), 4))
        self.residue_labels = self._parse_fixed(sections["RESIDUE_LABEL"],
                                                self._a_width(formats.get("RESIDUE_LABEL"), 4))
        self.residue_pointer = [int(tok) for line in sections["RESIDUE_POINTER"]
                                for tok in line.split()]

        self.natom = len(self.atom_names)
        self.nres = len(self.residue_labels)
        if len(self.residue_pointer) != self.nres:
            raise ValueError(f"{filename}: RESIDUE_POINTER has "
                             f"{len(self.residue_pointer)} entries but there are "
                             f"{self.nres} residues")

        # One name -> 1-based atom index map per residue.  First occurrence
        # wins, which is what matters for the duplicated names some caps carry.
        self._by_residue = []
        for r in range(self.nres):
            start = self.residue_pointer[r] - 1
            end = self.residue_pointer[r + 1] - 1 if r + 1 < self.nres else self.natom
            table = {}
            for a in range(start, end):
                table.setdefault(self.atom_names[a], a + 1)
            self._by_residue.append(table)

    @staticmethod
    def _read_sections(filename):
        sections, formats = {}, {}
        flag = None
        with open(filename) as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("%FLAG"):
                    flag = line.split()[1] if len(line.split()) > 1 else None
                    if flag is not None:
                        sections[flag] = []
                    continue
                if line.startswith("%FORMAT"):
                    if flag is not None:
                        formats[flag] = line
                    continue
                if line.startswith("%"):        # %COMMENT, %VERSION, ...
                    continue
                if flag is not None:
                    sections[flag].append(line)
        return sections, formats

    @staticmethod
    def _a_width(format_line, default):
        """Width of the a<width> field in a %FORMAT(20a4) line."""
        if not format_line:
            return default
        body = format_line[format_line.find("(") + 1:format_line.rfind(")")]
        pos = body.lower().find("a")
        if pos < 0:
            return default
        digits = ""
        for ch in body[pos + 1:]:
            if ch.isdigit():
                digits += ch
            else:
                break
        return int(digits) if digits else default

    @staticmethod
    def _parse_fixed(lines, width):
        out = []
        for line in lines:
            for i in range(0, len(line), width):
                field = line[i:i + width].strip()
                if field:
                    out.append(field)
        return out

    def residue_label(self, resnum):
        """1-based residue label."""
        if not 1 <= resnum <= self.nres:
            raise ValueError(f"residue {resnum} is outside 1..{self.nres}")
        return self.residue_labels[resnum - 1]

    def atom(self, resnum, name):
        """1-based atom index of `name` in 1-based residue `resnum`."""
        if not 1 <= resnum <= self.nres:
            raise KeyError(f"residue {resnum} is outside 1..{self.nres}")
        try:
            return self._by_residue[resnum - 1][name]
        except KeyError:
            raise KeyError(f"residue {resnum} ({self.residue_label(resnum)}) "
                           f"has no atom named {name!r}") from None


# --------------------------------------------------------------------------
# The restraint hierarchy
#
# Restraint  one &rst.  `r` is (r1, r2, r3, r4), already in Amber units --
#            Angstrom for a distance, degrees for a torsion.
# Group      what MELD ranks at the first selection level.
# Collection what MELD ranks at the second.
# --------------------------------------------------------------------------

Restraint = namedtuple("Restraint", "kind atoms r comment")
Group = namedtuple("Group", "restraints keep comment")
Collection = namedtuple("Collection", "name groups active keep scaler ramp k comment")


def distance_restraint(a1, a2, r2_nm, r3_nm, quad_cut_nm, comment):
    r = (0.0,
         r2_nm * NM_TO_ANG,
         r3_nm * NM_TO_ANG,
         (r3_nm + quad_cut_nm) * NM_TO_ANG)
    return Restraint("distance", (a1, a2), r, comment)


def torsion_restraint(a1, a2, a3, a4, target, width, comment):
    lo, hi = target - width, target + width
    r = (lo - TORSION_WING, lo, hi, hi + TORSION_WING)
    return Restraint("torsion", (a1, a2, a3, a4), r, comment)


# --------------------------------------------------------------------------
# Secondary structure
# --------------------------------------------------------------------------

def read_secondary_structure(filename):
    """A single H/E/. string, exactly as parse.py::_get_secondary_sequence."""
    chunks = []
    with open(filename) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            chunks.append(line)
    ss = "".join(chunks).replace(" ", "")
    bad = sorted({c for c in ss if c not in "HE."})
    if bad:
        raise ValueError(f"{filename}: unknown secondary structure type(s) "
                         f"{', '.join(repr(c) for c in bad)} -- only H, E and . are allowed")
    if not ss:
        raise ValueError(f"{filename}: no secondary structure read")
    return ss


def one_letter(label):
    """One-letter code for a prmtop residue label, N/C terminal forms included."""
    if label in AA3_TO_AA1:
        return AA3_TO_AA1[label]
    if len(label) == 4 and label[0] in "NC" and label[1:] in AA3_TO_AA1:
        return AA3_TO_AA1[label[1:]]
    return None


def protein_run_length(prmtop, first):
    """How many residues from `first` on are proteins rather than caps -- what
    an ss string starting there would have to be to cover the whole chain."""
    n = 0
    r = first
    while r <= prmtop.nres and prmtop.residue_label(r) not in CAP_RESIDUES:
        n += 1
        r += 1
    return n


def mapping_note(prmtop, first, n_ss):
    """'ss 1-56 -> prmtop 2-57  (ACE at 1, NHE at 58)'.

    Terminal caps are the easy way to get this off by one, so the run says out
    loud which residues it landed on and what sits either side of them.
    """
    last = first + n_ss - 1
    flank = []
    if first > 1:
        flank.append(f"{prmtop.residue_label(first - 1)} at {first - 1}")
    if last < prmtop.nres:
        flank.append(f"{prmtop.residue_label(last + 1)} at {last + 1}")
    text = f"ss 1-{n_ss}  ->  prmtop {first}-{last}"
    return text + (f"   ({', '.join(flank)})" if flank else "   (whole topology)")


def first_protein_residue(prmtop):
    """1-based prmtop residue number of the first non-cap residue."""
    for r in range(1, prmtop.nres + 1):
        if prmtop.residue_label(r) not in CAP_RESIDUES:
            return r
    raise ValueError(f"{prmtop.filename}: every residue looks like a cap")


def extract_runs(ss, ss_type, run_length, at_least):
    """parse.py::_extract_secondary_runs -- every window of `run_length`
    residues holding at least `at_least` of `ss_type`.  Windows overlap, which
    is deliberate: consecutive windows along one long helix each become their
    own group, so a partially formed helix can satisfy some of them."""
    marks = [1 if c == ss_type else 0 for c in ss]
    n = len(ss)
    if n < run_length:
        return []
    return [(i, i + run_length)
            for i in range(n - run_length + 1)
            if sum(marks[i:i + run_length]) >= at_least]


def strand_segments(ss, min_length):
    """Maximal runs of 'E', as inclusive 0-based (start, end) pairs."""
    segments = []
    i, n = 0, len(ss)
    while i < n:
        if ss[i] != "E":
            i += 1
            continue
        j = i
        while j < n and ss[j] == "E":
            j += 1
        if j - i >= min_length:
            segments.append((i, j - 1))
        i = j
    return segments


# --------------------------------------------------------------------------
# The three collections
# --------------------------------------------------------------------------

class Builder:
    """Turns sequence positions into atom indices, and complains usefully."""

    def __init__(self, prmtop, first_residue, warn):
        self.prmtop = prmtop
        self.first = first_residue
        self.warn = warn

    def resnum(self, i):
        """prmtop residue number of 0-based sequence position i.  i may be -1
        or len(sequence): a phi needs the C of the preceding residue and a psi
        the N of the following one, and at the chain ends those are the caps."""
        return self.first + i

    def try_atom(self, i, name):
        try:
            return self.prmtop.atom(self.resnum(i), name)
        except (KeyError, ValueError) as exc:
            self.warn(str(exc))
            return None


def build_secondary_structure(builder, ss, params):
    """One group per secondary structure window, helices first then extended
    runs -- the order parse.py builds them in.  KEEP all."""
    run_length = params["ss_run_length"]
    min_match = params["ss_min_match"]
    quad_cut = params["quadratic_cut"]
    groups = []
    for ss_type in ("H", "E"):
        spec = SS_KINDS[ss_type]
        for start, end in extract_runs(ss, ss_type, run_length, min_match):
            rests = []
            # Torsions for the interior residues: parse.py's
            # range(run.start + 1, run.end - 1).
            for index in range(start + 1, end - 1):
                res = index + 1                     # 1-based, for comments only
                c_prev = builder.try_atom(index - 1, "C")
                n_this = builder.try_atom(index, "N")
                ca = builder.try_atom(index, "CA")
                c_this = builder.try_atom(index, "C")
                n_next = builder.try_atom(index + 1, "N")
                if None not in (c_prev, n_this, ca, c_this):
                    rests.append(torsion_restraint(
                        c_prev, n_this, ca, c_this, *spec["phi"],
                        comment=f"phi res {res}"))
                if None not in (n_this, ca, c_this, n_next):
                    rests.append(torsion_restraint(
                        n_this, ca, c_this, n_next, *spec["psi"],
                        comment=f"psi res {res}"))
            # The three CA-CA distances of the window.
            for off_i, off_j, key in ((0, 3, "d13"), (1, 4, "d13"), (0, 4, "d14")):
                ai = builder.try_atom(start + off_i, "CA")
                aj = builder.try_atom(start + off_j, "CA")
                if None in (ai, aj):
                    continue
                r2_nm, r3_nm = spec[key]
                rests.append(distance_restraint(
                    ai, aj, r2_nm, r3_nm, quad_cut,
                    comment=f"CA {start + off_i + 1} - CA {start + off_j + 1}"))
            if rests:
                groups.append(Group(rests, "all",
                                    f"{spec['name']} window {start + 1}-{end}"))
    return groups


def build_strand_pairs(builder, segments, params):
    """One group per residue pair across two different strands, holding the
    N-O and the O-N distance.  KEEP 1: either hydrogen bond satisfies it."""
    r2_nm, r3_nm = params["sp_r2"], params["sp_r3"]
    quad_cut = params["quadratic_cut"]
    groups = []
    for a in range(len(segments) - 1):
        start_a, end_a = segments[a]
        for b in range(a + 1, len(segments)):
            start_b, end_b = segments[b]
            for i in range(start_a, end_a + 1):
                for j in range(start_b, end_b + 1):
                    rests = []
                    for name_i, name_j in (("N", "O"), ("O", "N")):
                        ai = builder.try_atom(i, name_i)
                        aj = builder.try_atom(j, name_j)
                        if None in (ai, aj):
                            continue
                        rests.append(distance_restraint(
                            ai, aj, r2_nm, r3_nm, quad_cut,
                            comment=f"{name_i} {i + 1} - {name_j} {j + 1}"))
                    if rests:
                        groups.append(Group(rests, "1",
                                            f"strand pair {i + 1} - {j + 1}"))
    return groups


def build_hydrophobic(builder, sequence, params):
    """One group per hydrophobic residue pair, holding every side-chain atom
    pair.  KEEP 1: any one contact satisfies the pair."""
    min_sep = params["hy_min_sep"]
    r2_nm, r3_nm = params["hy_r2"], params["hy_r3"]
    quad_cut = params["quadratic_cut"]
    groups = []
    n = len(sequence)
    for i in range(n):
        if sequence[i] not in HP_ATOMS:
            continue
        for j in range(i + min_sep, n):
            if sequence[j] not in HP_ATOMS:
                continue
            rests = []
            for name_i in HP_ATOMS[sequence[i]]:
                for name_j in HP_ATOMS[sequence[j]]:
                    ai = builder.try_atom(i, name_i)
                    aj = builder.try_atom(j, name_j)
                    if None in (ai, aj):
                        continue
                    rests.append(distance_restraint(
                        ai, aj, r2_nm, r3_nm, quad_cut,
                        comment=f"{name_i} {i + 1} - {name_j} {j + 1}"))
            if rests:
                groups.append(Group(rests, "1",
                                    f"{sequence[i]}{i + 1} - {sequence[j]}{j + 1}"))
    return groups


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def fmt(value):
    return f"{value:g}"


def write_disang(filename, collections, header_lines):
    """One &rst per restraint, collections in order.  pmemd echoes the leading
    '#' comment block to mdout, and skips any line whose first non-blank
    character is not '&', so the per-group comments are free."""
    index = 0
    with open(filename, "w") as fh:
        for line in header_lines:
            fh.write(f"# {line}\n" if line else "#\n")
        fh.write("#\n")
        for coll in collections:
            n_rest = sum(len(g.restraints) for g in coll.groups)
            fh.write(f"#\n# {'=' * 68}\n")
            fh.write(f"# collection {coll.name} -- {coll.comment}\n")
            fh.write(f"# {len(coll.groups)} groups, {n_rest} restraints, "
                     f"DISANG {index + 1}-{index + n_rest}\n")
            fh.write(f"# {'=' * 68}\n")
            for gnum, group in enumerate(coll.groups, start=1):
                fh.write(f"#\n# {coll.name} group {gnum}: {group.comment} "
                         f"({len(group.restraints)} restraints, keep {group.keep})\n")
                for rest in group.restraints:
                    index += 1
                    if rest.kind not in coll.k:
                        raise ValueError(
                            f"collection {coll.name} holds a {rest.kind} restraint "
                            f"but sets no force constant for that type")
                    k_amber = (coll.k["distance"] * K_DIST_TO_AMBER
                               if rest.kind == "distance"
                               else coll.k["torsion"] * K_TORS_TO_AMBER)
                    fh.write(f"#  {index}: {rest.kind} {rest.comment}\n")
                    fh.write(" &rst\n")
                    fh.write("   iat = " + ", ".join(str(a) for a in rest.atoms) + ",\n")
                    fh.write("   r1 = {:.3f}, r2 = {:.3f}, r3 = {:.3f}, r4 = {:.3f},\n"
                             .format(*rest.r))
                    fh.write(f"   rk2 = {k_amber:.5f}, rk3 = {k_amber:.5f},\n")
                    fh.write(" /\n")
        fh.write("#\n# end of restraints\n")
    return index


def run_length_encode(sizes):
    encoded = []
    for size in sizes:
        if encoded and encoded[-1][0] == size:
            encoded[-1][1] += 1
        else:
            encoded.append([size, 1])
    return [(size, count) for size, count in encoded]


def group_specs(groups):
    """GROUPS specs for a run of groups: '<size>' or '<size> x <count>'."""
    return [f"{size}" if count == 1 else f"{size} x {count}"
            for size, count in run_length_encode([len(g.restraints) for g in groups])]


def collection_records(coll, specs_per_line=20):
    """The COLL record for one collection.

    A one-line COLL is read settings-first, so K reaches every group whatever
    the order.  A record is capped at 512 fields and 2048 characters, though,
    which a collection whose groups differ in size can reach -- so a long spec
    list falls back to the COLLECTION/END block, where each GROUPS line is its
    own record.  Settings come first in the block, so they still reach every
    group below them.
    """
    settings = [f"ACTIVE {coll.active}", f"KEEP {coll.keep}"]
    if coll.scaler:
        settings.append(f"SCALER {coll.scaler}")
    if coll.ramp:
        settings.append(f"RAMP {coll.ramp}")
    k_fields = " ".join(f"{kind} {fmt(value)}" for kind, value in sorted(coll.k.items()))
    settings.append(f"K {k_fields}")
    settings = "  ".join(settings)

    specs = group_specs(coll.groups)
    if len(specs) <= specs_per_line:
        return [f"COLL {coll.name}  {settings}  GROUPS " + " ".join(specs)]

    lines = [f"COLLECTION {coll.name}  {settings}"]
    for start in range(0, len(specs), specs_per_line):
        lines.append("  GROUPS " + " ".join(specs[start:start + specs_per_line]))
    lines.append("END")
    return lines


def write_indxf(filename, collections, ladder_lines, header_lines):
    with open(filename, "w") as fh:
        for line in header_lines:
            fh.write(f"# {line}\n" if line else "#\n")
        fh.write("\nMELD 3\n\n") # REMOVE LATER --  MELD 3 is just new format (read in meld_wrapper.F90)
        # The force constants on the COLL records below are MELD's own units;
        # pmemd converts them per restraint type.
        fh.write("UNITS  meld\n\n")
        for line in ladder_lines:
            fh.write(line + "\n")
        for coll in collections:
            n_rest = sum(len(g.restraints) for g in coll.groups)
            fh.write(f"\n# {coll.comment}\n")
            fh.write(f"# {len(coll.groups)} groups, {n_rest} restraints\n")
            for line in collection_records(coll):
                fh.write(line + "\n")


# --------------------------------------------------------------------------
# Parameters: the template, and reading one back
# --------------------------------------------------------------------------

PARAM_HEADER = """\
# ==========================================================================
#  MELD restraint parameters -- template written by gen_restraints.py
#
#      gen_restraints.py ss.dat -i this_file
#
#  Anything left out falls back to the defaults compiled into the script.
#
#  FORCE CONSTANT UNITS.  The values below are in MELD's own units, exactly
#  as the python MELD (maccallumlab/meld) quotes them:
#
#      distance          kJ/mol/nm^2
#      torsion, angle    kJ/mol/deg^2
#
#  The generated index file carries a 'UNITS meld' record, so pmemd converts
#  them per restraint type on the way in:
#
#      rk2 [kcal/mol/A^2  ] = k [kJ/mol/nm^2 ] * 0.5/4.184/100        = k * 1.1950287e-3
#      rk2 [kcal/mol/rad^2] = k [kJ/mol/deg^2] * 0.5/4.184*(180/pi)^2 = k * 392.30477
#
#  The two factors differ by five orders of magnitude, which is why a
#  collection holding both kinds -- SS is one -- states them separately.
#
#  Distances are in nm here and are written to DISANG in Angstrom.
#
#  A count ('active') may be written as 'all', an integer, a fraction (0.45)
#  or a percentage (45%).
# ==========================================================================
"""

PARAM_LADDER_HEADER = """\
# ==========================================================================
#  THE ALPHA LADDER
#
#  Everything between LADDER and END_LADDER is copied into the .indxf file
#  verbatim, so it is plain MELD index syntax: anything the pmemd reader
#  accepts can go here.
#
#      alpha            = (run - 1)/(nrep - 1), from the -ng group position
#      force constant   = base_k x SCALER(alpha) x RAMP(exchange step)
#      bath temperature = TSCALE(alpha), which replaces temp0 from mdin
#
#  A ladder clamps outside the alpha range it covers, so one TSCALE record is
#  exactly the reference's single GeometricTemperatureScaler.  Several TSCALE
#  records make a piecewise ladder; they must tile their range with no gap and
#  no overlap.  The type comes last and is usually quoted:
#
#      SCALER <name>  [params]  "constant|linear|geometric|nonlinear|
#                                plateau|plateausmooth|plateaunonlinear"
#      RAMP   <name>  [params]  "constant_ramp|linear_ramp|nonlinear_ramp|
#                                ramp_switcher"
#      TSCALE a_min a_max t_min t_max  "constant|linear|geometric"
#      ADAPT  growth burn_in every stop_after min_acc -- work in progress
#             (present = adaptation ON, absent = OFF; there is no 'ADAPT off')
#      INFO   on|off
#
#  Only those keywords belong here -- the collections themselves are built
#  from the secondary structure file.
# ==========================================================================
"""


def write_params_template(filename):
    with open(filename, "w") as fh:
        fh.write(PARAM_HEADER)
        width = max(len(p.key) for p in PARAMS)
        seen = set()
        for param in PARAMS:
            if param.section not in seen:
                seen.add(param.section)
                title, notes = SECTION_TITLES[param.section]
                fh.write(f"\n# ---- {title} " + "-" * max(4, 66 - len(title)) + "\n")
                for note in notes:
                    fh.write(f"# {note}\n" if note else "#\n")
            value = param.default
            text = fmt(value) if isinstance(value, float) else str(value)
            fh.write(f"{param.key:<{width}} = {text:<10}  # {param.comment}\n")
        fh.write("\n\n" + PARAM_LADDER_HEADER + "\nLADDER\n")
        for line in DEFAULT_LADDER.splitlines():
            fh.write(("  " + line if line.strip() else "") + "\n")
        fh.write("END_LADDER\n")


def coerce(param, text, where):
    try:
        if param.kind == "int":
            return int(text)
        if param.kind == "float":
            return float(text)
        if param.kind == "auto":
            return "auto" if text.lower() == "auto" else text
        return text                      # name, count
    except ValueError:
        raise ValueError(f"{where}: {param.key} wants a{'n integer' if param.kind == 'int' else ' number'}, "
                         f"got {text!r}") from None


def read_params(filename):
    """Read a parameter file: 'key = value' lines plus one LADDER block."""
    by_key = {p.key: p for p in PARAMS}
    values = dict(DEFAULTS)
    ladder = []
    in_ladder = False

    with open(filename) as fh:
        for lineno, raw in enumerate(fh, start=1):
            where = f"{filename}:{lineno}"
            line = raw.split("#", 1)[0].strip()
            if line.upper() == "LADDER":
                in_ladder = True
                continue
            if line.upper() == "END_LADDER":
                in_ladder = False
                continue
            if in_ladder:
                # Ladder lines are copied into the .indxf whole -- blank lines
                # and trailing '#' comments included, since the index file
                # comments the same way -- so an edited ladder reaches the run
                # looking exactly as it was written.
                if line:
                    keyword = line.split()[0].upper()
                    if keyword not in LADDER_KEYWORDS:
                        raise ValueError(
                            f"{where}: {keyword} does not belong in the ladder block "
                            f"-- only {', '.join(sorted(LADDER_KEYWORDS))}. The "
                            f"collections are generated, not written here.")
                ladder.append(raw.rstrip("\n").strip())
                continue
            if not line:
                continue
            if "=" not in line:
                raise ValueError(f"{where}: expected 'key = value', got {line!r}")
            key, text = line.split("=", 1)
            key, text = key.strip().lower(), text.strip()
            if key not in by_key:
                raise ValueError(f"{where}: unknown parameter {key!r}. "
                                 f"Known: {', '.join(sorted(by_key))}")
            values[key] = coerce(by_key[key], text, where)

    if in_ladder:
        raise ValueError(f"{filename}: LADDER block is missing its END_LADDER")
    return values, (ladder if ladder else DEFAULT_LADDER.splitlines())


def check_ladder(ladder_lines, params):
    """Every SCALER/RAMP a collection names has to be defined in the ladder."""
    defined = {"SCALER": {"none"}, "RAMP": {"none"}}
    for line in ladder_lines:
        tokens = line.split("#", 1)[0].split()
        if len(tokens) >= 2 and tokens[0].upper() in defined:
            defined[tokens[0].upper()].add(tokens[1])
    for coll in ("ss", "sp", "hy"):
        for kind in ("scaler", "ramp"):
            name = params[f"{coll}_{kind}"]
            if name not in defined[kind.upper()]:
                raise ValueError(
                    f"{coll}_{kind} names {name!r}, which the ladder does not "
                    f"define. Defined {kind}s: "
                    f"{', '.join(sorted(defined[kind.upper()]))}")


def count_token(value, n_groups):
    """An ACTIVE/KEEP count as the index file spells it: 'all', an integer, a
    fraction or a percentage.  Absolute counts are clamped here so the file
    never asks for more groups than the collection holds."""
    token = str(value).strip()
    if token.lower() == "all":
        return "all", n_groups
    if token.endswith("%"):
        return token, int(round(float(token[:-1]) / 100.0 * n_groups))
    if "." in token:
        return token, int(round(float(token) * n_groups))
    count = min(int(token), n_groups)
    return str(count), count


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build the DISANG and MELD index files for a MELD run in pmemd. "
                    "Defaults live at the top of this script; --write-params dumps "
                    "them as an annotated file you can edit and pass back with -i.")
    p.add_argument("ss", help="secondary structure file: combination of \"H\", \"E\", \".\" per residue")
    p.add_argument("-p", "--prmtop", required=True,
                   help="AMBER topology (required)")
    p.add_argument("-o", "--output", default="restraints",
                   help="output prefix; writes <prefix>.disang and <prefix>.indxf "
                        "(default: restraints)")
    p.add_argument("-i", "--params",
                   help="parameter file overriding the built-in defaults")
    p.add_argument("--write-params", metavar="FILE", nargs="?", const="meld_params.in",
                   help="write the annotated parameter template and exit "
                        "(default: meld_params.in)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.write_params:
        write_params_template(args.write_params)
        print(f"wrote {args.write_params}")
        print(f"edit it, then: {os.path.basename(sys.argv[0])} "
              f"{args.ss} -p {args.prmtop or '<system.prmtop>'} "
              f"-i {args.write_params}")
        return 0

    warnings = []

    try:
        prmtop_path = args.prmtop
        if prmtop_path is None:
            raise ValueError(
                "no topology given -- pass the AMBER topology with "
                "-p/--prmtop <system.prmtop>")

        params, ladder_lines = (read_params(args.params) if args.params
                                else (dict(DEFAULTS), DEFAULT_LADDER.splitlines()))
        check_ladder(ladder_lines, params)

        if params["ss_min_match"] > params["ss_run_length"]:
            raise ValueError("ss_min_match cannot exceed ss_run_length")
        if params["ss_min_match"] > 5:
            raise ValueError("ss_min_match must be 5 or less (parse.py's own limit)")

        prmtop = Prmtop(prmtop_path)
        ss = read_secondary_structure(args.ss)

        first = (first_protein_residue(prmtop) if params["first_residue"] == "auto"
                 else int(params["first_residue"]))
        if first < 1 or first + len(ss) - 1 > prmtop.nres:
            raise ValueError(
                f"the secondary structure string is {len(ss)} residues and starts "
                f"at prmtop residue {first}, which does not fit inside the "
                f"{prmtop.nres} residues of {prmtop_path}")

        sequence = ""
        for offset in range(len(ss)):
            label = prmtop.residue_label(first + offset)
            code = one_letter(label)
            if code is None:
                raise ValueError(
                    f"prmtop residue {first + offset} is {label}, which the "
                    f"secondary structure string cannot describe -- set "
                    f"first_residue in a parameter file")
            sequence += code
    except (OSError, ValueError) as exc:
        sys.exit(f"error: {exc}")

    builder = Builder(prmtop, first, warnings.append)

    ss_groups = build_secondary_structure(builder, ss, params)
    segments = strand_segments(ss, params["sp_min_strand_length"])
    sp_groups = build_strand_pairs(builder, segments, params)
    hy_groups = build_hydrophobic(builder, sequence, params)

    n_extended = ss.count("E")
    n_hydrophobic = sum(1 for c in sequence if c in HP_ATOMS)

    sp_active = params["sp_active"]
    if sp_active == "auto":
        sp_active = max(1, int(params["sp_fraction"] * n_extended))
    hy_active = params["hy_active"]
    if hy_active == "auto":
        hy_active = max(1, int(params["hy_contacts_per_residue"] * n_hydrophobic))

    collections = []
    if ss_groups:
        collections.append(Collection(
            name="SS", groups=ss_groups,
            active=count_token(params["ss_active"], len(ss_groups))[0], keep="all",
            scaler=params["ss_scaler"], ramp=params["ss_ramp"],
            k={"distance": params["ss_k_distance"], "torsion": params["ss_k_torsion"]},
            comment="secondary structure: helix and extended windows, "
                    "6 phi/psi torsions + 3 CA-CA distances each"))
    if sp_groups:
        collections.append(Collection(
            name="SP", groups=sp_groups,
            active=count_token(sp_active, len(sp_groups))[0], keep="1",
            scaler=params["sp_scaler"], ramp=params["sp_ramp"],
            k={"distance": params["sp_k"]},
            comment="strand pairing: N-O and O-N per residue pair, keep either"))
    if hy_groups:
        collections.append(Collection(
            name="HY", groups=hy_groups,
            active=count_token(hy_active, len(hy_groups))[0], keep="1",
            scaler=params["hy_scaler"], ramp=params["hy_ramp"],
            k={"distance": params["hy_k"]},
            comment="hydrophobic contacts: every side-chain atom pair, keep one"))

    if not collections:
        sys.exit("error: no restraints were built -- check the secondary "
                 "structure file")

    disang_name = f"{args.output}.disang"
    indxf_name = f"{args.output}.indxf"

    header = [
        "MELD restraints, written by gen_restraints.py",
        f"topology  : {prmtop_path}",
        f"structure : {args.ss}",
        f"parameters: {args.params if args.params else 'script defaults'}",
        f"residues  : {mapping_note(prmtop, first, len(ss))}",
        "",
        f"DISANG    : {disang_name}",
        f"index     : {indxf_name}",
        "The two files describe the same restraints and must be used together.",
    ]

    try:
        n_written = write_disang(disang_name, collections, header)
        write_indxf(indxf_name, collections, ladder_lines, header)
    except (OSError, ValueError) as exc:
        sys.exit(f"error: {exc}")

    # ---------------------------------------------------------------- report
    print(f"topology            : {prmtop_path}")
    print(f"parameters          : {args.params if args.params else 'script defaults'}")
    print(f"residue mapping     : {mapping_note(prmtop, first, len(ss))}")
    n_protein = protein_run_length(prmtop, first)
    if n_protein != len(ss):
        print(f"  ! the ss string is {len(ss)} residues but there are {n_protein} "
              f"protein residues from {first} on.")
        print(f"  ! that is fine for part of a chain -- otherwise check the caps "
              f"and first_residue.")
    print(f"secondary structure : {len(ss)} residues, "
          f"{ss.count('H')} H, {n_extended} E, {ss.count('.')} coil")
    print(f"strand segments     : {len(segments)}"
          + (" (" + ", ".join(f"{a + 1}-{b + 1}" for a, b in segments) + ")" if segments else ""))
    print(f"hydrophobic residues: {n_hydrophobic}")
    print()
    print(f"{'collection':<12}{'groups':>8}{'restraints':>12}{'keep/group':>12}"
          f"{'active':>10}  force constants")
    for coll in collections:
        n_rest = sum(len(g.restraints) for g in coll.groups)
        kept = count_token(coll.active, len(coll.groups))[1]
        ks = ", ".join(
            f"{kind} {fmt(v)} -> {v * (K_DIST_TO_AMBER if kind == 'distance' else K_TORS_TO_AMBER):.5f}"
            for kind, v in sorted(coll.k.items()))
        print(f"{coll.name:<12}{len(coll.groups):>8}{n_rest:>12}"
              f"{coll.keep:>12}{coll.active + f' ({kept})':>10}  {ks}")
    print(f"{'total':<12}{sum(len(c.groups) for c in collections):>8}{n_written:>12}")
    print()
    print(f"wrote {disang_name} and {indxf_name}")

    if warnings:
        unique = sorted(set(warnings))
        print()
        print(f"{len(warnings)} restraint(s) skipped, a needed atom was missing:")
        for message in unique[:10]:
            print(f"  {message}")
        if len(unique) > 10:
            print(f"  ... and {len(unique) - 10} more")

    print()
    print("in mdin:")
    print(f"   nmropt=1, meld=1, indxf='{indxf_name}',")
    print("  /")
    print(f" DISANG={disang_name}")
    print()
    print("The RAMP record in the index file is what ramps the restraints in. "
          "Drop any\n&wt type='REST' block from mdin: MELD rewrites rk2/rk3 from "
          "the ladder every\nexchange and modwt would then scale them a second "
          "time on top of it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
