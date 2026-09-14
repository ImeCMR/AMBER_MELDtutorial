#!/usr/bin/env python3
"""
gen_md_inputs.py -- fan one Amber mdin out into a MELD replica exchange ladder.

    ./gen_md_inputs.py -i meld.mdin --indxf restraints.indxf -n 10

You write ONE ordinary Amber input file and keep owning it.  This script copies
it once per replica, changing only the two fields that have to differ, and
writes the groupfile pmemd reads:

    meld.mdin.001 .. meld.mdin.NNN   your file, per replica
    meld.groupfile                   the -groupfile for  pmemd.MPI -ng N

WHAT IT CHANGES, AND NOTHING ELSE
---------------------------------
  ig       every replica needs its own Langevin seed.  Left identical, all N
           replicas would integrate the same noise.
  temp0    set to the value MELD will actually apply -- TSCALE(alpha) from the
           index file given with --indxf, where alpha = (rung-1)/(N-1) comes
           from the replica's position in the -ng group order.  MELD overwrites
           temp0 during setup either way; writing the real number just stops
           the file from disagreeing with the run.

           --indxf is required rather than inferred: the temperature ladder is
           the one thing here that comes from outside your mdin, so it is named
           outright.  It is checked against the `indxf=` your mdin declares, and
           a disagreement is refused -- otherwise this would report one ladder
           while the run used another.

Every other line is copied through byte for byte, with one exception it tells
you about: a `&wt type='REST'` block is REMOVED.  MELD's own RAMP record already
fades the restraints in, and Amber's modwt would multiply rk2/rk3 a second time
on top of that.  The removed text is printed.

Nothing else is added -- no header, no injected comments.  A generated replica
differs from your sample only in the ways listed above.

WHAT IT REFUSES
---------------
The sample is the single source of truth, so anything -rem 6 needs but the file
does not have is an error naming what to add, not a silent fix-up:
nmropt=1, meld=1, indxf=, numexchg>0, a `&wt TYPE='END'` namelist, and a DISANG=
line immediately after it.  That last one matters more than it looks: Amber
stops reading redirections at the first line that is not one, so a comment or a
blank line between the `&wt` terminator and DISANG= silently costs you every
restraint in the run.

There is no submission script -- launch it however you normally do.  The
command is printed at the end.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import sys

# ==========================================================================
#  SETTINGS -- the few things that are not in your mdin.
# ==========================================================================

SETTINGS = {
    "prefix": "meld",
    "seed": None,             # None = fresh random ig per replica; an int = reproducible
    "temp0_fallback": 300.0,  # only if the index file carries no TSCALE record
}

# nmr_calls.F90: data redirc /'LISTIN','LISTOUT','DISANG','NOESY','SHIFTS',
#                             'DUMPAVE','PCSHIFT','DIPOLE'/
REDIRECTIONS = ("LISTIN", "LISTOUT", "DISANG", "NOESY", "SHIFTS",
                "DUMPAVE", "PCSHIFT", "DIPOLE")


# ==========================================================================
#  Reading a Fortran namelist file without pretending it is one
#
#  Amber mdins are hand-written, so the parsing here is deliberately literal:
#  it finds block boundaries and variable values, and otherwise leaves every
#  byte of the user's file alone.
# ==========================================================================

def split_comment(line):
    """(code, comment) for one line, with '!' inside quotes left alone."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "!":
            return line[:i], line[i:]
    return line, ""


def has_terminator(code, skip=0):
    """True if a namelist terminator ('/' or &end) appears outside quotes."""
    quote = None
    i = skip
    while i < len(code):
        ch = code[i]
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "/":
            return True
        elif ch in "&$" and code[i + 1:i + 4].lower() == "end":
            return True
        i += 1
    return False


def find_namelists(lines, name):
    """[(start, end)] inclusive line indices of every &name ... / block."""
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        code = split_comment(lines[i])[0]
        stripped = code.strip()
        if stripped[:1] in ("&", "$"):
            head = stripped[1:].split()
            if head and head[0].lower() == name.lower():
                after = code.index(stripped[0]) + 1 + len(head[0])
                j = i
                while j < n:
                    codej = split_comment(lines[j])[0]
                    if has_terminator(codej, after if j == i else 0):
                        break
                    j += 1
                blocks.append((i, min(j, n - 1)))
                i = j + 1
                continue
        i += 1
    return blocks


def split_commas(text):
    """Split on commas that are not inside quotes."""
    out, buf, quote = [], [], None
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch == ",":
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return out


def namelist_vars(lines, start, end):
    """{name: raw value} for one namelist block."""
    code = " ".join(split_comment(lines[k])[0] for k in range(start, end + 1))
    code = re.sub(r"^\s*[&$]\w+", " ", code)
    code = re.sub(r"(/|[&$][eE][nN][dD])\s*$", " ", code.strip())
    values = {}
    for part in split_commas(code):
        if "=" in part:
            key, val = part.split("=", 1)
            key = key.strip().lower()
            if key:
                values[key] = val.strip()
    return values


def unquote(text):
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    return text


def as_number(text):
    try:
        return float(unquote(text).replace("d", "e").replace("D", "e"))
    except (ValueError, AttributeError):
        return None


def set_variable(lines, start, end, name, value):
    """Rewrite `name = ...` in place inside a block, keeping its comment.

    Returns True if it replaced an existing assignment, False if it had to be
    inserted (which the caller reports -- an mdin with no `ig` would otherwise
    hand every replica Amber's fixed default seed).
    """
    rx = re.compile(r"(\b" + name + r"\s*=\s*)([^,!\s]+)", re.IGNORECASE)
    for k in range(start, end + 1):
        code, comment = split_comment(lines[k])
        if rx.search(code):
            lines[k] = rx.sub(lambda m: m.group(1) + value, code, count=1) + comment
            return True
    lines.insert(start + 1, f"   {name}={value},")
    return False


# ==========================================================================
#  The alpha ladder, read back out of the index file the mdin names
# ==========================================================================

TSCALE_KINDS = ("constant", "linear", "geometric")
SETTING_WORDS = {"ACTIVE", "KEEP", "SCALER", "RAMP", "K", "GROUPS", "MEMBERS"}


def read_ladder(path):
    """TSCALE segments, whether ADAPT is on, and the collection scalers."""
    segments, scalers, ramps, colls = [], [], [], []
    adapt = False
    if not os.path.exists(path):
        raise ValueError(f"{path} not found -- run gen_restraints.py first")

    with open(path) as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.split("#", 1)[0].split("!", 1)[0].strip()
            if not line:
                continue
            tok = line.replace(",", " ").split()
            kw = tok[0].upper()
            if kw == "ADAPT":
                adapt = True
            elif kw == "SCALER" and len(tok) > 1:
                scalers.append(tok[1])
            elif kw == "RAMP" and len(tok) > 1:
                ramps.append(tok[1])
            elif kw in ("COLL", "COLLECTION") and len(tok) > 1:
                colls.append(collection_scalers(tok))
            elif kw in ("TSCALE", "TEMPERATURE"):
                segments.append(parse_tscale(tok, kw, f"{path}:{lineno}"))
    return {"segments": segments, "adapt": adapt, "scalers": scalers,
            "ramps": ramps, "colls": colls}


def collection_scalers(tok):
    """(collection name, scaler, ramp) off a COLL/COLLECTION record.

    Only the names are read -- the scaler maths stays in pmemd, so nothing here
    can drift away from what the run actually applies.
    """
    name = tok[1] if tok[1].upper() not in SETTING_WORDS else "(unnamed)"
    scaler = ramp = "none"
    for i, t in enumerate(tok[:-1]):
        if t.upper() == "SCALER":
            scaler = tok[i + 1]
        elif t.upper() == "RAMP":
            ramp = tok[i + 1]
    return (name, scaler, ramp)


def parse_tscale(tok, kw, where):
    """One ladder segment as (kind, a_min, a_max, t_min, t_max)."""
    def num(t):
        try:
            return float(t)
        except ValueError:
            raise ValueError(f"{where}: {t!r} is not a number") from None

    strip = lambda t: t.strip('"').strip("'").lower()

    if kw == "TEMPERATURE":
        kind = strip(tok[1])                       # type comes second here
        if kind == "constant":
            return (kind, 0.0, 1.0, num(tok[2]), num(tok[2]))
        return (kind, num(tok[2]), num(tok[3]), num(tok[4]), num(tok[5]))

    kind = strip(tok[-1])                          # ... and last here
    if kind not in TSCALE_KINDS:
        raise ValueError(f"{where}: unknown temperature scaler {tok[-1]!r}")
    if len(tok) == 3:
        if kind != "constant":
            raise ValueError(f"{where}: only 'constant' takes a single temperature")
        return (kind, 0.0, 1.0, num(tok[1]), num(tok[1]))
    if len(tok) != 6:
        raise ValueError(f"{where}: TSCALE wants 'a_min a_max t_min t_max <type>'")
    return (kind, num(tok[1]), num(tok[2]), num(tok[3]), num(tok[4]))


def seg_temperature(seg, alpha):
    """meld_scalers_mod.F90::seg_temperature -- clamped outside its own range."""
    kind, a_min, a_max, t_min, t_max = seg
    if kind == "constant":
        return t_min
    if alpha <= a_min:
        return t_min
    if alpha > a_max:
        return t_max
    frac = (alpha - a_min) / (a_max - a_min)
    if kind == "linear":
        return t_min + frac * (t_max - t_min)
    return math.exp((math.log(t_max) - math.log(t_min)) * frac + math.log(t_min))


def temperature_at(segments, alpha):
    """meld_temperature_set: the segment owning this alpha, clamping at both ends."""
    if not segments:
        return None
    ordered = sorted(segments, key=lambda s: (s[1], s[2]))
    if alpha <= ordered[0][1]:
        return seg_temperature(ordered[0], alpha)
    for seg in ordered:
        if alpha <= seg[2]:
            return seg_temperature(seg, alpha)
    return seg_temperature(ordered[-1], alpha)


def alpha_of_rung(rung, n_rungs):
    """meld_wrapper.F90::meld_alpha_of_rung -- the -ng group order is the ladder."""
    if n_rungs <= 1:
        return 0.0
    return (rung - 1) / (n_rungs - 1)


# ==========================================================================
#  Reading and checking the sample mdin
# ==========================================================================

class MdinError(Exception):
    """The sample is missing something -rem 6 needs. Carries the whole list."""


def load_sample(path):
    """Parse the sample, refuse what -rem 6 cannot run, and report what it found."""
    with open(path) as fh:
        lines = fh.read().splitlines()

    cntrl = find_namelists(lines, "cntrl")
    if not cntrl:
        raise MdinError([f"{path} has no &cntrl namelist"])
    cntrl = cntrl[0]
    cvars = namelist_vars(lines, *cntrl)

    wt_blocks = find_namelists(lines, "wt")
    wt_types = [unquote(namelist_vars(lines, s, e).get("type", "")).upper()
                for s, e in wt_blocks]

    problems = []

    def need(name, want, note):
        got = as_number(cvars.get(name, ""))
        if name not in cvars:
            problems.append(f"{name} is missing -- add `{name}={want}` to &cntrl ({note})")
        elif got is None or got != float(want):
            problems.append(f"{name}={cvars[name]} -- must be {want} ({note})")

    need("nmropt", 1, "MELD rides on the NMR restraint machinery")
    need("meld", 1, "-rem 6 selects the MELD exchange and requires meld=1")

    if "indxf" not in cvars:
        problems.append("indxf is missing -- add `indxf='restraints.indxf'` to &cntrl "
                        "(the MELD index file; required whenever meld=1)")

    numexchg = as_number(cvars.get("numexchg", ""))
    if "numexchg" not in cvars:
        problems.append("numexchg is missing -- add `numexchg=<n>` to &cntrl "
                        "(pmemd rejects a REMD run with no exchanges)")
    elif numexchg is None or numexchg < 1:
        problems.append(f"numexchg={cvars['numexchg']} -- must be > 0")

    if "END" not in wt_types:
        problems.append("no `&wt TYPE='END'` namelist -- nmropt=1 aborts without the "
                        "terminator, even when there are no weight changes")

    # DISANG has to sit in the run of redirection lines that starts immediately
    # after the last &wt block. Amber stops at the first line that is not a
    # redirection, so a comment or a blank line in between costs every restraint.
    disang = None
    if wt_blocks:
        after = max(e for _, e in wt_blocks) + 1
        k = after
        while k < len(lines):
            code = split_comment(lines[k])[0].strip()
            name = code.split("=", 1)[0].strip().upper() if "=" in code else ""
            if name not in REDIRECTIONS:
                break
            if name == "DISANG":
                disang = code.split("=", 1)[1].strip()
                break
            k += 1
        if disang is None:
            offender = (repr(lines[k]) if k < len(lines) else "end of file")
            problems.append(
                "no DISANG= line in the redirection block after the last &wt "
                f"namelist (stopped at {offender}). It must follow the closing "
                "'/' with nothing between them -- not a comment, not a blank line")

    if problems:
        raise MdinError(problems)

    return {"lines": lines, "cntrl": cntrl, "vars": cvars,
            "wt_blocks": wt_blocks, "wt_types": wt_types,
            "indxf": unquote(cvars["indxf"]), "disang": disang,
            "numexchg": int(numexchg), "nstlim": as_number(cvars.get("nstlim", "0")),
            "dt": as_number(cvars.get("dt", "0.001"))}


def strip_rest_blocks(lines, wt_blocks, wt_types):
    """Remove every `&wt type='REST'` block; return the text that went."""
    removed = []
    for (start, end), kind in sorted(zip(wt_blocks, wt_types), reverse=True):
        if kind == "REST":
            removed.append("\n".join(lines[start:end + 1]))
            del lines[start:end + 1]
    return list(reversed(removed))


# ==========================================================================
#  Helpers
# ==========================================================================

def check_prmtop(path, dt):
    """Refuse a topology that is not there, and flag one the timestep outgrows.

    HMR moves mass from heavy atoms onto the hydrogens so a 4 fs step is stable;
    running dt > 2.5 fs on the un-repartitioned topology is the classic way to
    make a MELD run blow up hours in.  The name is the only clue available here,
    so this warns rather than refuses."""
    if not os.path.exists(path):
        sys.exit(f"error: topology {path} not found")
    if dt > 0.0025 and "hmr" not in os.path.basename(path).lower():
        return (f"WARNING: dt={dt:g} ps needs a hydrogen-mass-repartitioned "
                f"topology, and {os.path.basename(path)} is not named like one")
    return None




def main(argv=None):
    p = argparse.ArgumentParser(
        description="Fan one Amber mdin out into a MELD -rem 6 replica ladder. "
                    "Only ig and temp0 are changed; a &wt type='REST' block is "
                    "removed and reported.")
    p.add_argument("-i", "--mdin", required=True,
                   help="your Amber input file -- the sample every replica is copied from")
    p.add_argument("--indxf", required=True,
                   help="the MELD index file to read the temperature scaler from; "
                        "must be the same file your mdin names in indxf=")
    p.add_argument("-n", "--nreplicas", type=int, required=True,
                   help="replicas on the ladder (pmemd requires an even number)")
    p.add_argument("-c", "--coords", required=True,
                   help="starting coordinates, e.g. ../1_system_setup/min.rst; "
                        "use {rep} in the path for per-replica files")
    p.add_argument("-p", "--prmtop", required=True,
                   help="topology file")
    p.add_argument("--dry-run", action="store_true",
                   help="check the sample, report the ladder, write nothing")
    args = p.parse_args(argv)

    nrep = args.nreplicas
    if nrep < 2:
        sys.exit("error: a replica exchange run needs at least 2 replicas")
    if nrep % 2 != 0:
        sys.exit(f"error: {nrep} replicas -- pmemd refuses an odd replica count for "
                 f"-rem 6 (mdin_ctrl_dat.F90: 'REMD requires an even number of "
                 f"replicas!'). Use an even number, or set gremd_acyc=1 in your mdin.")

    try:
        sample = load_sample(args.mdin)
    except MdinError as exc:
        print(f"error: {args.mdin} cannot run under -rem 6:", file=sys.stderr)
        for problem in exc.args[0]:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    except OSError as exc:
        return sys.exit(f"error: {exc}")

    # pmemd resolves the mdin's indxf= relative to the run directory, which is
    # this one, so realpath is the honest comparison: a different spelling of the
    # same file is fine, a different file is not.
    if os.path.realpath(args.indxf) != os.path.realpath(sample["indxf"]):
        sys.exit(f"error: --indxf and the mdin disagree about the index file\n"
                 f"  --indxf      {args.indxf}\n"
                 f"  {args.mdin} says  indxf='{sample['indxf']}'\n"
                 f"pmemd reads the one in the mdin, so the ladder reported here "
                 f"would not be the ladder that runs.")

    try:
        ladder = read_ladder(args.indxf)
    except (OSError, ValueError) as exc:
        sys.exit(f"error: {exc}")

    prmtop = args.prmtop
    prmtop_warning = check_prmtop(prmtop, sample["dt"])

    prefix = SETTINGS["prefix"]
    groupfile = f"{prefix}.groupfile"

    rows = []
    for rung in range(1, nrep + 1):
        alpha = alpha_of_rung(rung, nrep)
        t = temperature_at(ladder["segments"], alpha)
        rows.append((rung, alpha, SETTINGS["temp0_fallback"] if t is None else t))

    # ---- report -----------------------------------------------------------
    total_steps = int((sample["nstlim"] or 0) * sample["numexchg"])
    print(f"sample mdin    : {args.mdin}")
    print(f"topology       : {prmtop}")
    if prmtop_warning:
        print(f"                 ! {prmtop_warning}")
    print(f"index file     : {args.indxf}"
          + (f"   scalers: {', '.join(ladder['scalers'])}" if ladder["scalers"] else ""))
    print(f"DISANG         : {sample['disang']}")
    for name, scaler, ramp in ladder["colls"]:
        print(f"                 {name:<4} scaler {scaler}, ramp {ramp}")
    print(f"replicas       : {nrep}")
    print(f"per exchange   : {sample['nstlim']:g} steps x {sample['dt']:g} ps = "
          f"{sample['nstlim'] * sample['dt']:g} ps")
    print(f"total          : {sample['numexchg']} exchanges, {total_steps:,} steps, "
          f"{total_steps * sample['dt'] / 1000:g} ns per replica")
    print("adaptation     : " + ("ON -- these alphas are the STARTING ladder; the "
                                 "adaptor re-spaces them at run time"
                                 if ladder["adapt"] else
                                 "off -- the ladder stays where the index file puts it"))
    if not ladder["segments"]:
        print(f"  ! no TSCALE record in {args.indxf}: temp0 stays at "
              f"{SETTINGS['temp0_fallback']} K on every replica")
    print()
    print(f"{'rung':>5}{'alpha':>10}{'temp0 (K)':>12}   mdin")
    for rung, alpha, t in rows:
        print(f"{rung:>5}{alpha:>10.4f}{t:>12.2f}   {prefix}.mdin.{rung:03d}")

    if args.dry_run:
        print("\n(dry run -- the sample checks out, nothing written)")
        return 0

    # ---- write ------------------------------------------------------------
    rng = random.Random(SETTINGS["seed"])
    removed_rest, inserted, missing = [], set(), []

    with open(groupfile, "w") as gf:
        for rung, alpha, t in rows:
            rep = f"{rung:03d}"
            lines = list(sample["lines"])
            wt_blocks, wt_types = sample["wt_blocks"], sample["wt_types"]

            gone = strip_rest_blocks(lines, wt_blocks, wt_types)
            if gone and not removed_rest:
                removed_rest = gone

            # &cntrl moves if a REST block above it was removed -- it never is,
            # but re-finding costs nothing and cannot go stale.
            cntrl = find_namelists(lines, "cntrl")[0]
            if not set_variable(lines, *cntrl, "ig", str(rng.randint(1, 999999))):
                inserted.add("ig")
            cntrl = find_namelists(lines, "cntrl")[0]
            if not set_variable(lines, *cntrl, "temp0", f"{t:.2f}"):
                inserted.add("temp0")

            name = f"{prefix}.mdin.{rep}"
            with open(name, "w") as fh:
                fh.write("\n".join(lines) + "\n")

            coords = args.coords.replace("{rep}", rep)
            if not os.path.exists(coords):
                missing.append(coords)

            gf.write(f"-O -rem 6 -remlog rem.log "
                     f"-i {name} -o {prefix}.mdout.{rep} -c {coords} "
                     f"-r {prefix}.rst.{rep} -x {prefix}.nc.{rep} "
                     f"-inf {prefix}.mdinfo.{rep} -p {prmtop}\n")
        gf.write("#\n")

    print()
    print(f"wrote {nrep} mdin files and {groupfile}")
    if removed_rest:
        print()
        print("removed from every replica -- MELD's RAMP already ramps the restraints,")
        print("and modwt would scale rk2/rk3 a second time on top of it:")
        for block in removed_rest:
            for line in block.splitlines():
                print(f"    {line}")
    if inserted:
        print()
        print(f"  ! {', '.join(sorted(inserted))} was not in {args.mdin}; "
              f"inserted into &cntrl")
        if "ig" in inserted:
            print("  ! without it every replica would share Amber's fixed default seed")
    if missing:
        print()
        print(f"  ! {len(missing)} starting coordinate file(s) do not exist yet, "
              f"first: {missing[0]}")
    print()
    print("launch with:")
    print(f"    srun --mpi=pmix_v5 $AMBERHOME/bin/pmemd.MPI "
          f"-ng {nrep} -groupfile {groupfile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
