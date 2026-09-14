#!/bin/bash

usage() {
    cat << USAGE
Usage: $(basename "$0") -s <sequence.fasta> [-o <output_prefix>] [-cap]

  -s     input FASTA file with the protein sequence (required)
  -o     prefix for the output prmtop/inpcrd files
         (default: the FASTA file name without its extension)
  -cap   add ACE / NHE terminal cap to the sequence
         (default: no cap)
  -h     show this help message

Example:
  $(basename "$0") -s 3gb1.fasta -o 3gb1
  -> writes leap.in (uncapped); run tleap -f leap.in to build
     3gb1.prmtop and 3gb1.inpcrd

  $(basename "$0") -s 3gb1.fasta -o 3gb1 -cap
  -> same, but the sequence is capped with ACE ... NHE
USAGE
}

fasta=""
prefix=""
cap="no"

while [ $# -gt 0 ]; do
    case $1 in
        -s) if [ -z "$2" ]; then echo "Error: option -s requires an argument" >&2; usage >&2; exit 1; fi
            fasta="$2"; shift 2 ;;
        -o) if [ -z "$2" ]; then echo "Error: option -o requires an argument" >&2; usage >&2; exit 1; fi
            prefix="$2"; shift 2 ;;
        -cap) cap="yes"; shift ;;
        -h) usage; exit 0 ;;
        *) echo "Error: invalid option $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [ -z "$fasta" ]; then
    echo "Error: no input FASTA given (-s)" >&2
    usage >&2
    exit 1
fi

if [ ! -f "$fasta" ]; then
    echo "Error: FASTA file '$fasta' not found" >&2
    exit 1
fi

# Default output prefix: FASTA name without extension
if [ -z "$prefix" ]; then
    prefix=$(basename "$fasta")
    prefix="${prefix%.*}"
fi

sequence=$(grep -v "^>" "$fasta" | tr -d '\n[:space:]')

if [ -z "$sequence" ]; then
    echo "Error: no sequence found in '$fasta'" >&2
    exit 1
fi

convert_aa() {
    case $1 in
        A) echo "ALA" ;;
        C) echo "CYS" ;;
        D) echo "ASP" ;;
        E) echo "GLU" ;;
        F) echo "PHE" ;;
        G) echo "GLY" ;;
        H) echo "HIS" ;;
        I) echo "ILE" ;;
        K) echo "LYS" ;;
        L) echo "LEU" ;;
        M) echo "MET" ;;
        N) echo "ASN" ;;
        P) echo "PRO" ;;
        Q) echo "GLN" ;;
        R) echo "ARG" ;;
        S) echo "SER" ;;
        T) echo "THR" ;;
        V) echo "VAL" ;;
        W) echo "TRP" ;;
        Y) echo "TYR" ;;
        *) echo "UNK" ;;
    esac
}

three_letter_seq=""
for (( i=0; i<${#sequence}; i++ )); do
    aa="${sequence:$i:1}"
    three_letter_seq="$three_letter_seq $(convert_aa $aa)"
done

# Add the terminal cap only if -cap
if [ "$cap" = "yes" ]; then
    leap_seq="ACE$three_letter_seq NHE"
else
    leap_seq="${three_letter_seq# }"
fi

# Create the leap.in file
cat > leap.in << EOF
source leaprc.protein.ff19SB
set default PBradii mbondi2
pro = sequence { $leap_seq }
saveamberparm pro $prefix.prmtop $prefix.inpcrd
quit
EOF

if [ $? -ne 0 ]; then
    echo "Error: failed to create leap.in" >&2
    exit 1
fi
echo "leap.in file created successfully!"
echo "  input  : $fasta"
if [ "$cap" = "yes" ]; then
    echo "  caps   : ACE / NHE"
else
    echo "  caps   : none"
fi

