# AMBER_MELDtutorial
This repository has a tutorial on how to implement Replica Exchange MELD simulation in AMBER molecular dynamics software packag.

## 1. System setup

In this section we will build the system via Leap and run minimization via Sander. Here a brief description of the system and the procedure used to generate the topology and coordinate files.

For this tutorial we will generate structure of the 1UAO protein using the sequence FASTA file. Use following command to download the sequence.
```
curl -o 3gb1.fasta https://www.rcsb.org/fasta/entry/3GB1
```
Now lets bild the system to simulate using Leap. Here we will use ff19SB forcefield and mbondii2 radii that are appropriate for the igb=5 option in sander. We use [makeLeap.x](1_system_setup/makeLeap.x) to make the Leap input file ([leap.in](1_system_setup/leap.in)) using the [1uoa.fasta](1_system_setup/1uoa.fasta) file we downloaded before.

* [makeLeap.x](1_system_setup/makeLeap.x)
```
#!/bin/bash

sequence=$(grep -v "^>" 1uoa.fasta | tr -d '\n')
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

# Create the leap.in file
cat > leap.in << EOF
source leaprc.protein.ff19SB
set default PBradii mbondi2
pro = sequence { ACE$three_letter_seq NHE }
saveamberparm pro 1uoa.prmtop 1uoa.inpcrd
quit
EOF

echo "leap.in file created successfully!"
```
Execute this with following commands for 3GB1 protein:
```
chmod +x makeLeap.x
./makeLeap.x -s 3gb1.fasta -o 3gb1 
```
This will create [leap.in](1_system_setup/leap.in). Use command ```tleap -f leap.in 2> leap.log``` to run this with a log file. Carefully look for any Errors, Warnings in the log file. This creates [3gb1.prmtop](1_system_setup/3gb1.prmtop) and [3gb1.inpcrd](1_system_setup/3gb1.inpcrd).\\
Here, we use  Hydrogen Mass Repartitioning ([HMR](https://pubs.acs.org/jctcce/article/11/4/1864/847792/Long-Time-Step-Molecular-Dynamics-through-Hydrogen)) to alow longer time step simulations. We can simply do this using built in *ParmED* package inside *AMBER*. Using ParmED input file [HMR.in](1_system_setup/HMR.in) create new topology file [3gb1_HMR.prmtop](1_system_setup/3gb1_HMR.prmtop).
```
parmed -i HMR.in
```
\\
Next, we move on to minimization. Our main goal is to use MELD to guid our protein to fold into it's native structure by imposing some restrains we already know. Therefore, we do a simple minimization. [min.in](1_system_setup/min.in)
```
energy minimization
 &cntrl
  imin=1, maxcyc=2000, ncyc=500,
  ntwr = 1000, ntpr = 100,
  cut = 999.0, rgbmax = 999.0,
  ntb = 0, igb = 5, saltcon = 0.0,
 /
 ```
