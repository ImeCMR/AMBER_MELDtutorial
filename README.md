# MODELING EMPLOYING LIMITED DATA (MELD) in AMBER

## Leraning Outcomes
* Be able to setup and run replica exchange MELD simulation in parallel on GPUs.

## Introduction

Predicting how a protein folds into its native structure from sequence alone is a long-standing problem in computational biology. In theory, molecular dynamics should be able to do so, since the force field establishes an energy landscape with the native state as its global minimum, and molecular dynamics provides not just a single predicted structure but also the populations and mechanisms involved. In reality, however, the bottleneck lies in sampling: the conformational space of even a small polypeptide is prohibitively large, and proteins fold on microsecond-to-millisecond timescales that brute-force atomistic MD cannot reach without specialized hardware.

MELD (Modeling Employing Limited Data) tackles this by combining physics with external information in a Bayesian framework: the force field acts as the prior, the external data provides the likelihood, and MELD then samples the resulting posterior. Unlike conventional restrained molecular dynamics, MELD is designed to make use of information that is vague, sparse, or only partly correct. Restraints are organized into groups and collections, and at any given time, only a certain fraction of them needs to be fulfilled. During the simulation, the best-satisfied subset of restraints is activated on the fly, so it works out which restraints are correct as the protein folds, rather than being forced to obey all the restraints. When folding is carried out on the basis of the amino acid sequence, the restraints are derived from the Coarse Physical Insights (CPI): general truths about globular proteins — namely, that they have hydrophobic cores, that they form secondary structure, and that their β-strands pair. The sampling is carried out using Hamiltonian and temperature replica exchange (H,T-REMD), with the hot replicas, which have weak restraints, exploring a wide range of conformations and the cool replicas, which have fully-scaled restraints, adopting structures that are similar to the native ones.<sup>[[1](https://www.pnas.org/doi/10.1073/pnas.1506788112),[2](https://www.pnas.org/doi/10.1073/pnas.1515561112)]</sup>

![3GB1 Preview](.assets/3gb1.gif)

In this tutorial you will fold the B1 domain of protein G (PDB: 3GB1) starting from its sequence. You will build the system with `tleap` (ff19SB, implicit solvent) and minimize it, generate CPI restraints, run an H,T-REMD MELD simulation in parallel across GPUs, and analyze the resulting trajectories to identify the folded state as a dominant, low-energy population.

**Assumptions**: This tutorial assumes that you have experience building systems using ```tleap```, running MD simulations in parallel with ```pmemd.cuda.MPI```, knowledge on terminology and typical variables used for standard MD simulations in ```AMBER```, and ```MELD```. <br>If you are not familiar with setting up a system, equilibration and running MD simulations please refer to the more basic tutorials before attempting to run ```MELD``` simulations.



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
## References
1. MacCallum, J. L.; Perez, A.; Dill, K. A. Determining Protein Structures by Combining Semireliable Data with Atomistic Physical Models by Bayesian Inference. *Proc. Natl. Acad. Sci.* **2015**, 112 (22), 6985–6990. https://doi.org/10.1073/pnas.1506788112.
2. Perez, A.; MacCallum, J. L.; Dill, K. A. Accelerating Molecular Simulations of Proteins Using Bayesian Inference on Weak Information. *Proc. Natl. Acad. Sci.* **2015**, 112 (38), 11846–11851. https://doi.org/10.1073/pnas.1515561112.