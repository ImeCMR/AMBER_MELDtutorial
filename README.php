<?php // AMBER-MELD Tutorial - README.php ?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AMBER MELD Tutorial</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            font-size: 14px;
            color: #000000;
            background: #ffffff;
            margin: 0;
            padding: 10px 20px;
            line-height: 1.6;
        }

        h1 {
            color: #003366;
            font-size: 1.4em;
            margin-bottom: 5px;
        }

        h2 {
            color: #003366;
            font-size: 1.1em;
            margin-top: 20px;
            margin-bottom: 5px;
        }

        h3, .numbered-step {
            color: #003399;
            font-size: 1.0em;
            margin-top: 15px;
            margin-bottom: 5px;
            font-weight: bold;
            display: block;
        }

        h4 {
            color: #003399;
            font-size: 1.0em;
            margin-top: 12px;
            margin-bottom: 4px;
        }

        p {
            margin: 8px 0;
            text-align: justify;
        }

        pre {
            background-color: #e8e8e8;
            font-family: "Courier New", Courier, monospace;
            font-size: 13px;
            padding: 8px 10px;
            margin: 8px 0;
            border: none;
            white-space: pre-wrap;
            word-wrap: break-word;
            width: 100%;
            box-sizing: border-box;
            display: block;
        }

        code {
            font-family: "Courier New", Courier, monospace;
            font-size: 13px;
        }

        p code, li code {
            background: none;
            font-style: italic;
        }

        ul, ol {
            margin: 5px 0 10px 20px;
            padding: 0;
        }

        ul li, ol li {
            margin-bottom: 4px;
        }

        a {
            color: #0000cc;
        }

        a:visited {
            color: #551a8b;
        }

        .note-box {
            border: 1px solid #aabbdd;
            background: #eef3fa;
            padding: 10px 14px;
            margin: 12px 0;
        }

        .note-box strong {
            color: #003366;
        }

        table {
            border-collapse: collapse;
            margin: 10px 0;
            width: 100%;
            max-width: 100%;
            overflow-x: auto;
            display: block;
        }

        th, td {
            border: 1px solid #cccccc;
            padding: 6px 10px;
            text-align: left;
            font-size: 13px;
        }

        th {
            background-color: #e8e8e8;
            color: #003366;
        }

        hr {
            border: none;
            border-top: 1px solid #cccccc;
            margin: 20px 0;
        }

        footer {
            margin-top: 30px;
            font-size: 12px;
            color: #333333;
            border-top: 1px solid #000000;
            padding-top: 4px;
        }

        sup {
            font-size: 0.8em;
            vertical-align: super;
        }

        em {
            font-style: italic;
        }
    </style>
</head>
<body>

<!-- ═══════════════════════════════════════════════
     TITLE
═══════════════════════════════════════════════ -->
<h1>MODELING EMPLOYING LIMITED DATA (MELD) in AMBER</h1>

<!-- ==============================================
     Learning Outcomes
     ============================================== -->
<h2>Learning Outcomes</h2>

<ul>
    <li>Be able to setup and run replica exchange MELD simulation in parallel on GPUs.</li>
</ul>

<!-- ==============================================
     Introduction
     ============================================== -->
<h2>Introduction</h2>

<p>
    Predicting how a protein folds into its native structure from sequence alone is a long-standing problem in computational biology. In theory, molecular dynamics should be able to do so, since the force field establishes an energy landscape with the native state as its global minimum, and molecular dynamics provides not just a single predicted structure but also the populations and mechanisms involved. In reality, however, the bottleneck lies in sampling: the conformational space of even a small polypeptide is prohibitively large, and proteins fold on microsecond-to-millisecond timescales that brute-force atomistic MD cannot reach without specialized hardware.
</p>

<p>
    MELD (Modeling Employing Limited Data) tackles this by combining physics with external information in a Bayesian framework: the force field acts as the prior, the external data provides the likelihood, and MELD then samples the resulting posterior. Unlike conventional restrained molecular dynamics, MELD is designed to make use of information that is vague, sparse, or only partly correct. Restraints are organized into groups and collections, and at any given time, only a certain fraction of them needs to be fulfilled. During the simulation, the best-satisfied subset of restraints is activated on the fly, so it works out which restraints are correct as the protein folds, rather than being forced to obey all the restraints. When folding is carried out on the basis of the amino acid sequence, the restraints are derived from the Coarse Physical Insights (CPI): general truths about globular proteins — namely, that they have hydrophobic cores, that they form secondary structure, and that their β-strands pair. The sampling is carried out using Hamiltonian and temperature replica exchange (H,T-REMD), with the hot replicas, which have weak restraints, exploring a wide range of conformations and the cool replicas, which have fully-scaled restraints, adopting structures that are similar to the native ones.<sup><a href="https://www.pnas.org/doi/10.1073/pnas.1506788112">1</a>,<a href="https://www.pnas.org/doi/10.1073/pnas.1515561112">2</a></sup>
</p>

<p>
    <img src=".assets/3gb1.gif" alt="3GB1 Preview" style="max-width:100%; height:auto;">
</p>

<p>
    In this tutorial you will fold the B1 domain of protein G (PDB: 3GB1) starting from its sequence. You will build the system with <code>tleap</code> (ff19SB, implicit solvent) and minimize it, generate CPI restraints, run an H,T-REMD MELD simulation in parallel across GPUs, and analyze the resulting trajectories to identify the folded state as a dominant, low-energy population.
</p>

<div class="note-box">
    <strong>Assumptions:</strong> This tutorial assumes that you have experience building systems using <code>tleap</code>, running MD simulations in parallel with <code>pmemd.cuda.MPI</code>, knowledge on terminology and typical variables used for standard MD simulations in <code>AMBER</code>, and <code>MELD</code>. <br>If you are not familiar with setting up a system, running MD simulations please refer to the more basic tutorials before attempting to run <code>Replica Exchange MELD</code> simulations.
</div>

<hr>

<!-- ═══════════════════════════════════════════════
     METHOD
═══════════════════════════════════════════════ -->
<h2>Method</h2>

<!-- ==============================================
     1. System setup
     ============================================== -->
<h2>1. System setup</h2>

<p>
    This is the earliest stage where we create the topology and the initial coordinate file for the intended system 3GB1. Because MELD begins from an extended chain and lets the restraints and the force field guide the fold, the starting geometry carries no structural information.
</p>

<p>
    For this tutorial we will generate initial peptide chain using the sequence alone. Use following command to download the FASTA entry for 3GB1 from the PDB.
</p>

<pre>curl -o 3gb1.fasta https://www.rcsb.org/fasta/entry/3GB1</pre>

<p>
    <em>Note that we take only the sequence from this entry. The deposited coordinates are never read; they are reserved for the final validation step, where the folded ensemble is compared against the experimental structure.</em>
</p>

<p>
    The chain is then built with <code>tleap</code>. We use the <code>ff19SB</code> force field with <code>mbondi2</code> radii, which are the radii appropriate for the <code>igb=5</code> generalized Born model. Rather than writing the <code>tleap</code> input by hand, the helper script <a href="1_system_setup/makeLeap.x">makeLeap.x</a> translates the one-letter FASTA sequence into the three-letter residue names that <code>tleap</code> expects.
</p>

<p>
    Run <a href="1_system_setup/makeLeap.x">makeLeap.x</a> with the following commands for 3GB1:
</p>

<pre>chmod +x makeLeap.x
./makeLeap.x -s 3gb1.fasta -o 3gb1</pre>

<p>
    This writes <a href="1_system_setup/leap.in">leap.in</a>. Execute it, capturing the output so that nothing is missed:
</p>

<pre>tleap -f leap.in 2> leap.log</pre>

<p>
    Read leap.log before continuing. Warnings about unperturbed charge for the built chain are ignored since implicit solvent simulation. However, any errors about unrecognized residues or missing parameters must be resolved at this stage because they will cause the system to be unrunnable later. A successful run produces <a href="1_system_setup/3gb1.prmtop">3gb1.prmtop</a> and <a href="1_system_setup/3gb1.inpcrd">3gb1.inpcrd</a>.
</p>

<h3>Hydrogen Mass Repartitioning (HMR)</h3>

<p>
    Next we apply Hydrogen Mass Repartitioning (HMR)<sup><a href="https://doi.org/10.1021/ct5010406">3</a></sup>. HMR shifts mass from heavy atoms to the hydrogens bonded to them, slowing the fastest bond vibrations without changing the total mass or thermodynamics of the system. This allows a longer timestep of about 4 fs instead of the usual 2 fs. The repartitioning is handled by <code>ParmED</code>, which ships with AMBER, using the input file <a href="1_system_setup/HMR.in">HMR.in</a>:
</p>

<pre>parm 3gb1.prmtop
hmassrepartition
outparm 3gb1_HMR.prmtop
quit</pre>

<p>To execute simply use command:</p>

<pre>parmed -i HMR.in</pre>

<p>
    This creates <a href="1_system_setup/3gb1_HMR.prmtop">3gb1_HMR.prmtop</a>, which is used for the remainder of the tutorial. Note that the coordinate file has not changed: only the masses have been redistributed.
</p>

<h3>Minimization</h3>

<p>
    Lastly, we carry out a minimization. Since MELD is in charge of guiding the chain to its native fold, we are not attempting to pre-organise the structure in any way; our only requirement is to eliminate the steric clashes and strained geometries that tleap leaves behind when it assembles the chain from scratch, so that the initial steps of the dynamics do not fail. A brief minimisation in implicit solvent is enough.
</p>

<pre>energy minimization
 &cntrl
   imin=1, maxcyc=2000, ncyc=500,
   ntwr = 1000, ntpr = 100,
   cut = 999.0, rgbmax = 999.0,
   ntb = 0, igb = 5, saltcon = 0.0,
  /</pre>

<p>
    The above settings follow from implicit-solvent treatment. The flag <code>ntb = 0</code> removes periodic boundaries while <code>igb = 5</code> selects the generalized Born model, so there is no solvent box and no cutoff is needed (<code>cut = 999.0</code>, <code>rgbmax = 999.0</code>).
</p>

<p>Run the minimization using <a href="1_system_setup/min.in">min.in</a>:</p>

<pre>srun $AMBERHOME/bin/pmemd -O -i min.in -p 3gb1_HMR.prmtop -c 3gb1.inpcrd -o min.out -r min.rst -inf min.info</pre>

<p>
    Check min.out and confirm that the energy has decreased smoothly and converged, with no <code>NaN</code> values. The resulting <a href="1_system_setup/min.rst">min.rst</a> is the coordinate file from which every replica of the MELD simulation will be launched, so it is worth inspecting visually before committing to the far more expensive stages ahead. What you should see is an extended, unfolded chain with clean bond geometry and no overlapping atoms.
</p>

<hr>

<!-- ==============================================
     2. MELD Replica Exchange MD
     ============================================== -->
<h2>2. MELD Replica Exchange MD</h2>

<p>
    With a topology and a relaxed starting structure in hand, we now proceed with constructing the information that will direct the folding. For this purpose two programs are used: <a href="2_meld_remd/gen_restraints.py">gen_restraints.py</a> converts a secondary structure string into the restraint files which tell MELD what it knows about the protein, and <a href="2_meld_remd/gen_md_inputs.py">gen_md_inputs.py</a> takes a single Amber input file and expands it out into the series of replicas that shows how this knowledge is applied throughout the ensemble.
</p>

<h3>2.1. CPI restraints:</h3>

<p>
    The Coarse Physical Insights introduced above are, in practice, primarily 3 collections of distance and torsion restraints. None of them requires knowledge of the native structure; all that is needed is the sequence and a prediction of the secondary structure.
</p>

<p>
    The secondary structure prediction is obtained from the sequence alone, using the <a href="https://bioinf.cs.ucl.ac.uk/psipred/">PSIPRED</a> server at UCL. Provided the sequence, the server returns per-residue assignment of helix, strand or coil (<code>H/E/.</code>) together with a confidence score for each. Reference file <a href="2_meld_remd/ss.dat">ss.dat</a> contains this structural information for 3GB1.
</p>

<p>
    <img src=".assets/psipredChart.jpg" alt="PSIPRED chart Preview" style="max-width:100%; height:auto;">
</p>

<h4>2.1.1. Secondary structure (SS)</h4>

<p>
    Then we use this prediction to generate distance and torsion restraints required to hold the secondary structure of the protein. The prediction is read as a single string of <strong><code>H</code></strong> (helix), <strong><code>E</code></strong> (extended) and <strong><code>&middot;</code></strong> (coil); one character per residue.
</p>

<p>
    When generating secondary structure (<code>SS</code>) restraints, <a href="2_meld_remd/gen_restraints.py">gen_restraints.py</a> scans <a href="2_meld_remd/ss.dat">ss.dat</a> for windows of five consecutive residues where at least four are the same type, and for each such window, it creates a restraint group.
</p>

<p>
    Each group holds nine restraints: the <i>&phi;</i> and <i>&psi;</i> torsion angles for the three inner residues, plus three C&alpha;&ndash;C&alpha; distances that define the window&rsquo;s geometry &mdash; specifically, pairs (i, i+3), (i+1, i+4), and (i, i+4). Helix and extended windows share the same setup; only their reference values differ. Within an active <code>SS</code> group, all nine restraints are retained. However, MELD activates only a selected fraction of the available groups. The default setting activates 85% of <code>SS</code> groups, allowing an incorrect local secondary-structure prediction to be ignored rather than forcing the simulation into an incorrect fold.
</p>

<h4>2.1.2. Strand pairing (SP)</h4>

<p>
    Secondary structure predictions don&rsquo;t tell you which strands pair together or the specific register&mdash;so the script checks every pairing option. For each residue pair from two different extended segments, it forms a group with two backbone hydrogen-bond distances: N(i)&ndash;O(j) and O(i)&ndash;N(j). Only one of the two needs to be satisfied, since a residue pair in a &beta;-sheet donates in one direction or the other depending on whether the pairing is parallel or antiparallel, and the group does not presume to know which.
</p>

<h4>2.1.3. Hydrophobic (HY)</h4>

<p>
    This collection encourages formation of a hydrophobic core. Hydrophobic residues are identified from the amino-acid sequence using the residue types Ala, Val, Leu, Ile, Phe, Trp, Met, and Pro. For every pair of hydrophobic residues sufficiently distant in sequence, the script creates one group. By default, residues must be separated by at least seven positions in the sequence; close sequence neighbors are excluded because they are already likely to be near one another and provide little information about the global fold.
</p>

<p>
    Each group contains all pairwise distances between selected side-chain atoms of the two residues. Only the best-satisfied distance is retained (KEEP 1). Therefore, a group is satisfied when any suitable side-chain atom pair forms a contact. As with strand pairing, only a limited number of candidate hydrophobic contacts are active at one time.
</p>

<h3>2.2. MELD parameter file:</h3>

<p>
    The restraint geometry and MELD replica-exchange protocol are controlled through <a href="2_meld_remd/meld_params.in">meld_params.in</a>. This file is read by <a href="2_meld_remd/gen_restraints.py">gen_restraints.py</a>, which uses it to generate both <code>restraints.disang</code> and <code>restraints.indxf</code>.
</p>

<p>
    The two output files have different roles:
</p>

<ul>
    <li><a href="2_meld_remd/restraints.disang">restraints.disang</a> contains the individual Amber distance and torsion restraint definitions, including atom indices, target ranges, and Amber-format force constants.</li>
    <li><a href="2_meld_remd/restraints.indxf">restraints.indxf</a> contains the MELD hierarchy: which restraints belong to each group and collection, how many groups are active, and how restraint strength and temperature vary across replicas and during the simulation.</li>
</ul>

<div class="note-box">
    <strong>Note:</strong> These files must be generated as a matched pair. <strong>Do not edit <code>DISANG</code> or <code>INDXF</code> files manually</strong> after generation. Instead, edit <a href="2_meld_remd/meld_params.in">meld_params.in</a> and rerun <a href="2_meld_remd/gen_restraints.py">gen_restraints.py</a>.
</div>

<p>
    A fully annotated parameter template can be written using:
</p>

<pre>./gen_restraints.py --write-params meld_params.in</pre>

<p>
    The parameter file has two sections:
</p>
<ol>
    <li>Ordinary <code>key=value</code> settings that define the restraint collections</li>
    <li><code>LADDER</code> block that defines the temperature ladder, restraint scalers, and time ramps.</li>
</ol>

<p>
    The ordinary parameter section uses one assignment per line. Parameter names are <strong>case-insensitive</strong>. Here, blank lines are ignored, and <code>#</code> begins a comment anywhere on a line.
</p>

<h4>ORDINARY settings:</h4>

<h5>SYSTEM:</h5>

<table>
    <thead>
        <tr>
            <th>Key</th>
            <th>Description</th>
            <th>Default</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>first_residue</code></td>
            <td>Residue number in the topology corresponding to position 1 of the secondary-structure string. <code>auto</code> skips recognized leading terminal caps and determines the first protein residue automatically.</td>
            <td><code>auto</code></td>
        </tr>
        <tr>
            <td><code>quadratic_cut</code></td>
            <td>Distances turn linear this far past r3, nm</td>
            <td><code>0.2</code></td>
        </tr>
    </tbody>
</table>

<h5>SS &mdash; Secondary Structure:</h5>

<table>
    <thead>
        <tr>
            <th>Key</th>
            <th>Description</th>
            <th>Default Value</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>ss_active</code></td>
            <td>Fraction of secondary-structure restraint groups retained. <code>85%</code> corresponds to <code>int(len(groups) * 0.85)</code></td>
            <td><code>85%</code></td>
        </tr>
        <tr>
            <td><code>ss_run_length</code></td>
            <td>Number of residues in each secondary-structure window.</td>
            <td><code>5</code></td>
        </tr>
        <tr>
            <td><code>ss_min_match</code></td>
            <td>Minimum number of residues within the <code>ss_run_length</code> window that must carry the specified secondary-structure type (<code>H</code> or <code>E</code>).</td>
            <td><code>4</code></td>
        </tr>
        <tr>
            <td><code>ss_k_distance</code></td>
            <td>C&alpha;&ndash;C&alpha; distance force constant in kJ mol<sup>-1</sup> nm<sup>-2</sup>.</td>
            <td><code>2500</code></td>
        </tr>
        <tr>
            <td><code>ss_k_torsion</code></td>
            <td>Backbone <i>&phi;</i>/<i>&psi;</i> torsional force constant in kJ mol<sup>-1</sup> deg<sup>-2</sup>.</td>
            <td><code>0.025</code></td>
        </tr>
        <tr>
            <td><code>ss_scaler</code></td>
            <td>Name of the <code>SCALER</code> used to scale secondary-structure restraints in the <code>LADDER</code> section.</td>
            <td><code>none</code></td>
        </tr>
        <tr>
            <td><code>ss_ramp</code></td>
            <td>Name of the <code>RAMP</code> used to control the strength of secondary-structure restraints in the <code>LADDER</code> section.</td>
            <td><code>none</code></td>
        </tr>
    </tbody>
</table>

<h5>SP &mdash; Strand Pairing:</h5>

<table>
    <thead>
        <tr>
            <th>Key</th>
            <th>Description</th>
            <th>Default Value</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>sp_active</code></td>
            <td>Number of strand-pairing restraint groups retained. With <code>auto</code>, this is determined from <code>sp_fraction</code> and the number of residues marked as extended (<code>E</code>).</td>
            <td><code>auto</code></td>
        </tr>
        <tr>
            <td><code>sp_fraction</code></td>
            <td>Fraction of eligible strand-pairing restraints selected when <code>sp_active = auto</code>.</td>
            <td><code>0.45</code></td>
        </tr>
        <tr>
            <td><code>sp_min_strand_length</code></td>
            <td>Minimum length of an <code>E</code> residue run required to be considered a &beta;-strand for pairing.</td>
            <td><code>1</code></td>
        </tr>
        <tr>
            <td><code>sp_r2</code></td>
            <td>Lower bound of the N&ndash;O distance flat-bottom region, in nm.</td>
            <td><code>0</code></td>
        </tr>
        <tr>
            <td><code>sp_r3</code></td>
            <td>Upper bound of the N&ndash;O distance flat-bottom region, in nm.</td>
            <td><code>0.35</code></td>
        </tr>
        <tr>
            <td><code>sp_k</code></td>
            <td>Strand-pairing distance force constant in kJ mol<sup>-1</sup> nm<sup>-2</sup>.</td>
            <td><code>250</code></td>
        </tr>
        <tr>
            <td><code>sp_scaler</code></td>
            <td>Name of the <code>SCALER</code> used to scale strand-pairing restraints in the <code>LADDER</code> section.</td>
            <td><code>none</code></td>
        </tr>
        <tr>
            <td><code>sp_ramp</code></td>
            <td>Name of the <code>RAMP</code> used to control the strength of strand-pairing restraints in the <code>LADDER</code> section.</td>
            <td><code>none</code></td>
        </tr>
    </tbody>
</table>

<h5>HY &mdash; Hydrophobic Contacts:</h5>

<table>
    <thead>
        <tr>
            <th>Key</th>
            <th>Description</th>
            <th>Default Value</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>hy_active</code></td>
            <td>Number of hydrophobic-contact restraint groups retained. With <code>auto</code>, this is determined from <code>hy_contacts_per_residue</code> and the number of hydrophobic residues.</td>
            <td><code>auto</code></td>
        </tr>
        <tr>
            <td><code>hy_contacts_per_residue</code></td>
            <td>Number of hydrophobic-contact restraints selected per hydrophobic residue when <code>hy_active = auto</code>.</td>
            <td><code>1.2</code></td>
        </tr>
        <tr>
            <td><code>hy_min_sep</code></td>
            <td>Minimum sequence separation between residues forming a hydrophobic contact.</td>
            <td><code>7</code></td>
        </tr>
        <tr>
            <td><code>hy_r2</code></td>
            <td>Lower bound of the hydrophobic-contact distance flat-bottom region, in nm.</td>
            <td><code>0</code></td>
        </tr>
        <tr>
            <td><code>hy_r3</code></td>
            <td>Upper bound of the hydrophobic-contact distance flat-bottom region, in nm.</td>
            <td><code>0.5</code></td>
        </tr>
        <tr>
            <td><code>hy_k</code></td>
            <td>Hydrophobic-contact distance force constant in kJ mol<sup>-1</sup> nm<sup>-2</sup>.</td>
            <td><code>250</code></td>
        </tr>
        <tr>
            <td><code>hy_scaler</code></td>
            <td>Name of the <code>SCALER</code> used to scale hydrophobic-contact restraints in the <code>LADDER</code> section.</td>
            <td><code>none</code></td>
        </tr>
        <tr>
            <td><code>hy_ramp</code></td>
            <td>Name of the <code>RAMP</code> used to control the strength of hydrophobic-contact restraints in the <code>LADDER</code> section.</td>
            <td><code>none</code></td>
        </tr>
    </tbody>
</table>

<h4>LADDER settings:</h4>

<p>
    This section starts with syntax <code>LADDER</code> and ends with <code>END_LADDER</code>. Currently this section has few functions as explained below.
</p>

<table>
    <thead>
        <tr>
            <th>Key</th>
            <th>Usage</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>INFO</td>
            <td><code>INFO [off | on]</code></td>
            <td>each replica writes meld.info</td>
        </tr>
        <tr>
            <td>TSCALE</td>
            <td><code>TSCALE [a_min] [a_max] [tempi] [temp0] ["constant | linear | geometric"]</code></td>
            <td>temperature scaler</td>
        </tr>
        <tr>
            <td>SCALER</td>
            <td><code>SCALER <name=ss_scaler|sp_scaler|hy_scaler> [parms..] "<type>"</code></td>
            <td>restraint scaler</td>
        </tr>
        <tr>
            <td>RAMP</td>
            <td><code>RAMP <name=ss_ramp|sp_ramp|hy_ramp> [params..] "<type>"</code></td>
            <td>restraint ramp</td>
        </tr>
    </tbody>
</table>

<p>
    The following table shows the required parameters for different types of <code>SCALER</code>s available:
</p>

<table>
    <thead>
        <tr>
            <th>Type</th>
            <th>Parameters</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>"constant"</code></td>
            <td><em>(none)</em></td>
        </tr>
        <tr>
            <td><code>"linear"</code></td>
            <td><code>a_min a_max</code></td>
        </tr>
        <tr>
            <td><code>"geometric"</code></td>
            <td><code>a_min a_max</code></td>
        </tr>
        <tr>
            <td><code>"nonlinear"</code></td>
            <td><code>a_min a_max factor</code></td>
        </tr>
        <tr>
            <td><code>"plateau"</code></td>
            <td><code>a_min a_one a_two a_max</code></td>
        </tr>
        <tr>
            <td><code>"plateausmooth"</code></td>
            <td><code>a_min a_one a_two a_max</code></td>
        </tr>
        <tr>
            <td><code>"plateaunonlinear"</code></td>
            <td><code>a_min a_one a_two a_max factor</code></td>
        </tr>
    </tbody>
</table>

<p>
    Every type except <code>constant</code> takes an <strong>optional trailing <code>s_min s_max</code></strong>, defaulting to <code>1.0</code> and <code>1e-3</code>.
</p>

<p>
    <code>s_min</code> is the strength at <code>a_min</code> (full), <code>s_max</code> the strength at <code>a_max</code> (essentially off). So a <code>linear</code> or <code>nonlinear</code> scaler holds restraints at full strength up to <code>a_min</code>, decays to <code>1e-3</code> by <code>a_max</code>, and stays there. <code>factor</code> sets how sharply the decay bends.
</p>

<p>
    The <code>plateau*</code> shapes use all four alphas: full strength below <code>a_min</code>, down to <code>s_min</code> across <code>a_min &rarr; a_one</code>, flat through <code>a_one &rarr; a_two</code>, back up across <code>a_two &rarr; a_max</code>. They are for restraints that should be weak in the <em>middle</em> of the ladder.
</p>

<p>
    <code>RAMP</code> parameters also change with the type of RAMP chosen:
</p>

<table>
    <thead>
        <tr>
            <th>Type</th>
            <th>Parameters</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>"constant_ramp"</code></td>
            <td><em>(none)</em></td>
        </tr>
        <tr>
            <td><code>"linear_ramp"</code></td>
            <td><code>t_start t_end w_start w_end</code></td>
        </tr>
        <tr>
            <td><code>"nonlinear_ramp"</code></td>
            <td><code>t_start t_end w_start w_end factor</code></td>
        </tr>
    </tbody>
</table>

<p>
    Weight is <code>w_start</code> before <code>t_start</code>, <code>w_end</code> after <code>t_end</code>, interpolated between.
</p>

<p>
    For this tutorial, we use the usual MELD parameters used to fold proteins. <a href="2_meld_remd/meld_params.in">Check meld_params.in</a>
</p>

<pre>LADDER
  INFO    off

  TSCALE  0.0 0.3  300.0 550.0  "geometric"

  SCALER  ss                       "constant"
  SCALER  prot  0.4 1.0 4.0        "nonlinear"
  RAMP    warmup 1 200 1e-3 1 4.0  "nonlinear_ramp"
END_LADDER</pre>

<ul>
    <li><code>INFO off</code> &mdash; no per-replica info files.</li>
    <li><code>TSCALE 0.0 0.3 300 550 "geometric"</code> &mdash; temperature rises geometrically from 300 K at <code>alpha = 0</code> to 550 K at <code>alpha = 0.3</code>, and <strong>clamps at 550 K above that</strong>.</li>
    <li><code>SCALER ss "constant"</code> &mdash; secondary structure restraints are at full strength on every replica, hot ones included. The CPI claim "this protein has secondary structure" is not something the hot replicas should be allowed to forget.</li>
    <li><code>SCALER prot 0.4 1.0 4.0 "nonlinear"</code> &mdash; strand pairing and hydrophobic restraints are at full strength up to <code>alpha = 0.4</code>, then decay nonlinearly to <code>1e-3</code> at <code>alpha = 1.0</code>. The hottest replicas are effectively unrestrained in their tertiary contacts and free to explore.</li>
    <li><code>RAMP warmup 1 200 1e-3 1 4.0 "nonlinear_ramp"</code> &mdash; every restraint starts at <code>1e-3</code> of its force constant and reaches full strength by exchange step 200, so a clashing extended chain is not yanked apart on step 1.</li>
</ul>

<p>
    Once parameters for the simulation is edited on template <a href="2_meld_remd/meld_params.in">meld_params.in</a>, run <code>gen_restraints.py</code>:
</p>

<pre>./gen_restraints.py ss.dat -i meld_params.in -p ../1_system_setup/3gb1.prmtop</pre>

<p>
    This will generate verbose as follows:
</p>

<pre>topology            : ../1_system_setup/3gb1_HMR.prmtop
parameters          : meld_params.in
residue mapping     : ss 1-56  ->  prmtop 2-57   (ACE at 1, NHE at 58)
secondary structure : 56 residues, 14 H, 24 E, 18 coil
strand segments     : 4 (2-7, 13-20, 42-46, 51-55)
hydrophobic residues: 18

collection    groups  restraints  keep/group    active  force constants
SS                28         252         all  85% (24)  distance 2500 -> 2.98757, torsion 0.025 -> 9.80762
SP               213         426           1   10 (10)  distance 250 -> 0.29876
HY               121        2471           1   21 (21)  distance 250 -> 0.29876
total            362        3149

wrote restraints.disang and restraints.indxf

in mdin:
   nmropt=1, meld=1, indxf='restraints.indxf',
  /
 DISANG=restraints.disang

The RAMP record in the index file is what ramps the restraints in. Drop any
&wt type='REST' block from mdin: MELD rewrites rk2/rk3 from the ladder every
exchange and modwt would then scale them a second time on top of it.</pre>

<h3>2.3. MELD INDXF file</h3>

<p>
    Above section generates two files; <a href="2_meld_remd/restraints.indxf">restraints.indxf</a> and <a href="2_meld_remd/restraints.disang">restraints.disang</a>. In this tutorial we will not be looking at AMBER style <code>DISANG</code> file format. However, <code>INDXF</code> file is the main MELD input file for the simulation that carries all the information about collections, groups, scalers and ramps etc.
</p>

<pre>MELD 3

UNITS  meld

INFO    off
TSCALE  0.0 0.3  300.0 550.0  "geometric"
SCALER  ss                       "constant"
SCALER  prot  0.4 1.0 4.0        "nonlinear"
RAMP    warmup 1 200 1e-3 1 4.0  "nonlinear_ramp"

# 28 groups, 252 restraints
COLL SS  ACTIVE 85%  KEEP all  SCALER ss  RAMP warmup  K distance 2500 torsion 0.025  GROUPS 9 x 28

# 213 groups, 426 restraints
COLL SP  ACTIVE 10  KEEP 1  SCALER prot  RAMP warmup  K distance 250  GROUPS 2 x 213

# 121 groups, 2471 restraints
COLLECTION HY  ACTIVE 21  KEEP 1  SCALER prot  RAMP warmup  K distance 250
  GROUPS 25 10 20 10 x 3 20 40 10 20 55 10 40 20 25 10 20 10 x 3 20 40 10 20
  GROUPS 55 10 40 20 10 20 10 x 3 20 40 10 20 55 10 40 20 10 20 10 x 3 20 40
  GROUPS 10 20 55 10 40 20 10 20 10 x 3 20 40 10 20 55 10 40 20 8 16 4
  GROUPS 8 22 4 16 8 16 32 8 16 44 8 32 16 x 2 4 8 22 4 16 8 4
  GROUPS 8 22 4 16 8 4 8 22 4 16 8 16 44 8 32 16 32 88 16 64
  GROUPS 32 22 4 16 8 x 2 32 16 88 44
END</pre>

<p>
    A <code>GROUPS</code> spec is a restraint count, and <code>x <n></code> repeats the preceding spec: <code>9 x 28</code> means 28 consecutive groups of 9 restraints each, claimed in DISANG order. <code>SS</code> and <code>SP</code> fit on one <code>COLL</code> line; <code>HY</code>'s group sizes vary, and a record is capped at 512 fields, so it falls back to the <code>COLLECTION &hellip; END</code> block form. The two forms behave identically &mdash; settings are read before the groups they apply to.
</p>

<h3>2.4. Generate AMBER input files</h3>

<p>
    In this section we create <code>30</code> input files to run MELD replica exchange MD in AMBER. To make things easy we first create a template input file <a href="2_meld_remd/meld.mdin">meld.mdin</a>.
</p>

<pre>Replica Exchange MELD (200 ns)
 &cntrl
   irest=0, ntx=1,
   nstlim=12500, dt=0.004, numexchg=4000,   ! 50ps blocks * 4000 = 200ns
   ntt=3, gamma_ln=2.0,                     ! Langevin, 2.0/ps (implicit solvent)
   temp0=XXXXX, ig=RANDOM_NUMBER,
   ntc=2, ntf=2,                            ! SHAKE on H
   ntb=0, igb=5,                            ! implicit solvent
   cut=999.0, rgbmax=999.0,
   ntpr=500, ntwx=500, ntwr=12500,
   nmropt=1, meld=1, indxf='restraints.indxf',   ! MELD flags
  /
  &wt TYPE='END'
  /
DISANG=restraints.disang</pre>

<p>
    As shown we simulate each replica for <code>200 ns</code> with exchanges happening every <code>50 ps</code>. Hydrogen Mass Repartitioning (HMR) allows a larger time step <code>4 fs</code>.
</p>

<p>
    For a <code>MELD</code> simulation, there are few essential input flags:
</p>

<table>
    <thead>
        <tr>
            <th>Requirement</th>
            <th>Reason</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>nmropt=1</code></td>
            <td>MELD rides on Amber's NMR restraint machinery</td>
        </tr>
        <tr>
            <td><code>meld=1</code></td>
            <td>Set MELD <code>on</code>  -- default <code>off</code></td>
        </tr>
        <tr>
            <td><code>indxf='&hellip;'</code></td>
            <td><code>INDXF</code> file; required whenever <code>meld=1</code></td>
        </tr>
        <tr>
            <td><code>numexchg > 0</code></td>
            <td><code>pmemd</code> rejects a REMD run with no exchanges</td>
        </tr>
        <tr>
            <td><code>DISANG='&hellip;'</code></td>
            <td>Amber readable restraint geometry file</td>
        </tr>
    </tbody>
</table>

<p>
    Note that the template input file has some placeholders for <code>temp0</code> and <code>ig</code>. These are to be edited via <a href="2_meld_remd/gen_md_inputs.py">gen_md_inputs.py</a> to match previously specified temperature scale and different random seeds, respectively.
</p>

<div class="note-box">
    <strong>Note:</strong> There is <strong>No <code>&wt type='REST'</code> block</strong>: As <code>INDXF</code> file directly pass <code>RAMP</code> information to AMBER's restraint calculations.
</div>

<p>
    For this tutorial we use the following command to create 30 replica <code>MDIN</code> files along with concatenated <code>GROUPFILE</code> <a href="2_meld_remd/meld.groupfile">meld.groupfile</a>.
</p>

<pre>./gen_md_inputs.py -i meld.mdin --indxf restraints.indxf -n 30 \
                   -c min.rst -p 3gb1_HMR.prmtop</pre>

<h3>2.5. Launch MELD REMD</h3>

<p>
    MELD groupfiles have a similar format to other AMBER REMD groupfiles. However, in MELD the replica ladder is based on both temperature (T) and Hamiltonian (H). Alpha scales T and H along the ladder. So to cope with this alpha based ladder, we use <code>-rem 6</code>; a new exchange protocol just for MELD.
</p>

<pre>-O -rem 6 -remlog rem.log -i meld.mdin.001 -o meld.mdout.001 -c ../1_system_setup/min.rst -r meld.rst.001 -x meld.nc.001 -inf meld.mdinfo.001 -p ../1_system_setup/3gb1_HMR.prmtop
-O -rem 6 -remlog rem.log -i meld.mdin.002 -o meld.mdout.002 -c ../1_system_setup/min.rst -r meld.rst.002 -x meld.nc.002 -inf meld.mdinfo.002 -p ../1_system_setup/3gb1_HMR.prmtop
.....
-O -rem 6 -remlog rem.log -i meld.mdin.030 -o meld.mdout.030 -c ../1_system_setup/min.rst -r meld.rst.030 -x meld.nc.030 -inf meld.mdinfo.030 -p ../1_system_setup/3gb1_HMR.prmtop</pre>

<p>
    Now we are all set to run this simulation with <code>pmemd.cuda.MPI</code>:
</p>

<pre>srun --mpi=pmix_v5 $AMBERHOME/bin/pmemd.cuda.MPI -ng 30 -groupfile meld.groupfile</pre>

<hr>

<!-- ==============================================
     References
     ============================================== -->
<h2>References</h2>

<ol>
    <li>MacCallum, J. L.; Perez, A.; Dill, K. A. Determining Protein Structures by Combining Semireliable Data with Atomistic Physical Models by Bayesian Inference. <em>Proc. Natl. Acad. Sci.</em> <strong>2015</strong>, 112 (22), 6985&ndash;6990. <a href="https://doi.org/10.1073/pnas.1506788112">https://doi.org/10.1073/pnas.1506788112</a>.</li>
    <li>Perez, A.; MacCallum, J. L.; Dill, K. A. Accelerating Molecular Simulations of Proteins Using Bayesian Inference on Weak Information. <em>Proc. Natl. Acad. Sci.</em> <strong>2015</strong>, 112 (38), 11846&ndash;11851. <a href="https://doi.org/10.1073/pnas.1515561112">https://doi.org/10.1073/pnas.1515561112</a>.</li>
    <li>Hopkins, C. W.; Le Grand, S.; Walker, R. C.; Roitberg, A. E. Long-Time-Step Molecular Dynamics through Hydrogen Mass Repartitioning. <em>J. Chem. Theory Comput.</em> <strong>2015</strong>, 11 (4), 1864&ndash;1874. <a href="https://doi.org/10.1021/ct5010406">https://doi.org/10.1021/ct5010406</a>.</li>
</ol>

<footer>
    by Namindu Rangana, Imesh Ranaweera, and Binod Perera<br>
    All materials copyrighted by authors. &copy; <?php echo date("Y"); ?>
</footer>

</body>
</html>