#!/bin/bash

# For running and analyzing the short simulation
# needed for getting early ISF decay

# ===== Config Section: Edit These As Needed =====

# Add more work directories to the list as needed by inserting a new line between the parentheses
# Only include the parent directory for the new run. For example: ../OCT/328K <- Do NOT include, for example, /5_nvt_prod_system
workdirs=(
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.0_V0.0_no_reservoir_N1/OCT/328K
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.0_V0.0_no_reservoir_N1/OCT/358K
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.2_V0.2_no_reservoir_N1/OCT/298K
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.2_V0.2_no_reservoir_N1/OCT/328K
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.2_V0.2_no_reservoir_N1/OCT/358K
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.4_V0.0_no_reservoir_N1/OCT/328K
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.4_V0.0_no_reservoir_N1/OCT/358K
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.2_V0.2_no_reservoir_N1/OCT/298K
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.2_V0.2_no_reservoir_N1/OCT/328K
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.2_V0.2_no_reservoir_N1/OCT/358K
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.2_V0.2_no_reservoir_N1/OCT/358K
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.2_V0.2_no_reservoir_N1/OCT/358R
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.4_V0.0_no_reservoir_N1/OCT/298K
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.4_V0.0_no_reservoir_N1/OCT/358K
)

# A list of q values to use with the corresponding directory in workdirs. The two lists MUST be the same length.
# You may get the q values from summary.txt for the desired simulation.
# Note 1: 1.0 is a dummy q value for testing purposes--if you see it, keep looking higher in the file until you find a valid number.
# Note 2: Leave this list empty if you do not wish to use a pre-determined q value.
q_values=(
    3.10850211356711
    4.694306182092584
    2.0640041580756376
    3.049674105241572
    4.970701360515549
    3.730476513978905
    7.262793407559881
    1.5148879819240033
    1.7513057876555778
    2.291061588058136
    2.729047474125677
    2.2197082967193604
    1.475873565144969
    9.229815700703762
)
SYS="Pore"
SEGMENTS=1000

# Define residues and their atom pairs: format is "RESIDUE ATOM1 ATOM2"
residue_atom_list=(
  "OCT O01 C0O"
)

# List for end-to-end of all PEG monomers (copy the lines with quotes only)
:<<PEG
  "DEG OAD1 OAD2"
  "REG OAR1 OAR2"
  "TEG OAT1 OAT2"
  "PEG OAP1 OAP2"
  "XEG OAX1 OAX2"
  "HEG OAH1 OAH2"
PEG
# List of possible Octanol (OPLS parameterizations) choices (do not copy the comments)
:<<OCT
  "OCT O01 H00"     # Hydroxy
  "OCT O01 C0O"     # End-to-end
OCT
# =====================================================

# ===== Universal Functions =====
write_mdp() {
    cat << EOF > $EXTDIR/mdp_nvt_prod_system_short_run.mdp
; very basics of the simulation 
integrator               = md              ; solve newtown's equation of motion 
dt                       = 0.002           ; integration time step / ps 
nsteps                   = 200000          ; number of steps (400 ps)

; remove drifts of the center of mass 
comm-mode                = None            ; do not remove COM translation due to pore absolute position restraint 

; control frequency of output 
nstvout                  = 0               ; write velocities to trajectory file every number of steps 
nstfout                  = 0               ; write forces to trajectory file every number of steps 
nstlog                   = 2000            ; update log file every number of steps 
nstcalcenergy            = -1              ; calculate energies/pressures every nstenergy steps 
nstenergy                = 0               ; write energies to energy file every number of steps 
nstxout-compressed       = 10              ; write positions using compression (saves memory, worse quality) 
compressed-x-precision   = 1000            ; precision to write compressed trajectory 

; next neighbor search and periodic boundary conditions 
nstlist                  = 20              ; freq. to update neighbor list & long range forces (>= 20 w/ GPUs) 
rlist                    = 1.0             ; short-range neighbor list cutoff (nm) 
cutoff-scheme            = Verlet          ; atom-based neighbor search with an implicit buffer region 
pbc                      = xyz             ; periodicity in x, y, and z 
periodic-molecules       = yes             ; silica pore is infinitely bonded to itself through pbc 

; coulomb interaction 
coulombtype              = PME             ; particle-mesh ewald summation for long range (>rcoulomb) 
rcoulomb                 = 1.0             ; short-range electrostatic cutoff (in nm) (with PME,rcoulomb >= rvdw) 
coulomb-modifier         = Potential-shift ; shifts potential by constant so potential is 0 at cut-off 
fourierspacing           = 0.12            ; spacing of FFT in reciprocal space in PME long range treatment (in nm) 
pme-order                = 4               ; cubic PME interpolation order 

; lennard-jones potential handling 
vdwtype                  = PME             ; particle-mesh ewald summation for long range (>rvdw) 
rvdw                     = 1.0             ; short-range vdw cutoff (in nm) 
vdw-modifier             = Potential-shift ; shifts potential by constant so potential is 0 at cut-off 

; temperature coupling 
tcoupl                   = v-rescale       ; the algorithm to use, v-rescale generates correct canonical ensemble 
tc-grps                  = pore_graft solvent ; groups to couple to temperature bath 
tau-t                    = 1.0 1.0         ; time constant (in ps), meaning varies by algorithm 
ref-t                    = $TEMP $TEMP     ; temperature for coupling (K) 
nsttcouple               = 1               ; frequency to couple temperature 

; velocity generation 
gen-vel                  = no              ; generate velocities according to Maxwell distr. (no for a continuation run) 

; pressure coupling 
pcoupl                   = no              ; the algorithm to use (no for NVT) 

; constraints 
constraints              = h-bonds         ; constrains bonds only involving hydrogen 
constraint_algorithm     = lincs           ; algorithm to use, lincs should NOT be used for angle constraining 
lincs-order              = 4               ; highest order in constraint coupling matrix expansion 
lincs-iter               = 1               ; accuracy of lincs algorithm 
continuation             = yes             ; no for applying constraints at start of run 
EOF
}

run_python_script() {
    Q="$1"  # First argument is the optional q-value

    shift 1 # Shift the remaining arguments to $1, $2, $3...

    if [ -n "$Q" ]; then
        python main.py -w "$EXTDIR" -sys "$SYS" -s "$SEGMENTS" -a "$2" "$3" -r "$1" -q "$Q" -m
    else
        python main.py -w "$EXTDIR" -sys "$SYS" -s "$SEGMENTS" -a "$2" "$3" -r "$1" -m
    fi

    status=$?
    if [ $status -ne 0 ]; then
        echo "Uh oh! It seems there was an issue! Python exited with status code $status."
        exit $status
    fi
}

# =====================================================

echo "Running full loop over workdirs and residue-atom sets..."

use_q_values=false

if [ ${#q_values[@]} -gt 0 ]; then
    use_q_values=true
fi

for i in "${!workdirs[@]}"; do
    WORKDIR="${workdirs[$i]}"
    echo "In workdir: $WORKDIR"

    if $use_q_values; then
        Q="${q_values[$i]}"
        echo "  Using q-value: $Q"
    fi

    IFS='/' read -ra array <<< "$WORKDIR"

    TEMP="${array[-1]}"
    TEMP=${TEMP::-1}

    EXTDIR="$WORKDIR"/5_nvt_prod_system_short_run
    mkdir -p $EXTDIR/out
    cp $WORKDIR/5_nvt_prod_system/{gro_nvt_prod_system.gro,prev_sim.cpt,tc_grps.ndx,top_nvt_prod_system.top} $EXTDIR
    write_mdp

    if [ ! -e "$EXTDIR/out/out.gro" ]; then
        gmx grompp \
            -f $EXTDIR/mdp_nvt_prod_system_short_run.mdp \
            -c $EXTDIR/gro_nvt_prod_system.gro \
            -r $EXTDIR/gro_nvt_prod_system.gro \
            -t $EXTDIR/prev_sim.cpt \
            -n $EXTDIR/tc_grps.ndx \
            -p $EXTDIR/top_nvt_prod_system.top \
            -o $EXTDIR/run.tpr \
            -po $EXTDIR/out/mdout.mdp \
            -quiet no 

        gmx mdrun \
            -s $EXTDIR/run.tpr \
            -o $EXTDIR/out/traj.trr \
            -x $EXTDIR/out/traj.xtc \
            -c $EXTDIR/out/out.gro \
            -e $EXTDIR/out/energy.edr \
            -g $EXTDIR/out/log.log \
            -cpo $EXTDIR/out/state.cpt \
            -cpi $EXTDIR/out/state.cpt \
            -cpt 5 \
            -quiet no -v 

        mkdir -p "$EXTDIR"/analysis/graphs/{MSD,ISF,RDF}
        mkdir -p "$EXTDIR"/analysis/data_files/{MSD,ISF,RDF}
    fi

    for entry in "${residue_atom_list[@]}"; do
        read -r RESIDUE ATOM1 ATOM2 <<< "$entry"
        echo "  → Running for residue: $RESIDUE with atoms $ATOM1, $ATOM2"

        if $use_q_values; then
            run_python_script "$Q" "$RESIDUE" "$ATOM1" "$ATOM2"
        else
            run_python_script "" "$RESIDUE" "$ATOM1" "$ATOM2"       # Needs the empty argument if Q is not available!
        fi
    done

done