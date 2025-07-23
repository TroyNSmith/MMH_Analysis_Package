#!/bin/bash

# ===== Config Section: Edit These As Needed =====

# Add more work directories to the list as needed by inserting a new line between the parentheses
# Make sure the path contains the directory being analyzed. For example: ../OCT/328K/5_nvt_prod_system <- The 5_nvt_prod_system must be here
# Specify a q value for each simulation. If q_values is empty, they will be independently generated instead.

workdirs=(
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.0_V0.0_no_reservoir_N1/OCT/328K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.0_V0.0_no_reservoir_N1/OCT/358K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.2_V0.2_no_reservoir_N1/OCT/298K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.2_V0.2_no_reservoir_N1/OCT/328K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.2_V0.2_no_reservoir_N1/OCT/358K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.4_V0.0_no_reservoir_N1/OCT/328K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.4_V0.0_no_reservoir_N1/OCT/358K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.2_V0.2_no_reservoir_N1/OCT/298K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.2_V0.2_no_reservoir_N1/OCT/328K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.2_V0.2_no_reservoir_N1/OCT/358K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.2_V0.2_no_reservoir_N1/OCT/358K/7_nvt_prod2_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.2_V0.2_no_reservoir_N1/OCT/358R/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.4_V0.0_no_reservoir_N1/OCT/298K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.4_V0.0_no_reservoir_N1/OCT/358K/5_nvt_prod_system
)

SYS="Pore"
SEGMENTS=1000

q_values=(
    12.57
)

# Define residues and their atom pairs: format is "RESIDUE ATOM1 ATOM2"
residue_atom_list=(
    "OCT O01 H00"
)

# ===== References: Copy and Paste As Needed =====

# Pore Work Directories on Desktop 2
:<<EOC
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.0_V0.0_no_reservoir_N1/OCT/328K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.0_V0.0_no_reservoir_N1/OCT/358K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.2_V0.2_no_reservoir_N1/OCT/298K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.2_V0.2_no_reservoir_N1/OCT/328K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.2_V0.2_no_reservoir_N1/OCT/358K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.4_V0.0_no_reservoir_N1/OCT/328K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.0_A0.4_V0.0_no_reservoir_N1/OCT/358K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.2_V0.2_no_reservoir_N1/OCT/298K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.2_V0.2_no_reservoir_N1/OCT/328K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.2_V0.2_no_reservoir_N1/OCT/358K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.2_V0.2_no_reservoir_N1/OCT/358K/7_nvt_prod2_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.2_V0.2_no_reservoir_N1/OCT/358R/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.4_V0.0_no_reservoir_N1/OCT/298K/5_nvt_prod_system
    /media/mmh/ExtraSpace/Final_Pore_Analysis_Organized_TNS/pore_D3_L6_W2_S5.0_E0.2_A0.4_V0.0_no_reservoir_N1/OCT/358K/5_nvt_prod_system
EOC

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

# Function: Show help
print_help() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options (user input mode):"
  echo "  -w, --workdir    Working directory of simulation"
  echo "  -sys, --system   Type of system being analyzed (pore, bulk, ...)"
  echo "  -s, --segments   Number of segments"
  echo "  -a, --atoms      Atom1 and Atom2 (space separated)"
  echo "  -r, --residue    Residue name"
  echo "  -h, --help       Show help"
  echo ""
  echo "If no arguments are provided, the script runs the manual block below."
}

# Parse command-line arguments if provided
use_manual=true
if [ "$#" -gt 0 ]; then
    use_manual=false
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            -w|--workdir) WORKDIR="$2"; shift 2 ;;
            -sys|--system) SYS="$2"; shift 2 ;;
            -s|--segments) SEGMENTS="$2"; shift 2 ;;
            -a|--atoms) ATOM1="$2"; ATOM2="$3"; shift 3 ;;
            -r|--residue) RESIDUE="$2"; shift 2 ;;
            -h|--help) print_help; exit 0 ;;
            *) echo "Unknown option: $1"; print_help; exit 1 ;;
        esac
    done
fi

# ========== Main Logic ==========

run_python_script() {
    LOGTMP="$(mktemp)"
    if [ ${#q_values[@]} -gt 0 ]; then
        for q in "${q_values[@]}"; do
            python main.py -w "$WORKDIR" -sys "$SYS" -s "$SEGMENTS" -a "$ATOM1" "$ATOM2" -r "$RESIDUE" -q "$q" \
                > >(tee "$LOGTMP") \
                2> >(tee -a "$LOGTMP" >&2)
        done
    else
        python main.py -w "$WORKDIR" -sys "$SYS" -s "$SEGMENTS" -a "$ATOM1" "$ATOM2" -r "$RESIDUE" \
            > >(tee "$LOGTMP") \
            2> >(tee -a "$LOGTMP" >&2)
    fi

    status=$?
    if [ $status -ne 0 ]; then
        cp "$LOGTMP" "$WORKDIR/error.log"
        echo "Uh oh! It seems there was an issue when processing $WORKDIR! Python exited with status code $status."
        echo "Full terminal output written to $WORKDIR/error.log =)"
        exit $status
    fi
}

if [ "$use_manual" = false ]; then
    echo "Running user input procedure..."
    run_python_script
else
    echo "Running full loop over workdirs and residue-atom sets..."
    for WORKDIR in "${workdirs[@]}"; do
        echo "In workdir: $WORKDIR"
        mkdir -p "$WORKDIR"/analysis/graphs/{MSD,ISF,RDF,Rotation,nonGauss,vanHove,Susceptibility,HBonds,End_to_End,Z_Axis,Spatial_Density,Dihedrals,Gyration}
        mkdir -p "$WORKDIR"/analysis/data_files/{MSD,ISF,RDF,Rotation,nonGauss,vanHove,Susceptibility,HBonds,End_to_End,Z_Axis,Spatial_Density,Dihedrals,Gyration}
        for entry in "${residue_atom_list[@]}"; do
            read -r RESIDUE ATOM1 ATOM2 <<< "$entry"
            echo "  → Running for residue: $RESIDUE with atoms $ATOM1, $ATOM2"
            run_python_script
        done
        echo ""
    done
fi