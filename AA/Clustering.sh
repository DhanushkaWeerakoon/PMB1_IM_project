#!/bin/bash

source /usr/local/gromacs/bin/GMXRC

#!/bin/bash

tpr_list=(1457_1644 1645_1832 2021_2208 2397_2584 2585_2772 2773_2960 3149_3336 3337_3524 4089_4276 4277_4464 4653_4840 4841_5028 5029_5216)

for i in "${tpr_list[@]}"
do
    gmx convert-tpr -s ../Trajectories_and_tprs/PMB1s.tpr -o ../Trajectories_and_tprs/PMB1_"$i".tpr -n ../Trajectories_and_tprs/PMB1s.ndx

    gmx cluster -f ../Trajectories_and_tprs/PMB1_interfacial_"$i"_aligned.xtc -s ../Trajectories_and_tprs/PMB1_"$i".tpr -method gromos -cl ../Clustering/cluster_PMB1_"$i"_full.pdb -g ../Clustering/cluster_PMB1_"$i"_full.log -cutoff 0.2 -n ../Trajectories_and_tprs/PMB1_"$i".ndx -skip 40 -b 100000 -dist ../Clustering/rmsd_PMB1_"$i"_full.xvg

    gmx cluster -f ../Trajectories_and_tprs/PMB1_interfacial_"$i"_aligned.xtc -s ../Trajectories_and_tprs/PMB1_"$i".tpr -method gromos -cl ../Clustering/cluster_PMB1_"$i"_end.pdb -g ../Clustering/cluster_PMB1_"$i"_end.log -cutoff 0.2 -n ../Trajectories_and_tprs/PMB1_"$i".ndx -skip 10 -b 400000 -dist ../Clustering/rmsd_PMB1_"$i"_end.xvg

done

