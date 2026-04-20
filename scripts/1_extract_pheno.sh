#!/bin/bash

InDir="/proj/yunligrp/ukbiobank/clinical"  # ukb data
OutDir="/proj/yunligrp/users/qinwen/R_works/LLM_embedding/data/ukb_phenotype"  # you should update this to your directory

##colum, fid, description
#426, 1239, Current tobacco smoking
#429, 1249, past tobacco smoking
#432, 1259, Smoking/smokers in household
#435, 1269, Exposure to tobacco smoke at home
#438, 1279, Exposure to tobacco smoke outside home  
#750, 2644, Light smokers, at least 100 smokes in lifetime
#813, 2867, Age started smoking in former smokers
#816, 2877, Type of tobacco previously smoked
#822, 2897, Age stopped smoking
#825, 2907, Ever stopped smoking for 6+ months
#876, 3436, Age started smoking in current smokers
#879, 3446, Type of tobacco currently smoked
#882, 3456, Number of cigarettes currently smoked daily (current cigarette smokers)
#888, 3476, Difficulty not smoking for 1 day
#891, 3486, Ever tried to stop smoking

cut -f1,427,430,433,436,439,751,814,817,823,826,877,880,883,889,892 $InDir/ukb9630.tab > $OutDir/extracted_pheno_smoke.txt


##get diabetes

cut -f1,730 $InDir/ukb9630.tab > $OutDir/extracted_pheno_diabetes.txt


##get two more dieases
#copd
cut -f1,3586 /proj/yunligrp/UKBB_phen_29983/ukb32796.tab > $OutDir/COPD_new_IID.txt
#fev/fvc 
cut -f1,356 /proj/yunligrp/UKBB_phen_29983/ukb48839.tab > $OutDir/FEV_FVC_new_IID.txt