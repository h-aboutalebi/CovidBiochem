#!/bin/bash

trj_len="200"
seed_number="0"
seed_shadow1='seed_5'
url_seed_shadow1='~/projects/rrg-dprecup/samin/learning_output/Hopper-v2/1000000/4000000/20/5/200/buffers/*'
seed_shadow2='seed_100'
url_seed_shadow2='~/projects/rrg-dprecup/samin/learning_output/Hopper-v2/1000000/4000000/20/100/200/buffers/*'
seed_target1='seed_75'
url_seed_seed_target1='~/projects/rrg-dprecup/samin/learning_output/Hopper-v2/1000000/4000000/200/75/200/buffers/*'
seed_target2='seed_700'
url_seed_seed_target2='~/projects/rrg-dprecup/samin/learning_output/Hopper-v2/1000000/4000000/200/700/200/buffers/*'

cd /home/hossein.aboutalebi/data/PrivAttack-Data

mkdir $trj_len
cd $trj_len
mkdir $seed_number
cd $seed_number
mkdir shadow
mkdir target
cd shadow
mkdir $seed_shadow1
cd $seed_shadow1
sshpass -p 'Mywoodencottage1' scp -r samin@beluga.calculquebec.ca:$url_seed_shadow1 .
cd ..
mkdir $seed_shadow2
cd $seed_shadow2
sshpass -p 'Mywoodencottage1' scp -r samin@beluga.calculquebec.ca:$url_seed_shadow2 .
cd ../../target
mkdir $seed_target1
cd $seed_target1
sshpass -p 'Mywoodencottage1' scp -r samin@beluga.calculquebec.ca:$url_seed_seed_target1 .
cd ..
mkdir $seed_target2
cd $seed_target2
sshpass -p 'Mywoodencottage1' scp -r samin@beluga.calculquebec.ca:$url_seed_seed_target2 .

