#!/bin/bash

trj_len="100"
seed_number="0"
seed_shadow1='seed_5'
seed_shadow2='seed_100'
seed_target1='seed_75'
seed_target2='seed_700'

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
sshpass -p 'Mywoodencottage1' scp -r samin@beluga.calculquebec.ca:~/projects/rrg-dprecup/samin/learning_output/Hopper-v2/1000000/4000000/20/5/100/buffers/* .
cd ..
mkdir $seed_shadow2
sshpass -p 'Mywoodencottage1' scp -r samin@beluga.calculquebec.ca:~/projects/rrg-dprecup/samin/learning_output/Hopper-v2/1000000/4000000/20/100/100/buffers/* .
cd ../../target
mkdir $seed_target1
cd $seed_target1
sshpass -p 'Mywoodencottage1' scp -r samin@beluga.calculquebec.ca:~/projects/rrg-dprecup/samin/learning_output/Hopper-v2/1000000/4000000/200/75/100/buffers/* .
cd ..
mkdir $seed_target2
cd $seed_target2
sshpass -p 'Mywoodencottage1' scp -r samin@beluga.calculquebec.ca:~/projects/rrg-dprecup/samin/learning_output/Hopper-v2/1000000/4000000/200/700/100/buffers/* .

