#!/bin/bash

trj_len="100"
seed_shadow1='seed_100'
seed_shadow2='seed_5'
seed_target1='seed_75'
seed_target2='seed_700'

cd /home/hossein.aboutalebi/data/PrivAttack-Data
mkdir $trj_len
cd $trj_len
mkdir shadow
mkdir target
cd shadow
mkdir $seed_shadow1
mkdir $seed_shadow2
cd ../target
mkdir $seed_target1
mkdir $seed_target2
