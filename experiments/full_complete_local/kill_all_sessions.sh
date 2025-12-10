#!/bin/bash
exp_date_base=$1
for i in {1..14}; do
    tmux kill-session -t sample-$exp_date_base-$i
done