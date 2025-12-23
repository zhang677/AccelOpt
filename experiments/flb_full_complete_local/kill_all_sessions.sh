#!/bin/bash
exp_date_base="12-21-17-05"
for i in {1..7}; do
    tmux kill-session -t sample-$exp_date_base-$i
done