#!/bin/bash
exp_date_base=$1
exp_dir="../checkpoints/$exp_date_base"

# Collect scripts from the experiment directory
mapfile -t scripts < <(cd "$exp_dir" && printf '%s\n' resume_single_loop_*.sh)

if (( ${#scripts[@]} == 0 )); then
  echo "No scripts matching resume_single_loop_*.sh in: $exp_dir" >&2
  exit 1
fi

for i in "${!scripts[@]}"; do
  sess="resume-$exp_date_base-$((i+1))"
  # Use bash -lc so 'source' works (it's a bash builtin) and PATH/profile apply
  tmux new-session -d -s "$sess" \
    "bash -lc 'cd \"$exp_dir\" \
      && source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate \
      && export ACCELOPT_BASE_DIR=\"${ACCELOPT_BASE_DIR}\" \
      && bash \"${scripts[$i]}\"; exec bash'"
done
