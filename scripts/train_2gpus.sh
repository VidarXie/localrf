#!/usr/bin/env bash
set -e

DATA_DIR=data
EXP=logs

DATA_PREFIX=s_h
SCENES=( forest2 forest3 garden1 garden2 garden3 \
         university1 university2 university3 )
FOVS=( 89 69 59 69 69 \
         85 73 73 )

N_GPU=2                    # <-- only 2 GPUs
N_SCENES=${#SCENES[@]}

run_scene () {
    local gpu=$1 idx=$2
    local scene=${DATA_PREFIX}/${SCENES[$idx]}
    local fov=${FOVS[$idx]}

    local scene_dir=${DATA_DIR}/${scene}
    local log_root=${EXP}/${scene}
    local log_main=${log_root}/main
    local log_wo_reg=${log_root}/wo_reg
    mkdir -p "$log_main" "$log_wo_reg"

    echo "[GPU $gpu]  $scene  (FoV $fov)  -- starting…"

    # variant-A
    python -u localTensoRF/train.py \
        --datadir "${scene_dir}"    \
        --logdir  "${log_main}"     \
        --fov     "${fov}"          \
        --device  "cuda:${gpu}"     \
        --N_voxel_init  216000      \
        --N_voxel_final 27000000

    # variant-B
    python -u localTensoRF/train.py \
        --datadir "${scene_dir}"    \
        --logdir  "${log_wo_reg}"      \
        --fov     "${fov}"          \
        --device  "cuda:${gpu}"     \
        --N_voxel_init  27000000    \
        --N_voxel_final 27000000

    echo "[GPU $gpu]  $scene  -- done."
}

# -------------------------------------------------------------------------
# One subshell per GPU.  Each subshell loops over *its* half of the scenes
# and therefore runs 12 × 2 = 24 jobs sequentially (no overlap).
# -------------------------------------------------------------------------
for gpu in $(seq 0 $((N_GPU-1))); do
(
    for idx in $(seq 0 $((N_SCENES-1))); do
        if (( idx % N_GPU == gpu )); then
            run_scene "$gpu" "$idx"
        fi
    done
) >"worker_gpu${gpu}.out" 2>&1 &       # stdout+stderr go to worker log
done

echo "Launched two background workers (worker_gpu0.out / worker_gpu1.out)."
echo "Each worker will process its 12 scenes in sequence."
