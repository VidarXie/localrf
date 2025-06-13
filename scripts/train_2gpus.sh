#!/usr/bin/env bash
set -e          # exit on first unhandled error
set -o pipefail # exit if any part of a pipeline fails
set -u          # exit on use of undefined variables

###########################################################################
# User-configurable parameters
###########################################################################
DATA_DIR=data
EXP=logs                    # top-level log folder
DATA_PREFIX=s_h

# Scenes and matching FoVs ──> keep indices aligned
SCENES=( forest2 forest3 garden1 garden2 garden3 \
         university1 university2 university3 )
FOVS=( 89 69 59 69 69 85 73 73 )

N_GPU=2                                     # two RTX 4090 cards
N_SCENES=${#SCENES[@]}

###########################################################################
run_scene () {
    local gpu=$1 idx=$2
    local scene=${DATA_PREFIX}/${SCENES[$idx]}
    local fov=${FOVS[$idx]}

    # --------------------------------------------------------------------
    # Paths for this scene
    # --------------------------------------------------------------------
    local scene_dir=${DATA_DIR}/${scene}       # data/s_h/forest2 …
    local log_root=${EXP}/${scene}             # logs/s_h/forest2 …
    local log_main=${log_root}/main            # variant-A
    local log_wo_reg=${log_root}/wo_reg        # variant-B
    mkdir -p "$log_main" "$log_wo_reg"

    echo "[GPU ${gpu}] ${scene}  FoV=${fov}  -- starting …"

    # --------------------------------------------------------------------
    # Variant-A (regular settings)  – runs in the background
    # --------------------------------------------------------------------
    python -u localTensoRF/train.py \
        --datadir "${scene_dir}"    \
        --logdir  "${log_main}"     \
        --fov     "${fov}"          \
        --device  "cuda:${gpu}"     \
        --N_voxel_init  216000      \
        --N_voxel_final 27000000 \
        > "${log_main}/logs.out" 2>&1 &

    pid_a=$!

    # --------------------------------------------------------------------
    # Variant-B (27 M voxel) – also in the background
    # --------------------------------------------------------------------
    python -u localTensoRF/train.py \
        --datadir "${scene_dir}"    \
        --logdir  "${log_wo_reg}"   \
        --fov     "${fov}"          \
        --device  "cuda:${gpu}"     \
        --N_voxel_init  27000000    \
        --N_voxel_final 27000000 \
        > "${log_wo_reg}/logs.out" 2>&1 &

    pid_b=$!

    # --------------------------------------------------------------------
    # Wait for *both* variants to finish before moving to the next scene
    # --------------------------------------------------------------------
    wait "${pid_a}" "${pid_b}"

    echo "[GPU ${gpu}] ${scene}  -- done."
}
export -f run_scene          # needed if you ever call from xargs/parallel

###########################################################################
# One subshell (background worker) per GPU
###########################################################################
for gpu in $(seq 0 $((N_GPU-1))); do
(
    for idx in $(seq 0 $((N_SCENES-1))); do
        if (( idx % N_GPU == gpu )); then
            run_scene "${gpu}" "${idx}"
        fi
    done
) > "worker_gpu${gpu}.out" 2>&1 &
done

echo "Launched ${N_GPU} background workers (worker_gpu0.out, worker_gpu1.out)."
echo "Each worker processes its assigned scenes; two variants run concurrently on the same GPU."
echo "Use 'tail -f worker_gpu0.out' or 'tail -f logs/s_h/<scene>/<variant>/logs.out' to monitor progress."
