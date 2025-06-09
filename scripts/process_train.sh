DATA_DIR=data/s_h
LOG_DIR=log/s_h

SCENES=(forest1 forest2 forest3 garden1 garden2 garden3 indoor playground university1 university2 university3 university4)
FOVS=(59 89 69 59 69 69 69 69 85 73 73 69)

for i in "${!SCENES[@]}"; do
    SCENE=${SCENES[$i]}
    FOV=${FOVS[$i]}
    SCENE_DIR=${DATA_DIR}/${SCENE}
    RES_DIR=${LOG_DIR}/${SCENE}

    echo "Processing scene: ${SCENE} with FOV: ${FOV}"

    python scripts/run_flow.py --data_dir ${SCENE_DIR}
    python DPT/run_monodepth.py --input_path ${SCENE_DIR}/images --output_path ${SCENE_DIR}/depth --model_type dpt_large
    python localTensoRF/train.py --datadir ${SCENE_DIR} --logdir ${RES_DIR} --fov ${FOV}
done
