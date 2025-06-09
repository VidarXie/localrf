# rf-reg-exp

### Steps:
- create environment: 
```
cd localrf
git submodule update --init --recursive
conda create -n localrf python=3.8 -y
conda activate localrf
pip install torch torchvision 
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard imageio easydict matplotlib scipy==1.6.1 plyfile joblib timm
```
- download data
Download the [hike scenes](https://drive.google.com/file/d/1DngTRNuZZXpho8-2cjpToa3lGWzxgwqL/view?usp=drive_link).

- download weights
```
bash scripts/download_weights.sh
```

- run pre-process for all data 
```
bash scripts/preprocess_all.sh
```

- run comparison exps
```
python localTensoRF/train.py --datadir ${SCENE_DIR} --logdir ${RES_DIR} --fov ${FOV} --N_voxel_init 216000 --N_voxel_final 27000000
python localTensoRF/train.py --datadir ${SCENE_DIR} --logdir ${RES_DIR} --fov ${FOV} --N_voxel_init 27000000 --N_voxel_final 27000000
```

```${SCENE_DIR}``` is like ```data/s_h/forest1```

```${RES_DIR}``` is like ```logs/s_h/forest1```

scenes and fovs are listed as 
```
SCENES=(forest1 forest2 forest3 garden1 garden2 garden3 indoor playground university1 university2 university3 university4)
FOVS=(59 89 69 59 69 69 69 69 85 73 73 69)
```
