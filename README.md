### Environment Setup
```bash
conda create -n piano python=3.10
conda activate piano
conda install -c conda-forge -y portaudio pyaudio=0.2.14 fluidsynth ffmpeg alsa-lib wget quadprog=0.1.11

git clone https://github.com/sNiper-Qian/pianomime.git
cd pianomime

bash scripts/install_deps_cluster.sh

python -m pip install -U pip setuptools wheel
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade "jax==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -U "jaxlib==0.4.23+cuda12.cudnn89" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install tensorboard
python -m pip install torchviz

conda install -c conda-forge "numpy=1.26.4" -y
```


### Train Specialist
```bash
bash pianomime/scripts/run_ppo.sh
```

### Train high level DDPM
```bash
CUDA_VISIBLE_DEVICES=5 python pianomime/multi_task/train_high_level.py dataset_hl.zarr
```


### Train low level DDPM
```bash
CUDA_VISIBLE_DEVICES=7 python pianomime/multi_task/train_low_level.py dataset_ll.zarr
```


### Train high level flow matching
```bash
CUDA_VISIBLE_DEVICES=2 python pianomime/multi_task/flow_matching/train_high_level_flow.py dataset_hl.zarr
```


### Train low level flow matching
```bash
CUDA_VISIBLE_DEVICES=3 python pianomime/multi_task/flow_matching/train_low_level_flow.py dataset_ll.zarr
```



### Calculate Metrics
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
CUDA_VISIBLE_DEVICES=2 python pianomime/multi_task/eval_high_level.py TwinkleTwinkleRousseau --ae-ckpt given_ckpt/checkpoint_ae.ckpt --ckpt-path given_ckpt/checkpoint_high_level.ckpt --use-midi
CUDA_VISIBLE_DEVICES=2 python pianomime/multi_task/eval_low_level.py TwinkleTwinkleRousseau --ae-ckpt given_ckpt/checkpoint_ae.ckpt --ckpt-path given_ckpt/checkpoint_low_level.ckpt --use-midi
```


### Calculate DDPM on the given checkpoint
```bash
CUDA_VISIBLE_DEVICES=0 python pianomime/eval_metrics.py Wednesday_1 --policy ddpm --ae-ckpt given_ckpt/checkpoint_ae.ckpt --high-level-ckpt given_ckpt/checkpoint_high_level.ckpt --low-level-ckpt given_ckpt/checkpoint_low_level.ckpt --label ddpm_given
```


### Calculate DDPM on ours reproduced checkpoint
```bash
CUDA_VISIBLE_DEVICES=4 python pianomime/eval_metrics.py Wednesday_1 --policy ddpm --ae-ckpt reproduced_ckpt/checkpoint_ae.ckpt --high-level-ckpt reproduced_ckpt/dataset_hl_without_fingering.ckpt --low-level-ckpt reproduced_ckpt/dataset_ll.ckpt --label ddpm
```


### Calculate Flow Matching
```bash
CUDA_VISIBLE_DEVICES=3 python pianomime/eval_metrics.py Wednesday_1 --policy flow --ae-ckpt reproduced_ckpt/checkpoint_ae.ckpt --high-level-ckpt flow/ckpts/checkpoint_FM-HL-dataset_hl_without_fingering.ckpt --low-level-ckpt flow/ckpts/checkpoint_FM-LL-dataset_ll.ckpt --flow-steps 20 --flow-solver euler --flow-clip-mode final --label fm20
```


### Calculate the MIDI songs
```bash
CUDA_VISIBLE_DEVICES=3 python pianomime/eval_metrics.py TwinkleTwinkleRousseau --policy ddpm --ae-ckpt given_ckpt/checkpoint_ae.ckpt --high-level-ckpt given_ckpt/checkpoint_high_level.ckpt --low-level-ckpt given_ckpt/checkpoint_low_level.ckpt --label ddpm_given --use-midi
CUDA_VISIBLE_DEVICES=3 python pianomime/eval_metrics.py TwinkleTwinkleRousseau --policy ddpm --ae-ckpt reproduced_ckpt/checkpoint_ae.ckpt --high-level-ckpt reproduced_ckpt/dataset_hl_without_fingering.ckpt --low-level-ckpt reproduced_ckpt/dataset_ll.ckpt --label ddpm --use-midi
CUDA_VISIBLE_DEVICES=3 python pianomime/eval_metrics.py TwinkleTwinkleRousseau --policy flow --ae-ckpt reproduced_ckpt/checkpoint_ae.ckpt --high-level-ckpt flow/ckpts/checkpoint_FM-HL-dataset_hl_without_fingering.ckpt --low-level-ckpt flow/ckpts/checkpoint_FM-LL-dataset_ll.ckpt --flow-steps 20 --flow-solver euler --flow-clip-mode final --label fm20 --use-midi
```


### Run all experiments
```bash
ENABLE_IK=0 bash pianomime/eval_metric_multi.sh
```