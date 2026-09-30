终端 1：LightNav server

(lightnav)
conda activate lightnav
cd ~/projects/LightNav-0
PORT=8050 CUDA_VISIBLE_DEVICES=1 \
lightnav-serve \
  --task vln \
  --model_path checkpoints/LightNav-0 \
  --backend vllm_local \
  --gpu_memory_utilization 0.75


conda activate lightnav
cd ~/projects/LightNav-0/mujoco_demo
  MUJOCO_GL=egl ./run.sh \
    --vln-server ws://127.0.0.1:8050 \
    --no-trajectory-map

  MUJOCO_GL=egl ./run.sh \
    --vln-server ws://127.0.0.1:8050 \
    --trajectory-map

  http://127.0.0.1:8088

ssh -N -o ExitOnForwardFailure=yes \
    -p 9086 \
    -L 127.0.0.1:18088:127.0.0.1:8088 \
    xrq@10.134.88.113
