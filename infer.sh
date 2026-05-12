#!/usr/bin/env bash

ENTRY="infer.py"
LOGDIR="./logs/reveal"

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/diffusers/src:${PYTHONPATH}"

mkdir -p "${LOGDIR}"

mid=${1:-0}
bins=8
tols=8
processes_per_gpu=1
gpus=8

sid=$((mid * bins))

for ((gpu_id=0; gpu_id<gpus; gpu_id++)); do
    for ((j=0; j<processes_per_gpu; j++)); do
        i=$((gpu_id * processes_per_gpu + j))
        if [ ${i} -ge ${bins} ]; then
            break
        fi
        cid=$((sid + i))
        echo "GPU ${gpu_id}, process ${j}, cid ${cid}, tols ${tols}"
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1 \
        nohup /home/jovyan/boomcheng-work-shcdt/zhaoshihao/anaconda/envs/flux/bin/python -u "${PROJECT_ROOT}/${ENTRY}" \
            --cid ${cid} \
            --tols ${tols} \
            > "${LOGDIR}/proc${mid}_${gpu_id}_${j}.log" 2>&1 &
    done
done