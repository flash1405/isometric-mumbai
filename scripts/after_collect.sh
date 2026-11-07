#!/usr/bin/env bash
# When the teacher set is complete: build the dataset and start training on the B200.
cd "$(dirname "$0")/.."
N=${N:-800}
until [ "$(ls out/mumbai/teacher/*_out.png 2>/dev/null | wc -l)" -ge "$N" ]; do sleep 60; done
echo "teacher set complete $(date +%H:%M)"
uv run python pipeline/build_teacher_dataset.py mumbai --variants 3 && \
uv run python pipeline/lambda_gpu.py train && echo "TRAINING STARTED $(date +%H:%M)"
