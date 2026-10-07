#!/usr/bin/env bash
# Nsight Systems traces for a FEW representative conditions only (never for the sweep).
#   scripts/nsys.sh [outdir] [names...]      default outdir results/nsys, default all three conditions
# Each trace: 4-GPU set from results/env/topology.json, NCCL env from the env stage, 5 warmup + 10 steps.
set -euo pipefail
cd "$(dirname "$0")/.."
NSYS=${NSYS:-$(command -v nsys || echo /usr/local/cuda/bin/nsys)}
OUT=${1:-results/nsys}; shift || true
busy=$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader | wc -l)
if [ "$busy" != 0 ]; then echo "refusing to trace: GPUs have running compute processes (shared machine)"; exit 2; fi

mkdir -p "$OUT"
read -r GPUS NCCL_ENV < <(python -c "import json; t=json.load(open('results/env/topology.json')); \
print(','.join(map(str,t['settings']['4gpu'])), ' '.join(f'{k}={v}' for k,v in t['nccl_env'].items()))")
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=4 PYTHONPATH=$PWD NCCL_DEBUG=WARN
[ -n "${NCCL_ENV:-}" ] && export $NCCL_ENV
N=$(echo "$GPUS" | tr , '\n' | wc -l)

run() {  # name model bucket delay
  timeout 900 "$NSYS" profile -o "$OUT/$1" --force-overwrite true --trace=cuda,nvtx --sample=none \
    --cpuctxsw=none python -m torch.distributed.run --standalone --nproc_per_node="$N" -m hddp.worker \
    --model "$2" --conditions "[[$3, $4]]" --warmup 5 --steps 10 --out "$OUT/$1.jsonl" --meta '{"stage": "nsys"}'
  "$NSYS" stats --report cuda_gpu_kern_sum --format csv --output "$OUT/$1" "$OUT/$1.nsys-rep" > /dev/null || true
  rm -f "$OUT/$1.sqlite"
  # a trace taken while another process shared the GPUs is not representative: flag it loudly
  python -c "import json,sys; r=[json.loads(l) for l in open('$OUT/$1.jsonl')]; \
f=[k['foreign_pids'] for x in r for k in x['ranks'] if k['foreign_pids']]; \
print('CONTAMINATED trace $1 (foreign GPU processes):', f) if f else print('clean trace $1'); sys.exit(3 if f else 0)" \
    || mv "$OUT/$1.nsys-rep" "$OUT/$1.CONTAMINATED.nsys-rep"
}

declare -A COND=([resnet50_b25_d0]="resnet50 25 0" [resnet50_b25_d20]="resnet50 25 20" [vit_b_16_b25_d20]="vit_b_16 25 20")
NAMES=("$@"); [ ${#NAMES[@]} -eq 0 ] && NAMES=(resnet50_b25_d0 resnet50_b25_d20 vit_b_16_b25_d20)
for n in "${NAMES[@]}"; do run "$n" ${COND[$n]}; done
