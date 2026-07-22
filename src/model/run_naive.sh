#!/bin/bash
# ============================================================
# run_pipeline.sh — Full pipeline, one command
#
# Stage 1: naive → frozen + singles + allfree
# Stage 2: (after singles finish) rebuild candidates with pairs → run pairs
#
# All automatic via SLURM dependency chain.
#
# After completion (on your PC):
#   python3 analyze_and_plot.py
#   python3 generate_traces.py
#
# Usage: bash run_pipeline.sh
# ============================================================
# TODO: set for your cluster — the account (p32424), mail-user (*.northwestern.edu),
# working directory (~/Documents/EphysModeling/...), and virtualenv path below are
# HPC/Quest-specific; edit them to match your own cluster environment.
cd ~/Documents/EphysModeling/LiEtAlModels
source ~/Documents/EphysModeling/bpop_env/bin/activate
mkdir -p logs/robust/results logs/robust/slurm

# ============================================================
# BUILD SINGLES-ONLY CANDIDATES
# ============================================================
python3 build_candidates.py
N_CANDS=$(python3 -c "import json; print(len(json.load(open('candidates.json'))))")
TOTAL_CAND=$(( N_CANDS * 10 ))

# ============================================================
# STAGE 1: NAIVE (30 jobs)
# ============================================================
JOB_NAIVE=$(sbatch --parsable <<'NAIVE'
#!/bin/bash
#SBATCH --job-name=naive
#SBATCH --output=logs/robust/slurm/naive_%a_%j.out
#SBATCH --error=logs/robust/slurm/naive_%a_%j.err
#SBATCH --account=p32424
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --array=1-3
#SBATCH --export=ALL,NGEN=50,OFFSPRING=128
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jan7154@ads.northwestern.edu

source ~/Documents/EphysModeling/bpop_env/bin/activate
cd ~/Documents/EphysModeling/LiEtAlModels

TASK=${SLURM_ARRAY_TASK_ID}
SEED=$(( (TASK - 1) / 3 + 1 ))
CLUSTER=$(( (TASK - 1) % 3 + 1 ))
export SEED

cp optimize_robust_naive.py _naive_cl${CLUSTER}_s${SEED}.py
sed -i "s/^CLUSTER_TO_FIT = .*/CLUSTER_TO_FIT = ${CLUSTER}/" _naive_cl${CLUSTER}_s${SEED}.py
python3 _naive_cl${CLUSTER}_s${SEED}.py
rm -f _naive_cl${CLUSTER}_s${SEED}.py
NAIVE
)
echo "Naive: $JOB_NAIVE (3 jobs)"
