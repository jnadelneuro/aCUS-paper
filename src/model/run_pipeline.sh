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
# working directory (~/Documents/EphysModeling/...), and virtualenv path used here and
# inside every SBATCH heredoc below are HPC/Quest-specific; edit to match your cluster.
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
#SBATCH --cpus-per-task=50
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=1-30
#SBATCH --export=ALL,NGEN=200,OFFSPRING=600
#SBATCH --mail-type=END,FAIL
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
echo "Naive: $JOB_NAIVE (30 jobs)"

# ============================================================
# STAGE 1: FROZEN (30 jobs, after naive)
# ============================================================
JOB_FROZEN=$(sbatch --parsable --dependency=afterany:${JOB_NAIVE} <<'FROZEN'
#!/bin/bash
#SBATCH --job-name=frozen
#SBATCH --output=logs/robust/slurm/frozen_%a_%j.out
#SBATCH --error=logs/robust/slurm/frozen_%a_%j.err
#SBATCH --account=p32424
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --array=1-30
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jan7154@ads.northwestern.edu

source ~/Documents/EphysModeling/bpop_env/bin/activate
cd ~/Documents/EphysModeling/LiEtAlModels

TASK=${SLURM_ARRAY_TASK_ID}
NAIVE_SEED=$(( (TASK - 1) / 3 + 1 ))
CLUSTER=$(( (TASK - 1) % 3 + 1 ))
export MODE=frozen CLUSTER NAIVE_SEED STRESS_SEED=1

python3 optimize_robust_stress.py
FROZEN
)
echo "Frozen: $JOB_FROZEN (30 jobs)"

# ============================================================
# STAGE 1: SINGLES (N_CANDS x 10 seeds, after naive)
# ============================================================
JOB_SINGLES=$(sbatch --parsable --dependency=afterany:${JOB_NAIVE} <<SINGLES
#!/bin/bash
#SBATCH --job-name=singles
#SBATCH --output=logs/robust/slurm/cand_%a_%j.out
#SBATCH --error=logs/robust/slurm/cand_%a_%j.err
#SBATCH --account=p32424
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=10G
#SBATCH --time=02:00:00
#SBATCH --array=1-${TOTAL_CAND}
#SBATCH --export=ALL,NGEN=200,OFFSPRING=300
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jan7154@ads.northwestern.edu

source ~/Documents/EphysModeling/bpop_env/bin/activate
cd ~/Documents/EphysModeling/LiEtAlModels

TASK=\${SLURM_ARRAY_TASK_ID}
NAIVE_SEED=\$(( (TASK - 1) / ${N_CANDS} + 1 ))
CANDIDATE_IDX=\$(( (TASK - 1) % ${N_CANDS} ))
export MODE=candidate NAIVE_SEED CANDIDATE_IDX STRESS_SEED=\${NAIVE_SEED}

python3 optimize_robust_stress.py candidates.json
SINGLES
)
echo "Singles: $JOB_SINGLES ($TOTAL_CAND jobs)"

# ============================================================
# STAGE 1: ALL-FREE (30 jobs, after naive)
# ============================================================
JOB_FREE=$(sbatch --parsable --dependency=afterany:${JOB_NAIVE} <<'ALLFREE'
#!/bin/bash
#SBATCH --job-name=allfree
#SBATCH --output=logs/robust/slurm/allfree_%a_%j.out
#SBATCH --error=logs/robust/slurm/allfree_%a_%j.err
#SBATCH --account=p32424
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=10G
#SBATCH --time=03:00:00
#SBATCH --array=1-30
#SBATCH --export=ALL,NGEN=200,OFFSPRING=600
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jan7154@ads.northwestern.edu

source ~/Documents/EphysModeling/bpop_env/bin/activate
cd ~/Documents/EphysModeling/LiEtAlModels

TASK=${SLURM_ARRAY_TASK_ID}
NAIVE_SEED=$(( (TASK - 1) / 3 + 1 ))
CLUSTER=$(( (TASK - 1) % 3 + 1 ))
export MODE=allfree CLUSTER NAIVE_SEED STRESS_SEED=${NAIVE_SEED}

python3 optimize_robust_stress.py
ALLFREE
)
echo "All-free: $JOB_FREE (30 jobs)"

# ============================================================
# BRIDGE: rebuild candidates with pairs, submit pair jobs
# Runs as a single job AFTER all singles finish.
# ============================================================
JOB_BRIDGE=$(sbatch --parsable --dependency=afterok:${JOB_SINGLES} <<'BRIDGE'
#!/bin/bash
#SBATCH --job-name=bridge
#SBATCH --output=logs/robust/slurm/bridge_%j.out
#SBATCH --error=logs/robust/slurm/bridge_%j.err
#SBATCH --account=p32424
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jan7154@ads.northwestern.edu

source ~/Documents/EphysModeling/bpop_env/bin/activate
cd ~/Documents/EphysModeling/LiEtAlModels

echo "=== BRIDGE: rebuilding candidates with pairs ==="
python3 build_candidates.py

N_SINGLES=$(python3 -c "
import json
cands = json.load(open('candidates.json'))
print(sum(1 for c in cands if len(c['channels']) == 1))
")
N_TOTAL=$(python3 -c "import json; print(len(json.load(open('candidates.json'))))")
N_PAIRS=$(( N_TOTAL - N_SINGLES ))

if [ "$N_PAIRS" -eq 0 ]; then
    echo "ERROR: No pairs generated. Check singles results."
    exit 1
fi

TOTAL_PAIR_JOBS=$(( N_PAIRS * 10 ))
echo "Submitting $N_PAIRS pairs x 10 seeds = $TOTAL_PAIR_JOBS jobs"

sbatch <<PAIRS
#!/bin/bash
#SBATCH --job-name=pairs
#SBATCH --output=logs/robust/slurm/cand_%a_%j.out
#SBATCH --error=logs/robust/slurm/cand_%a_%j.err
#SBATCH --account=p32424
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=10G
#SBATCH --time=03:00:00
#SBATCH --array=1-${TOTAL_PAIR_JOBS}
#SBATCH --export=ALL,NGEN=50,OFFSPRING=500
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jan7154@ads.northwestern.edu

source ~/Documents/EphysModeling/bpop_env/bin/activate
cd ~/Documents/EphysModeling/LiEtAlModels

TASK=\${SLURM_ARRAY_TASK_ID}
NAIVE_SEED=\$(( (TASK - 1) / ${N_PAIRS} + 1 ))
PAIR_OFFSET=\$(( (TASK - 1) % ${N_PAIRS} ))
CANDIDATE_IDX=\$(( PAIR_OFFSET + ${N_SINGLES} ))
export MODE=candidate NAIVE_SEED CANDIDATE_IDX STRESS_SEED=\${NAIVE_SEED}

python3 optimize_robust_stress.py candidates.json
PAIRS

echo "=== BRIDGE COMPLETE ==="
BRIDGE
)
echo "Bridge: $JOB_BRIDGE (submits pairs after singles finish)"

echo ""
echo "============================================"
echo "FULL PIPELINE SUBMITTED"
echo "  Naive:    $JOB_NAIVE    (30 jobs)"
echo "  Frozen:   $JOB_FROZEN   (30 jobs, after naive)"
echo "  Singles:  $JOB_SINGLES  ($TOTAL_CAND jobs, after naive)"
echo "  Allfree:  $JOB_FREE     (30 jobs, after naive)"
echo "  Bridge:   $JOB_BRIDGE   (1 job, after singles → submits pairs)"
echo ""
echo "Stage 1: $(( 30 + 30 + TOTAL_CAND + 30 )) jobs"
echo "Stage 2: pairs (auto-submitted by bridge)"
echo ""
echo "After ALL jobs finish (on your PC):"
echo "  python3 analyze_and_plot.py"
echo "  python3 generate_traces.py"
echo "============================================"
