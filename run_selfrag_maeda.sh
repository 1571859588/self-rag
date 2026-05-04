#!/bin/bash
# ============================================================================
# Self-RAG on MAEDA-benchmark-300 (Plan A: pre-retrieved docs)
# ============================================================================
# Usage:
#   bash run_selfrag_maeda.sh                  # full pipeline
#   bash run_selfrag_maeda.sh convert          # step 1 only
#   bash run_selfrag_maeda.sh inference        # step 2 only
#   bash run_selfrag_maeda.sh postprocess      # step 3 only
#   bash run_selfrag_maeda.sh eval             # step 4 only
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
BENCH_DIR="$SCRIPT_DIR/../.."

# ---- Conda env check ----
# Only inference step requires vllm; other steps just need basic Python
check_vllm() {
    if ! python3 -c "import vllm" 2>/dev/null; then
        echo "Error: vllm not found. Activate conda env: conda activate huada_docqa_demo_release_v1"
        exit 1
    fi
}

# ---- Config ----
MODEL_PATH="/mnt/public/sichuan_a/nyt/models/Self-RAG/models/selfrag_llama2_7b"
BENCHMARK="$BENCH_DIR/MAEDA-benchmark-300.json"
DATA_DIR="$SCRIPT_DIR/maeda_selfrag_data"
RESULTS_DIR="$SCRIPT_DIR/results"
NDOCS=5
MAX_NEW_TOKENS=300
THRESHOLD=0.2
MODE="adaptive_retrieval"
WORLD_SIZE=1
DTYPE="half"

info()  { echo -e "\033[32m[$(date '+%H:%M:%S') INFO]\033[0m $*"; }
step()  { echo -e "\033[36m[$(date '+%H:%M:%S') STEP]\033[0m $*"; }
warn()  { echo -e "\033[33m[$(date '+%H:%M:%S') WARN]\033[0m $*"; }

# ======================== Step 1: Data Conversion ========================
step_convert() {
    step "Step 1: Convert MAEDA benchmark → Self-RAG format"
    mkdir -p "$DATA_DIR"
    python3 "$SCRIPT_DIR/convert_maeda_to_selfrag.py" \
        --input "$BENCHMARK" \
        --output_dir "$DATA_DIR" \
        --ndocs "$NDOCS"
}

# ======================== Step 2: Self-RAG Inference ========================
step_inference() {
    step "Step 2: Run Self-RAG inference"
    check_vllm
    if [ ! -f "$DATA_DIR/selfrag_input.json" ]; then
        echo "Error: $DATA_DIR/selfrag_input.json not found. Run 'convert' first."
        exit 1
    fi
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Error: Model not found at $MODEL_PATH"
        exit 1
    fi

    mkdir -p "$RESULTS_DIR"
    cd "$SCRIPT_DIR/retrieval_lm"

    # Fix CUDA fork issue with vllm >= 0.8
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

    python3 run_short_form.py \
        --model_name "$MODEL_PATH" \
        --input_file "$DATA_DIR/selfrag_input.json" \
        --mode "$MODE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --threshold "$THRESHOLD" \
        --output_file "$RESULTS_DIR/selfrag_raw_results.json" \
        --metric match \
        --ndocs "$NDOCS" \
        --use_groundness \
        --use_utility \
        --use_seqscore \
        --dtype "$DTYPE" \
        --world_size "$WORLD_SIZE" \
        ${RESUME_FILE:+--resume_file "$RESUME_FILE"}

    if [ ! -f "$RESULTS_DIR/selfrag_raw_results.json" ]; then
        echo "Error: Inference failed — no output file generated."
        exit 1
    fi
    info "Inference done. Raw results: $RESULTS_DIR/selfrag_raw_results.json"
}

# ======================== Step 3: Post-process to MAEDA evaluator format ========================
step_postprocess() {
    step "Step 3: Post-process Self-RAG output → MAEDA evaluator input"
    if [ ! -f "$RESULTS_DIR/selfrag_raw_results.json" ]; then
        echo "Error: $RESULTS_DIR/selfrag_raw_results.json not found. Run 'inference' first."
        exit 1
    fi
    python3 "$SCRIPT_DIR/postprocess_selfrag_output.py" \
        --raw_results "$RESULTS_DIR/selfrag_raw_results.json" \
        --gt_data "$DATA_DIR/selfrag_gt.json" \
        --output "$RESULTS_DIR/selfrag_maeda_eval_input.json"
}

# ======================== Step 4: Run MAEDA Evaluation ========================
step_eval() {
    step "Step 4: Run MAEDA evaluation"
    if [ ! -f "$RESULTS_DIR/selfrag_maeda_eval_input.json" ]; then
        echo "Error: $RESULTS_DIR/selfrag_maeda_eval_input.json not found. Run 'postprocess' first."
        exit 1
    fi

    cd "$PROJ_ROOT"
    python3 -m experiments.regression_test.run_eval \
        --config "$SCRIPT_DIR/eval_config_selfrag.json" \
        --input_path "$RESULTS_DIR/selfrag_maeda_eval_input.json" \
        --gt_path "$RESULTS_DIR/selfrag_maeda_eval_input.json" \
        --output_path "$RESULTS_DIR/selfrag_maeda_eval_result.json"
}

# ======================== Pipeline ========================
STEPS=(convert inference postprocess eval)
run_step() { case $1 in
    convert) step_convert;;
    inference) step_inference;;
    postprocess) step_postprocess;;
    eval) step_eval;;
esac; }

STEP="${1:-all}"

if [ "$STEP" = "all" ]; then
    info "========== Self-RAG × MAEDA Benchmark Pipeline =========="
    info "Model: $MODEL_PATH"
    info "Benchmark: $BENCHMARK"
    info "Mode: $MODE, ndocs: $NDOCS, threshold: $THRESHOLD"
    info "Python: $(which python3) ($(python3 --version))"
    echo ""
    for s in "${STEPS[@]}"; do
        run_step "$s" || { warn "Step $s failed, stopping"; exit 1; }
    done
    info "========== Pipeline Complete =========="
else
    run_step "$STEP"
fi
