#!/bin/bash
# ============================================================================
# Self-RAG on MAEDA-benchmark-300 (with BGE retrieval)
# ============================================================================
# Usage:
#   bash run_selfrag_maeda.sh                  # full pipeline (5 steps)
#   bash run_selfrag_maeda.sh retrieve         # step 1 only (BGE retrieval)
#   bash run_selfrag_maeda.sh convert          # step 2 only (data conversion)
#   bash run_selfrag_maeda.sh inference        # step 3 only (Self-RAG inference)
#   bash run_selfrag_maeda.sh postprocess      # step 4 only
#   bash run_selfrag_maeda.sh eval             # step 5 only
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
BGE_MODEL_PATH="/mnt/public/sichuan_a/nyt/models/RAG-EDA/models/finetuned-models/embedding/bge-large-en-v1.5/output_flagembedding"
CORPUS_PATH="$PROJ_ROOT/resources/knowledge_openroad_MAEDA.json"
FAISS_INDEX_DIR="$PROJ_ROOT/resources/faiss_bge_selfrag"
BENCHMARK="$BENCH_DIR/MAEDA-benchmark-300.json"
DATA_DIR="$SCRIPT_DIR/maeda_selfrag_data"
RESULTS_DIR="$SCRIPT_DIR/results"
RETRIEVAL_RESULTS="$DATA_DIR/bge_retrieval_results.json"
NDOCS=5
MAX_NEW_TOKENS=300
THRESHOLD=0.2
MODE="adaptive_retrieval"
WORLD_SIZE=1
DTYPE="half"
RETRIEVAL_TOPK=10

info()  { echo -e "\033[32m[$(date '+%H:%M:%S') INFO]\033[0m $*"; }
step()  { echo -e "\033[36m[$(date '+%H:%M:%S') STEP]\033[0m $*"; }
warn()  { echo -e "\033[33m[$(date '+%H:%M:%S') WARN]\033[0m $*"; }

# ======================== Step 1: BGE Retrieval ========================
step_retrieve() {
    step "Step 1: BGE retrieval — build FAISS index & retrieve passages"
    if [ ! -f "$CORPUS_PATH" ]; then
        echo "Error: Knowledge corpus not found at $CORPUS_PATH"
        exit 1
    fi
    mkdir -p "$DATA_DIR"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
    python3 "$SCRIPT_DIR/retrieve_with_bge.py" \
        --corpus "$CORPUS_PATH" \
        --benchmark "$BENCHMARK" \
        --model_name "$BGE_MODEL_PATH" \
        --index_dir "$FAISS_INDEX_DIR" \
        --top_k "$RETRIEVAL_TOPK" \
        --output "$RETRIEVAL_RESULTS" \
        --device "$CUDA_VISIBLE_DEVICES"
    if [ ! -f "$RETRIEVAL_RESULTS" ]; then
        echo "Error: Retrieval failed — no output file generated."
        exit 1
    fi
    info "Retrieval done. Results: $RETRIEVAL_RESULTS"
}

# ======================== Step 2: Data Conversion ========================
step_convert() {
    step "Step 2: Convert MAEDA benchmark → Self-RAG format (with BGE retrieval)"
    mkdir -p "$DATA_DIR"
    if [ -f "$RETRIEVAL_RESULTS" ]; then
        python3 "$SCRIPT_DIR/convert_maeda_to_selfrag.py" \
            --input "$BENCHMARK" \
            --output_dir "$DATA_DIR" \
            --ndocs "$NDOCS" \
            --retrieval_results "$RETRIEVAL_RESULTS"
    else
        warn "No BGE retrieval results found ($RETRIEVAL_RESULTS). Falling back to reranked_knowledge (CHEATING!)"
        python3 "$SCRIPT_DIR/convert_maeda_to_selfrag.py" \
            --input "$BENCHMARK" \
            --output_dir "$DATA_DIR" \
            --ndocs "$NDOCS"
    fi
}

# ======================== Step 3: Self-RAG Inference ========================
step_inference() {
    step "Step 3: Run Self-RAG inference"
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
    # Disable vLLM v1 engine — it has msgspec serialization bug with large logprobs
    # (ValidationError: Expected `float`, got `array`)
    export VLLM_USE_V1=0
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

# ======================== Step 4: Post-process to MAEDA evaluator format ========================
step_postprocess() {
    step "Step 4: Post-process Self-RAG output → MAEDA evaluator input"
    if [ ! -f "$RESULTS_DIR/selfrag_raw_results.json" ]; then
        echo "Error: $RESULTS_DIR/selfrag_raw_results.json not found. Run 'inference' first."
        exit 1
    fi
    python3 "$SCRIPT_DIR/postprocess_selfrag_output.py" \
        --raw_results "$RESULTS_DIR/selfrag_raw_results.json" \
        --gt_data "$DATA_DIR/selfrag_gt.json" \
        --output "$RESULTS_DIR/selfrag_maeda_eval_input.json"
}

# ======================== Step 5: Run MAEDA Evaluation ========================
step_eval() {
    step "Step 5: Run MAEDA evaluation"
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
STEPS=(retrieve convert inference postprocess eval)
run_step() { case $1 in
    retrieve) step_retrieve;;
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
