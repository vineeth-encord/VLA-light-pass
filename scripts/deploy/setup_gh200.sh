#!/usr/bin/env bash
# =============================================================================
# setup_gh200.sh — One-time setup for the Lambda Labs GH200 instance
#
# Installs vLLM, creates a systemd service to serve Qwen2.5-VL-7B-Instruct,
# then installs the Encord agent dependencies.
#
# Run once as the ubuntu user (or your Lambda Labs user):
#   chmod +x setup_gh200.sh && bash setup_gh200.sh
#
# After it completes:
#   sudo systemctl start vllm        # start the inference server
#   sudo systemctl status vllm       # verify it's running
#   python scripts/05_vla_agent.py   # run the annotation agent
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Config — adjust if needed
# ---------------------------------------------------------------------------
VLLM_PORT=8000
MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
VENV_DIR="$HOME/vllm-env"
AGENT_VENV_DIR="$HOME/agent-env"
# HuggingFace cache (GH200 has large NVMe; keep models there)
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
GPU_MEM_UTIL=0.85        # leave 15% for OS + agent process overhead
MAX_MODEL_LEN=16384       # context window (tokens)

CYAN="\033[0;36m"; BOLD="\033[1m"; RESET="\033[0m"
step() { echo -e "\n${BOLD}${CYAN}▶ $*${RESET}"; }

# ---------------------------------------------------------------------------
# 0. Preflight
# ---------------------------------------------------------------------------
step "Preflight checks"
python3 --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Architecture: $(uname -m)"

# ---------------------------------------------------------------------------
# 1. System deps
# ---------------------------------------------------------------------------
step "System packages"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3-venv python3-pip git curl screen

# ---------------------------------------------------------------------------
# 2. vLLM virtual environment
# ---------------------------------------------------------------------------
step "Creating vLLM venv at $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# GH200 (Grace Hopper) has ARM + H200 GPU.
# vLLM 0.7+ ships pre-built wheels for CUDA 12.x on aarch64.
pip install --upgrade pip wheel setuptools

step "Installing vLLM"
# Install vLLM — the wheel index covers aarch64 + CUDA 12 for GH200
pip install vllm

# Verify Qwen2.5-VL is supported
python3 - <<'EOF'
from vllm import LLM
print("vLLM import OK")
EOF

deactivate

# ---------------------------------------------------------------------------
# 3. Pre-download the model weights (optional but recommended)
# ---------------------------------------------------------------------------
step "Pre-downloading $MODEL_ID weights (this may take ~15 min)"
source "$VENV_DIR/bin/activate"
python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$MODEL_ID",
    cache_dir="$HF_CACHE",
    ignore_patterns=["*.pt", "*.bin"],   # prefer safetensors
)
print("Download complete.")
EOF
deactivate

# ---------------------------------------------------------------------------
# 4. systemd service for vLLM
# ---------------------------------------------------------------------------
step "Creating systemd service: vllm.service"

# Write the service file
sudo tee /etc/systemd/system/vllm.service > /dev/null <<EOF
[Unit]
Description=vLLM — Qwen2.5-VL-7B-Instruct inference server
After=network.target

[Service]
Type=simple
User=$USER
Environment="HF_HOME=$HF_CACHE"
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=$VENV_DIR/bin/python -m vllm.entrypoints.openai.api_server \\
    --model $MODEL_ID \\
    --host 127.0.0.1 \\
    --port $VLLM_PORT \\
    --dtype bfloat16 \\
    --max-model-len $MAX_MODEL_LEN \\
    --gpu-memory-utilization $GPU_MEM_UTIL \\
    --trust-remote-code \\
    --served-model-name qwen-vl
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
KillSignal=SIGTERM
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable vllm

echo "vllm.service installed. Start with: sudo systemctl start vllm"
echo "Logs: sudo journalctl -u vllm -f"

# ---------------------------------------------------------------------------
# 5. Agent virtual environment
# ---------------------------------------------------------------------------
step "Creating agent venv at $AGENT_VENV_DIR"
python3 -m venv "$AGENT_VENV_DIR"
source "$AGENT_VENV_DIR/bin/activate"
pip install --upgrade pip

# The agent no longer loads models locally — it calls vLLM via HTTP.
# So we only need lightweight deps here.
pip install \
    "encord>=0.1.130" \
    "encord-agents[vision]>=0.1.0" \
    "openai>=1.30.0" \
    "Pillow>=9.4.0" \
    "numpy>=1.24.4" \
    "python-dotenv>=1.0.0" \
    "requests>=2.28.0"

deactivate

# ---------------------------------------------------------------------------
# 6. Helper scripts
# ---------------------------------------------------------------------------
step "Writing helper scripts"

# Quick smoke-test: curl the vLLM server once it's up
cat > "$HOME/test_vllm.sh" <<'TESTSCRIPT'
#!/usr/bin/env bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
TESTSCRIPT
chmod +x "$HOME/test_vllm.sh"

# Activate agent env shortcut
cat > "$HOME/activate_agent.sh" <<ACTSCRIPT
#!/usr/bin/env bash
source "$AGENT_VENV_DIR/bin/activate"
echo "Agent env active. Run: python scripts/05_vla_agent.py"
ACTSCRIPT
chmod +x "$HOME/activate_agent.sh"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. sudo systemctl start vllm          # start the inference server"
echo "  2. sudo journalctl -u vllm -f         # watch startup logs (~2 min)"
echo "  3. ~/test_vllm.sh                     # verify server is up"
echo "  4. source ~/activate_agent.sh         # activate agent env"
echo "  5. export ENCORD_SSH_KEY_PATH=~/.ssh/id_ed25519"
echo "  6. python scripts/05_vla_agent.py     # run the annotation agent"
echo "======================================================================"
