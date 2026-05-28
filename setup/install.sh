#!/bin/bash
set -e   # stop immediately if any command fails

pip install -r setup/requirements.txt

# auto-gptq needs separate index URL
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

# verify all four libraries loaded correctly
python -c "import bitsandbytes; print('bitsandbytes:', bitsandbytes.__version__)"
python -c "import awq; print('autoawq OK')"
python -c "import auto_gptq; print('auto-gptq:', auto_gptq.__version__)"
python -c "import lm_eval; print('lm-eval OK')"