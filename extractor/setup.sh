# 1. Kill the NGC environment variables causing the DNS timeouts
unset PIP_EXTRA_INDEX_URL
unset PIP_INDEX_URL
export PIP_NO_CACHE_DIR=1

# 3. Upgrade pip directly from standard PyPI
pip install --upgrade pip --index-url https://pypi.org/simple

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip3 install tqdm --index-url https://pypi.org/simple