cat >> ~/.bashrc <<'EOF'

# Initialize Conda in interactive Apptainer shells.
if [ -n "$APPTAINER_CONTAINER" ] && [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    . /opt/conda/etc/profile.d/conda.sh
fi
EOF

source ~/.bashrc