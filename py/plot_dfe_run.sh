#!/bin/bash

# Load necessary modules
module load Python/3.10.8-GCCcore-12.2.0
module load matplotlib/3.7.0-gfbf-2022b

# Delete previous output files
rm out_plot_dfe.txt err_plot_dfe.txt

# Navigate to the script directory
cd /home/labs/pilpel/yotamlif/Spin_Glass_Evolution/py

# Print Python path for debugging
echo "Python executable: $(which python)"
echo "Python path: $(python -c 'import sys; print(sys.path)')"

# Run the Python script with arguments
python plot_dfe.py 10 1 1e-3 5e-3 10000 20000 30000 40000 50000 60000 70000 80000 90000 99000