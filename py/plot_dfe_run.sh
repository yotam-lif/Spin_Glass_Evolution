#!/bin/bash

# Load necessary modules
module load Python/3.10.8-GCCcore-12.2.0
module load matplotlib/3.7.0-gfbf-2022b

# Delete previous output files
rm out_graph.txt err_graph.txt

# Navigate to the script directory
cd /home/labs/pilpel/yotamlif/Spin_Glass_Evolution/py

# Print environment and module information for debugging
echo "Loaded modules:" > out_graph.txt
module list >> out_graph.txt

# Print Python executable and path for debugging
echo "Python executable: $(which python)" >> out_graph.txt
echo "Python version: $(python --version)" >> out_graph.txt
echo "Python path: $(python -c 'import sys; print(sys.path)')" >> out_graph.txt

# Run the Python script with arguments: nexps, nsamples, dir_name, fit(bool), beneficial(bool) [times]
python plot_dfe.py 1 1 40 "lenski_data" --no-fit --no-beneficial 0 10000 20000 30000 40000 49990 >> out_graph.txt 2>> err_graph.txt

# Check if the script ran successfully
if [ $? -eq 0 ]; then
  echo "Script ran successfully." >> out_graph.txt
else
  echo "Script encountered an error." >> out_graph.txt
fi