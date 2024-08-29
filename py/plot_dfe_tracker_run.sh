#!/bin/bash

# Load necessary modules
module load Python/3.10.8-GCCcore-12.2.0
module load matplotlib/3.7.0-gfbf-2022b

# Navigate to the script directory
cd /home/labs/pilpel/yotamlif/Spin_Glass_Evolution/py

# Delete previous output files
rm out_dfe_track.txt err_dfe_track.txt

# Print environment and module information for debugging
echo "Loaded modules:" > out_dfe_track.txt
module list >> out_dfe_track.txt

# Print Python executable and path for debugging
echo "Python executable: $(which python)" >> out_dfe_track.txt
echo "Python version: $(python --version)" >> out_dfe_track.txt
echo "Python path: $(python -c 'import sys; print(sys.path)')" >> out_dfe_track.txt

# Run the Python script with arguments nexps, nsamples, nbins, border, dir_name, beneficial, init_day, [times]
python plot_dfe_tracker.py 4 1 20 0 "lenski_data_10-6" --beneficial 0 0 500000 1000000 2000000 3000000 >> out_dfe_track.txt 2>> err_dfe_track.txt

# Check if the script ran successfully
if [ $? -eq 0 ]; then
  echo "Script ran successfully." >> out_dfe_track.txt
else
  echo "Script encountered an error." >> out_dfe_track.txt
fi