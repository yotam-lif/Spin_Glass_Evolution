#!/bin/bash

# Load necessary modules
# shellcheck disable=SC2129
echo "Loading modules..." >> out_$LSB_JOBID.txt 2>> err_$LSB_JOBID.txt
module load Boost/1.82.0-GCC-12.3.0 >> out_$LSB_JOBID.txt 2>> err_$LSB_JOBID.txt
module load GSL/2.7-GCC-12.3.0 >> out_$LSB_JOBID.txt 2>> err_$LSB_JOBID.txt

# Export environment variables
echo "Setting environment variables..." >> out_$LSB_JOBID.txt 2>> err_$LSB_JOBID.txt
export BOOST_INCLUDE=/apps/easybd/easybuild/software/Boost/1.82.0-GCC-12.3.0/include
export BOOST_LIB=/apps/easybd/easybuild/software/Boost/1.82.0-GCC-12.3.0/lib
export GSL_INCLUDE=/apps/easybd/easybuild/software/GSL/2.7-GCC-12.3.0/include
export GSL_LIB=/apps/easybd/easybuild/software/GSL/2.7-GCC-12.3.0/lib
export LD_LIBRARY_PATH=$GSL_LIB:$BOOST_LIB:$LD_LIBRARY_PATH

# Print environment variables for debugging
echo "BOOST_INCLUDE: $BOOST_INCLUDE" >> out_$LSB_JOBID.txt 2>> err_$LSB_JOBID.txt
echo "BOOST_LIB: $BOOST_LIB" >> out_$LSB_JOBID.txt 2>> err_$LSB_JOBID.txt
echo "GSL_INCLUDE: $GSL_INCLUDE" >> out_$LSB_JOBID.txt 2>> err_$LSB_JOBID.txt
echo "GSL_LIB: $GSL_LIB" >> out_$LSB_JOBID.txt 2>> err_$LSB_JOBID.txt
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH" >> out_$LSB_JOBID.txt 2>> err_$LSB_JOBID.txt

# Clean and build the project
echo "Running make clean..." >> out_$LSB_JOBID.txt 2>> err_$LSB_JOBID.txt
make clean >> out_$LSB_JOBID.txt 2>> err_$LSB_JOBID.txt

echo "Running make..." >> out_$LSB_JOBID.txt 2>> err_$LSB_JOBID.txt
make >> out_$LSB_JOBID.txt 2>> err_$LSB_JOBID.txt

# Check if the executable is created
if [ ! -f "./lenski_main" ]; then
    echo "Executable lenski_main not found!" >> err_$LSB_JOBID.txt
    exit 1
fi

# Print the current directory and its contents
echo "Current directory: $(pwd)" >> out_$LSB_JOBID.txt 2>> err_$LSB_JOBID.txt
echo "Directory contents:" >> out_$LSB_JOBID.txt 2>> err_$LSB_JOBID.txt
ls -la >> out_$LSB_JOBID.txt 2>> err_$LSB_JOBID.txt

# Run the executable with arguments
echo "Running lenski_main..." >> out_$LSB_JOBID.txt 2>> err_$LSB_JOBID.txt

# Run the executable with arguments
# L, N_0, N_f, ndays, nexps, dt, p_val (p*L prob per division), output_interval, base_folder, init_rank, rank_interval, rho, beta, delta, hoc
./lenski_main 1e3 5e6 5e8 1e5 1 0.01 1e-6 1e4 "lenski_data" 100 10000 0.05 0.75 0.005 0

# Capture the exit code
EXIT_CODE=$?
echo "lenski_main exit code: $EXIT_CODE" >> out_$LSB_JOBID.txt 2>> err_$LSB_JOBID.txt

echo "Finished running lenski_main" >> out_$LSB_JOBID.txt 2>> err_$LSB_JOBID.txt

# Rename the output and error files to include the job name
mv out_$LSB_JOBID.txt out_$LSB_JOBNAME.txt
mv err_$LSB_JOBID.txt err_$LSB_JOBNAME.txt

# Exit with the same code as lenski_main
exit $EXIT_CODE