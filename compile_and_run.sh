#!/bin/bash

# Load necessary modules
echo "Loading modules..." >> out.txt 2>> err.txt
module load Boost/1.82.0-GCC-12.3.0 >> out.txt 2>> err.txt
module load GSL/2.7-GCC-12.3.0 >> out.txt 2>> err.txt

# Export environment variables
echo "Setting environment variables..." >> out.txt 2>> err.txt
export BOOST_INCLUDE=/apps/easybd/easybuild/software/Boost/1.82.0-GCC-12.3.0/include
export BOOST_LIB=/apps/easybd/easybuild/software/Boost/1.82.0-GCC-12.3.0/lib
export GSL_INCLUDE=/apps/easybd/easybuild/software/GSL/2.7-GCC-12.3.0/include
export GSL_LIB=/apps/easybd/easybuild/software/GSL/2.7-GCC-12.3.0/lib
export LD_LIBRARY_PATH=$GSL_LIB:$BOOST_LIB:$LD_LIBRARY_PATH

# Print environment variables for debugging
echo "BOOST_INCLUDE: $BOOST_INCLUDE" >> out.txt 2>> err.txt
echo "BOOST_LIB: $BOOST_LIB" >> out.txt 2>> err.txt
echo "GSL_INCLUDE: $GSL_INCLUDE" >> out.txt 2>> err.txt
echo "GSL_LIB: $GSL_LIB" >> out.txt 2>> err.txt
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH" >> out.txt 2>> err.txt

# Clean and build the project
echo "Running make clean..." >> out.txt 2>> err.txt
make clean >> out.txt 2>> err.txt

echo "Running make..." >> out.txt 2>> err.txt
make >> out.txt 2>> err.txt

# Check if the executable is created
if [ ! -f "./lenski_main" ]; then
    echo "Executable lenski_main not found!" >> err.txt
    exit 1
fi

# Print the current directory and its contents
echo "Current directory: $(pwd)" >> out.txt 2>> err.txt
echo "Directory contents:" >> out.txt 2>> err.txt
ls -la >> out.txt 2>> err.txt

# Run the executable with arguments
echo "Running lenski_main..." >> out.txt 2>> err.txt

# Run the executable with arguments
# L, N_0, N_f, ndays, nexps, dt, p_val (p*L prob per division), output_interval, base_folder, init_rank, rank_interval, rho, beta, delta, hoc
./lenski_main 1e3 5e6 5e8 1e5 10 0.01 8.9e-6 1e3 "lenski_data" 100 1e3 1 0.05 0.75 0.005 0

# Capture the exit code
EXIT_CODE=$?
echo "lenski_main exit code: $EXIT_CODE" >> out.txt 2>> err.txt

echo "Finished running lenski_main" >> out.txt 2>> err.txt

# Exit with the same code as lenski_main
exit $EXIT_CODE