#!/bin/bash

# Load necessary modules
module load Boost/1.82.0-GCC-12.3.0
module load GSL/2.7-GCC-12.3.0

# Export environment variables
export BOOST_INCLUDE=/apps/easybd/easybuild/software/Boost/1.82.0-GCC-12.3.0/include
export BOOST_LIB=/apps/easybd/easybuild/software/Boost/1.82.0-GCC-12.3.0/lib
export GSL_INCLUDE=/apps/easybd/easybuild/software/GSL/2.7-GCC-12.3.0/include
export GSL_LIB=/apps/easybd/easybuild/software/GSL/2.7-GCC-12.3.0/lib
export LD_LIBRARY_PATH=$GSL_LIB:$BOOST_LIB:$LD_LIBRARY_PATH

# Print environment variables for debugging
echo "BOOST_INCLUDE: $BOOST_INCLUDE"
echo "BOOST_LIB: $BOOST_LIB"
echo "GSL_INCLUDE: $GSL_INCLUDE"
echo "GSL_LIB: $GSL_LIB"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Clean and build the project
make clean
make

# Run the executable with arguments
# L, N_0, N_f, ndays, nexps, dt, p_val (p*L prob per division), output_interval, base_folder, init_rank, rank_interval, rho, beta, delta, hoc
./lenski_main 1e3 5e6 5e8 1e5 10 0.01 8.9e-6 1e3 lenski_data 100 1e3 1 0.05 0.75 0.005 0