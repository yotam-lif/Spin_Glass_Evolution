# Load necessary modules
module load Boost/1.82.0-GCC-12.3.0
module load GSL/2.7-GCC-12.3.0

# Export environment variables
export BOOST_INCLUDE=/apps/easybd/easybuild/software/Boost/1.82.0-GCC-12.3.0/include
export BOOST_LIB=/apps/easybd/easybuild/software/Boost/1.82.0-GCC-12.3.0/lib
export GSL_INCLUDE=/apps/easybd/easybuild/software/GSL/2.7-GCC-12.3.0/include
export GSL_LIB=/apps/easybd/easybuild/software/GSL/2.7-GCC-12.3.0/lib

# Run the make command
make