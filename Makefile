CXX = g++
CXXFLAGS = -fopenmp -g -Wno-long-long -Wall -lm -pedantic -std=c++14
CXXFLAGS += -O3

# Use environment variables for include and library paths
CXXFLAGS += -I$(BOOST_INCLUDE) -I$(GSL_INCLUDE)
LDFLAGS = -L$(BOOST_LIB) -L$(GSL_LIB) -lgsl -lgslcblas

# Object files and executables
OBJS = lenski_sim.o
EXECS = lenski_main lenski_vary_epi lenski_vary_clonal

# Default target
all: $(EXECS)

# Pattern rule for object files
%.o: %.cc %.hh
	$(CXX) -c -o $@ $< $(CXXFLAGS)

# Individual executable targets
lenski_main: lenski_main.cc $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

lenski_vary_epi: lenski_vary_epi.cc $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

lenski_vary_clonal: lenski_vary_clonal.cc $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

# Phony target for cleanup
.PHONY: clean
clean:
	rm -f *.o $(EXECS) out.txt err.txt