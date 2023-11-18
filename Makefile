## How to use this Makefile
## You should not have to modify anything except the OBJS variable
## 	by adding .o file names if you have more .cpp/.hpp files.

CXX := mpic++
CXXFLAGS := -Wall -Wextra -Wpedantic -std=c++20 -O3 -DOMPI_SKIP_MPICXX
LDLIBS := -lm

# Generate output files
# Main executable, 2 debug level versions of your code, and a reference sequential executable
OUTPUT := distr-sched
OUTPUT_DEBUG1 := distr-sched-debug1
OUTPUT_DEBUG2 := distr-sched-debug2
SEQUENTIAL := distr-sched-seq

# STUDENT TODO: Append to this list if you have more files to compile
# E.g., if you create the files `custom.cpp` and `custom.hpp`, add 'custom.o' to the list below. 
# You should also add them to the `main.o` dependancy list (after `tasks.hpp`)
OBJS := main.o runner.o
SEQ_OBJS := main.o runner_seq.o

all: $(OUTPUT) $(OUTPUT_DEBUG1) $(OUTPUT_DEBUG2) $(SEQUENTIAL)

# Your main executable at DEBUG level 0
$(OUTPUT): $(OBJS) tasks.0.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

# DEBUG level 1 version: execution "trace"
$(OUTPUT_DEBUG1): $(OBJS) tasks.1.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

# DEBUG level 2 version: execution "trace" plus more MPI information
$(OUTPUT_DEBUG2): $(OBJS) tasks.2.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

# Sequential version for checking trace outputs
$(SEQUENTIAL): $(SEQ_OBJS) tasks.1.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

tasks.0.o: tasks.cpp tasks.hpp 
	$(CXX) $(CXXFLAGS) -DDEBUG=0 -c $< -o $@

tasks.1.o: tasks.cpp tasks.hpp 
	$(CXX) $(CXXFLAGS) -DDEBUG=1 -c $< -o $@

tasks.2.o: tasks.cpp tasks.hpp 
	$(CXX) $(CXXFLAGS) -DDEBUG=2 -c $< -o $@

main.o: main.cpp runner.hpp tasks.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cpp %.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(OUTPUT) $(OUTPUT_DEBUG1) $(OUTPUT_DEBUG2) $(OBJS) *.o
