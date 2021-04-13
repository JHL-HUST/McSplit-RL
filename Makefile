CXX := g++
CXXFLAGS := -O3 -march=native
all: mcsp

mcsp: mcsplit+RL.cpp graph.cpp graph.h
	$(CXX) $(CXXFLAGS) -Wall -std=c++11 -o mcsp graph.cpp mcsplit+RL.cpp -pthread
