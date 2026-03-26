CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-sign-compare -Wno-reorder

HEADERS = voronoi.hpp simulation.hpp vor_types.hpp nbrlist.hpp

TARGETS = testBuildRebuild testBuild_Trial testSimulation testDroplet testInterface testViscous testStaticVoronoi

.PHONY: all clean

all: $(TARGETS)

testBuildRebuild: testBuildRebuild.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $<

testBuild_Trial: testBuild_Trial.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $<

testSimulation: testSimulation.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $<

testDroplet: testDroplet.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $<

testInterface: testInterface.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $<

testViscous: testViscous.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $<

testStaticVoronoi: testStaticVoronoi.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) *.o
