CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -pthread

.PHONY: all clean
all: bookd

bookd: src/bookd.cpp
	$(CXX) $(CXXFLAGS) -o bookd src/bookd.cpp

clean:
	rm -f bookd
