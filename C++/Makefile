CXX = g++
CC = gcc

BUILDDIR := 

CXXFLAGS := -O3 

INCS := -I/usr/local/include/opencv2
LIBS := $(shell pkg-config --libs opencv)

MAIN := ./main

CXX_SRC := $(wildcard *.cpp)

#CXX_OBJS := $(addprefix $(BUILDDIR)/, $(CXX_SRC:.cpp = .o))
CXX_OBJS := gaborKernel.o roadDetector.o main.o

$(MAIN): $(CXX_OBJS)
	$(CXX) $(CXXFLAGS) $(INCS) -o $(MAIN) $(CXX_OBJS) $(LIBS)


$(CXX_OBJS): %.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCS) $< -o $@

.PHONY: clean
clean:
	rm -f *.o
	rm -f $(MAIN)
