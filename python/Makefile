FLAGS = -std=c++11 -Xcompiler #-O3
CC = nvcc
SRCS = $(wildcard ../*.cpp) 

CU_SRCS = $(wildcard ../*.cu)
OBJS = $(SRCS:.cpp=.cpp.o)
CU_OBJS = $(CU_SRCS:.cu=.cu.o)

BOOST_INC=/usr/include
BOOST_LIB=/usr/lib/x86_64-linux-gnu
PYTHON_VERSION=2.7
PYTHON_INC=/usr/include/python$(PYTHON_VERSION)



Sim:
	$(CC) -I$(BOOST_INC) -I../ -I$(PYTHON_INC) $(FLAGS) -fpic -c Sim.cpp -o Sim.o


lib: 
	$(CC) -shared -L$(BOOST_LIB) -o Sim.so $(CU_OBJS) $(OBJS) Sim.o -lboost_python -lpython2.7 -lpugixml

%.cpp.o: %.cpp
	$(CC) $(FLAGS) -fPIC $(DEBUG) -I$(PYTHON_INC) -I$(BOOST_INC) -c -o $@ $< -lboost_python -lpython2.7


%.cu.o: %.cu
	$(CC) $(FLAGS) $(DEBUG) -I$(PYTHON_INC) -I$(BOOST_INC) -c -o $@ $< -lboost_python -lpython2.7 -lpugixml

clean:
	rm *.cpp.o

dist:
	cp Sim.so ../../lib/

all:
	make Sim
	make lib
	make dist
