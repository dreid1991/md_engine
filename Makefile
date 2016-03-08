FLAGS = -std=c++11 -Xcompiler -fpic -g  --use_fast_math --prec-div=true# -Winline
CU_CC = nvcc
DEPDIR := deps
OBJDIR := objs
LIBDIR := lib
POSTCOMPILE = mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d
$(shell mkdir -p $(DEPDIR) >/dev/null)
$(shell mkdir -p $(OBJDIR) >/dev/null)
$(shell mkdir -p $(LIBDIR) >/dev/null)
H = $(wildcard $(MD)*.h)
C_SRCS = $(wildcard $(MD)*.cpp) 
CU_SRCS = $(wildcard $(MD)*.cu)

C_OBJS = $(patsubst %,%.o,$(basename $(C_SRCS)))
CU_OBJS = $(patsubst %,%.o,$(basename $(CU_SRCS)))


DEPFILES = $(patsubst %,$(DEPDIR)/%.d,$(basename $(C_SRCS) $(CU_SRCS)))
#$(info Serial result = $(DEPFILES))
$(foreach path,$(DEPFILES),$(shell if [ ! -e $(path) ] ; then touch $(path) ; fi ))

BOOST_INC=/usr/include
BOOST_LIB=/usr/lib/#x86_64-linux-gnu
PYTHON_VERSION=2.7
PYTHON_INC=/usr/include/python$(PYTHON_VERSION)

main: $(C_OBJS) $(CU_OBJS) 
	$(CU_CC) $(FLAGS) $(C_OBJS) $(CU_OBJS) -o a.out -lpython2.7 -lpugixml -lboost_python -lcufft

%.o: %.cpp ./$(DEPDIR)/%.d
	$(CU_CC) $(FLAGS) $(DEBUG) -I$(PYTHON_INC) -I$(BOOST_INC) --output-file $(DEPDIR)/$*.Td -M $< -lboost_python -lpython2.7 -lpugixml
	$(CU_CC) $(FLAGS) $(DEBUG) -I$(PYTHON_INC) -I$(BOOST_INC) -c -o $@ $< -lboost_python -lpython2.7 -lpugixml
	$(POSTCOMPILE)


%.o: %.cu ./$(DEPDIR)/%.d
	$(CU_CC) $(FLAGS) $(DEBUG) -I$(PYTHON_INC) -I$(BOOST_INC) --output-file $(DEPDIR)/$*.Td -M $< -lboost_python -lpython2.7 -lpugixml
	$(CU_CC) $(FLAGS) $(DEBUG) -I$(PYTHON_INC) -I$(BOOST_INC) -c -o $@ $< -lboost_python -lpython2.7 -lpugixml
	$(POSTCOMPILE)


%.d: ;

clean:
	rm -f *.o



Sim:
	$(CU_CC) -I. -I$(BOOST_INC) -I$(PYTHON_INC) $(FLAGS)  -c python/Sim.cpp -o python/Sim.o
lib: 
	$(CU_CC) -shared -L$(BOOST_LIB) -o python/Sim.so $(CU_OBJS) $(C_OBJS) python/Sim.o -lboost_python -lpython2.7 -lpugixml
dist:
	cp python/Sim.so lib/

py-all:
	make Sim
	make lib
	make dist

all:
	make main
	make py-all
-include $(DEPFILES)

.PHONY: lib 
