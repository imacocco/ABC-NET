EXE = MCMC.x

LFLAGS = -fPIC -lgsl -lgslcblas -lm
CXXFLAGS = -I include -g -std=c++20 -O3 -Wall -Wextra -fPIC
CXX = c++

SRC = MCMC.cpp src/utils.cpp src/actions.cpp

OBJ = ${SRC:.cpp=.o}

LIB = include/func.hpp include/timer.hpp
all: $(EXE) 

%.o: %.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

$(EXE): $(OBJ)
	$(CXX) $^ -o $(EXE) $(LFLAGS)

src/utils.o: $(LIB)

clean:
	rm -rf src/*.o $(EXE) */*~ *~ latex html *.o

.PHONY: clean all documentation
