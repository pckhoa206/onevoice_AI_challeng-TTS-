CXX ?= clang++
CXXFLAGS = -O3 -std=c++17 -fPIC -Iinclude -Isrc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    TARGET = libQnnSupertonicTokenizer.dylib
    LDFLAGS = -dynamiclib -Wl,-undefined,dynamic_lookup
else
    TARGET = libQnnSupertonicTokenizer.so
    LDFLAGS = -shared
endif

SRCS = src/indexer_table.cpp \
       src/supertonic_tokenizer_kernel.cpp \
       src/QnnSupertonicTokenizerPackage.cpp

OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

# Hexagon cross-compilation target (requires Hexagon SDK)
hexagon:
	hexagon-clang++ -mv75 -O3 -fPIC -shared -Iinclude -Isrc $(SRCS) -o libQnnSupertonicTokenizer_v75.so

.PHONY: all clean hexagon
