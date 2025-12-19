#---------------------------------------------------------------------
# Makefile
#---------------------------------------------------------------------

# 1. (Source and Object Definitions)
#---------------------------------------------------------------------
SRC = Base58.cpp IntGroup.cpp main.cpp Random.cpp \
      Timer.cpp Int.cpp IntMod.cpp Point.cpp SECP256K1.cpp \
      Vanity.cpp GPU/GPUGenerate.cpp hash/ripemd160.cpp \
      hash/sha256.cpp hash/sha512.cpp hash/ripemd160_sse.cpp \
      hash/sha256_sse.cpp Bech32.cpp Wildcard.cpp

OBJDIR = obj

OBJET = $(addprefix $(OBJDIR)/, \
        Base58.o IntGroup.o main.o Random.o Timer.o Int.o \
        IntMod.o Point.o SECP256K1.o Vanity.o GPU/GPUGenerate.o \
        hash/ripemd160.o hash/sha256.o hash/sha512.o \
        hash/ripemd160_sse.o hash/sha256_sse.o \
        GPU/GPUEngine.o Bech32.o Wildcard.o)

# 2. (Compilers and Tools)
#---------------------------------------------------------------------
CXX        = g++
CUDA       = /usr/local/cuda
CXXCUDA    = /usr/bin/g++
NVCC       = $(CUDA)/bin/nvcc

# 3. (Compilation and Linker Flags)
#---------------------------------------------------------------------

CUDA_ARCH = -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_120,code=sm_120

# --- CPU  ---
ifdef debug
CXXFLAGS   = -static -msse4.1 -Wno-write-strings -g -I. -I$(CUDA)/include
else

CXXFLAGS   = -static -msse4.1 -Wno-write-strings -O3 -march=native -I. -I$(CUDA)/include
endif

# --- GPU NVCC  ---
# Debug 
NVCC_FLAGS_DEBUG = \
	-G \
	-g \
	--compiler-options -fPIC \
	-ccbin $(CXXCUDA) \
	-m64 \
	-I$(CUDA)/include \
	$(CUDA_ARCH)

# Release 
NVCC_FLAGS_RELEASE = \
	-O3 \
	--use_fast_math \
	--fmad=true \
	-maxrregcount=0 \
	--ptxas-options=--allow-expensive-optimizations=true \
	--resource-usage \
	--compiler-options -fPIC \
	-ccbin $(CXXCUDA) \
	-m64 \
	-I$(CUDA)/include \
	$(CUDA_ARCH)
	# ------
	--ftz=true
	--prec-div=false
	--prec-sqrt=false

# ------
LFLAGS     = -lpthread -L$(CUDA)/lib64 -lcudart

# 4. (Build Rules)
#---------------------------------------------------------------------

# ------
all: kk

kk: $(OBJET)
	@echo "==> Linking executable: kk..."
	$(CXX) $(OBJET) $(LFLAGS) -o kk
	@echo "==> Build finished: ./kk"

# --- GPU Kernel ---
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
ifdef debug
	@echo "==> Compiling GPU Kernel (Debug): $<"
	$(NVCC) $(NVCC_FLAGS_DEBUG) --compile -o $@ -c $<
else
	@echo "==> Compiling GPU Kernel (Release): $<"
	$(NVCC) $(NVCC_FLAGS_RELEASE) --compile -o $@ -c $<
endif

# --- C++ ---
$(OBJDIR)/%.o : %.cpp
	@echo "==> Compiling C++ Source: $<"
	$(CXX) $(CXXFLAGS) -o $@ -c $<

# 5. (Directory Creation & Clean Rules)
#---------------------------------------------------------------------
$(OBJET): | $(OBJDIR) $(OBJDIR)/GPU $(OBJDIR)/hash

$(OBJDIR):
	@mkdir -p $(OBJDIR)

$(OBJDIR)/GPU:
	@mkdir -p $(OBJDIR)/GPU

$(OBJDIR)/hash:
	@mkdir -p $(OBJDIR)/hash

clean:
	@echo "==> Cleaning project..."
	@rm -rf $(OBJDIR) kk
	@echo "==> Done."

.PHONY: all clean
