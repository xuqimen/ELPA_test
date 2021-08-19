
# MKLROOT = /opt/intel/compilers_and_libraries_2017.4.196/linux/mkl

ELPA_INC = ${ELPA_ROOT}/include/elpa_openmp-2019.05.002

FFLAGS = 

CPPFLAGS = -m64 -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -ldl -lrt -O3
CPPFLAGS += -I${ELPA_INC} -L${ELPA_ROOT}/lib -lm -lelpa_openmp -lrt -O3


FPPFLAGS = 

OBJSC = elpatest.o

override CC=mpicc

all: test_eig

test_eig: $(OBJSC)
	$(CC) -o $@ $^ $(CPPFLAGS)

.PHONY: clean
clean:
	rm -f  $(OBJSC) test_eig
