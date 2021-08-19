
# MKLROOT = /opt/intel/compilers_and_libraries_2017.4.196/linux/mkl

ELPA_INC = ${ELPA_ROOT}/include/elpa_openmp-2019.05.002

FFLAGS = 

#CPPFLAGS = -m64 -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -ldl -lrt -O3
#CPPFLAGS += -I${ELPA_INC} -L${ELPA_ROOT}/lib -lm -lelpa_openmp -lrt -O3

LDFLAGS = -L${MKLROOT}/lib/intel64 -L${ELPA_ROOT}/lib

CPPFLAGS = -I${MKLROOT}/include -I${ELPA_INC}

CFLAGS = -std=gnu99 -O3 -fopenmp

# Load ELPA
LDLIBS  = -lelpa_openmp
# Load MKL
LDLIBS += -Wl,--no-as-needed -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -ldl -lrt

FPPFLAGS = 

OBJSC = elpatest.o

override CC=mpicc

LIBBASE = test_elpa

all: ${LIBBASE}

#test_elpa: $(OBJSC)
#	$(CC) -o $@ $^ $(CPPFLAGS)

${LIBBASE}: $(OBJSC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(LIBBASE) $^ $(LDLIBS)

.PHONY: clean
clean:
	rm -f  $(OBJSC) ${LIBBASE}
