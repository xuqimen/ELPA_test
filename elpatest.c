#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <elpa/elpa.h>

#include "mkl.h"
#include <mkl_scalapack.h>
#include "mkl_lapacke.h"
#include <mkl_cblas.h>

#include <mkl_pblas.h>
//#include <mkl_scalapack.h>
#include <mkl_blacs.h>

extern void   pdlawrite_();
extern void   pdelset_();
extern double pdlamch_();
extern int    indxg2p_();
extern int    indxg2l_();
extern int    numroc_();
extern void   descinit_();
extern void   pdlaset_();
extern double pdlange_();
extern void   pdlacpy_();
extern int    indxg2p_();

extern void   pdgemr2d_();
extern void   pdgemm_();
extern void   pdsygvx_();
extern void   pdgesv_();
extern void   pdgesvd_();

extern void   pzgemr2d_();
extern void   pzgemm_();
extern void   pzhegvx_();

extern void   Cblacs_pinfo( int* mypnum, int* nprocs);
extern void   Cblacs_get( int context, int request, int* value);
extern int    Cblacs_gridinit( int* context, char * order, int np_row, int np_col);
extern void   Cblacs_gridinfo( int context, int*  np_row, int* np_col, int*  my_row, int*  my_col);
extern void   Cblacs_gridexit( int context);
extern void   Cblacs_exit( int error_code);
extern void   Cblacs_gridmap (int *ConTxt, int *usermap, int ldup, int nprow0, int npcol0);
extern int    Csys2blacs_handle(MPI_Comm comm);
extern void   Cfree_blacs_system_handle(int handle);
#define max(x,y) (((x) > (y)) ? (x) : (y))
#define min(x,y) (((x) > (y)) ? (y) : (x))


// #include <cblas.h>
// #include <PBblacs.h>
//#include <src/helpers/scalapack_interfaces.h>
extern void descinit_();
extern int numroc_();
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int n = atoi(argv[1]);
  int rank,size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  int global_num_cols = n;
  int ierr;

  //Use up to a 16 grid
  int naive_width = floor(sqrt(size));
  int sqrt_nprocs_for_eigensolve = (int)fmin(4, naive_width);


  if (0) {
  volatile int gdbcheck = 0;
  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  if (rank == 0) printf("PID %d on %s ready for attach | %i\n", getpid(), hostname, rank);
  fflush(stdout);
    while (0 == gdbcheck)
      sleep(5);
  }
  if (rank == 0) printf("sqrt: %i\n", sqrt_nprocs_for_eigensolve);
  if (rank  < (sqrt_nprocs_for_eigensolve * sqrt_nprocs_for_eigensolve) ) {
    printf("Inside here!\n");


    MPI_Comm eigen_cart_comm;
    int dims[2];
    dims[0] = sqrt_nprocs_for_eigensolve;
    dims[1] = sqrt_nprocs_for_eigensolve;
    int periods[2];
    periods[0] = sqrt_nprocs_for_eigensolve;
    periods[1] = sqrt_nprocs_for_eigensolve;

    if (rank == 0) printf("Create comm\n");
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &eigen_cart_comm);
    int cart_rank;
    int cart_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &cart_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &cart_size);

    int coords[2];
    MPI_Cart_coords(eigen_cart_comm, cart_rank, 2, coords);

    int nblk = ceil((double)global_num_cols/sqrt_nprocs_for_eigensolve);

    const int src_srow = 0;
    const int src_scol = 0;
    const int src_nrow = (rank==0)?global_num_cols:0;
    const int src_ncol = (rank==0)?global_num_cols:0;
    const int req_srow = coords[0] * nblk;
    const int req_scol = coords[1] * nblk;
    const int req_nrow = (int)fmin(nblk, global_num_cols-req_srow);
    const int req_ncol = (int)fmin(nblk, global_num_cols-req_scol);
    double* local_H_s = calloc(req_nrow * req_ncol,sizeof(double));
    double* local_M_s = calloc(req_nrow * req_ncol,sizeof(double));
    double* local_Z_s = calloc(req_nrow * req_ncol,sizeof(double));

    srand(rank);
    //if(coords[0] == coords[1]) {
      for(int i = 0; i < req_nrow; i++) {
        for(int j = 0; j < req_ncol; j++) {
          local_H_s[i*req_nrow + j] = ((double)rand())/RAND_MAX-0.5;
          local_Z_s[i*req_nrow + j] = ((double)rand())/RAND_MAX-0.5;
        }
      }
    //}
    assert(req_nrow == req_ncol);

    if(coords[0] == coords[1]) {
      if(1) {
        cblas_dgemm(CblasColMajor, CblasTrans,CblasNoTrans,req_nrow, req_nrow, req_nrow, 1, local_Z_s, req_nrow, local_Z_s, req_nrow, 0, local_M_s, req_nrow);
      } else 
      {
        for(int j = 0; j < req_ncol; j++) {
          local_M_s[j*req_nrow + j] = fabs(local_Z_s[j*req_nrow + j]);
        }
      }
    }



    Cblacs_pinfo(&cart_rank, &cart_size);
    if (rank == 0) printf("A: %i,B: %i\n", cart_rank,cart_size);

    int blacs_ctx;
    int nprow, npcol, my_prow, my_pcol;
    Cblacs_get(-1,0,&blacs_ctx);
    if (rank == 0) printf("ctx: %i\n", blacs_ctx);
    Cblacs_gridinit(&blacs_ctx, "Col", dims[0], dims[1]);
    Cblacs_gridinfo(blacs_ctx, &nprow, &npcol, &my_prow, &my_pcol);
    if (rank == 0) printf("nprow : %i, npcol: %i, myprow: %i, mypcol: %i, dims: %i, %i\n",
        nprow,npcol,my_prow, my_pcol, dims[0], dims[1]);
    assert(nprow == dims[0]);
    assert(npcol == dims[1]);

    int desc[9];
    int ZERO=0;
    int na_rows = numroc_(&global_num_cols, &nblk, &my_prow, &ZERO, &dims[0]);
    int na_cols = numroc_(&global_num_cols, &nblk, &my_pcol, &ZERO, &dims[1]);
    if (rank == 0) printf("NA ROWS: %i\n", na_rows);
    descinit_(desc, &global_num_cols, &global_num_cols, &nblk, &nblk,&ZERO,&ZERO,&blacs_ctx, &req_nrow, &ierr);
    assert(ierr == 0);
    //assert(my_prow== coords[0]);
    //assert(my_pcol== coords[1]);

    elpa_t handle;
    const int na = global_num_cols;
    const int nev = global_num_cols;
    const int local_nrows = req_nrow;
    const int local_ncols = req_ncol;
    const int process_row = coords[0];
    const int process_col = coords[1];
    if (rank == 0) printf("na: %i, nev: %i, local_nrows: %i, local_ncols: %i, nblk: %i, process_row: %i, process_col: %i\n",
        na, nev, local_nrows, local_ncols, nblk, process_row,process_col);

    if (elpa_init(20171202) != ELPA_OK) {
      printf("Invalid elpa version\n");
      exit(-1);
    } else {
      printf("ABCD\n");
    }
    handle = elpa_allocate(&ierr);
    if (ierr != ELPA_OK) {
      printf("Unable to handle elpa alloc\n");
      exit(-1);
    }

    MPI_Comm mpi_comm_rows;
    ierr = MPI_Comm_split(MPI_COMM_WORLD, process_col, process_row, &mpi_comm_rows);
    assert(ierr==0);
    int mpi_comm_rows_size;
    int mpi_comm_rows_rank;

    MPI_Comm_rank(mpi_comm_rows,&mpi_comm_rows_rank);
    MPI_Comm_size(mpi_comm_rows,&mpi_comm_rows_size);
    if (rank == 0) printf("com rows: %i/%i\n", mpi_comm_rows_rank, mpi_comm_rows_size);

    MPI_Comm mpi_comm_cols;
    ierr = MPI_Comm_split(MPI_COMM_WORLD, process_row, process_col, &mpi_comm_cols);
    assert(ierr==0);
    int mpi_comm_cols_size;
    int mpi_comm_cols_rank;

    MPI_Comm_rank(mpi_comm_cols,&mpi_comm_cols_rank);
    MPI_Comm_size(mpi_comm_cols,&mpi_comm_cols_size);
    if (rank == 0) printf("com cols: %i/%i\n", mpi_comm_cols_rank, mpi_comm_cols_size);

    if (rank == 0) printf("Setting up some base stuff\n");
    elpa_set_integer(handle, "cannon_for_generalized", 0,&ierr);
    assert(ierr==ELPA_OK);
    if (rank == 0) printf("SKIPPING CANNON\n");
    elpa_set_integer(handle, "na", na,&ierr);
    assert(ierr==ELPA_OK);
    elpa_set_integer(handle, "nev", nev,&ierr);
    assert(ierr==ELPA_OK);
    elpa_set_integer(handle, "local_nrows", local_nrows,&ierr);
    assert(ierr==ELPA_OK);
    elpa_set_integer(handle, "local_ncols", local_ncols,&ierr);
    assert(ierr==ELPA_OK);
    elpa_set_integer(handle, "nblk", nblk,&ierr);
    assert(ierr==ELPA_OK);
    elpa_set_integer(handle, "mpi_comm_parent", (int)(MPI_Comm_c2f(MPI_COMM_WORLD)),&ierr);
    assert(ierr==ELPA_OK);
    elpa_set_integer(handle, "mpi_comm_rows", (int)(MPI_Comm_c2f(mpi_comm_rows)),&ierr);
    assert(ierr==ELPA_OK);
    elpa_set_integer(handle, "mpi_comm_cols", (int)(MPI_Comm_c2f(mpi_comm_cols)),&ierr);
    assert(ierr==ELPA_OK);
    elpa_set_integer(handle, "blacs_context", (int) blacs_ctx,&ierr);
    assert(ierr==ELPA_OK);
    //elpa_set_integer(handle, "process_col", process_col,&ierr);
    //assert(ierr==ELPA_OK);
    if (rank == 0) printf("Running setup\n");
    ierr = elpa_setup(handle);
    if (ierr != ELPA_OK) {
      printf("Unable to setup elpa \n");
      exit(-1);
    }


    elpa_set_integer(handle, "solver", ELPA_SOLVER_2STAGE, &ierr);
    if (ierr != ELPA_OK) {
      printf("Unable to setup solver \n");
      exit(-1);
    }
    //elpa_set(handle, "real_kernel", ELPA_1STAGE_REAL_AVX_BLOCK2, &ierr);
    //if (ierr != ELPA_OK) {
    //  printf("Unable to set kernel \n");
    //  exit(-1);
    //}
    elpa_set_integer(handle, "debug", 1, &ierr);
    if (ierr != ELPA_OK) {
      printf("Unable to set debug \n");
      exit(-1);
    }
    int value;
    elpa_get_integer(handle, "solver", &value, &ierr);
    if (rank == 0) printf("Solver is :%d\n", value);

    elpa_get_integer(handle, "num_processes", &value, &ierr);
    if (rank == 0) printf("NP is :%d\n", value);

    elpa_get_integer(handle, "process_id", &value, &ierr);
    if (rank == 0) printf("procid is :%d\n", value);

    elpa_get_integer(handle, "mpi_comm_cols", &value, &ierr);
    if (rank == 0) printf("mpi_comm_cols is :%d\n", value);

    elpa_get_integer(handle, "mpi_comm_rows", &value, &ierr);
    if (rank == 0) printf("mpi_comm_rows is :%d\n", value);

    elpa_get_integer(handle, "process_col", &value, &ierr);
    if (rank == 0) printf("process_col is :%d\n", value);

    elpa_get_integer(handle, "process_row", &value, &ierr);
    if (rank == 0) printf("process_row is :%d\n", value);

/*
    for(int k = 0; k < 4; k++) {
      if(cart_rank == k) {
        printf("Rank %i: \n", k);

        printf("H\n");
        for(int i = 0; i < local_nrows; i++) {
          for(int j = 0; j < local_ncols; j++) {
            printf("%f ", local_H_s[j*local_nrows + i]);
          }
          printf("\n");
        }
        printf("M\n");
        for(int i = 0; i < local_nrows; i++) {
          for(int j = 0; j < local_ncols; j++) {
            printf("%f ", local_M_s[j*local_nrows + i]);
          }
          printf("\n");
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
*/
    double* eigenvectors = malloc(global_num_cols * global_num_cols * sizeof(double));
    double* eigvals = calloc(global_num_cols,sizeof(double));
    if (rank == 0) printf("Starting to run\n");
    //sleep(rank);
    double t1, t2;
    t1 = MPI_Wtime();
    elpa_generalized_eigenvectors_d(handle, local_H_s, local_M_s, eigvals, eigenvectors, 0, &ierr);
    t2 = MPI_Wtime();
    if (rank == 0) printf("Time for ELPA generalized eigensolver: %.3f ms\n", (t2-t1)*1e3);
    //elpa_eigenvectors_d(handle, local_M_s, eigvals, eigenvectors, &ierr);
    if (rank == 0) printf("After Run\n");
    if (ierr != ELPA_OK) {
      printf("Unable to solve \n");
      exit(-1);
    }
/*
    for(int k = 0; k < 4; k++) {
      if(cart_rank == k) {
        for(int i = 0; i < global_num_cols; i++) {
          printf(" (%i): %i: %.15f\n",cart_rank, i, eigvals[i]);
        }

        for(int i = 0; i < local_ncols; i++) {
          for(int j = 0; j < local_ncols; j++) {
            printf("%f ", eigenvectors[i * local_ncols + j]);
          }
          printf("\n");
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
*/

  }
  else {
    printf("BLAH!!!!\n");
  }
  MPI_Finalize();
}
