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


void elpa_generalized_solver(
	double *a, int desca[9], double *b, int descb[9],
	int nev, double *ev, double *z, MPI_Comm comm);

void elpa_std_solver(
	double *a, int desca[9], int nev, double *ev, double *z, MPI_Comm comm);

// #include <cblas.h>
// #include <PBblacs.h>
//#include <src/helpers/scalapack_interfaces.h>
extern void descinit_();
extern int numroc_();
int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	int n = atoi(argv[1]);
	int type = 1;
	if (argc == 3) {
		type = atoi(argv[2]); // read type: 0 - standard, 1 - generalized
 	}

	int rank,size;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	int global_num_cols = n;
	int ierr;

	int naive_width = floor(sqrt(size));
	//Use up to a 16 grid
	// int sqrt_nprocs_for_eigensolve = (int)fmin(4, naive_width);
	int sqrt_nprocs_for_eigensolve = naive_width;

	if (0) {
			volatile int gdbcheck = 0;
			char hostname[256];
			gethostname(hostname, sizeof(hostname));
			if (rank == 0) printf("PID %d on %s ready for attach | %i\n", getpid(), hostname, rank);
			fflush(stdout);
			while (0 == gdbcheck) sleep(5);
	}


	if (rank == 0) printf("sqrt(nproc): %i\n", sqrt_nprocs_for_eigensolve);
	if (rank  < (sqrt_nprocs_for_eigensolve * sqrt_nprocs_for_eigensolve) ) {
		int dims[2];
		dims[0] = sqrt_nprocs_for_eigensolve;
		dims[1] = sqrt_nprocs_for_eigensolve;
		int periods[2];
		periods[0] = periods[1] = 1;

		if (rank == 0) printf("Create comm\n");
		MPI_Comm eigen_cart_comm;
		MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &eigen_cart_comm);
		int cart_rank;
		int cart_size;
		MPI_Comm_rank(MPI_COMM_WORLD, &cart_rank);
		MPI_Comm_size(MPI_COMM_WORLD, &cart_size);

		int coords[2];
		MPI_Cart_coords(eigen_cart_comm, cart_rank, 2, coords);

		Cblacs_pinfo(&cart_rank, &cart_size);
		if (rank == 0) printf("cart_rank: %i,cart_size: %i\n", cart_rank,cart_size);

		int blacs_ctx;
		int nprow, npcol, my_prow, my_pcol;
		Cblacs_get(-1,0,&blacs_ctx);
		if (rank == 0) printf("ctx: %i\n", blacs_ctx);
		Cblacs_gridinit(&blacs_ctx, "Row", dims[0], dims[1]);
		Cblacs_gridinfo(blacs_ctx, &nprow, &npcol, &my_prow, &my_pcol);
		if (rank == 0) printf("nprow : %i, npcol: %i, myprow: %i, mypcol: %i, dims: %i, %i\n",
				nprow,npcol,my_prow, my_pcol, dims[0], dims[1]);
		assert(nprow == dims[0]);
		assert(npcol == dims[1]);
		
		//int nblk = ceil((double)global_num_cols/sqrt_nprocs_for_eigensolve-1e12);
		// int nblk = (global_num_cols - 1) / sqrt_nprocs_for_eigensolve + 1;
		int nblk = 64;

		// set up descriptor
		int desc[9];
		int ZERO=0, ONE=1;
		int na_rows = numroc_(&global_num_cols, &nblk, &my_prow, &ZERO, &dims[0]);
		int na_cols = numroc_(&global_num_cols, &nblk, &my_pcol, &ZERO, &dims[1]);
		int lld = max(1,na_rows);
		if (rank == 0) printf("NA ROWS: %i\n", na_rows);
		descinit_(&desc[0], &global_num_cols, &global_num_cols, &nblk, &nblk,&ZERO,&ZERO,&blacs_ctx, &lld, &ierr);
		assert(ierr == 0);
		//assert(my_prow== coords[0]);
		//assert(my_pcol== coords[1]);


		// set up local matrices

		// const int src_srow = 0;
		// const int src_scol = 0;
		// const int src_nrow = (rank==0)?global_num_cols:0; // not used!
		// const int src_ncol = (rank==0)?global_num_cols:0; // not used!
		// const int req_srow = coords[0] * nblk;
		// const int req_scol = coords[1] * nblk;
		// const int req_nrow = (int)fmin(nblk, global_num_cols-req_srow);
		// const int req_ncol = (int)fmin(nblk, global_num_cols-req_scol);
		
		int req_nrow = numroc_(&global_num_cols, &nblk, &my_prow, &ZERO, &dims[0]);
		int req_ncol = numroc_(&global_num_cols, &nblk, &my_pcol, &ZERO, &dims[1]);

		double* local_H_s = calloc(max(req_nrow * req_ncol,1),sizeof(double));
		double* local_M_s = calloc(max(req_nrow * req_ncol,1),sizeof(double));
		double* local_Z_s = calloc(max(req_nrow * req_ncol,1),sizeof(double));
		assert(local_H_s != NULL && local_M_s != NULL && local_Z_s != NULL);

		srand(rank+1);
		// if(coords[0] == coords[1]) {
			for(int j = 0; j < req_ncol; j++) {
				for(int i = 0; i < req_nrow; i++) {
					local_H_s[j*req_nrow + i] = ((double)rand())/RAND_MAX;
					local_Z_s[j*req_nrow + i] = ((double)rand())/RAND_MAX;
				}
			}
		// }

		// for(int k = 0; k < cart_size; k++) {
		// 	if(cart_rank == k && k == 0) {
		// 		printf("rank = %d = (%d,%d), original rand A = \n", rank,my_prow,my_pcol);
		// 		for(int i = 0; i < min(req_nrow,10); i++) {
		// 			for(int j = 0; j < min(req_ncol,10); j++) {
		// 				printf("%f ", local_H_s[j * req_nrow + i]);
		// 			}
		// 			printf("\n");
		// 		}
		// 	}
		// 	MPI_Barrier(MPI_COMM_WORLD);
		// }
		// MPI_Barrier(MPI_COMM_WORLD);

		// if (req_nrow != req_ncol)
				// printf("rank = %d, req_nrow = %d, req_ncol = %d\n", rank, req_nrow, req_ncol);
		//assert(req_nrow == req_ncol);

		// if(coords[0] == coords[1]) {
		//   if(1) {
		//     cblas_dgemm(CblasColMajor, CblasTrans,CblasNoTrans,req_nrow, req_nrow, req_ncol,
		//         1, local_Z_s, req_nrow, local_Z_s, req_nrow, 0, local_M_s, req_nrow);
		//   } else {
		//     for(int j = 0; j < req_ncol; j++) {
		//       local_M_s[j*req_nrow + j] = fabs(local_Z_s[j*req_nrow + j]);
		//     }
		//   }
		// }

		// matrix multiplication M = H'*H, H = Z'*Z
		double alpha = 1.0, beta = 0.0;
		int M = global_num_cols;
		int N = global_num_cols;
		int descH[9], descM[9], descZ[9];
		for (int i = 0; i < 9; i++) {
			descH[i] = descM[i] = descZ[i] = desc[i];
		}
		pdgemm_("T", "N", &N, &N, &M, &alpha, local_H_s, &ONE, &ONE, descH,
				local_H_s, &ONE, &ONE, descH, &beta, local_M_s, &ONE, &ONE, descM);
		pdgemm_("T", "N", &N, &N, &M, &alpha, local_Z_s, &ONE, &ONE, descZ,
				local_Z_s, &ONE, &ONE, descZ, &beta, local_H_s, &ONE, &ONE, descH);

		// for(int k = 0; k < cart_size; k++) {
		// 	if(cart_rank == k && k == 0) {
		// 		printf("rank = %d = (%d,%d), H = \n", rank,my_prow,my_pcol);
		// 		for(int i = 0; i < min(req_nrow,10); i++) {
		// 			for(int j = 0; j < min(req_ncol,10); j++) {
		// 				printf("%f ", local_H_s[j * req_nrow + i]);
		// 			}
		// 			printf("\n");
		// 		}
		// 	}
		// 	MPI_Barrier(MPI_COMM_WORLD);
		// }
		// MPI_Barrier(MPI_COMM_WORLD);

		// for(int k = 0; k < cart_size; k++) {
		// 	if(cart_rank == k && k == 0) {
		// 		printf("rank = %d = (%d,%d), M = \n", rank,my_prow,my_pcol);
		// 		for(int i = 0; i < min(req_nrow,10); i++) {
		// 			for(int j = 0; j < min(req_ncol,10); j++) {
		// 				printf("%f ", local_M_s[j * req_nrow + i]);
		// 			}
		// 			printf("\n");
		// 		}
		// 	}
		// 	MPI_Barrier(MPI_COMM_WORLD);
		// }
		// MPI_Barrier(MPI_COMM_WORLD);

		// elpa_generalized_solver(
		// 	double *a, int desca[9], double *b, int descb[9],
		// 	int nev, double *ev, double *z, MPI_Comm comm);
		
		// create a copy of H and M, ELPA seems to change both H and M within the solver!
		double *A, *B;
		A = calloc(max(req_nrow * req_ncol,1),sizeof(double));
		B = calloc(max(req_nrow * req_ncol,1),sizeof(double));
		assert(A != NULL && B != NULL);
		for (int i = 0; i < req_nrow * req_ncol; i++) {
			A[i] = local_H_s[i];
			B[i] = local_M_s[i];
		}

		// elpa_generalized_eigenvectors_d(handle, local_H_s, local_M_s, eigvals, eigenvectors, 0, &ierr);
		const int nev = global_num_cols;
		double* eigvec = malloc(max(req_nrow * req_ncol,1) * sizeof(double));
		double* lambda = calloc(global_num_cols,sizeof(double));
		assert(eigvec != NULL && lambda != NULL);

		double t1, t2;
		// type: 0 - standard eigenproblem, 1 - generalized eigenproblem
		if (type == 1) {
			t1 = MPI_Wtime();
			elpa_generalized_solver(
				A, descH, B, descM, nev, lambda, eigvec, MPI_COMM_WORLD);
			t2 = MPI_Wtime();
			if (rank == 0) printf("Time for ELPA generalized eigensolver: %.3f ms\n", (t2-t1)*1e3);
		} else if (type == 0) {
			t1 = MPI_Wtime();
			elpa_std_solver(
				A, descH, nev, lambda, eigvec, MPI_COMM_WORLD);
			t2 = MPI_Wtime();
			if (rank == 0) printf("Time for ELPA standard eigensolver: %.3f ms\n", (t2-t1)*1e3);
		}
// #define SOLVE_FLAG
#ifdef SOLVE_FLAG
{
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
			exit(1);
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
		// elpa_set_integer(handle, "mpi_comm_rows", (int)(MPI_Comm_c2f(mpi_comm_rows)),&ierr);
		// assert(ierr==ELPA_OK);
		// elpa_set_integer(handle, "mpi_comm_cols", (int)(MPI_Comm_c2f(mpi_comm_cols)),&ierr);
		// assert(ierr==ELPA_OK);
		elpa_set(handle, "process_row", my_prow, &ierr);                             // row coordinate of MPI process
		elpa_set(handle, "process_col", my_pcol, &ierr);                             // column coordinate of MPI process

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

		// elpa_get_integer(handle, "mpi_comm_cols", &value, &ierr);
		// if (rank == 0) printf("mpi_comm_cols is :%d\n", value);

		// elpa_get_integer(handle, "mpi_comm_rows", &value, &ierr);
		// if (rank == 0) printf("mpi_comm_rows is :%d\n", value);

		elpa_get_integer(handle, "process_col", &value, &ierr);
		if (rank == 0) printf("process_col is :%d\n", value);

		elpa_get_integer(handle, "process_row", &value, &ierr);
		if (rank == 0) printf("process_row is :%d\n", value);

		// double* eigenvectors = malloc(global_num_cols * global_num_cols * sizeof(double));
		double* eigenvectors = malloc(max(req_nrow * req_ncol,1) * sizeof(double));
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

		// for(int k = 0; k < 4; k++) {
		// 	if(cart_rank == k && k == 0) {
		// 		for(int i = 0; i < min(global_num_cols,10); i++) {
		// 			printf(" (%i): %i: %.15f\n",cart_rank, i, eigvals[i]);
		// 		}
		// 	}
		// 	MPI_Barrier(MPI_COMM_WORLD);
		// }

		// for(int k = 0; k < cart_size; k++) {
		// 	if(cart_rank == k) {
		// 		printf("rank = %d = (%d,%d), V = \n", rank,my_prow,my_pcol);
		// 		for(int i = 0; i < req_nrow; i++) {
		// 			for(int j = 0; j < req_ncol; j++) {
		// 				printf("%8.4f ", eigenvectors[j * req_nrow + i]);
		// 			}
		// 			printf("\n");
		// 		}
		// 	}
		// 	MPI_Barrier(MPI_COMM_WORLD);
		// }
		
		double err_eigval = 0.0;
		for (int i = 0; i < nev; i++) {
			err_eigval = min(err_eigval, fabs(lambda[i] - eigvals[i]));
		}
		printf("err_eigval = %.3e\n", err_eigval);

		free(eigenvectors);
		free(eigvals);
		elpa_deallocate(handle,&ierr);
		elpa_uninit(&ierr);
}
#endif
		free(A);
		free(B);
		free(eigvec);
		free(lambda);
		free(local_H_s);
		free(local_M_s);
		free(local_Z_s);
	}
	else {
		printf("BLAH!!!!\n");
	}
	MPI_Finalize();
}


void elpa_generalized_solver(
	double *a, int desca[9], double *b, int descb[9],
	int nev, double *ev, double *z, MPI_Comm comm)
{
	int ictxt = desca[1];
	int nprow, npcol, my_prow, my_pcol;
	Cblacs_gridinfo(ictxt, &nprow, &npcol, &my_prow, &my_pcol);
	int na = desca[2];
	int nblk = desca[4];
	int ZERO = 0;
	int na_rows = numroc_(&na, &nblk, &my_prow, &ZERO, &nprow);
	int na_cols = numroc_(&na, &nblk, &my_pcol, &ZERO, &npcol);

	elpa_t handle;
	int ierr;

	if (elpa_init(20171202) != ELPA_OK) {
		printf("Invalid elpa version\n");
		exit(1);
	}

	handle = elpa_allocate(&ierr);
	if (ierr != ELPA_OK) {
		printf("Unable to handle elpa alloc\n");
		exit(-1);
	}

	// MPI_Comm mpi_comm_rows;
	// ierr = MPI_Comm_split(comm, process_col, process_row, &mpi_comm_rows);
	// assert(ierr==0);
	// int mpi_comm_rows_size;
	// int mpi_comm_rows_rank;
	// MPI_Comm_rank(mpi_comm_rows,&mpi_comm_rows_rank);
	// MPI_Comm_size(mpi_comm_rows,&mpi_comm_rows_size);

	// MPI_Comm mpi_comm_cols;
	// ierr = MPI_Comm_split(comm, process_row, process_col, &mpi_comm_cols);
	// assert(ierr==0);
	// int mpi_comm_cols_size;
	// int mpi_comm_cols_rank;
	// MPI_Comm_rank(mpi_comm_cols,&mpi_comm_cols_rank);
	// MPI_Comm_size(mpi_comm_cols,&mpi_comm_cols_size);
	
	elpa_set_integer(handle, "cannon_for_generalized", 0,&ierr);
	assert(ierr==ELPA_OK);
	elpa_set_integer(handle, "na", na,&ierr);
	assert(ierr==ELPA_OK);
	elpa_set_integer(handle, "nev", nev,&ierr);
	assert(ierr==ELPA_OK);
	elpa_set_integer(handle, "local_nrows", na_rows,&ierr);
	assert(ierr==ELPA_OK);
	elpa_set_integer(handle, "local_ncols", na_cols,&ierr);
	assert(ierr==ELPA_OK);
	elpa_set_integer(handle, "nblk", nblk,&ierr);
	assert(ierr==ELPA_OK);
	elpa_set_integer(handle, "mpi_comm_parent", (int)(MPI_Comm_c2f(MPI_COMM_WORLD)),&ierr);
	assert(ierr==ELPA_OK);
	// elpa_set_integer(handle, "mpi_comm_rows", (int)(MPI_Comm_c2f(mpi_comm_rows)),&ierr);
	// assert(ierr==ELPA_OK);
	// elpa_set_integer(handle, "mpi_comm_cols", (int)(MPI_Comm_c2f(mpi_comm_cols)),&ierr);
	// assert(ierr==ELPA_OK);
	elpa_set(handle, "process_row", my_prow, &ierr);                             // row coordinate of MPI process
	elpa_set(handle, "process_col", my_pcol, &ierr);                             // column coordinate of MPI process

	elpa_set_integer(handle, "blacs_context", (int) ictxt,&ierr);
	assert(ierr==ELPA_OK);
	//elpa_set_integer(handle, "process_col", process_col,&ierr);
	//assert(ierr==ELPA_OK);

	ierr = elpa_setup(handle);

	elpa_set_integer(handle, "solver", ELPA_SOLVER_2STAGE, &ierr);
	//elpa_set(handle, "real_kernel", ELPA_1STAGE_REAL_AVX_BLOCK2, &ierr);
	elpa_set_integer(handle, "debug", 1, &ierr);

	int value;
	elpa_get_integer(handle, "solver", &value, &ierr);
	elpa_get_integer(handle, "num_processes", &value, &ierr);
	elpa_get_integer(handle, "process_id", &value, &ierr);
	// elpa_get_integer(handle, "mpi_comm_cols", &value, &ierr);
	// if (rank == 0) printf("mpi_comm_cols is :%d\n", value);
	// elpa_get_integer(handle, "mpi_comm_rows", &value, &ierr);
	// if (rank == 0) printf("mpi_comm_rows is :%d\n", value);
	elpa_get_integer(handle, "process_col", &value, &ierr);
	elpa_get_integer(handle, "process_row", &value, &ierr);

	elpa_generalized_eigenvectors_d(handle, a, b, ev, z, 0, &ierr);
	if (ierr != ELPA_OK) {
		printf("Unable to solve \n");
		exit(-1);
	}

	elpa_deallocate(handle,&ierr);
	elpa_uninit(&ierr);
}


void elpa_std_solver(
	double *a, int desca[9], int nev, double *ev, double *z, MPI_Comm comm)
{
	int ictxt = desca[1];
	int nprow, npcol, my_prow, my_pcol;
	Cblacs_gridinfo(ictxt, &nprow, &npcol, &my_prow, &my_pcol);
	int na = desca[2];
	int nblk = desca[4];
	int ZERO = 0;
	int na_rows = numroc_(&na, &nblk, &my_prow, &ZERO, &nprow);
	int na_cols = numroc_(&na, &nblk, &my_pcol, &ZERO, &npcol);

	elpa_t handle;
	int error;

	if (elpa_init(20171201) != ELPA_OK) {                          // put here the API version that you are using
		fprintf(stderr, "Error: ELPA API version not supported");
		exit(1);
	}

	handle = elpa_allocate(&error);
	if (error != ELPA_OK) {
		/* react on the error code */
		/* we urge the user to always check the error codes of all ELPA functions */
	}


	/* Set parameters the matrix and it's MPI distribution */
	elpa_set(handle, "na", na, &error);                                           // size of the na x na matrix
	elpa_set(handle, "nev", nev, &error);                                         // number of eigenvectors that should be computed ( 1<= nev <= na)
	elpa_set(handle, "local_nrows", na_rows, &error);                             // number of local rows of the distributed matrix on this MPI task 
	elpa_set(handle, "local_ncols", na_cols, &error);                             // number of local columns of the distributed matrix on this MPI task
	elpa_set(handle, "nblk", nblk, &error);                                       // size of the BLACS block cyclic distribution
	elpa_set(handle, "mpi_comm_parent", MPI_Comm_c2f(comm), &error);    // the global MPI communicator
	elpa_set(handle, "process_row", my_prow, &error);                             // row coordinate of MPI process
	elpa_set(handle, "process_col", my_pcol, &error);                             // column coordinate of MPI process

	/* Setup */
	error = elpa_setup(handle);

	/* if desired, set any number of tunable run-time options */
	/* look at the list of possible options as detailed later in
		USERS_GUIDE.md */

	elpa_set(handle, "solver", ELPA_SOLVER_2STAGE, &error);

	// set the AVX BLOCK2 kernel, otherwise ELPA_2STAGE_REAL_DEFAULT will
	// be used
	elpa_set(handle, "real_kernel", ELPA_2STAGE_REAL_AVX_BLOCK2, &error);

	/* use method solve to solve the eigenvalue problem */
	/* other possible methods are desribed in USERS_GUIDE.md */
	elpa_eigenvectors(handle, a, ev, z, &error);

	/* cleanup */
	elpa_deallocate(handle, &error);
	elpa_uninit(&error);
}
