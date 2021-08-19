#### Usage
- Load ELPA by `module load elpa'
- Compile the code by `make clean; make`
- run a test to solve an eigenvalue problem A*x = lambda*B*x of size `N x N` with `p` processes by
```
mpirun -np p ./test_elpa N
```
Note that when p is not a square number, the program only uses a subset of processors (largest square number) to run the test.
