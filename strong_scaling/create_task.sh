#!/bin/bash

# use seq -f "%.2f" <start> <step> <end> to generate floating point sequence 
#for i in 11 12 14 15 18 22 27 36 54
#for h in `seq -f "%.2f" 0.1 .05 0.9`

#for i in `seq -f "%.0f" 50 10 300`
#for i in 1 4 9 16 25 36 64 81 100 196 400 625 784
#for i in 25 36 64 81 100 196 400 625 784
for i in 400 625 784
do
    echo $i
    cp -r template/ np_$i
    nnode=`python3 -c "import math; print(int(math.ceil(${i}/24.0)))"` # find number of nodes
    sed -i "s|__nnode__|${nnode}|g" np_$i/run_hive.pbs
    sed -i "s|nproc=24|nproc=$i|g" np_$i/run_hive.pbs
    # NP=`python3 -c "import math; n=math.sqrt(${i}); n=math.floor(n); print(n*n)"`
    # echo $NP
    # sed -i "s|NSTATES: 1|NSTATES: $i|g" np_$i/sprc-calc.inpt 
done



