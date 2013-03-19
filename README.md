CUDA_Floyd_Warshall_
====================

CUDA implementation of the Floyd-Warshall All pairs shortest path graph algorithm(with path reconstruction)

This is a very simple implementation of the Floyd-Warshall all-pairs-shortest-path algorithm written in two versions,
a standard serial CPU version and a CUDA GPU version. Both implementations also calculate and store the full edge 
path and their respective weights from the start vertex to the end vertex(if such a path exists).

UPDATE:

Ran some tests using this code on a K20c GPU and was able to get a 30% speedup over the 680 (using 32 bit ints)

here are the results of a test on a highly connected graph with 10000 vertices;

--------------------------------------------------------------------------------------------------------------------

Success! The GPU Floyd-Warshall result and the CPU Floyd-Warshall results are identical
(both final adjacency matrix and path matrix).

N= 10000 , and the total number of elements(for Adjacency Matrix and Path Matrix) was 100000000 .
Matrices are int full dense format(row major) with a minimum of 25000000 valid directed edges.

The CPU timing for all was 3794.15 seconds, and the GPU timing(including all device memory operations
(allocations,copies etc) ) for all was 198.931 seconds.

The GPU result was 19.0727 faster than the CPU version.

----------------------------------------------------------------------------------------------------------------------


The other tests done in the project folder were using the GTX 680, but in general for this task the Tesla K20c 
is at least 30% faster.

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

This type of dynamic programming algorithm generally does not lend itself as well to the parallel computing model,
but still is able to get a consistent 7 to 13 times speedup over the CPU version(including all host-device 
and device-host  memory allocations and copies for the CUDA version). Also this implementation does seem to scale well,
and so far has tested as generating the same results as the CPU version for all tested data sets.

This algorithm is intended to be used on directed graphs with non-negative edge weights.

The testing generates a adjacency Matrix in row-major form with initial random weights assigned to apx 25% of the edges,
and M[i][i]=0. All other entries are set to 'infinity' to indicate no known path from vertex i to vertex j.

Since no sparse format is used to store the matrix, it seems this algorithm is best suited for highly-connected graphs.
If there is a low-level of connectivity the CUDA version of BFS is better suited for that type of graph.

Included in the project file is a .txt file which orginates from a Wikipedia data set, and is in the form of an edge list.
A function has been created to read in this file as well for testing on a real-world data set. 

Also some initial test results are included which show the speedup using CUDA.

The project was created in Visual Studio 2010 as a 64 bit application.

The CPU used in an Intel I-7 3770 3.5 ghz with 3.9 ghz target, and a single Nvidia GTX 680 2GB GPU.
