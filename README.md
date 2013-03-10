CUDA_Floyd_Warshall_
====================

CUDA implementation of the Floyd-Warshall All pairs shortest path graph algorithm(with path reconstruction)

This is a very simple implementation of the Floyd-Warshall all-pairs-shortest-path algorithm written in two versions,
a standard serial CPU version and a CUDA GPU version. Both implementations also calculate and store the full edge 
path and their respective weights from the start vertex to the end vertex(if such a path exists).

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
