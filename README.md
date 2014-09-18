CUDA_Floyd_Warshall_
====================

CUDA implementation of the Floyd-Warshall All pairs shortest path graph algorithm(with path reconstruction)

UPDATE:  Made a new table with times for Tesla GPU. Make sure you run in release mode for full speed!

This is a very simple implementation of the Floyd-Warshall all-pairs-shortest-path algorithm written in two versions,
a standard serial CPU version and a CUDA GPU version. Both implementations also calculate and store the full edge 
path and their respective weights from the start vertex to the end vertex(if such a path exists).

Uses two adjacency matrices, one for path values, one for path reconstruction.

привет Белгорода и Волгограда!

調布の日本の友人が、私のコードを主演してください！

NOTE: no overlocking of GPU, is running at stock 700 Mhz

<b>Running Times CPU vs GPU for Floyd-Warshall APSP with full Path cache:</b>

___________________________

<table>
  <tr>
    <th>Total Vertices</th><th>Size of Adjacency Matrices</th><th>GPU time(s)</th>
  </tr>
  <tr>
    <td> 1,000</td><td> 1,000,000 </td><td> 0.061s </td>
  </tr>
  <tr>
    <td> 2,000</td><td> 4,000,000 </td><td> 0.392s </td>
  </tr>
  <tr>
    <td> 4,000</td><td> 16,000,000 </td><td> 2.8s </td>
  </tr>
  <tr>
    <td> 8,000</td><td> 64,000,000 </td><td> 22.6s </td>
  </tr
  <tr>
 
</table> 
____ 

<b>Tesla K40x initial tests</b>

<table>
<tr>
    <th>Total Vertices</th><th>Size of Adjacency Matrices</th><th>GPU time(s)</th>
  </tr>
<tr>
    <td> 15,000</td><td> 225,000,000 </td><td> 211.4s</td>
  </tr>
  <tr>
    <td> 20,000</td><td> 400,000,000 </td><td> 490.2s</td>
  </tr>

</table>

___


This type of dynamic programming algorithm generally does not lend itself as well to the parallel computing model,
but still is able to get a consistent 37 to 51 times speedup over the CPU version(including all host-device 
and device-host  memory allocations and copies for the CUDA version). Also this implementation does seem to scale well,
and so far has tested as generating the same results as the CPU version for all tested data sets.

This algorithm is intended to be used on directed graphs with non-negative edge weights.

The testing generates a adjacency Matrix in row-major form with initial random weights assigned to apx 25% of the edges,
and M[i][i]=0. All other entries are set to 'infinity' to indicate no known path from vertex i to vertex j.

Since no sparse format is used to store the matrix, it seems this algorithm is best suited for highly-connected graphs.
If there is a low-level of connectivity the CUDA version of BFS is better suited for that type of graph.

The CPU used in an Intel I-7 3770 3.5 ghz with 3.9 ghz target, and a single Nvidia GTX 680 2GB GPU. 

<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-43459430-1', 'github.com');
  ga('send', 'pageview');

</script>

[![githalytics.com alpha](https://cruel-carlota.pagodabox.com/18de473bf04c6f431030e67ad1744eaa "githalytics.com")](http://githalytics.com/OlegKonings/CUDA_Floyd_Warshall_)
