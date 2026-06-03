# LUT Sampler
Multi Party Noise Sampling with Lookup Tables. This repository contains the implementation of the paper: [Accelerating Multiparty Noise Generation Using Lookups](https://eprint.iacr.org/2025/805). 

# License notice
Portions of this file are adapted from:
https://github.com/KULeuven-COSIC/maestro/
Copyright © 2024 COSIC-KU Leuven and Concordium AG
Licensed under the MIT License



# Compiling the programm
The main executables are `src/lut_fill.rs` and `src/sampler.rs` containing the code for filling the lookup table and running the MPC sampling respectively. For compilation replace `<bin>` with either `lut_fill` or `sampler`. In case the machine supports it, add `--features='clmul'` to greatly increase verification performance.
```
RUSTFLAGS='-C target-cpu=native' cargo build --release --bin <bin>
```
The compiled files are now in `src/target/release/`. `lut_fill` is a non-interactive algorithm run with a single process, whereas `sampler` expects 3 parties to interact.

# Running the Executables
After compiling with the command above you can run the implementation. For `lut_fill` you directly run the executable from the terminal (with parameters as discussed below in the [Table 1](#tbl-lutfillparams)). For `sampler` a good way to test is to open three terminal windows and run the command as follows. Everytime something is given in `<.>` replace by the parameters according to the  [Table 2](#tbl-samplerparams):

```
./target/release/lut_fill --k <k> --l <l> --ber <ber> --eps <eps> --path <path>
```
For the sampler run the three lines in seperate windows and again replace parameters according to the table:
```
./target/release/sampler --config p1.toml --simd <simd> <--mal-sec> <--network> --rep <rep> --bench <bench>
./target/release/sampler --config p2.toml --simd <simd> <--mal-sec> <--network> --rep <rep> --bench <bench>
./target/release/sampler --config p3.toml --simd <simd> <--mal-sec> <--network> --rep <rep> --bench <bench>
```
### LUT Fill parameter table
The choice of target distribution is made by either selecting eps (=Laplace) or sig (=Gauss).
<a id="tbl-lutfillparams"></a>
|name         | options  | description                                                                        | type             | 
|-------------|----------|------------------------------------------------------------------------------------|------------------|
|`k`          | `[k_i,]` | Up to 3 integers giving the bit width of each dimension                            | list of integers |
|`n`          | positive | Denotes the bound of values in the table. (max of 2^16-1)                          | Integer          |
|`error`      | positive | Denotes the value of λ (Stopping condition of search)                              | Integer          |
|`search`     | set      | Flag enabling full grid search up to the given k                                   | Flag             |
|`l`          | `[0..k]` | Number of _active_ bits in the index. List for grid search                         | list of integers |
|`ber`        | `[1..12]`| negative log of bernoulli bias used in grid search                                 | list of integers |
|`eps`        | positive | p_lap is computed e^-eps                                                           | list of decimals |
|`sig`        | positive | variance is taken as sigma^2=sig                                                   | list of decimals |
|`path`       | path     | if provided, and count of eps and sig is 1 total, prints resulting matrix to path  | String           |
|`bench-info` | set      | if set, displays runtime breakdown                                                 | flag             |
|`v`          | set      | if set, displays intermediate computations                                         | flag             |
|`debug`      | set      | if set, ber and eps are ignored and prints debug table                             | flag             |

### Overview of approximation accuracy
In our paper in Figure 7: "Lowest approximation error achieved for different table sizes and target distributions. We depict $\mathcal{N}_\mathbb{Z}$" in red with $\sigma^2=\frac{1}{\epsilon^2}$ and $DLap$ in blue with $p_{DLap}=e^{-\epsilon}$", we depict the approximation accuracy of A_fill for different privacy parameters and table sizes for the Laplace and Gauss distributions. Here, we explain how to reproduce the results, in later sections we will go into the process of finding a good approximation for a single setting.

To get the approximation results for various Gauss parameters $\sigma^2=\frac{1}{\epsilon^2}$ evaluate the following (the command line option sig, refers to the variance, i.e., $\sigma^2$):
```
./target/release/lut_fill --k 24 --n 65535 --ber 2 3 4 5 6 7 8 9 10 11 12 --sig 1 16 256 4096 65536 1048576 --search  --error 128
```
The precomputed output of this specific command can be found in `benches/accuracy/gauss_grid.txt`. Equivalently, for the same values of epsilon we can get the best approximations of the Laplace distribution via (the parameter p_lap is internally computed via $p_{lap} = e^{\epsilon}$):
```
./target/release/lut_fill --k 24 --n 65535 --ber 2 3 4 5 6 7 8 9 10 11 12 --eps 1 0.25 0.0625 0.015625 0.00390625 0.0009765625 --search  --error 128
```
The result of this call is stored in `benches/accuracy/laplace_grid.txt`. Note that especially the Laplace grid search may run for a while especially for larger tables. Further note that in both output files, we manually trimmed the resulting values as the output is overly precise.

When looking for an approximation for a specific $\epsilon$ merely replace the respecitve value of either `--eps` of `--sig`, depending on the distribution. In the next section, we explain how to generate a table for use in the MPC implementation.

### LUT Fill table generation
We have created and stored all tables relevant to recreate the experiments of our paper. The following guide is to create an approximation table for different parameters, or to understand how we created the tables and verify our claims of reasonable approximation runtime. 

To create a table refer to the output of the grid search and the smallest k that reached sufficient accuracy. Let e.g. sigma = 4 and sigma^2 = 16, we refer to the respective file in the benchmarks to find the computed conservative bound and the LUT size `k`:
```
Generating approximations with type: u16
	With Variance Parameters
        ...
    	"1.6000e1" Truncation error at 63: 2.72234e-54
        ...
```
Now we now that a bound of 255 (the u8 datatype) will be sufficient. Next let us check for the dimension of the LUT:
```
With k: 14 and index probabilities: 2^-2 2^-3 2^-4 2^-5 2^-6 2^-7 Increasing index bias has no further impact
Best Approximations for k: 14 and target: Gaussian with Variance parameter
    ...
    v 1.6000e1: 1 tables with k: 14, l: 10, p: 2^-5,     delta: 1.493e-13 ~ 2^-42, considered range: [0,63]
    ...
```
Given `k=14` and the datatype size `w=8`, we ca run `py optimal_d3.py --w 8` and read the line `k 14: (6,6,2) cost (prep, online, total): (115, 54, 169) bits` to find that the optimal dimensional setup is `6,6,2`, meaning a lookup cube with 2 dimensions of size $2^6$ and one dimension of size $2^4$. 
At the moment the last dimension must fulfil $w\cdot 2^{k_3} \geq 64$ since we pack these values into u64 values. Thus, we have to use $w=16$ at the cost of 40 bits in the evaluation. Create the correct table as follows:
```
./target/release/lut_fill --k 6 6 2 --n 65535 --l 10 --ber 5 --sig 16 --path src/lut_sampler/tables/my_table
```
Then add the new table by adding `pub mod my_table` to the file `src/lut_sampler/tables/mod.rs`. Now the file is accesible from `src/sampler.rs` via `tables::my_table::MyTable` and is either an object of type `Cube` or type `Matrix` depending on the parameter `k` (in our example it will be a cube).

### Sampler parameter table
Note that performance of the scheme does not depend on the concrete table used but only on the bias in the index and the size of the table. At this time, the chosen table has to be hardcoded in `sampler.rs`. For ease of verification, we provide functions that run all tables necessary for the respective tables and figures of our paper. The options are as follows:
- `size` Benchmarks tables with increasing sizes of no particular approximation. These results did not make it into the paper.
- `table4` The table used for results presented in table 4.  
- `table3`The table used for results presented in table 3. 
- `variance` Tables as presented in Figure 8
- `lambda` Tables as presented in Figure 7
- `all` Benchmarks all of the above

<a id="tbl-samplerparams"></a>
|name      | py | options  | description                              | type    | 
|----------|----|----------|------------------------------------------|---------|
| `config` |  - | Path     | Indicates the path to the network config | String  |
|`simd`    |`-s`| 1..16384 | Number of samples evaluated at once      | Integer |
|`rep`     |`-r`| posiitve | Number of repetitions of benchmarks      | integer |
|`mal-sec` |`-m`| set      | if set, everything is verified           | flag    |
|`debug`   |`-d`| set      | if set, protocol output is revealed, composed and checked | flag |
|`network` |`-n`| set      | if set, rep is overwritten to one and network output is printed   | flag |
|`bench`   |Enum| Enum     | selects predefined benchmark suite: (size, table4, table3 variance, lambda, all) | Enum |

To run a given setting adapt the network configuration files and run the three parties individually:
```
./target/release/sampler --config p1.toml --simd 1000 (--mal-sec) (--network) --bench <bench>
./target/release/sampler --config p2.toml --simd 1000 (--mal-sec) (--network) --bench <bench>
./target/release/sampler --config p3.toml --simd 1000 (--mal-sec) (--network) --bench <bench>
```
Given the largest tables the commands above may lead to stackoverflows in some settings. Either remove the largest tables or run: `ulimit -s 262144` to increase the stack size of the current terminal session. (This needs to be run in all terminal windows)

To run all clients on a single machine, we provide the helper pyhton file `test.py` where parameters can be passed after the flags listed in [Table 2](#tbl-samplerparams), the network config will be set to the one present in this repository, the flag `-b` builds the underlying rust files.

### Testing the network settings
All our experiments ran on the localhost. Our reduced network setting was achieved through the `tc` command for RTT=1ms:
```
sudo tc qdisc add dev lo root handle 1: htb default 12 r2q 1000
sudo tc class add dev lo parent 1: classid 1:1 htb rate 1gbit
sudo tc class add dev lo parent 1:1 classid 1:12 htb rate 1gbit
sudo tc qdisc add dev lo parent 1:12 handle 10: netem delay 0.5ms
```
And for RTT=100ms:
```
sudo tc qdisc add dev lo root handle 1: htb default 12 r2q 100
sudo tc class add dev lo parent 1: classid 1:1 htb rate 100mbit
sudo tc class add dev lo parent 1:1 classid 1:12 htb rate 100mbit
sudo tc qdisc add dev lo parent 1:12 handle 10: netem delay 50ms
```
In the end remove any introduced delay by:
`sudo tc qdisc del dev lo root`
