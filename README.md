# LUT Sampler
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20530104.svg)](https://doi.org/10.5281/zenodo.20530104)

Multi-Party Noise Sampling with Lookup Tables. This repository contains the implementation of the paper [Accelerating Multiparty Noise Generation Using Lookups](https://eprint.iacr.org/2025/805).

## 1. License notice
Portions of this repository are adapted from:
https://github.com/KULeuven-COSIC/maestro/
Copyright © 2024 COSIC-KU Leuven and Concordium AG
Licensed under the MIT License

## 2. Prerequisites
- A recent stable [Rust](https://rustup.rs) toolchain (edition 2021), including `cargo`.
- Python 3 for the helper scripts. `test.py` and `optimal_d3.py` only use the standard library; `plot_histogram.py` additionally requires `matplotlib`.
- The three parties communicate over TLS. Self-signed certificates for a localhost setup are provided in `keys/` and referenced by the network configuration files `p1.toml`, `p2.toml` and `p3.toml`. For a distributed setup, adapt the addresses and ports and provide your own certificates.
- (Optional) The network emulation described [below](#testing-the-network-settings) uses the Linux `tc` tool.

## 3. Compiling the program
The main executables are `src/lut_fill.rs` and `src/sampler.rs`, containing the code for filling the lookup table and running the MPC sampling respectively. For compilation, replace `<bin>` with either `lut_fill` or `sampler`.
```
RUSTFLAGS='-C target-cpu=native' cargo build --release --bin <bin>
```
The compiled binaries are now in `target/release/`. `lut_fill` is a non-interactive algorithm run with a single process, whereas `sampler` expects 3 parties to interact.

The feature `clmul` (enabled by default) uses carry-less multiplication instructions to greatly increase verification performance. On CPUs without support for these instructions, disable it by adding `--no-default-features` to the build command.

## 4. Running the executables
After compiling with the command above you can run the implementation. `lut_fill` is run directly from the terminal, with parameters as described in [Table 1](#tbl-lutfillparams). 

```
./target/release/lut_fill --k <k> --n <n> --l <l> --ber <ber> --eps <eps> --error <error> --path <path>
```
For `sampler`, a good way to test is to open three terminal windows and run one of the commands below in each. Every time something is given in `<.>`, replace it with a parameter according to [Table 2](#tbl-samplerparams):
```
./target/release/sampler --config p1.toml --simd <simd> <--mal-sec> <--network> --rep <rep> --bench <bench>
./target/release/sampler --config p2.toml --simd <simd> <--mal-sec> <--network> --rep <rep> --bench <bench>
./target/release/sampler --config p3.toml --simd <simd> <--mal-sec> <--network> --rep <rep> --bench <bench>
```
### 4.1 LUT Fill parameter table
The choice of target distribution is made by selecting either eps (=Laplace) or sig (=Gauss).
<a id="tbl-lutfillparams"></a>
|name         | options  | description                                                                        | type             |
|-------------|----------|------------------------------------------------------------------------------------|------------------|
|`k`          | `[k_i,]` | Up to 3 integers giving the bit width of each dimension                            | list of integers |
|`n`          | positive | Denotes the bound of values in the table (max of 2^16-1)                           | integer          |
|`error`      | positive | Denotes the value of λ (stopping condition of the search)                          | integer          |
|`search`     | set      | Flag enabling full grid search up to the given k                                   | flag             |
|`l`          | `[0..k]` | Number of _active_ bits in the index. List for grid search                         | list of integers |
|`ber`        | `[1..12]`| Negative log of the Bernoulli bias used in the grid search                         | list of integers |
|`eps`        | positive | p_lap is computed as $e^{-\epsilon}$                                               | list of decimals |
|`sig`        | positive | The variance is taken as $\sigma^2 \gets sig$                                      | list of decimals |
|`path`       | path     | If provided, and the count of eps and sig is 1 in total, prints the resulting matrix to path | string |
|`bench-info` | set      | If set, displays a runtime breakdown                                               | flag             |
|`v`          | set      | If set, displays intermediate computations                                         | flag             |
|`debug`      | set      | If set, ber and eps are ignored and a debug table is printed                       | flag             |

### 4.2 Overview of approximation accuracy
In our paper in Figure 6: "Lowest approximation error achieved for different table sizes and target distributions. We depict $\mathcal{N}_\mathbb{Z}$ in red with $\sigma^2=\frac{1}{\epsilon^2}$ and $DLap$ in blue with $p_{DLap}=e^{-\epsilon}$", we depict the approximation accuracy of A_fill for different privacy parameters and table sizes for the Laplace and Gauss distributions. Here, we explain how to reproduce the results; in later sections we go into the process of finding a good approximation for a single setting.

To get the approximation results for various Gauss parameters $\sigma^2=\frac{1}{\epsilon^2}$, evaluate the following (the command-line option `sig` refers to the variance, i.e., $\sigma^2$):
```
./target/release/lut_fill --k 24 --n 65535 --ber 2 3 4 5 6 7 8 9 10 11 12 --sig 1 16 256 4096 65536 1048576 --search  --error 128
```
The precomputed output of this specific command can be found in `benches/accuracy/gauss_grid.txt`. Equivalently, for the same values of epsilon we can get the best approximations of the Laplace distribution via (the parameter p_lap is internally computed as $p_{lap} = e^{-\epsilon}$):
```
./target/release/lut_fill --k 24 --n 65535 --ber 2 3 4 5 6 7 8 9 10 11 12 --eps 1 0.25 0.0625 0.015625 0.00390625 0.0009765625 --search  --error 128
```
The result of this call is stored in `benches/accuracy/laplace_grid.txt`. Note that the Laplace grid search in particular may run for a while, especially for larger tables. Further note that in both output files, we manually trimmed the resulting values as the output is overly precise.

When looking for an approximation for a specific $\epsilon$, simply replace the respective value of either `--eps` or `--sig`, depending on the distribution. In the next section, we explain how to generate a table for use in the MPC implementation.

### 4.3 LUT Fill table generation
We have created and stored all tables relevant to recreate the experiments of our paper. The following guide shows how to create an approximation table for different parameters, or to understand how we created the tables and to verify our claims of reasonable approximation runtime.

To create a table, refer to the output of the grid search and the smallest k that reached sufficient accuracy. Let e.g. sigma = 4 and sigma^2 = 16; we refer to the respective file in the benchmarks to find the computed conservative bound and the LUT size `k`:
```
Generating approximations with type: u16
	With Variance Parameters
        ...
    	"1.6000e1" Truncation error at 63: 2.72234e-54
        ...
```
Now we know that a bound of 255 (the u8 datatype) will be sufficient. Next, let us check the dimensions of the LUT:
```
With k: 14 and index probabilities: 2^-2 2^-3 2^-4 2^-5 2^-6 2^-7 Increasing index bias has no further impact
Best Approximations for k: 14 and target: Gaussian with Variance parameter
    ...
    v 1.6000e1: 1 tables with k: 14, l: 10, p: 2^-5,     delta: 1.493e-13 ~ 2^-42, considered range: [0,63]
    ...
```
Given `k=14` and the datatype size `w=8`, we can run `python3 optimal_d3.py --w 8` and read the line `k 14: (6,6,2) cost (prep, online, total): (115, 54, 169) bits` to find that the optimal dimensional setup is `6,6,2`, meaning a lookup cube with 2 dimensions of size $2^6$ and one dimension of size $2^2$.
At the moment, the last dimension must satisfy $w\cdot 2^{k_3} \geq 64$ since we pack these values into u64 values. Thus, we have to use $w=16$ at the cost of 40 bits in the evaluation. Create the correct table as follows:
```
./target/release/lut_fill --k 6 6 2 --n 65535 --l 10 --ber 5 --sig 16 --path src/lut_sampler/tables/my_table
```
Then add the new table by adding `pub mod my_table` to the file `src/lut_sampler/tables/mod.rs`. Now the table is accessible from `src/sampler.rs` via `tables::my_table::MyTable` and is either an object of type `Cube` or type `Matrix`, depending on the parameter `k` (in our example it will be a cube).

### 4.4 Sampler parameter table
Note that the performance of the scheme does not depend on the concrete table used, but only on the bias in the index and the size of the table. At this time, the chosen table has to be hardcoded in `sampler.rs`. For ease of verification, we provide benchmark suites that run all tables necessary for the respective tables and figures of our paper. The options are as follows:
- `size` Benchmarks tables with increasing sizes of no particular approximation. These results did not make it into the paper.
- `table4` The tables used for the results presented in Table 4.
- `table3` The tables used for the results presented in Table 3.
- `variance` Tables as presented in Figure 8.
- `lambda` Tables as presented in Figure 7.
- `all` Benchmarks all of the above.

<a id="tbl-samplerparams"></a>
|name      | options  | description                                                        | type    |
|----------|----------|--------------------------------------------------------------------|---------|
|`config`  | path     | Indicates the path to the network config                           | string  |
|`simd`    | 1..16384 | Number of samples evaluated at once                                | integer |
|`rep`     | positive | Number of repetitions of the benchmarks                            | integer |
|`mal-sec` | set      | If set, everything is verified                                     | flag    |
|`debug`   | set      | If set, the protocol output is revealed, composed and checked      | flag    |
|`network` | set      | If set, rep is overwritten to one and the network cost is printed  | flag    |
|`bench`   | enum     | Selects a predefined benchmark suite: (size, table4, table3, variance, lambda, all) | enum |

To run a given setting, adapt the network configuration files and run the three parties individually:
```
./target/release/sampler --config p1.toml --simd 1000 (--mal-sec) (--network) --bench <bench>
./target/release/sampler --config p2.toml --simd 1000 (--mal-sec) (--network) --bench <bench>
./target/release/sampler --config p3.toml --simd 1000 (--mal-sec) (--network) --bench <bench>
```
Given the largest tables, the commands above may lead to stack overflows in some settings. Either remove the largest tables or run `ulimit -s 262144` to increase the stack size of the current terminal session. (This needs to be run in all terminal windows.)

### 4.5 Running all parties on one machine
To run all three parties on a single machine, we provide the helper Python script `test.py`. It launches the three sampler processes with the network configuration present in this repository (only the output of party 1 is shown) and automatically raises the stack limit, so the `ulimit` workaround above is not needed. The benchmark suite is passed as a positional argument; all other parameters are passed via the following flags:

<a id="tbl-testpyparams"></a>
|flag | long             | description                                                        | type    |
|-----|------------------|---------------------------------------------------------------------|---------|
|`-s` | `--simd`         | Number of samples evaluated at once (required)                     | integer |
|`-r` | `--repetitions`  | Number of repetitions of the benchmarks                            | integer |
|`-m` | `--mal-sec`      | If set, everything is verified                                     | flag    |
|`-n` | `--network`      | If set, rep is overwritten to one and the network cost is printed  | flag    |
|`-d` | `--debug`        | If set, debug binaries are used and the protocol output is revealed, composed and checked | flag |
|`-t` | `--trace`        | If set, trace logging is enabled (`RUST_LOG=lut_sampler=trace`)    | flag    |
|`-b` | `--build`        | If set, the sampler binary is (re)built before running             | flag    |
|`bench` | (positional)  | Benchmark suite to run: (size, table4, table3, variance, lambda, all) | enum |

For example, to build and run the Table 3 benchmarks with 1000 parallel samples:
```
python3 test.py -b -s 1000 table3
```

### 4.6 Testing the network settings
All our experiments ran on localhost. Our reduced network setting was achieved through the `tc` command, for RTT=1ms:
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
In the end, remove any introduced delay by:
`sudo tc qdisc del dev lo root`

## 5. Reference Performance
In Table 4 and Figure 8 and 7 of our paper we use data obtained by evaluating related works. The implementations can be found at:
- Franzese et al.: "Secure Noise Sampling for Differentially Private Collaborative Learning" https://github.com/cleverhans-lab/Secure_Noise_Sampling_DP_CL
- Fu and Wang: "Benchmarking Secure Sampling Protocols for Differential Privacy" https://github.com/yuchengxj/Secure-sampling-benchmark
- Meisingseth et al.: "Practical Two-party Computational Differential Privacy with Active Security" https://github.com/Fable95/laplace_sampler 

### 5.1 Benchmark Results
For completeness we provide the results of our evaluations that we uesd in the folder `benches/related_works`, divided into subdirectories `Franzese` `FuWang` `Meisingseth` for each respective implementation.

### 5.2 Implementation Divergence
For a faithful representation, we sticked as close as possible to the provided implementations.
For the Mixed-Mode implementation, we did make some adjustments that allowed us to make broader tests in terms of table size and amount of lookup tables evaluated. 
The respective files can be found in the directory `reference_adaptation/`.
