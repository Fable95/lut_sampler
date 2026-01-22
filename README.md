# LUT Sampler
Multi Party Noise Sampling with Lookup Tables. This repository contains the implementation of the paper: [Accelerating Multiparty Noise Generation Using Lookups](https://eprint.iacr.org/2025/805). 

# License notice
Portions of this file are adapted from:
https://github.com/KULeuven-COSIC/maestro/
Copyright © 2024 COSIC-KU Leuven and Concordium AG
Licensed under the MIT License


# Additional Implementation material
We provide our implementation as part of the MPC library facilitating the lookup tables. The most convenient way to check it is to expand it into the underlying framework.


## Compiling the programm
The main executables are `src/lut_fill.rs` and `src/sampler.rs` containing the code for filling the lookup table and running the MPC sampling respectively. For compilation replace `<bin>` with either `lut_fill` or `sampler`. In case the machine supports it, add `--features='clmul'` to greatly increase verification performance.
```
RUSTFLAGS='-C target-cpu=native' cargo build --release --bin <bin>
```
The compiled files are now in `src/target/release/`. `lut_fill` is a non-interactive algorithm run with a single process.

## Running the Executables
After compiling with the command above you can run the implementation. For `lut_fill` you directly run the executable from the terminal (with parameters as discussed below). For `sampler` a good way to test is to open three terminal windows and run the command as follows. Everytime something is given in `<.>` replace by the parameters according to the table:

```
./target/release/lut_fill --k <k> --l <l> --ber <ber> --eps <eps> --path <path>
```
For the sampler run the three lines in seperate windows and again replace parameters according to the table:
```
./target/release/sampler --config p1.toml --skew <skew> --simd <simd> <--mal-sec>
./target/release/sampler --config p2.toml --skew <skew> --simd <simd> <--mal-sec>
./target/release/sampler --config p3.toml --skew <skew> --simd <simd> <--mal-sec>
```
### LUT Fill parameter table
The choice of target distribution is made by either selecting eps (=Laplace) or sig (=Gauss).
|name         | options  | description                                                                        | type             | 
|-------------|----------|------------------------------------------------------------------------------------|------------------|
|`k`          | `[k_i,]` | Up to 3 integers giving the bit width of each dimension                            | list of integers |
|`n`          | positive | Denotes the bound of values in the table. (max of 2^16-1)                          | Integer          |
|`error`      | positive | Denotes the value of λ (Stopping condition of search)                              | Integer          |
|`searcg`     | set      | Flag enabling full grid search up to the given k                                   | Flag             |
|`l`          | `[0..k]` | Number of _active_ bits in the index. List for grid search                         | list of integers |
|`ber`        | `[1..12]`| negative log of bernoulli bias used in grid search                                 | list of integers |
|`eps`        | positive | p_lap is computed e^-eps                                                           | list of decimals |
|`sig`        | positive | variance is taken as sigma^2=sig                                                   | list of decimals |
|`path`       | path     | if provided, and count of eps and sig is 1 total, prints resulting matrix to path  | String           |
|`bench-info` | set      | if set, displays runtime breakdown                                                 | flag             |
|`v`          | set      | if set, displays intermediate computations                                         | flag             |
|`debug`      | set      | if set, ber and eps are ignored and prints debug table                             | flag             |

To get the best approximation results for various Gauss parameters sig^2 evaluate the following:
```
./target/release/lut_fill --k 24 --n 65535 --ber 2 3 4 5 6 7 8 9 10 11 12 --sig 1 16 256 4096 65536 1048576 --search  --error 128
```
The output of this can be found in `benches/accuracy/gauss_grid.txt`. Before we use any of these results, we need to construct the table.

### LUT Fill table generation
To create a table refer to the output of the grid search and the smallest k that reached sufficient accuracy. Let e.g. sigma = 4 and sigma^2 = 16, we refer to the respective file in the benchmarks to find the computed conservative bound and the LUT size `k`:
```
Generating approximations with type: u16
	With Variance Parameters
        ...
    	"1.6000e1" Truncation error at 63: 2.72234e-54
        ...
```
Now we now that a bound of 255 (the u8 datatype will be sufficient). Next let us check for the dimension of the LUT:
```
With k: 14 and index probabilities: 2^-2 2^-3 2^-4 2^-5 2^-6 2^-7 Increasing index bias has no further impact
Best Approximations for k: 14 and target: Gaussian with Variance parameter
    ...
    v 1.6000e1: 1 tables with k: 14, l: 10, p: 2^-5,     delta: 1.493e-13 ~ 2^-42, considered range: [0,63]
    ...
```
Given `k=14` and the datatype size `w=8`, we ca run `py optimal_d3.py --w 8` and read the line `k 14: (6,6,2) cost: 155 bits` to find that the optimal setup. 
At the moment the last dimension must fulfil $w*2^{k_3} \geq 64$ and we have to use $w=16$ at the cost of 195 bits in the evaluation. Create the correct table as follows:
```
./target/release/lut_fill --k 6 6 2 --n 65535 --l 10 --ber 5 --sig 16 --path src/lut_sampler/tables/my_table
```
Then add the new table by adding `pub mod my_table` to the file `src/lut_sampler/tables/mod.rs`. Now the file is accesible from `src/sampler.rs` via `tables::my_table::MyTable` and is either an object of type `Cube` or type `Matrix` depending on the parameter `k` (in our example it will be a cube).

### Sampler parameter table
Note that performance of the scheme does not depend on the concrete table used. At this time, the chosen table has to be hardcoded in `sampler.rs`. Currently there are some selections for bias parameters from 2 to 6.
|name      | options  | description                              | type    | 
|----------|----------|------------------------------------------|---------|
| `config` | Path     | Indicates the path to the network config | String  |
|`simd`    | 1..16384 | Number of samples evaluated at once      | Integer |
|`rep`     | posiitve | Number of repetitions of benchmarks      | integer |
|`mal-sec` | set      | if set, everything is verified           | flag    |
|`debug`   | set      | if set, protocol output is revealed, composed and checked | flag |
|`network` | set      | if set, rep is set to one and network output is printed   | flag |
|`bench`   | Enum     | selects predefined benchmark suite: (size, table4, table3 variance, lambda, all) | Enum |

To run all pregenerated tables open 3 terminal windows and run the following commands
```
./target/release/sampler --config p1.toml --simd 1000 (--mal-sec) (--network) --bench all
./target/release/sampler --config p2.toml --simd 1000 (--mal-sec) (--network) --bench all
./target/release/sampler --config p3.toml --simd 1000 (--mal-sec) (--network) --bench all
```
Given the largest tables the commands above may lead to stackoverflows in some settings. Either remove the largest tables or run: `ulimit -s 262144` to increase the stack size of the current terminal session. (This needs to be run in all terminal windows)

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
