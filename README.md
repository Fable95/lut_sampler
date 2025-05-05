# LUT Sampler
Multi Party Noise Sampling with Lookup Tables. This repository contains the implementation of the paper: _Accelerating Multiparty Noise Generation Using Lookups_. 

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
./target/release/lut_fill --k <k> --samplings <samplings> --ber <ber> --eps <eps> --bench-info --path <path>
```
For the sampler run the three lines in seperate windows and again replace parameters according to the table:
```
./target/release/sampler --config p1.toml --skew <skew> --simd <simd> <--mal-sec>
./target/release/sampler --config p2.toml --skew <skew> --simd <simd> <--mal-sec>
./target/release/sampler --config p3.toml --skew <skew> --simd <simd> <--mal-sec>
```
### LUT Fill parameter table
|name      |options|description|type| 
|----------|-------|-----------|----|
|`k`     | 4..10 | log of the cell count | integer |
|`samplings`| `[s,u]` | length 3 list of index samplings e.g.( `s s s`) per dimension | list of length 3 |
|`ber`     | 1..7 | negative log of bernoulli bias | list of integers |
|`eps`     | positive | p_lap is computed e^eps | list of decimals  |
|`path` | path | if provided, and count of eps and ber is 1 each, prints resulting matrix to path  | String |
|`bench-info` | set | if set, displays runtime breakdown | flag |
|`v` | set | if set, displays intermediate computations | flag |
|`debug` | set | if set, ber and eps are ignored and prints debug table | flag |

The results from Table 2 in Section 6.1 Filling the LUT can be recreated by running:
```
./target/release/lut_fill --ber 1 2 3 4 5 6 7 --eps 3 2 1 0.5 0.1
```
### Sampler parameter table
|name      |options|description|type| 
|----------|-------|-----------|----|
|`simd`    | 1..16384 | Number of samples evaluated at once | list of integers |
|`rep`    | 1.. | Number of repetitions to run the benchmark | integer |
|`dim`    | 3 | Dimension of the table, currently fixed to 3 | integer |
|`k`      | 8 | log of dimension length, currently fixed to 8 | integer |
|`mal-sec`| set | if set, all multiplications and dot-products are verified | flag |
|`index-dist`| `[s,u]` | list of 3 index distributions, bernoulli (s) or uniform (u) | list of 3 |
|`skew`      | 1..7 | negative log of bernoulli bias | integer |

The results of Table 4, and the pi_Z related results from table 3 can be recreated with the following command. (Again, each line in a different terminal window on the same machine). The brackets indicate choices for single and amortized performance as well as malicious (including the flag) or semi-honest.
```
./target/release/sampler --config p1.toml --skew 4 --simd (1/1000) (--mal-sec)
./target/release/sampler --config p2.toml --skew 4 --simd (1/1000) (--mal-sec)
./target/release/sampler --config p3.toml --skew 4 --simd (1/1000) (--mal-sec)
```
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
