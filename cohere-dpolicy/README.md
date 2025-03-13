<a name="readme-top"></a>
# DPolicy Integration with Cohere DP Management System

This project integrates DPolicy with the [Cohere DP Management System](https://github.com/pps-lab/cohere), featuring a simplified Cohere DP Planner.
This adapted planner exclusively supports the DPK resource allocation algorithm.
A key addition is the `cohere-dpolicy/dp-planner/src/measure_traps.rs` Rust binary, which analyzes the Cohere simulation results to calculate privacy loss for specific subcomponents (e.g., releases accessing a specific attribute).
Crucially, it considers parallel composition via partitioning attributes.


### Built With


* [![Rust][rust-shield]][rust-url]
* [![Cargo][cargo-shield]][cargo-url]
* [![Gurobi][gurobi-shield]][gurobi-url]



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites


*  Installed `curl`, `git`, and `m4`:
    ```
    sudo apt-get install curl git m4
    ```
* [Rustup](https://rustup.rs/)
    ```sh
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```



* [Gurobi](https://docs.gurobi.com/current/)
The [dp-planner](./dp-planner) relies on Gurobi for solving resource allocation problems, which requires a Gurobi license.
Free academic licenses can be obtained from the following link: https://www.gurobi.com/downloads/end-user-license-agreement-academic/
The acquired license keys still need to be activated on the designated machine with `grbgetkey`.

* Local clone of the repository (with submodules)
    ```sh
    git clone --recurse-submodules git@github.com:pps-lab/dpolicy.git
    ```


<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Installation

1. Install [Rust Version 1.65](https://blog.rust-lang.org/2022/11/03/Rust-1.65.0.html):
    ```sh
    rustup install 1.65 && rustup override set 1.65
    ```

2. Install [Gurobi Version 9.5.1](https://support.gurobi.com/hc/en-us/articles/4429974969105-Gurobi-9-5-1-released)

    For Ubuntu 22.04 LTS:
    ```sh
    curl -o gurobi951.tar.gz https://packages.gurobi.com/9.5/gurobi9.5.1_linux64.tar.gz
    ```

    ```sh
    sudo tar -xzf gurobi951.tar.gz -C /opt
    ```

    Gurobi should now be installed under: `/opt/gurobi951`

3. Build the DP-Planner:
    ```sh
    cargo build --release
    ```


### Running Basic Example Workload

- Set your current working directory to the directory containing this `README.md` file.

- Execute the following command. You may need to adjust the path specified in the `$OUT_DIR` environment variable (or others) at the beginning of the command to match your system setup.
```sh
export \
    GUROBI_HOME=/opt/gurobi951/linux64 \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/gurobi951/linux64/lib \
    DATA_DIR=./dp-planner/resources/applications/minimal \
    OUT_DIR=./../doe-suite-results-super-reproduce/minimal \
    RUST_LOG=info \
    CACHE_REQUEST_COST=1; \
mkdir -p $OUT_DIR && \
cargo run --release --bin dp_planner -- \
    --schema $DATA_DIR/schema.json \
    --requests $DATA_DIR/requests_0.json \
    --blocks $DATA_DIR/blocks_0.json \
    --req-log-output $OUT_DIR/request_log.csv \
    --round-log-output $OUT_DIR/round_log.csv \
    --runtime-log-output $OUT_DIR/runtime_log.csv \
    --stats-output $OUT_DIR/stats.json \
    simulate --timeout-rounds 1 \
        efficiency-based --block-selector-seed 1000 dpk --eta 0.05 --kp-solver gurobi \
        block-composition-pa --num-threads 4 \
        unlocking-budget --trigger round --slack 0.4 --n-steps 12 \
        --alphas 1.5 1.75 2 2.5 3 4 5 6 8 16 32 64 1e6 1e10 \
        --convert-block-budgets \
&& cargo run --release --bin measure_traps -- \
    --dir $OUT_DIR \
    --privacy-unit user \
    --alphas 1.5 1.75 2 2.5 3 4 5 6 8 16 32 64 1e6 1e10 --delta 1.0e-7

```

This command runs a minimal simulation locally. The raw results will be written to the directory specified by the `$OUT_DIR` variable you set in the command.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Experiment Commands


Additional command examples are available from DPolicy's evaluation ([configuration](../doe-suite-config)).
All the commands can be listed by executing the commands below from the root directory of the repository.
Please note that for these commands, it is necessary to update the file paths (schema, requests, blocks, ...) to match your local environment.


List all commands for the context scenario:
```sh
make cmd-context
```

List all commands for the scope scenario:
```sh
make cmd-scope
```

List all commands for the time-based privacy unit scenario:
```sh
make cmd-timeunit
```



<p align="right">(<a href="#readme-top">back to top</a>)</p>


[rust-shield]: https://img.shields.io/badge/rust-grey?style=for-the-badge&logo=rust
[rust-url]: https://www.rust-lang.org/


[cargo-shield]: https://img.shields.io/badge/cargo-grey?style=for-the-badge&logo=rust
[cargo-url]: https://doc.rust-lang.org/stable/cargo/


[gurobi-shield]: https://img.shields.io/badge/gurobi-grey?style=for-the-badge&logo=gurobi
[gurobi-url]: https://www.gurobi.com/
