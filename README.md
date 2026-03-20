# Green Compressed Storage

An energy-optimised key-value store for massive source code archives, delivering Pareto-optimal compression with order-of-magnitude gains in throughput and energy efficiency.

## 📝 Short Description

**Green Compressed Storage** is an innovative, energy-aware key-value store designed to handle massive source code datasets. Built on **RocksDB**, it specialises in optimising the trade-off between space, time, and energy consumption. It achieves superior data density and high-speed retrieval by utilising finely-tuned **zstd** configurations, making it ideal for large-scale archival and analysis of code.

## 📋 Prerequisites

The following dependencies are required to compile and run the project.

1. Compilation Tools and Perf

Install the C++ compiler (Clang, G++), CMake, and the perf performance analysis tool suite.

### Compilation Tools
```bash
sudo apt install clang g++ gcc cmake
```

### Energy Profiling Tools (Perf)
```bash
sudo apt install linux-tools-common linux-tools-generic linux-tools-`uname -r`
```

### Arrow

The project utilises Apache Arrow for handling Parquet data. The Arrow C++ libraries must be installed for compilation. If you have the official Arrow repository configured, use the following commands:

```bash
sudo apt update
sudo apt install -y libarrow-dev libparquet-dev
```

### Python Dependencies

The supporting scripts require Python and several libraries.

```bash
sudo apt install python3 python3-pip
pip3 install pandas pyarrow
```

## 🛠️ Build Instructions

The project uses **CMake** for building. Ensure you have `cmake` and a C++ compiler installed.

1.  **Create a build directory and run CMake:**

    ```bash
    git submodule update --init --recursive
    mkdir build
    cd build
    cmake ..
    ```

2.  **Compile the project:**

    ```bash
    make
    ```

The main executable, `green-compressed-storage`, will be located in a directory like `./cmake-build-release/` (depending on your build configuration).

---

## 🏃 Example Run

This section demonstrates how to build the database, generate test keys, and execute single-get retrieval tests.

The `sample_data` directory contains dummy code files (extracted from the [MediaWiki repository](https://www.mediawiki.org/wiki/Download)) in a Parquet format for testing purposes. The file has two columns: `inverted_filepath` (key) and `content` (source code). In this example, we will be using the first as key and the second as value in our database.

### 1. Generate Query Keys

First, you must **generate the key sample** for the uniform and power-law (Zipfian-like) distributions, as described in the accompanying paper. The sample keys will be written to a new Parquet file in the `sample_data` directory.

The command below samples 100 keys for uniform single-gets and 100 for power-law single-gets (named `single-get-zipf` in the code).

```bash
cd scripts
python3 -u generate_query_data_shuffle.py ../sample_data/mediawiki10k.parquet inverted_filepath 0.0 42
cd ..
````

> **Note**: The parameter `0.0` ensures all selected keys are for retrieval, and `42` is the random seed for key selection.

### 2\. Build the Database (Insert Operation)

Execute the primary application to insert the data into the key-value store. This example uses Zstandard with level 6 and a block size of 64 KiB.

```bash
# Define the path to your project root (replace <PATH_TO_PROJECT> with the actual path)
PROJECT_PATH=$(pwd)
EXECUTABLE_PATH="./build/green-compressed-storage"

$EXECUTABLE_PATH \
    --parquetfile=$PROJECT_PATH/sample_data/mediawiki10k \
    --db-path=$PROJECT_PATH/zstd_6_65536 \
    --key-column=inverted_filepath \
    --compression=zstd \
    --compression-level=6 \
    --block-size=65536 \
    --run-test=insert \
    --sampling-rate-zipf=1.5 \
    --sampling-rate=1.0 \
    --probability=0.0
```

### 3\. Run Retrieval Tests (Single-Gets)

Execute single-get retrieval tests using the generated key sample.

**Uniformly Distributed Keys:**

```bash
$EXECUTABLE_PATH \
    --parquetfile=$PROJECT_PATH/sample_data/mediawiki10k-s42 \
    --db-path=$PROJECT_PATH/zstd_6_65536 \
    --key-column=inverted_filepath \
    --compression=zstd \
    --compression-level=6 \
    --block-size=65536 \
    --run-test=single-get \
    --nt=0 \
    --sampling-rate-zipf=1.5 \
    --sampling-rate=1.0 \
    --probability=0.0 
```

**Power Law Distributed Keys:**

To test retrieval with power law-distributed keys, simply substitute `--run-test=single-get` with **`--run-test=single-get-zipf`** in the command above.

> **Additional Test Modes:** You may also try **`--run-test=multi-get`** and **`--run-test=multi-get-zipf`** for multi-key retrieval tests.

### 4\. Profiling Energy Consumption

To profile the energy package consumption, ensure the **Perf suite** is installed on your system and prepend the execution command with `perf stat`.

For each test, prepend `perf stat -a -e power/energy-pkg/` to estimate the package-level consumption.

To conclude the `README.md` for this project, it is standard practice to include sections for citing the work, acknowledging contributors or funding, and specifying the licence.

Given the academic nature of the paper, here is a professional way to structure the end of your file:

### Citation

If you use this software or the data from our experiments in your research, please cite our paper:

```bibtex
@inproceedings{ferragina2026energy,
  author    = {Ferragina, Paolo and Tosoni, Francesco},
  title     = {The Energy-Throughput Trade-off in Lossless-Compressed Source Code Storage},
  booktitle = {2026 IEEE International Conference on Software Analysis, Evolution and Reengineering - Companion (SANER-C)},
  year      = {2026},
  pages     = {157--164},
  doi       = {10.1109/SANER-C67878.2026.00027},
  publisher = {IEEE}
}
```

### Acknowledgements

This work was supported by the L'EMbeDS Department at the Sant'Anna School of Advanced Studies, Pisa, Italy. All the computations presented in this paper were performed
using the GRICAD infrastructure ([https://gricad.univ-grenoble-alpes.fr](https://gricad.univ-grenoble-alpes.fr)), which is supported by Grenoble research communities. We thank SOS Gricad and the Software Heritage team for valuable insights, suggestions, and continuous support for our work.

### Licence

This project is licensed under the Apache 2.0 Licence - see the [LICENSE.md](LICENSE.md) file for details.

-----

*For any questions or further information regarding the experiments or the compressed key-value store design, please contact the authors at the Sant'Anna School of Advanced Studies.*
