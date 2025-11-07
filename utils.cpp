#include <filesystem>
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/table.h>
#include <rocksdb/write_batch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>

#include <array>
#include <vector>
#include <string>
#include <string_view>
#include <utility>

const unsigned kiB = 1024;
const unsigned MiB = 1024 * 1024;

#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#include <arrow/api.h>
#include <parquet/arrow/reader.h>

// #include "shard.h"



class ThreadPool {
public:
    ThreadPool(int num_threads) : stop_(false) {
        for (int i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) {
                            return;
                        }
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (std::thread& worker : workers_) {
            worker.join();
        }
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result_t<F, Args...>> {
        using return_type = typename std::invoke_result_t<F, Args...>;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
            );
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks_.emplace([task]() { (*task)(); });
        }
        condition_.notify_one();
        return res;
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;
};

// Helper function to get base path without extension
std::string get_basepath(const std::string& path) {
    size_t last_slash = path.find_last_of("/\\");
    size_t last_dot = path.find_last_of(".");

    if (last_dot != std::string::npos && (last_slash == std::string::npos || last_dot > last_slash)) {
        if (last_slash == std::string::npos) {
            return path.substr(0, last_dot);
        }
        return path.substr(0, last_dot);
    }

    return path;
}

/////////////////////////////////////
//
// Reading query data parquets
//
/////////////////////////////////////

// Helper function to check if file exists
bool file_exists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

// Helper function to read a single column from parquet file
std::vector<std::string> read_single_column_parquet(const std::string& file_path, const std::string& column_name) {
    std::shared_ptr<arrow::io::ReadableFile> infile;
    PARQUET_ASSIGN_OR_THROW(infile, arrow::io::ReadableFile::Open(file_path));

    std::unique_ptr<parquet::arrow::FileReader> reader;
    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

    std::shared_ptr<arrow::Table> table;
    PARQUET_THROW_NOT_OK(reader->ReadTable(&table));

    auto column = table->GetColumnByName(column_name);
    auto string_array = std::static_pointer_cast<arrow::StringArray>(column->chunk(0));

    std::vector<std::string> result;
    for (int64_t i = 0; i < string_array->length(); ++i) {
        result.push_back(string_array->GetString(i));
    }

    return result;
}

// Helper function to read multiple columns from parquet file
void read_multi_column_parquet(const std::string& file_path,
                              std::vector<std::string>& col1,
                              std::vector<std::string>& col2,
                              std::vector<std::string>& col3,
                              const std::string& col1_name,
                              const std::string& col2_name,
                              const std::string& col3_name) {
    std::shared_ptr<arrow::io::ReadableFile> infile;
    PARQUET_ASSIGN_OR_THROW(infile, arrow::io::ReadableFile::Open(file_path));

    std::unique_ptr<parquet::arrow::FileReader> reader;
    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

    std::shared_ptr<arrow::Table> table;
    PARQUET_THROW_NOT_OK(reader->ReadTable(&table));

    auto column1 = table->GetColumnByName(col1_name);
    auto string_array1 = std::static_pointer_cast<arrow::StringArray>(column1->chunk(0));
    for (int64_t i = 0; i < string_array1->length(); ++i) {
        col1.push_back(string_array1->GetString(i));
    }

    auto column2 = table->GetColumnByName(col2_name);
    auto string_array2 = std::static_pointer_cast<arrow::StringArray>(column2->chunk(0));
    for (int64_t i = 0; i < string_array2->length(); ++i) {
        col2.push_back(string_array2->GetString(i));
    }

    auto column3 = table->GetColumnByName(col3_name);
    auto string_array3 = std::static_pointer_cast<arrow::StringArray>(column3->chunk(0));
    for (int64_t i = 0; i < string_array3->length(); ++i) {
        col3.push_back(string_array3->GetString(i));
    }
}

// Function to read "ordinary" data
void read_ordinary_data(
    const std::string& base_name,
    const std::string& probability,
    const std::string& sampling_rate,
    std::vector<std::string>& ktgs,
    std::vector<std::string>& ktrs,
    std::vector<std::string>& ktis,
    std::vector<std::string>& vtis
) {
    std::string regular_suffix = "-" + probability + "-" + sampling_rate;
    std::string in_file_path;

    // Read get data
    in_file_path = base_name + ".get" + regular_suffix + ".parquet";
    if (!file_exists(in_file_path)) {
        throw std::runtime_error("File not found: " + in_file_path);
    }
    ktgs = read_single_column_parquet(in_file_path, "keys_to_get");
    std::cout << "Read ordinary get data from " << in_file_path << std::endl;

    // Read update data
    in_file_path = base_name + ".update" + regular_suffix + ".parquet";
    if (!file_exists(in_file_path)) {
        throw std::runtime_error("File not found: " + in_file_path);
    }
    read_multi_column_parquet(in_file_path, ktrs, ktis, vtis,
                                "keys_to_remove", "keys_to_insert", "values_to_insert");
    std::cout << "Read ordinary update data from " << in_file_path << std::endl;
}

// Function to read Zipfian data
void read_zipfian_data(
    const std::string& base_name,
    const std::string& probability,
    const std::string& sampling_rate_zipf,
    std::vector<std::string>& ktgs_zipf
) {
    std::string zipf_alpha = "1.5";
    std::string zipf_suffix = "-" + probability + "-" + sampling_rate_zipf + "-" + zipf_alpha;
    std::string in_file_path;

    // Read zipf get data
    in_file_path = base_name + ".getzipf" + zipf_suffix + ".parquet";
    if (!file_exists(in_file_path)) {
        throw std::runtime_error("File not found: " + in_file_path);
    }
    ktgs_zipf = read_single_column_parquet(in_file_path, "keys_to_get");
    std::cout << "Read Zipfian get data from " << in_file_path << std::endl;
}

/////////////////////////////////////////////////////////////////////////

void print_options(const rocksdb::Options& options) {
    std::cout << "create_if_missing: " << options.create_if_missing << std::endl;
    std::cout << "error_if_exists: " << options.error_if_exists << std::endl;
    std::cout << "paranoid_checks: " << options.paranoid_checks << std::endl;
    std::cout << "max_open_files: " << options.max_open_files << std::endl;
    std::cout << "write_buffer_size: " << options.write_buffer_size << std::endl;
    std::cout << "max_background_compactions: " << options.max_background_compactions << std::endl;
    std::cout << "compression_opts.enabled: " << options.compression_opts.enabled << std::endl;
    std::cout << "compression_opts.max_dict_bytes: " << options.compression_opts.max_dict_bytes << std::endl;
    std::cout << "compression_opts.zstd_max_train_bytes: " << options.compression_opts.zstd_max_train_bytes << std::endl;
    std::cout << "compression_opts.strategy: " << options.compression_opts.strategy << std::endl;
    std::cout << "allow_mmap_reads: " << options.allow_mmap_reads << std::endl;
    std::cout << "allow_mmap_writes: " << options.allow_mmap_writes << std::endl;
    std::cout << "use_adaptive_mutex: " << options.use_adaptive_mutex << std::endl;
    // Aggiungi altre opzioni che desideri stampare
}

// Helper function to convert compression type to string
const char* compression_type_to_string(rocksdb::CompressionType type) {
    switch(type) {
        case rocksdb::kNoCompression: return "None";
        case rocksdb::kSnappyCompression: return "Snappy";
        case rocksdb::kZlibCompression: return "Zlib";
        case rocksdb::kBZip2Compression: return "BZip2";
        case rocksdb::kLZ4Compression: return "LZ4";
        case rocksdb::kLZ4HCCompression: return "LZ4HC";
        case rocksdb::kXpressCompression: return "Xpress";
        case rocksdb::kZSTD: return "ZSTD";
        case rocksdb::kDisableCompressionOption: return "Disabled";
        default: return "Unknown";
    }
}

rocksdb::Options get_my_rocksdb_options(size_t block_size = 16 * 1024,
                                  rocksdb::CompressionType compr = rocksdb::kZSTD,
                                  int compr_level = 12,
                                  size_t max_dict_bytes = 0,
                                  size_t zstd_max_train_bytes = 0,
                                  rocksdb::CompressionType wal_compression = rocksdb::kNoCompression) {
    rocksdb::Options options;
    options.allow_mmap_reads = true;
    options.allow_mmap_writes = true;
    options.paranoid_checks = false;
    options.use_adaptive_mutex = true;

    // Compression
    options.compression = compr;
    options.compression_opts.enabled = (compr != rocksdb::kNoCompression);
    options.compression_opts.max_dict_bytes = max_dict_bytes;
    options.compression_opts.zstd_max_train_bytes = zstd_max_train_bytes;
    if (compr == rocksdb::kZSTD) {
        options.compression_opts.strategy = rocksdb::kZSTD;
    }
    if (compr_level != 0) {
        options.compression_opts.level = compr_level;
    }

    // WAL Compression (only ZSTD supported)
    if (wal_compression != rocksdb::kNoCompression && wal_compression != rocksdb::kZSTD) {
        std::cerr << "Warning: WAL compression only supports ZSTD. Disabling.\n";
        wal_compression = rocksdb::kNoCompression;
    }
    options.wal_compression = wal_compression;

    // Write Optimizations (assuming these are desired for both create and open)
    options.max_total_wal_size = 64UL * 1024 * 1024 * 1024; // 64 GiB
    options.wal_bytes_per_sync = 8UL * 1024 * 1024; // 8 MB
    options.bytes_per_sync = 8UL * 1024 * 1024; // 8 MB
    options.write_buffer_size = 2UL * 1024 * 1024 * 1024; // 2 GiB
    options.max_write_buffer_number = 4; // Up to 8 GiB memtables
    options.min_write_buffer_number_to_merge = 2;
    options.allow_concurrent_memtable_write = true;
    options.enable_pipelined_write = true;
    options.max_write_batch_group_size_bytes = 16UL * 1024 * 1024; // 16 MB
    options.avoid_flush_during_shutdown = true;
    options.avoid_flush_during_recovery = true; // Warning: May lose some WAL entries on crash

    options.max_background_jobs = 6;       // More flush/compaction threads
    options.max_subcompactions = 4;        // Parallelize L0→L1 compaction
    options.memtable_prefix_bloom_size_ratio = 0.0;  // Disables the feature (disabled also by default)

    // Table Options
    rocksdb::BlockBasedTableOptions table_options;
    table_options.block_size = block_size;
    // table_options.block_cache = rocksdb::NewLRUCache(1UL << 30); // 1GB cache
    options.optimize_filters_for_hits = true;  // If DB is write-once/read-many
    options.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_options));

    return options;
}

void print_my_rocksdb_config(const rocksdb::Options& options, size_t block_size,
                   rocksdb::CompressionType compr, int compr_level,
                   size_t max_dict_bytes, size_t zstd_max_train_bytes,
                   rocksdb::CompressionType wal_compression) {
    std::cout << "\nDatabase Configuration:\n";
    std::cout << "--------------------------------\n";
    std::cout << "Block size: " << block_size << " bytes\n";
    std::cout << "Compression: " << compression_type_to_string(compr) << "\n";
    std::cout << "Compression level: " << compr_level << "\n";
    std::cout << "Compression options enabled: " << std::boolalpha << options.compression_opts.enabled << "\n";
    std::cout << "Max dict bytes: " << max_dict_bytes << "\n";
    std::cout << "ZSTD max train bytes: " << zstd_max_train_bytes << "\n";
    std::cout << "WAL compression: " << compression_type_to_string(wal_compression) << "\n";
    std::cout << "--------------------------------\n\n";
}

class DB_PPC {
public:
    DB_PPC(const std::string& path) : db_path(path), db(nullptr) {}

    void create_db(size_t block_size = 16 * 1024,
                   rocksdb::CompressionType compr = rocksdb::kZSTD,
                   int compr_level = 12,
                   size_t max_dict_bytes = 0,
                   size_t zstd_max_train_bytes = 0,
                   rocksdb::CompressionType wal_compression = rocksdb::kNoCompression) {
        rocksdb::Options options = get_my_rocksdb_options(block_size, compr, compr_level,
                                                    max_dict_bytes, zstd_max_train_bytes,
                                                    wal_compression);
        options.create_if_missing = true;
        // Note: Removed options.error_if_exists = false; to ensure creation fails if DB exists

        print_my_rocksdb_config(options, block_size, compr, compr_level, max_dict_bytes,
                      zstd_max_train_bytes, wal_compression);

        rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db);
        if (!status.ok()) {
            throw std::runtime_error("Failed to create database: " + status.ToString());
        }
    }

    void compact_db() {
        if (!db) {
            throw std::runtime_error("Database is not open");
        }
        rocksdb::CompactRangeOptions options;
        db->CompactRange(options, nullptr, nullptr);
    }

    void print_db_stats() {
        std::string stats;
        db->GetProperty("rocksdb.stats", &stats);
        std::cout << stats << std::endl; // Stampa tutte le statistiche
    }

    double get_compression_ratio(size_t keys_size, size_t values_size) {
        // Start the timer before flushing
        auto start = std::chrono::high_resolution_clock::now();

        // FLush data to SST tables
        rocksdb::Status s = db->Flush(rocksdb::FlushOptions()); // Generate SST files
        if (!s.ok()) {
            std::cerr << "Flush failed: " << s.ToString() << std::endl;
            return 0.0; // Or handle the error as appropriate
        }

        // Stop the timer after flushing
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> flush_duration = end - start;
        std::cout << "Flush time: " << flush_duration.count() << " seconds" << std::endl;

        // Get total raw data size (keys + values)
        size_t raw_data_size = keys_size + values_size;
        if (raw_data_size == 0) return 0.0; // Avoid division by zero

        // 1. Get SST file sizes
        std::uintmax_t total_sst_size = 0;
        std::uintmax_t total_db_files_size = 0;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(db_path)) {
            if (entry.is_regular_file()){
                try {
                    if (entry.path().extension() == ".sst") {
                        std::uintmax_t file_size = entry.file_size();
                        total_db_files_size += file_size;
                        total_sst_size += file_size;
                    } else {
                        std::cout << "File: " << entry.path() << " is not an SST file." << std::endl;
                    }
                } catch (const std::filesystem::filesystem_error& e) {
                    std::cerr << "Warning: Could not access " << entry.path() << ": " << e.what() << std::endl;
                    // Continue with next file
                }
            }
        }

        // 2. Get memtable sizes from RocksDB statistics
        std::string stats;
        db->GetProperty("rocksdb.stats", &stats); // Get general stats

        uint64_t memtable_total_size = 0;
        db->GetIntProperty("rocksdb.cur-size-all-mem-tables", &memtable_total_size);

        // 3. Get WAL file sizes (approximate via statistics)
        uint64_t wal_size = 0;
        db->GetIntProperty("rocksdb.total-wal-size", &wal_size);

        // 4. Calculate total storage footprint
        uint64_t total_storage_footprint = total_sst_size + memtable_total_size + wal_size;

        // Calculate ratios
        double compression_ratio = static_cast<double>(total_storage_footprint) / raw_data_size * 100;
        double sst_only_ratio = static_cast<double>(total_sst_size) / raw_data_size * 100;
        double mem_wal_ratio = static_cast<double>(memtable_total_size + wal_size) / raw_data_size * 100;

        std::cout << "Compression statistics:\n";
        std::cout << "Keys size: " << keys_size << " bytes\n";
        std::cout << "Value size: " << values_size << " bytes\n";
        std::cout << "Raw data size: " << raw_data_size << " bytes\n";
        std::cout << "Total storage footprint: " << total_storage_footprint << " bytes\n";
        std::cout << "  - SST files: " << total_sst_size << " bytes\n";
        std::cout << "    --- Total DB files including LOCK, LOG, etc.: " << total_db_files_size << " bytes\n";
        std::cout << "  - Memtables: " << memtable_total_size << " bytes\n";
        std::cout << "  - WAL files: " << wal_size << " bytes\n";
        std::cout << "Overall compression ratio: " << compression_ratio << "%\n";
        std::cout << "SST-only compression ratio: " << sst_only_ratio << "%\n";
        std::cout << "Mem+WAL overhead: " << mem_wal_ratio << "%\n";

        return compression_ratio;
    }

    void open_db(size_t block_size = 16 * 1024,
                 rocksdb::CompressionType compr = rocksdb::kZSTD,
                 int compr_level = 12,
                 size_t max_dict_bytes = 0,
                 size_t zstd_max_train_bytes = 0,
                 rocksdb::CompressionType wal_compression = rocksdb::kNoCompression) {
        rocksdb::Options options = get_my_rocksdb_options(block_size, compr, compr_level,
                                                    max_dict_bytes, zstd_max_train_bytes,
                                                    wal_compression);
        options.create_if_missing = true;
        options.error_if_exists = false; // Allows opening an existing DB

        print_my_rocksdb_config(options, block_size, compr, compr_level, max_dict_bytes,
                      zstd_max_train_bytes, wal_compression);

        rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db);
        if (!status.ok()) {
            throw std::runtime_error("Failed to open database: " + status.ToString());
        }
    }

    void close_db() {
        if (this->db != nullptr) {
            // WARNING: Any existing iterators will become invalid after this call
            // and may cause crashes if used. Caller must ensure no iterators are active.
            delete this->db;
            this->db = nullptr;
        }
    }

    template <typename InputIterator>
    void insert_single(const std::string& key, InputIterator begin, InputIterator end) {
        if (!db) {
            throw std::runtime_error("Database is not open");
        }
        std::vector<char> value(begin, end);
        rocksdb::Slice value_slice(value.data(), value.size());
        rocksdb::Status status = db->Put(rocksdb::WriteOptions(), key, value_slice);
        if (!status.ok()) {
            throw std::runtime_error("Failed to insert key-value pair: " + status.ToString());
        }
    }

    std::string single_get(const std::string& key) {
        if (!db) {
            throw std::runtime_error("Database is not open");
        }
        std::string value;
        rocksdb::Status status = db->Get(rocksdb::ReadOptions(), key, &value);
        if (!status.ok()) {
            if (status.IsNotFound()) {
                return "";
            }
            throw std::runtime_error("Failed to get value: " + status.ToString());
        }
        return value;
    }

    bool single_remove(const std::string& key) {
        if (!db) {
            throw std::runtime_error("Database is not open");
        }
        rocksdb::Status status = db->Delete(rocksdb::WriteOptions(), key);
        if (!status.ok()) {
            if (status.IsNotFound()) {
                return false;
            }
            throw std::runtime_error("Failed to remove key: " + status.ToString());
        }
        return true;
    }

    std::vector<std::string> multi_get(const std::vector<std::string>& keys) {
        if (!db) {
            throw std::runtime_error("Database is not open");
        }
    
        // Crea un vettore di slices per le chiavi
        std::vector<rocksdb::Slice> slices;
        slices.reserve(keys.size());
        for (const auto& key : keys) {
            slices.emplace_back(key);
        }
    
        // Crea un vettore per i valori
        std::vector<std::string> values(keys.size());
    
        // Esegue la MultiGet per recuperare i valori associati alle chiavi
        std::vector<rocksdb::Status> statuses = db->MultiGet(rocksdb::ReadOptions(), slices, &values);
    
        // Verifica lo stato di ogni operazione di recupero
        for (const auto& status : statuses) {
            if (!status.ok() && !status.IsNotFound()) {
                throw std::runtime_error("Failed to get values: " + status.ToString());
            }
        }
    
        // Restituisce i valori recuperati
        return values;
    }

    void delete_key(const std::string& key) {
        if (!db) {
            throw std::runtime_error("Database is not open");
        }
        rocksdb::Status status = db->Delete(rocksdb::WriteOptions(), key);
        if (!status.ok()) {
            throw std::runtime_error("Failed to delete key: " + status.ToString());
        }
    }

    bool is_db_open() const {
        return db != nullptr;
    }

    // Simple iterator implementation
    class Iterator {
    private:
        rocksdb::Iterator* it;

    public:
        explicit Iterator(rocksdb::DB* db) : it(db->NewIterator(rocksdb::ReadOptions())) {}
        ~Iterator() { delete it; }

        void SeekToFirst() { it->SeekToFirst(); }
        void SeekToLast() { it->SeekToLast(); }
        void Seek(const rocksdb::Slice& target) { it->Seek(target); }
        void Next() { it->Next(); }
        void Prev() { it->Prev(); }

        bool Valid() const { return it->Valid(); }
        rocksdb::Slice key() const { return it->key(); }
        rocksdb::Slice value() const { return it->value(); }

        // Disallow copying
        Iterator(const Iterator&) = delete;
        Iterator& operator=(const Iterator&) = delete;
    };

    std::unique_ptr<Iterator> NewIterator() {
        if (!db) {
            throw std::runtime_error("Database is not open");
        }
        return std::unique_ptr<Iterator>(new Iterator(db));
    }

    rocksdb::DB* db;

private:
    std::string db_path;
};

/////////////////////////////////////
//
// Managing shard files
//
/////////////////////////////////////

/*
bool shard_to_vectors(
    shard_t* shard,
    std::vector<std::array<char, SHARD_KEY_LEN>>& keys,
    std::vector<std::string>& values
) {
    shard_index_t shard_idx;
    uint64_t size;

    keys.clear();
    values.clear();
    // Pre-allocate memory for performance
    const uint64_t estimated_count = shard->header.index_size / (32 + 8);
    keys.reserve(estimated_count);
    values.reserve(estimated_count);

    // Create a zero key for comparison
    std::array<char, SHARD_KEY_LEN> zero_key; // Use std::array for zero_key
    zero_key.fill(0); // Initialize with zeros
    size_t skipped = 0;

    for (uint64_t i = 0; i < estimated_count; i++) {
        shard_index_get(shard, i, &shard_idx);
        if (shard_idx.object_offset == (uint64_t)-1) {
            continue; // Skip empty entries
        }

        // New optimization: Check if key is all zeros
        // Use std::equal for comparison with std::array
        if (std::equal(std::begin(shard_idx.key), std::end(shard_idx.key), zero_key.begin())) {
            ++skipped;
            continue; // Skip this entry altogether
        }

        // Read the value
        if (shard_find_object(shard, shard_idx.key, &size) != 0) {
            fprintf(stderr, "Error: key not found during loading\n");
            return false;
        }

        std::string value;
        value.resize(size);
        if (shard_read_object(shard, value.data(), size) != 0) {
            fprintf(stderr, "Error: failed to read object\n");
            return false;
        }

        // Directly add the key (copying the array)
        std::array<char, SHARD_KEY_LEN> current_key;
        std::memcpy(current_key.data(), shard_idx.key, SHARD_KEY_LEN);
        keys.emplace_back(current_key); // Emplace the std::array

        values.emplace_back(std::move(value));
    }

    if (skipped)
        fprintf(stderr, "Warning: Skipped %zu entries with zero keys\n", skipped);

    return true;
}
    */