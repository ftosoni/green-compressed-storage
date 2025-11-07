#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <chrono>
#include <vector>
#include "utils.cpp"
#include <CLI/CLI.hpp>

#include <random>  // For better RNG
#include <cmath>
#include <filesystem>
#include <future>

#include <numeric> // For std::accumulate
#include <thread> // For std::this_thread
#include <algorithm> // For std::find

// Helper function to print truncated strings
void print_truncated(const std::string& s, const size_t max_len = 100) {
    if (s.size() <= max_len) {
        std::cout << s;
    } else {
        std::cout << s.substr(0, max_len) << "...[+" << (s.size() - max_len) << " more chars]";
    }
}

void test_iterate_all(DB_PPC& db, std::string &test_name) {
    std::cout << "[" << test_name << "] Performing a full database scan." << std::endl;

    size_t total_keys = 0;
    size_t keys_size = 0;
    size_t values_size = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        auto it = db.NewIterator();
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            total_keys++;

            // Get key and value
            std::string key = it->key().ToString();
            std::string value = it->value().ToString();

            // Update size statistics
            keys_size += key.size();
            values_size += value.size();

            // Print truncated key-value pair
            std::cout << "[" << test_name << "] Key " << total_keys << ": ";
            print_truncated(key);
            std::cout << " -> Value: ";
            print_truncated(value);
            std::cout << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "[" << test_name << "] Iterator error: " << e.what() << "\n";
        return;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    double total_size = keys_size + values_size;

    std::cout << "[" << test_name << "] Summary:\n";
    std::cout << "[" << test_name << "] Total keys: " << total_keys << "\n";
    std::cout << "[" << test_name << "] Total keys size: " << keys_size << " bytes\n";
    std::cout << "[" << test_name << "] Total values size: " << values_size << " bytes\n";
    std::cout << "[" << test_name << "] Total data size: " << total_size << " bytes\n";
    std::cout << "[" << test_name << "] Scan time: " << elapsed.count() << " seconds\n";

    if (elapsed.count() > 0) {
        double throughput_bps = total_size / elapsed.count();
        double throughput_kibps = throughput_bps / 1024;
        double throughput_mibps = throughput_kibps / 1024;

        std::cout << "[" << test_name << "] Scan throughput: " << throughput_bps << " B/s\n";
        std::cout << "[" << test_name << "] Scan throughput: " << throughput_kibps << " kiB/s\n";
        std::cout << "[" << test_name << "] Scan throughput: " << throughput_mibps << " MiB/s\n";
    }
}

std::pair<size_t, size_t> insert_batch_local_size(rocksdb::DB* db, const std::shared_ptr<arrow::StringArray>& key_array,
                                                 const std::shared_ptr<arrow::StringArray>& value_array) {
    size_t local_keys_size = 0;
    size_t local_values_size = 0;

    rocksdb::WriteBatch write_batch;
    for (int64_t i = 0; i < key_array->length(); ++i) {
        if (key_array->IsNull(i) || value_array->IsNull(i)) {
            std::cerr << "Skipping null key or value at index " << i << std::endl;
            continue;
        }
        std::string key = key_array->GetString(i);
        std::string value = value_array->GetString(i);

        local_keys_size += key.size();
        local_values_size += value.size();

        write_batch.Put(key, value);
    }
    rocksdb::Status status = db->Write(rocksdb::WriteOptions(), &write_batch);
    if (!status.ok()) {
        std::cerr << "Error inserting batch: " << status.ToString() << std::endl;
    }
    return {local_keys_size, local_values_size};
}

inline void insert_single_parquet_to_db(const std::string& parquet_file, DB_PPC& db,
                                        size_t& keys_size, size_t& values_size, const std::string& key_column_name,
                                        const std::string& value_column_name, int max_threads) {
    size_t file_keys_size = 0;
    size_t file_values_size = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Open Parquet file and create reader
    std::shared_ptr<arrow::io::ReadableFile> infile;
    PARQUET_ASSIGN_OR_THROW(infile, arrow::io::ReadableFile::Open(parquet_file, arrow::default_memory_pool()));
    std::unique_ptr<parquet::arrow::FileReader> reader;
    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));
    std::shared_ptr<arrow::Schema> schema;
    PARQUET_THROW_NOT_OK(reader->GetSchema(&schema));
    int key_index = schema->GetFieldIndex(key_column_name);
    int value_index = schema->GetFieldIndex(value_column_name);
    if (key_index == -1) {
        throw std::runtime_error("Required column " + key_column_name + " not found in the Parquet file");
    }
    if (value_index == -1) {
        throw std::runtime_error("Required column " + value_column_name + " not found in the Parquet file");
    }

    std::unique_ptr<arrow::RecordBatchReader> batch_reader;
    PARQUET_THROW_NOT_OK(reader->GetRecordBatchReader(&batch_reader));

    if (max_threads != 0) {
        std::cout << "Async insertion" << std::endl;
        // Asynchronous execution
        ThreadPool pool(max_threads);
        std::vector<std::future<std::pair<size_t, size_t>>> futures;
        std::shared_ptr<arrow::RecordBatch> batch;

        while (batch_reader->ReadNext(&batch).ok() && batch != nullptr) {
            auto key_array = std::static_pointer_cast<arrow::StringArray>(batch->column(key_index));
            auto value_array = std::static_pointer_cast<arrow::StringArray>(batch->column(value_index));

            futures.push_back(pool.enqueue([&db, key_array, value_array]() {
                return insert_batch_local_size(db.db, key_array, value_array);
            }));
        }

        // Wait for all batch insertions to complete and accumulate sizes
        for (auto& future : futures) {
            std::pair<size_t, size_t> kv_sizes = future.get();
            file_keys_size += kv_sizes.first;
            file_values_size += kv_sizes.second;
        }

    // } else {
    //     std::cout << "Sync insertion" << std::endl;
    //     // Synchronous execution
    //     std::shared_ptr<arrow::RecordBatch> batch;
    //     while (batch_reader->ReadNext(&batch).ok() && batch != nullptr) {
    //         auto key_array = std::static_pointer_cast<arrow::StringArray>(batch->column(key_index));
    //         auto value_array = std::static_pointer_cast<arrow::StringArray>(batch->column(value_index));
    //         std::pair<size_t, size_t> local_sizes = insert_batch_local_size(db.db, key_array, value_array);
    //         file_keys_size += local_sizes.first;
    //         file_values_size += local_sizes.second;
    //     }
    } else {
        std::cout << "Sync insertion" << std::endl;
        // Synchronous execution
        std::shared_ptr<arrow::RecordBatch> batch;
        double total_read_time = 0;
        double total_process_time = 0;

        while (true) {
            auto read_start = std::chrono::high_resolution_clock::now();
            auto status = batch_reader->ReadNext(&batch);
            auto read_end = std::chrono::high_resolution_clock::now();
            if (!status.ok() || batch == nullptr) break;

            total_read_time += std::chrono::duration<double>(read_end - read_start).count();

            auto process_start = std::chrono::high_resolution_clock::now();
            auto key_array = std::static_pointer_cast<arrow::StringArray>(batch->column(key_index));
            auto value_array = std::static_pointer_cast<arrow::StringArray>(batch->column(value_index));
            std::pair<size_t, size_t> local_sizes = insert_batch_local_size(db.db, key_array, value_array);
            file_keys_size += local_sizes.first;
            file_values_size += local_sizes.second;
            auto process_end = std::chrono::high_resolution_clock::now();

            total_process_time += std::chrono::duration<double>(process_end - process_start).count();
        }

        std::cout << "Time spent reading batches: " << total_read_time << " s ("
                  << (total_read_time/(total_read_time+total_process_time))*100 << "%)" << std::endl;
        std::cout << "Time spent processing batches: " << total_process_time << " s ("
                  << (total_process_time/(total_read_time+total_process_time))*100 << "%)" << std::endl;
    }
    //end modified code

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> ins_time = end_time - start_time;

    keys_size += file_keys_size;
    values_size += file_values_size;

    double total_size_mib = static_cast<double>(file_keys_size + file_values_size) / (1024 * 1024);
    double throughput_mib_s = total_size_mib / ins_time.count();

    std::cout << "Parquet: " << parquet_file << std::endl;
    std::cout << "Keys size: " << file_keys_size << " bytes" << std::endl;
    std::cout << "Values size: " << file_values_size << " bytes" << std::endl;
    std::cout << "Total size: " << file_keys_size + file_values_size << " bytes" << std::endl;
    std::cout << "Insertion time: " << ins_time.count() << " s" << std::endl;
    std::cout << "Insertion throughput: " << throughput_mib_s << " MiB/s" << std::endl;
}

void test_insertion(
    DB_PPC& db,
    const std::string& parquet_file_path,
    std::string& test_name,
    const std::string& key_column_name = "",
    const std::string& value_column_name = "",
    int max_threads = 0
) {
    std::cout << "[" << test_name << "] Inserting data from parquet file: " << parquet_file_path << std::endl;

    size_t keys_size = 0;
    size_t values_size = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        insert_single_parquet_to_db(parquet_file_path, db, keys_size, values_size, key_column_name, value_column_name, max_threads);
    } catch (const std::exception& e) {
        std::cerr << "[" << test_name << "] Error: " << e.what() << std::endl;
        return;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    double total_size = keys_size + values_size;

    std::cout << "[" << test_name << "] inserted keys size: " << keys_size << "\n";
    std::cout << "[" << test_name << "] inserted values size: " << values_size << "\n";
    std::cout << "[" << test_name << "] inserted tot size: " << total_size << "\n";
    std::cout << "[" << test_name << "] time: " << elapsed.count() << "\n";

    if (elapsed.count() > 0) {
        double throughput_bps = total_size / elapsed.count();
        double throughput_kibps = throughput_bps / 1024;
        double throughput_mibps = throughput_kibps / 1024;

        std::cout << "[" << test_name << "] throughput: " << throughput_bps << " B/s\n";
        std::cout << "[" << test_name << "] throughput: " << throughput_kibps << " kiB/s\n";
        std::cout << "[" << test_name << "] throughput: " << throughput_mibps << " MiB/s\n";
    }

    // Calculate and print compression ratio
    double ratio = db.get_compression_ratio(keys_size, values_size);
    std::cout << "[" << test_name << "] compression ratio: " << ratio << "\n";
}

//sync
void test_single_gets_sync(DB_PPC& db, const std::vector<std::string>& keys_to_get, std::string &test_name) {
    std::cout << "Synchronous execution" << std::endl;
    if (keys_to_get.empty()) {
        std::cout << "[" << test_name << "] No keys to retrieve\n";
        return;
    }

    size_t retrieved_keys = 0;
    size_t keys_size = 0;
    size_t values_size = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (const auto& key : keys_to_get) {
        std::string value = db.single_get(key);
        if (!value.empty()) {
            retrieved_keys++;
            keys_size += key.size();
            values_size += value.size();
        }else {
            std::cout << "Key not found: " << key << "\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    double total_size = keys_size + values_size;

    std::cout << "[" << test_name << "] keys to get: " << keys_to_get.size() << "\n";
    std::cout << "[" << test_name << "] retrieved keys number: " << retrieved_keys << "\n";
    std::cout << "[" << test_name << "] retrieved keys size: " << keys_size << "\n";
    std::cout << "[" << test_name << "] retrieved values size: " << values_size << "\n";
    std::cout << "[" << test_name << "] retrieved tot size: " << total_size << "\n";
    std::cout << "[" << test_name << "] time: " << elapsed.count() << "\n";

    if (elapsed.count() > 0) {
        double throughput_bps = total_size / elapsed.count();
        double throughput_kibps = throughput_bps / 1024;
        double throughput_mibps = throughput_kibps / 1024;

        std::cout << "[" << test_name << "] throughput: " << throughput_bps << " B/s\n";
        std::cout << "[" << test_name << "] throughput: " << throughput_kibps << " kiB/s\n";
        std::cout << "[" << test_name << "] throughput: " << throughput_mibps << " MiB/s\n";
    }
}

//async
void test_single_gets(DB_PPC& db, const std::vector<std::string>& keys_to_get,
                     std::string &test_name, int max_threads) {
    if (max_threads == 0) {
        //single-threaded version
        test_single_gets_sync(db,keys_to_get,test_name);
        return;
    }
    std::cout << "Asynchronous execution" << std::endl;
    if (keys_to_get.empty()) {
        std::cout << "[" << test_name << "] No keys to retrieve\n";
        return;
    }

    size_t retrieved_keys = 0;
    size_t keys_size = 0;
    size_t values_size = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    ThreadPool pool(max_threads);
    std::vector<std::future<std::string>> futures;
    futures.reserve(keys_to_get.size());

    for (const auto& key : keys_to_get) {
        futures.push_back(pool.enqueue([&db, key]() {
            return db.single_get(key);
        }));
    }

    // Collect results from the futures
    for (size_t i = 0; i < futures.size(); ++i) {
        std::string value = futures[i].get();
        if (!value.empty()) {
            retrieved_keys++;
            keys_size += keys_to_get[i].size();
            values_size += value.size();
        } else {
            std::cout << "Key not found: " << keys_to_get[i] << "\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    double total_size = keys_size + values_size;

    std::cout << "[" << test_name << "] keys to get: " << keys_to_get.size() << "\n";
    std::cout << "[" << test_name << "] retrieved keys number: " << retrieved_keys << "\n";
    std::cout << "[" << test_name << "] retrieved keys size: " << keys_size << "\n";
    std::cout << "[" << test_name << "] retrieved values size: " << values_size << "\n";
    std::cout << "[" << test_name << "] retrieved tot size: " << total_size << "\n";
    std::cout << "[" << test_name << "] time: " << elapsed.count() << "\n";

    if (elapsed.count() > 0) {
        double throughput_bps = total_size / elapsed.count();
        double throughput_kibps = throughput_bps / 1024;
        double throughput_mibps = throughput_kibps / 1024;

        std::cout << "[" << test_name << "] throughput: " << throughput_bps << " B/s\n";
        std::cout << "[" << test_name << "] throughput: " << throughput_kibps << " kiB/s\n";
        std::cout << "[" << test_name << "] throughput: " << throughput_mibps << " MiB/s\n";
    }
}

//sync multi-get
void test_multi_get_sync(DB_PPC& db, const std::vector<std::string>& keys_to_get,
                        std::string &test_name, size_t batch_size) {
    std::cout << "Synchronous multi-get execution" << std::endl;
    if (keys_to_get.empty()) {
        std::cout << "[" << test_name << "] No keys to retrieve\n";
        return;
    }

    size_t total_keys_size = 0;
    size_t total_values_size = 0;
    size_t total_retrieved_keys = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < keys_to_get.size(); i += batch_size) {
        auto batch_start = keys_to_get.begin() + i;
        auto batch_end = (i + batch_size) < keys_to_get.size()
                                         ? keys_to_get.begin() + i + batch_size
                                         : keys_to_get.end();
        std::vector<std::string> batch_keys(batch_start, batch_end);

        // Calculate batch keys size
        for (const auto& key : batch_keys) {
            total_keys_size += key.size();
        }

        std::vector<std::string> values = db.multi_get(batch_keys);

        // Process results for the current batch
        for (const auto& value : values) {
            if (!value.empty()) {
                total_retrieved_keys++;
                total_values_size += value.size();
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    double total_size = total_keys_size + total_values_size;

    std::cout << "[" << test_name << "] keys to get: " << keys_to_get.size() << "\n";
    std::cout << "[" << test_name << "] retrieved keys number: " << total_retrieved_keys << "\n";
    std::cout << "[" << test_name << "] retrieved keys size: " << total_keys_size << "\n";
    std::cout << "[" << test_name << "] retrieved values size: " << total_values_size << "\n";
    std::cout << "[" << test_name << "] retrieved tot size: " << total_size << "\n";
    std::cout << "[" << test_name << "] time: " << elapsed.count() << "\n";

    if (elapsed.count() > 0) {
        double throughput_bps = total_size / elapsed.count();
        double throughput_kibps = throughput_bps / 1024;
        double throughput_mibps = throughput_kibps / 1024;

        std::cout << "[" << test_name << "] throughput: " << throughput_bps << " B/s\n";
        std::cout << "[" << test_name << "] throughput: " << throughput_kibps << " kiB/s\n";
        std::cout << "[" << test_name << "] throughput: " << throughput_mibps << " MiB/s\n";
    }
}

//async multi-get
void test_multi_get(DB_PPC& db, const std::vector<std::string>& keys_to_get,
                         std::string &test_name, int max_threads=0, size_t batch_size=100) {
    if (max_threads == 0) {
        //single-threaded version
        test_multi_get_sync(db, keys_to_get, test_name, batch_size);
        return;
    }
    std::cout << "Asynchronous multi-get execution (max threads: " << max_threads << ")" << std::endl;
    if (keys_to_get.empty()) {
        std::cout << "[" << test_name << "] No keys to retrieve\n";
        return;
    }

    size_t total_keys_size = 0;
    size_t total_values_size = 0;
    size_t total_retrieved_keys = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    ThreadPool pool(max_threads);
    std::vector<std::future<std::vector<std::string>>> futures;

    for (size_t i = 0; i < keys_to_get.size(); i += batch_size) {
        auto batch_start = keys_to_get.begin() + i;
        auto batch_end = (i + batch_size) < keys_to_get.size()
                                         ? keys_to_get.begin() + i + batch_size
                                         : keys_to_get.end();
        std::vector<std::string> batch_keys(batch_start, batch_end);

        // Calculate batch keys size
        for (const auto& key : batch_keys) {
            total_keys_size += key.size();
        }

        futures.push_back(pool.enqueue([&db, batch_keys]() {
            return db.multi_get(batch_keys);
        }));
    }

    // Collect results from the futures
    for (auto& future : futures) {
        std::vector<std::string> values = future.get();
        for (const auto& value : values) {
            if (!value.empty()) {
                total_retrieved_keys++;
                total_values_size += value.size();
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    double total_size = total_keys_size + total_values_size;

    std::cout << "[" << test_name << "] keys to get: " << keys_to_get.size() << "\n";
    std::cout << "[" << test_name << "] retrieved keys number: " << total_retrieved_keys << "\n";
    std::cout << "[" << test_name << "] retrieved keys size: " << total_keys_size << "\n";
    std::cout << "[" << test_name << "] retrieved values size: " << total_values_size << "\n";
    std::cout << "[" << test_name << "] retrieved tot size: " << total_size << "\n";
    std::cout << "[" << test_name << "] time: " << elapsed.count() << "\n";

    if (elapsed.count() > 0) {
        double throughput_bps = total_size / elapsed.count();
        double throughput_kibps = throughput_bps / 1024;
        double throughput_mibps = throughput_kibps / 1024;

        std::cout << "[" << test_name << "] throughput: " << throughput_bps << " B/s\n";
        std::cout << "[" << test_name << "] throughput: " << throughput_kibps << " kiB/s\n";
        std::cout << "[" << test_name << "] throughput: " << throughput_mibps << " MiB/s\n";
    }
}

void test_delete_operations(DB_PPC& db,
                          const std::vector<std::string>& keys_to_delete,
                          std::string &test_name) {
    if (keys_to_delete.empty()) {
        throw std::invalid_argument("keys_to_delete must not be empty");
    }

    size_t delete_count = 0;
    size_t keys_size = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (const auto& key : keys_to_delete) {
        if (db.single_remove(key)) {
            delete_count++;
            keys_size += key.size();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Check if all deletes were performed
    if (delete_count != keys_to_delete.size()) {
        std::cout << "[" << test_name << "] Not all delete operations were successful\n";
        std::cout << "[" << test_name << "] Attempted: " << keys_to_delete.size()
                  << ", Succeeded: " << delete_count << "\n";
    }

    std::cout << "[" << test_name << "] total delete operations: " << delete_count << "\n";
    std::cout << "[" << test_name << "] processed keys size: " << keys_size << "\n";
    std::cout << "[" << test_name << "] time: " << elapsed.count() << " seconds\n";

    if (elapsed.count() > 0) {
        double throughput_bps = keys_size / elapsed.count();
        double throughput_kibps = throughput_bps / 1024;
        double throughput_mibps = throughput_kibps / 1024;

        std::cout << "[" << test_name << "] throughput: " << throughput_bps << " B/s\n";
        std::cout << "[" << test_name << "] throughput: " << throughput_kibps << " kiB/s\n";
        std::cout << "[" << test_name << "] throughput: " << throughput_mibps << " MiB/s\n";
    }
}

void test_mixed_operations(DB_PPC& db,
                         const std::vector<std::string>& keys_to_remove,
                         const std::vector<std::string>& keys_to_insert,
                         const std::vector<std::string>& values_to_insert,
                         const std::vector<std::string>& keys_to_get,
                         std::string &test_name) {
    if (keys_to_remove.size() != keys_to_insert.size()) {
        std::string keys_to_remove_size_str, keys_to_insert_size_str;
        keys_to_remove_size_str = std::to_string(keys_to_remove.size());
        keys_to_insert_size_str = std::to_string(keys_to_insert.size());
        throw std::invalid_argument("keys_to_remove (" + keys_to_remove_size_str + ") and keys_to_insert (" + keys_to_insert_size_str + ") must have the same size");
    }
    if (keys_to_insert.size() != values_to_insert.size()) {
        std::string keys_to_insert_size_str, values_to_insert_size_str;
        keys_to_insert_size_str = std::to_string(keys_to_insert.size());
        values_to_insert_size_str = std::to_string(values_to_insert.size());
        throw std::invalid_argument("keys_to_insert (" + keys_to_insert_size_str + ") and values_to_insert (" + values_to_insert_size_str + ") must have the same size");
    }

    assert(keys_to_remove.size() == keys_to_insert.size());

    // Create and shuffle operation flags
    std::vector<bool> operations;
    operations.resize(keys_to_remove.size() + keys_to_get.size());
    std::fill(operations.begin(), operations.begin() + keys_to_remove.size(), true);
    std::fill(operations.begin() + keys_to_remove.size(), operations.end(), false);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(operations.begin(), operations.end(), gen);

    size_t operations_count = 0;
    size_t remove_count = 0;
    size_t insert_count = 0;
    size_t get_count = 0;
    size_t inserted_keys_size = 0, inserted_values_size = 0, got_keys_size = 0, got_values_size = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    size_t remove_idx = 0;
    size_t insert_idx = 0;
    size_t get_idx = 0;

    for (bool do_remove_insert : operations) {
        if (do_remove_insert) {
            // Remove
            const std::string old_key = keys_to_remove[remove_idx];
            if (db.single_remove(old_key)) {
                remove_count++;
            }

            //insert
            const std::string& new_key = keys_to_insert[insert_idx];
            std::string new_value = values_to_insert[insert_idx];
            db.insert_single(new_key, new_value.begin(), new_value.end());
            insert_count++;

            //got sizes
            inserted_keys_size += new_key.size();
            inserted_values_size += new_value.size();

            //counting params
            remove_idx++;
            insert_idx++;
        } else {
            //single-get
            const std::string& key = keys_to_get[get_idx];
            std::string value;
            value = db.single_get(key);

            //got sizes
            got_keys_size += key.size();
            got_values_size += value.size();

            //counting params
            get_count++;
            get_idx++;
        }

        operations_count++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    const size_t inserted_total_size = inserted_keys_size + inserted_values_size;
    const size_t got_total_size = got_keys_size + got_values_size;

    //check if all gets are performed
    if (get_idx != keys_to_get.size()) {
        std::cout << "[" << test_name << "] Not all get operations were performed\n";
    }
    //check if all remove are performed
    if (remove_idx != keys_to_remove.size()) {
        std::cout << "[" << test_name << "] Not all remove operations were performed\n";
    }
    //check if all insert are performed
    if (insert_idx != keys_to_insert.size()) {
        std::cout << "[" << test_name << "] Not all insert operations were performed\n";
    }


    std::cout << "[" << test_name << "] total operations: " << operations_count << "\n";
    std::cout << "[" << test_name << "] #remove operations: " << remove_count << "\n";
    std::cout << "[" << test_name << "] #insert operations: " << insert_count << "\n";
    std::cout << "[" << test_name << "] #get operations: " << get_count << "\n";

    std::cout << "[" << test_name << "] inserted keys size: " << inserted_keys_size << "\n";
    std::cout << "[" << test_name << "] inserted values size: " << inserted_values_size << "\n";
    std::cout << "[" << test_name << "] inserted tot size: " << inserted_total_size << "\n";

    std::cout << "[" << test_name << "] got keys size: " << got_keys_size << "\n";
    std::cout << "[" << test_name << "] got values size: " << got_values_size << "\n";
    std::cout << "[" << test_name << "] got tot size: " << got_total_size << "\n";

    std::cout << "[" << test_name << "] time: " << elapsed.count() << "\n";

    if (elapsed.count() > 0) {
        const double inserted_throughput_bps = 1.0 * inserted_total_size / elapsed.count();
        const double inserted_throughput_kibps = 1.0 * inserted_throughput_bps / 1024;
        const double inserted_throughput_mibps = 1.0 * inserted_throughput_kibps / 1024;

        std::cout << "[" << test_name << "] insertion throughput: " << inserted_throughput_bps << " B/s\n";
        std::cout << "[" << test_name << "] insertion throughput: " << inserted_throughput_kibps << " kiB/s\n";
        std::cout << "[" << test_name << "] insertion throughput: " << inserted_throughput_mibps << " MiB/s\n";

        const double got_throughput_bps = 1.0 * got_total_size / elapsed.count();
        const double got_throughput_kibps = 1.0 * got_throughput_bps / 1024;
        const double got_throughput_mibps = 1.0 * got_throughput_kibps / 1024;

        std::cout << "[" << test_name << "] get throughput: " << got_throughput_bps << " B/s\n";
        std::cout << "[" << test_name << "] get throughput: " << got_throughput_kibps << " kiB/s\n";
        std::cout << "[" << test_name << "] get throughput: " << got_throughput_mibps << " MiB/s\n";
    }
}

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " [options]\n"
              << "Options:\n"
              << "  --parquetfile <path>         Path to parquet file (required)\n"
              << "  --db-path <path>             Database directory path (required)\n"
              << "  --key-column <name>          Key column name (required)\n"
              << "  --block-size <size>          Block size in bytes (required)\n"
              << "  --compression <type>         Compression type (required)\n"
              << "  --compression-level <level>  Compression level (required)\n"
              << "  --noktis                    Exclude keys to insert (ktis) from insertion (default: false)\n"
              << "  --initial-cleanup            Enable initial cleanup (default: false)\n"
              << "  --final-cleanup              Enable final cleanup (default: false)\n"
              << "  --max-dict-bytes <bytes>     Maximum bytes for compression dictionary (default: 0)\n"
              << "  --zstd-max-train-bytes <bytes> Maximum bytes to train ZSTD compression (default: 0)\n"
              << "  --run-test <test>            Test to run (default: all)\n"
              << "  --probability <value>        Probability for mixed test\n"
              << "  --sampling-rate <value>      Sampling rate for mixed test\n"
              << "  --sampling-rate-zipf <value> Sampling rate (zipf) for mixed test\n"
              << "Example:\n"
              << "  " << prog_name << " --parquetfile data.parquet --db-path ./db --key-column id \\\n"
              << "     --block-size 4096 --compression zstd --compression-level 3 --run-test insert \\\n"
              << "     --max-dict-bytes 1024 --zstd-max-train-bytes 1024\n";
}

rocksdb::CompressionType parse_compression(const std::string& compr_str) {
    if (compr_str == "none") return rocksdb::kNoCompression;  // Option 1: No compression
    if (compr_str == "zstd") return rocksdb::kZSTD;
    if (compr_str == "zlib") return rocksdb::kZlibCompression;
    if (compr_str == "snappy") return rocksdb::kSnappyCompression;
    if (compr_str == "lz4") return rocksdb::kLZ4Compression;
    if (compr_str == "lz4hc") return rocksdb::kLZ4HCCompression;
    if (compr_str == "bzip2") return rocksdb::kBZip2Compression;
    if (compr_str == "xpress") return rocksdb::kXpressCompression;
    throw std::invalid_argument("Invalid compression type: " + compr_str);
}

int main(int argc, char** argv) {
    CLI::App app{"PPC methods for source-code data based upon RocksDB"};

    // Default values
    std::string parquet_file, db_path, key_column_name, value_column_name;
    size_t block_size = 0;
    std::string compression_str;
    int compression_level = 0;
    bool noktis = false;  // Added noktis parameter with default false
    std::string run_test = "all", probability, sampling_rate, sampling_rate_zipf;
    size_t max_dict_bytes = 0;      // Added max_dict_bytes with default 0
    size_t zstd_max_train_bytes = 0; // Added zstd_max_train_bytes with default 0
    size_t batch_size = 100000;        // Added batch_size parameter with default 100
    int nt = 1;                     // set default to 1

    // Add options
    app.add_option("--parquetfile", parquet_file, "Path to parquet file");
    app.add_option("--db-path", db_path, "Database path");
    app.add_option("--key-column", key_column_name, "Key column name")->default_val("key");
    app.add_option("--value-column", value_column_name, "Value column name")->default_val("content");
    app.add_option("--block-size", block_size, "Block size");
    app.add_option("--compression", compression_str, "Compression type");
    app.add_option("--compression-level", compression_level, "Compression level");
    app.add_flag("--noktis", noktis, "Exclude keys to insert (ktis) from insertion")->default_str("false");

    // Add dictionary training options
    app.add_option("--max-dict-bytes", max_dict_bytes, "Maximum bytes for compression dictionary")->default_val("0");
    app.add_option("--zstd-max-train-bytes", zstd_max_train_bytes, "Maximum bytes to train ZSTD compression")->default_val("0");

    // Replace single cleanup flag with two separate flags
    bool initial_cleanup = false;
    bool final_cleanup = false;
    app.add_flag("--initial-cleanup", initial_cleanup, "Enable initial cleanup")->default_str("false");
    app.add_flag("--final-cleanup", final_cleanup, "Enable final cleanup")->default_str("false");

    // Add run-test option with validation
    auto run_test_option = app.add_option("--run-test", run_test, "Test to run")
        ->default_val("all");
        // ->check(CLI::IsMember({"all", "iterate-all", "single-get", "multi-get", "mixed", "insert", "delete"}));

    // Add probability and sampling_rate options
    std::string probability_str = "0.0";
    std::string sampling_rate_str = "0.0";
    auto probability_option = app.add_option("--probability", probability, "Probability for mixed test");
    auto sampling_rate_option = app.add_option("--sampling-rate", sampling_rate, "Sampling rate for mixed test");
    auto sampling_rate_zipf_option = app.add_option("--sampling-rate-zipf", sampling_rate_zipf, "Sampling rate for mixed test");

    // Add new parameters
    app.add_option("--nt", nt, "Number of threads")->default_val("1");
    app.add_option("--batch-size", batch_size, "Batch size for operations")->default_val("100");

    CLI11_PARSE(app, argc, argv);

    ///////////////////////////////////////

    // Validate required arguments
    if (parquet_file.empty()) {
        std::cerr << "Error: Missing required argument --parquetfile\n";
        print_usage(argv[0]);
        return 1;
    }
    if (db_path.empty()) {
        std::cerr << "Error: Missing required argument --db-path\n";
        print_usage(argv[0]);
        return 1;
    }
    if (key_column_name.empty()) {
        std::cerr << "Error: Missing required argument --key-column\n";
        print_usage(argv[0]);
        return 1;
    }
    if (block_size == 0) {
        std::cerr << "Error: Missing required argument --block-size\n";
        print_usage(argv[0]);
        return 1;
    }
    if (compression_str.empty()) {
        std::cerr << "Error: Missing required argument --compression\n";
        print_usage(argv[0]);
        return 1;
    }

    ///////////////////////////////////////

    std::cout << "parsed args" << std::endl;

    try {
        auto compression = parse_compression(compression_str);

        // Prepare DB directory
        if (initial_cleanup) {
            std::string command;

            command = "rm -rf " + db_path;
            system(command.c_str());

            command = "mkdir -p " + db_path;
            system(command.c_str());
        }

        // Initialise query data
        std::vector<std::string> ktgs, ktgs_zipf;
        std::vector<std::string> ktrs;
        std::vector<std::string> ktis;
        std::vector<std::string> vtis;

        std::string basepath = get_basepath(parquet_file);
        if (run_test == "insert") {
            if (noktis) {
                // Construct the noktis filename format
                parquet_file = basepath +
                               ".noktis-" +
                               probability +
                               "-" +
                               sampling_rate +
                               ".parquet";

                std::cout << "Using noktis parquet file: " << parquet_file << std::endl;
            }else {
                parquet_file = basepath +
                   ".parquet";
                std::cout << "Using parquet file: " << parquet_file << std::endl;
            }
        }
        if (run_test == "all" or run_test == "single-get" or run_test == "multi-get" or run_test == "delete" or run_test == "mixed") {
            // Read the data from parquet files
            std::cout << "debug: reading data" << std::endl;
            read_ordinary_data(basepath, probability, sampling_rate, ktgs, ktrs, ktis, vtis);

            std::cout << "Successfully read ordinary query data from parquet files with base: "
                      << get_basepath(parquet_file) << std::endl;
        }
        if (run_test == "all" or run_test == "single-get-zipf" or run_test == "multi-get-zipf") {
            // Read the data from parquet files
            read_zipfian_data(basepath, probability, sampling_rate_zipf, ktgs_zipf);

            std::cout << "Successfully read zipfian query data from parquet files with base: "
                      << get_basepath(parquet_file) << std::endl;
        }

        // Print test configuration
        std::cout << "Test Configuration:\n"
                  << "  Dataset: " << parquet_file << "\n"
                  << "  Directory: " << db_path << "\n"
                  << "  Key column: " << key_column_name << "\n"
                  << "  Value column: " << value_column_name << "\n"
                  << "  Block size: " << block_size << " bytes\n"
                  << "  Compression: " << compression_str << "\n"
                  << "  Compression level: " << compression_level << "\n"
                  << "  Exclude KTIS: " << (noktis ? "yes" : "no") << "\n"
                  << "  Max dictionary bytes: " << max_dict_bytes << " bytes\n"
                  << "  Max ZSTD train bytes: " << zstd_max_train_bytes << " bytes\n"
                  << "  Initial cleanup: " << (initial_cleanup ? "yes" : "no") << "\n"
                  << "  Final cleanup: " << (final_cleanup ? "yes" : "no") << "\n"
                  << "  Tests to run: " << run_test << "\n"
                  << "  Probability: " << probability << "\n"
                  << "  Sampling rate: " << sampling_rate << "\n"
                  << "  Sampling rate (Zipf): " << sampling_rate_zipf << "\n"
                  << "  Par. degree: " << nt << "\n"
                  << "  Batch size: " << batch_size << "\n";

        DB_PPC db(db_path);
        std::string test_name;

        if (initial_cleanup) {
            db.create_db(block_size, compression, compression_level, max_dict_bytes, zstd_max_train_bytes);
            std::cout << "DB created successfully\n";
        }else{
            db.open_db(block_size, compression, compression_level, max_dict_bytes, zstd_max_train_bytes);
            std::cout << "DB opened successfully\n";
        }

        if (run_test == "all" || run_test == "iterate-all") {
            test_name = "Iterate all";
            std::cout << "Testing " << test_name << "..." << std::endl;
            test_iterate_all(db, test_name);
        }

        if (run_test == "all" || run_test == "insert") {
            test_name = "Insertion";
            std::cout << "Testing " << test_name << "..." << std::endl;
            test_insertion(db, parquet_file, test_name, key_column_name, value_column_name, nt);
        }

        if (run_test == "all" || run_test == "single-get") {
            test_name = "Single gets";
            std::cout << "Testing " << test_name << "..." << std::endl;
            test_single_gets(db, ktgs, test_name, nt);
        }

        if (run_test == "all" || run_test == "single-get-zipf") {
            test_name = "Single gets Zipf";
            std::cout << "Testing " << test_name << "..." << std::endl;
            test_single_gets(db, ktgs_zipf, test_name, nt);
        }

        if (run_test == "all" || run_test == "multi-get") {
            test_name = "Multi gets";
            std::cout << "Testing " << test_name << "..." << std::endl;
            test_multi_get(db, ktgs, test_name, nt, batch_size);
        }

        if (run_test == "all" || run_test == "multi-get-zipf") {
            test_name = "Multi gets Zipf";
            std::cout << "Testing " << test_name << "...\n";
            test_multi_get(db, ktgs_zipf, test_name, nt, batch_size);
        }

        if (run_test == "delete") {
            // We assume that ktis are in the DB
            test_name = "Delete";
            std::cout << "Testing " << test_name << "...\n";
            test_delete_operations(db, ktis, test_name);
        }

        if (run_test == "mixed") {
            // We assume that ktrs and ktgs are in the DB
            // We assume that ktis are NOT in the DB
            test_name = "Mixed operations";
            std::cout << "Testing " << test_name << "...\n";
            test_mixed_operations(db, ktrs, ktis, vtis, ktgs, test_name);
        }

        db.close_db();

        if (final_cleanup) {
            std::string command;

            command = "rm -rf " + db_path;
            system(command.c_str());

            command = "mkdir -p " + db_path;
            system(command.c_str());
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}