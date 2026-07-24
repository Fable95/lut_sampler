#include "emp-aby/io/multi-io.hpp"
#include "include/vector_gen.h"

using namespace emp;

#include <typeinfo>

int party, port;

const static int threads = 8;

int num_party;


template <typename IO>
std::pair<double, double> test(MPIOChannel<IO> *io, VectorGen<IO> *vectorGen,
                            HE<IO> *he, int pool_size, int lambda = 40) {
  std::vector<std::vector<int64_t>> tables;
  for (int lam = 0; lam < lambda; ++lam) {
    tables.push_back(std::vector<int64_t>((1 << pool_size), 0));  
    for (int i = 0; i < (1 << pool_size); ++i) {
      tables[lam][i] = (i + lam * (1 << pool_size)) % he->q;
    }
  }
  int64_t *result = new int64_t[vectorGen->batch_size];
  memset(result, 0, vectorGen->batch_size * sizeof(int64_t));
  double offset = io->get_total_bytes_sent();
  auto start = clock_start();
  vectorGen->getResult(tables, result);
  double t = time_from(start) / (vectorGen->batch_size * 1000);
  double comm = io->get_total_bytes_sent() - offset;
  // std::cout << "generated shares \n";
  int64_t *indices = vectorGen->indices.data();

  if (party == ALICE) {
    int64_t *resultSH = new int64_t[vectorGen->batch_size];
    int64_t *indicesSh = new int64_t[vectorGen->batch_size];
    for (int i = 2; i <= num_party; ++i) {
      io->recv_data(i, indicesSh, vectorGen->batch_size * sizeof(int64_t));
      io->recv_data(i, resultSH, vectorGen->batch_size * sizeof(int64_t));

      for (int j = 0; j < vectorGen->batch_size; ++j) {
        indices[j] = (indices[j] + indicesSh[j]) % he->q;
        result[j] = (result[j] + resultSH[j]) % he->q;
        indices[j] = (indices[j] + he->q) % he->q;
        result[j] = (result[j] + he->q) % he->q;
      }
    }

    for (int i = 0; i < vectorGen->batch_size; ++i) {
      if ((tables[0][indices[i] % (1 << pool_size)] + he->q) % he->q != (result[i] + he->q) % he->q) {
        std::cout << "failed " << indices[i] << " "
                  << tables[0][indices[i] % (1 << pool_size)] << " " << result[i]
                  << " " << (result[i] + he->q) % he->q << "\n";
        return std::make_pair(t, comm);
      }
    }
    // std::cout << "Passed\n";

  } else {
    io->send_data(1, indices, vectorGen->batch_size * sizeof(int64_t));
    io->send_data(1, result, vectorGen->batch_size * sizeof(int64_t));
  }
  delete[] result;
  tables[0].clear();
  tables.clear();
  return std::make_pair(t, comm);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cout << "Format: test_vec_gen PartyID port num_parties [pool_size] [lambda]" << std::endl;
    exit(0);
  }
  parse_party_and_port(argv, &party, &port);
  num_party = atoi(argv[3]);

  std::vector<std::pair<std::string, unsigned short>> net_config;

  // if (argc >= 5) {
  //   const char *file = argv[4];
  //   FILE *f = fopen(file, "r");
  //   for (int i = 0; i < num_party; ++i) {
  //     char *c = (char *)malloc(15 * sizeof(char));
  //     uint p;
  //     fscanf(f, "%s %d\n", c, &p);
  //     std::string s(c);
  //     net_config.push_back(std::make_pair(s, p));
  //     fflush(f);
  //   }
  //   fclose(f);
  // } else {
    for (int i = 0; i < num_party; ++i) {
      std::string s = "127.0.0.1";
      uint p = (port + 4 * num_party * i);
      net_config.push_back(std::make_pair(s, p));
    }
  // }
  int pool_size = 9;
  if(argc >= 5){
    pool_size = atoi(argv[4]);
    // std::cout << "setting pool size to " << pool_size << std::endl;
  }
  int lambda = 1;
  if(argc >= 6){
    lambda = atoi(argv[5]);
  }
  MultiIO *io = new MultiIO(party, num_party, net_config);
  // std::cout << "io setup" << std::endl;
  ThreadPool *pool = new ThreadPool(threads);
  const long long int modulus = (1L << 32) - (1L << 30) + 1;
  int mult_depth = ceil(log2(pool_size)) * 2;
  if (mult_depth < num_party)
    mult_depth = num_party;
  HE<MultiIOBase> *he = new HE<MultiIOBase>(
      num_party, io, pool, party, modulus, mult_depth, true, false, false, 100);
  he->multiplication_keygen();
  he->rotation_keygen();
  auto start = clock_start();
  VectorGen<MultiIOBase> *vector_gen =
      new VectorGen<MultiIOBase>(num_party, party, io, pool, he, pool_size);
  double timeused = time_from(start) / (vector_gen->batch_size * 1000);
  // std::cout << "VectorGen setup time: " << timeused << " ms\n";
  double comm = io->get_total_bytes_sent();

  auto p = test<MultiIOBase>(io, vector_gen, he, pool_size, lambda);
  timeused += p.first;
  comm += p.second;
  
  std::cout << party << "\ttime used\t"
            << timeused
            << " ms\tcommunication\t" << comm / (vector_gen->batch_size)
            << " KB" << std::endl << std::endl;

  delete vector_gen;
  delete he;
  delete io;
}
