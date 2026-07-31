#include "emp-aby/emp-aby.h"

template <typename IO>
class VectorGen
{
private:
  LUT<IO> *bit_to_a;
  MPIOChannel<IO> *io;
  ThreadPool *pool;
  HE<IO> *he;

public:
  int num_party, party, batch_size;
  // std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>
  //     bitwise_ctxts;
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> bitwise_ctxts,
      minus_one_ctxts;
  std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> products;
  int pool_size;
  int lambda;
  int curr_table;
  // P : c_0 *..*c_i, Q: (1 - c_0) *..* (1 - c_i)
  // indices for testing
  std::vector<int64_t> indices;

  // void getCts();
  // void compute_partial_products();
  // void compute_one_hot_vector_helper(int start, int end);
  VectorGen(int num_party, int party, MPIOChannel<IO> *io, ThreadPool *pool,
            HE<IO> *he, int pool_size)
  {
    this->pool = pool;
    this->io = io;
    this->num_party = num_party;
    this->party = party;
    this->he = he;
    this->pool_size = pool_size;

    // std::cout << "lut\n";
    this->batch_size = he->cc->GetCryptoParameters()
                           ->GetElementParams()
                           ->GetCyclotomicOrder() /
                       2;
    // std::cout << "get cts\n";
    // std::cout << "one hot vector size " << one_hot_vector.size() << "\n";
  }

  ~VectorGen()
  {
    delete bit_to_a;
    bitwise_ctxts.clear();
    minus_one_ctxts.clear();
    products.clear();
  }

  void getCts()
  {
    indices.resize(batch_size, 0);
    std::vector<int64_t> tmp(batch_size, 1);
    auto minusOne = he->cc->MakePackedPlaintext(tmp);
    tmp.resize(batch_size);
    for (int i = 0; i < pool_size * lambda; ++i)
    {
      for (int j = 0; j < batch_size; ++j)
      {
        tmp[j] = bit_to_a->lut_share[(i * batch_size + j) * 2];
        indices[j] = (indices[j] + (tmp[j] << i) % he->q) % he->q;
      }
      auto plain1 = he->cc->MakePackedPlaintext(tmp);
      auto ctxt = he->cc->Encrypt(he->pk, plain1);
      bitwise_ctxts.push_back(ctxt);
    }
    indices.clear();

    if (party == ALICE)
    {
      std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> tmp1;
      for (int i = 2; i <= num_party; ++i)
      {
        he->deserialize_recv(tmp1, i, 0, emp::NORM_MSG, false);
        for (size_t j = 0; j < tmp1.size(); ++j)
          he->cc->EvalAddInPlace(bitwise_ctxts[j], tmp1[j]);
      }
      he->serialize_sendall(bitwise_ctxts);
    }
    else
    {
      he->serialize_send(bitwise_ctxts, 1, 0, emp::NORM_MSG, false);
      he->deserialize_recv(bitwise_ctxts, 1, 0, emp::NORM_MSG, false);
    }

    for (size_t i = 0; i < bitwise_ctxts.size(); ++i)
    {
      auto ctxt = he->cc->EvalSub(minusOne, bitwise_ctxts[i]);
      minus_one_ctxts.push_back(ctxt);
    }
  }

  void compute_one_hot_vector_helper(int start, int end)
  {
    int findCommon = (start ^ end) ^ ((1 << pool_size) - 1);
    // std::cout << start << " " << end << "\n";
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ctxt = nullptr;
    products.clear();
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> commons;
    for (size_t i = 0; i < pool_size; ++i)
    {
      if (((findCommon >> i) & 1) == 1)
      {
        auto tmp = minus_one_ctxts[pool_size * curr_table + i];
        if (((start >> i) & 1) == 1)
          tmp = bitwise_ctxts[pool_size * curr_table + i];

        commons.push_back(tmp);
        // if (ctxt == nullptr)
        //   ctxt = tmp;
        // else
        // {
        //   auto tmp2 = he->cc->EvalMult(ctxt, tmp);
        //   ctxt = tmp2;
        // }
      }
      else
      {
        products.push_back({minus_one_ctxts[pool_size * curr_table + i], bitwise_ctxts[pool_size * curr_table + i]});
      }
    }

    while (commons.size() > 1)
    {
      std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> prev_commons(commons);

      commons.clear();
      for (uint i = 0; i < prev_commons.size(); i += 2)
      {
        if(i + 1 > prev_commons.size())
          commons.push_back(prev_commons[i]);
        else{
          auto tmp = he->cc->EvalMult(prev_commons[i], prev_commons[i + 1]);
          commons.push_back(tmp);
        }
      }
    }
    if(commons.size() == 1)
      ctxt = commons[0];

    if (ctxt != nullptr)
      products.insert(products.begin(), {ctxt});
    while (products.size() > 2 && products[0].size() < 128)
    {
      compute_product(products);
    }
  }

  void compute_product(
      std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>
          &products)
  {
    size_t l = products.size();
    std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>
        new_products;
    for (size_t i = 0; i < l; i = i + 2)
    {
      std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> x;
      for (size_t j = 0; j < products[i].size(); ++j)
      {
        if (i + 1 < l)
        {
          for (size_t k = 0; k < products[i + 1].size(); ++k)
          {
            auto ctxt = he->cc->EvalMult(products[i][j], products[i + 1][k]);
            x.push_back(ctxt);
          }
        }
        else
        {
          x.push_back(products[i][j]);
        }
      }
      new_products.push_back(x);
    }
    products.clear();
    for (auto v : new_products)
    {
      products.push_back(v);
    }
    new_products.clear();
  }

  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> get_product(int index)
  {
    int l = products.size();
    if (l == 1)
    {
      return products[0][index];
    }
    if (l < 1)
    {
      std::cout << "returning nullptr\n";
      exit(1);
      return nullptr;
    }
    int running_index = index;
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ctxts;
    for (int i = l - 1; i >= 0; --i)
    {
      int innerSize = products[i].size();
      ctxts.push_back(products[i][running_index % innerSize]);
      running_index = floor(index / innerSize);
    }

    while (ctxts.size() != 1)
    {
      std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> next_ctxts;
      for (size_t i = 0; i < ctxts.size(); i = i + 2)
      {
        if (i + 1 < ctxts.size())
        {
          auto ctxt = he->cc->EvalMult(ctxts[i], ctxts[i + 1]);
          next_ctxts.push_back(ctxt);
        }
        else
          next_ctxts.push_back(ctxts[i]);
      }
      ctxts.clear();
      for (auto ctxt : next_ctxts)
        ctxts.push_back(ctxt);
      next_ctxts.clear();
    }

    return ctxts[0];
  }

  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> sampleEnc(std::vector<int64_t> table, int start, int end, lbcrypto::Ciphertext<lbcrypto::DCRTPoly> prev_res = nullptr)
  {
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> resVec;
    int n = end - start;
    int threads = pool->size();
    vector<std::future<void>> futs;

    // table.size() > end, we have to wait in a thread
    // start < table.size() < end wait in a thread and also do both computations
    // table.size() > start, compute sum and send
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> priv_multiplier = nullptr;
    if (table.size() < end)
    {
      uint i = table.size() > start ? table.size() : start;
      priv_multiplier = get_product(i - start);
      for (i += 1; i < end; ++i)
      {
        he->cc->EvalAddInPlace(priv_multiplier, get_product(i));
      }
    }

    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> sums;
    sums.resize(threads, nullptr);
    int num_steps = ceil((double)n / (double)threads);
    for (int t = 0; t < threads; ++t)
    {
      futs.push_back(
          pool->enqueue([this, table, &sums, start, end, num_steps, t]
                        {
            vector<int64_t> tmp;
            for (int step = 0; step < num_steps; ++step) {
              int i = t * num_steps + step;
              if(table.size() <= start + i)
                break;
              if (i >= end)
                break;
              tmp.resize(batch_size, table[start + i] % he->q);
              auto plain = he->cc->MakePackedPlaintext(tmp);
              auto prod = he->cc->EvalMult(plain, get_product(i));
              if (sums[t] == nullptr) {
                sums[t] = prod;
              } else {
                he->cc->EvalAddInPlace(sums[t], prod);
              }
            }
            tmp.clear(); }));
    }

    for (auto &v : futs)
      v.get();
    futs.clear();

    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> res = sums[0];
    for (int t = 1; t < threads; ++t)
    {
      if (sums[t] != nullptr)
        he->cc->EvalAddInPlace(res, sums[t]);
    }

    if (priv_multiplier != nullptr)
    {
      assert(prev_res != nullptr);
      auto tmp = he->cc->EvalMult(priv_multiplier, prev_res);
      if(res != nullptr)
        he->cc->EvalAddInPlace(res, tmp);
      else
        res = tmp;
    }

    resVec.push_back(res);

    if (party == ALICE)
    {
      std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> tmp;
      for (int i = 2; i <= num_party; ++i)
      {
        he->deserialize_recv(tmp, i);
        for (size_t j = 0; j < resVec.size(); ++j)
          he->cc->EvalAddInPlace(resVec[j], tmp[j]);
      }
    }
    else
    {
      he->serialize_send(resVec, ALICE);
    }
    return resVec[0];
  }

  void getResult(std::vector<std::vector<int64_t>> &tables, int64_t *result)
  {
    assert(tables[0].size() == (1 << pool_size));
    if (this->party == ALICE) {
      std::cout << "Batch size: " << this->batch_size <<std::endl;
    }
    // std::cout << "sizes: " << tables.size() << " ";
    // for (auto &v : tables)
    //   std::cout << v.size() << " ";
    // std::cout << "\n";
    int64_t table[2] = {0, 1};
    this->bit_to_a =
        new LUT<IO>(num_party, party, io, pool, he, table, pool_size * tables.size());
    this->lambda = tables.size();
    this->getCts();
    // std::cout << "create LUT elements done" << std::endl;
    int start, end;
    int x = 1 << (int)ceil(log2(num_party));
    int n = (1 << pool_size) / x;
    if (party <= (x - num_party))
    {
      start = n * 2 * (party - 1);
      end = n * 2 * party;
    }
    else
    {
      int used = x - num_party;
      start = n * 2 * (x - num_party) + n * (party - used - 1);
      end = n * 2 * (x - num_party) + n * (party - used);
    }
    n = end - start;
          std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> resVec;

    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> res = nullptr;
    for (uint i = 0; i < tables.size(); ++i)
    {
      // std::cout << "Compute OHV " << std::endl;
      this->curr_table = i;
      compute_one_hot_vector_helper(start, end - 1);
      // std::cout << "Sample from table" << std::endl;
      res = sampleEnc(tables[i], start, end, res);

      if(party == ALICE){
        resVec.clear();
        resVec.push_back(res);
      }

      if(i != (tables.size() - 1)){
        he->bootstrap(resVec, ALICE, ALICE);
      }
      if (party == ALICE)
        he->serialize_sendall(resVec);
      else
        he->deserialize_recv(resVec, ALICE);
    }

    he->enc_to_share(resVec, result, batch_size);
  }
};
