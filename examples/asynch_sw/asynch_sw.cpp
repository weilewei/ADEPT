#include "driver.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <bits/stdc++.h>
#include <functional>

// Pull in the reference implementation of P2300:
#include <stdexec/execution.hpp>

#include "exec/static_thread_pool.hpp"
#include <thread>
#include <chrono>

using namespace std;

using namespace stdexec;
using stdexec::sync_wait;

constexpr int MAX_REF_LEN    =      1200;
constexpr int MAX_QUERY_LEN  =       300;
constexpr int GPU_ID         =         0;

constexpr unsigned int DATA_SIZE = std::numeric_limits<unsigned int>::max();;

// scores
constexpr short MATCH          =  3;
constexpr short MISMATCH       = -3;
constexpr short GAP_OPEN       = -6;
constexpr short GAP_EXTEND     = -1;

// thread pool size
constexpr int THREAD_POOL_SIZE =  4;

// ------------------------------------------------------------------------------------ //

//
// verify correctness
//

bool verify_correctness(string file1, string file2);

// ------------------------------------------------------------------------------------ //

//
// main function
//
int
main(int argc, char* argv[])
{

  //
  // print banner and sanity checks
  //
  sync_wait(then(just(), [&]()
  {
    std::cout <<                               std::endl;
    std::cout << "-----------------------" <<  std::endl;
    std::cout << "       ASYNC DNA       " <<  std::endl;
    std::cout << "-----------------------" <<  std::endl;
    std::cout <<                               std::endl;

    // check command line arguments
    if (argc < 5)
    {
        cout << "USAGE: asynch_sw <reference_file> <query_file> <output_file> <res_file>" << endl;
        exit(-1);
    }
  }));

// ------------------------------------------------------------------------------------ //

  // variables
  string refFile = argv[1];
  string queFile = argv[2];
  string out_file = argv[3];
  string res_file = argv[4];

  unsigned batch_size;
  int total_alignments;

  // sequence vectors
  std::vector<string> ref_sequences, que_sequences;

  // adept driver pointer
  std::unique_ptr<ADEPT::driver> sw_driver;

  // vector for cpu counters
  std::vector<int> works(THREAD_POOL_SIZE);

  // initialize a thread pool
  exec::static_thread_pool ctx{THREAD_POOL_SIZE};

// ------------------------------------------------------------------------------------ //

  // get a scheduler from the thread pool
  scheduler auto sch = ctx.get_scheduler();

  // get a sender from the thread pool scheduler
  sender auto begin = schedule(sch);

// ------------------------------------------------------------------------------------ //

  // read sequences from files
  sender auto readseqs = then(begin, [&]()
  {
    // open files
    ifstream ref_file(refFile);
    ifstream quer_file(queFile);

    unsigned largestA = 0, largestB = 0;

    int totSizeA = 0, totSizeB = 0;
    std::string   lineR, lineQ;

    // extract reference sequences
    if(ref_file.is_open() && quer_file.is_open())
    {
      while(getline(ref_file, lineR))
      {
        getline(quer_file, lineQ);

        if(lineR[0] == '>')
        {
          if (lineR[0] == '>')
            continue;
          else
          {
            std::cout << "FATAL: Mismatch in lines" << std::endl;
            exit(-2);
          }
        }
        else
        {
          if (lineR.length() <= MAX_REF_LEN && lineQ.length() <= MAX_QUERY_LEN)
          {
            ref_sequences.push_back(lineR);
            que_sequences.push_back(lineQ);

            totSizeA += lineR.length();
            totSizeB += lineQ.length();

            if(lineR.length() > largestA)
              largestA = lineR.length();

            if(lineQ.length() > largestA)
              largestB = lineQ.length();
          }
        }
        // check sanctity
        if (ref_sequences.size() == DATA_SIZE)
            break;
      }

      // update total alignments
      total_alignments = ref_sequences.size();

      // close the files
      ref_file.close();
      quer_file.close();
    }
  });

// ------------------------------------------------------------------------------------------ //

  // initialize the adept driver. cannot be done
  // in parallel to file reading as total_alignments is needed.
  sender auto adept_init = then(readseqs, [&]()
  {
    // learned the hard way: this must be called before the driver is instantiated
    batch_size = ADEPT::get_batch_size(GPU_ID, MAX_QUERY_LEN, MAX_REF_LEN, 100);// batch size per GPU

    // instantiate ADEPT driver only after calling the get_batch_size
    sw_driver  = std::unique_ptr<ADEPT::driver>(new ADEPT::driver());

    // vector of scores.
    std::vector<short> scores = {MATCH, MISMATCH};

    // ADEPT::gap scores object
    ADEPT::gap_scores gaps(GAP_OPEN, GAP_EXTEND);

    // initialize the ADEPT::driver
    sw_driver->initialize(scores, gaps, ADEPT::options::ALG_TYPE::SW, ADEPT::options::SEQ_TYPE::DNA,
                          ADEPT::options::CIGAR::YES, ADEPT::options::SCORING::ALNS_AND_SCORE,
                          MAX_REF_LEN, MAX_QUERY_LEN, total_alignments, batch_size, GPU_ID);
  });

// ------------------------------------------------------------------------------------------ //

  // launch adept kernel here. TODO: Go inside the kernel and port to Sx/Rx model.
  sender auto adept_launch = then(adept_init, [&]()
  {
        sw_driver->kernel_launch(ref_sequences, que_sequences);
  });

// ------------------------------------------------------------------------------------------ //

  // bulk wait for kernel to finish

  sender auto kernel_wait =
  /* FIXME: Getting the following linker error when using bulk.
  undefined reference to `stdexec::get_completion_scheduler<stdexec::__receivers::set_value_t>'

  transfer_just(sch)
  | bulk(THREAD_POOL_SIZE, [&](size_t k)
  {
    while(sw_driver->kernel_done() != true)
      works[k]++;
  });
  * work around using this */
  then(adept_launch, [&]()
  {
    while(sw_driver->kernel_done() != true)
      works[0]++;
  });

// ------------------------------------------------------------------------------------------ //

  // copy results from device to host
  sender auto dth_launch = then(kernel_wait, [&]()
  {
    sw_driver->mem_cpy_dth();
  });

// ------------------------------------------------------------------------------------------ //

  // bulk wait for dth to finish
  sender auto dth_wait =
  /* FIXME: Getting the following linker error when using bulk.
  undefined reference to `stdexec::get_completion_scheduler<stdexec::__receivers::set_value_t>'

  transfer_just(sch)
  | bulk(THREAD_POOL_SIZE, [&](size_t k)
  {
    while(sw_driver->dth_done() != true)
      works[k]++;
  })
  | then([&]()
  {
    int total_work = 0;

    for(auto &work : works)
      total_work += work;

    return total_work;
  });
  *
  * work around using this */
  then(dth_launch, [&]()
  {
    while(sw_driver->dth_done() != true)
      works[0]++;
    return works[0];
  });

  auto [work_cpu] = sync_wait(std::move(dth_wait)).value();

// ------------------------------------------------------------------------------------------ //

  // get and write results
  sender auto results = then(just(), [&]()
  {
    // get results from GPU
    auto results = sw_driver->get_alignments();

    ofstream results_file(out_file);

    std::cout << std::endl << "STATUS: Writing results..." << std::endl;

    // write the results header
    results_file << "alignment_scores\t"     << "reference_begin_location\t" << "reference_end_location\t"
               << "query_begin_location\t" << "query_end_location"         << std::endl;

    for(int k = 0; k < ref_sequences.size(); k++){
      results_file<<results.top_scores[k]<<"\t"<<results.ref_begin[k]<<"\t"<<results.ref_end[k] - 1<<
      "\t"<<results.query_begin[k]<<"\t"<<results.query_end[k] - 1<<endl;
    }

    // free the results and close the files
    results.free_results(ADEPT::options::SCORING::ALNS_AND_SCORE);
    results_file.flush();
    results_file.close();
  })
  // clean up the driver as well
  | then([&]()
  {
    sw_driver->cleanup();
  });

// ------------------------------------------------------------------------------------------ //

  // print the work done by cpu. printing here instead of at the end for legacy compatibility

  sync_wait(then(just(work_cpu), [](int &&wcpu)
  {
    std::cout << "total " << THREAD_POOL_SIZE << " x cpu pool work (counts) done while GPU was busy: "<< wcpu << std::endl;
  }));

// ------------------------------------------------------------------------------------------ //

  // verify correctness
  auto [status] = sync_wait(then(results, [&]()
  {
    // return state to verify correctness
    int correct = 0;

    if(!verify_correctness(res_file, out_file))
      correct = -1;

    if(!correct)
      std::cout << "STATUS: Correctness test passed" << std::endl;
    else
      std::cout << "STATUS: Correctness test failed" << std::endl;

    return correct;

  })).value();

  return status;
}

// ------------------------------------------------------------------------------------------ //

// function to verify correctness
bool verify_correctness(string file1, string file2)
{
  // open the ground truth results and the generated results
  ifstream ref_file(file1);
  ifstream test_file(file2);

  // strings to hold the lines
  string ref_line, test_line;

  // compare the ground truths and the generated results line by line
  if(ref_file.is_open() && test_file.is_open())
  {
    while(getline(ref_file, ref_line) && getline(test_file, test_line)){
      if(test_line != ref_line){
        return false;
      }
    }
    // close the files
    ref_file.close();
    test_file.close();
  }
  else
    // false if can't open either of the files
    return false;

  // all good if we reach here
  return true;
}

// ------------------------------------------------------------------------------------ //
