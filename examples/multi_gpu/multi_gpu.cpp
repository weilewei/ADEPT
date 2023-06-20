#include "driver.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <bits/stdc++.h>
#include <thread>
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

constexpr unsigned int DATA_SIZE = std::numeric_limits<unsigned int>::max();

// scores
constexpr short MATCH          =  3;
constexpr short MISMATCH       = -3;
constexpr short GAP_OPEN       = -6;
constexpr short GAP_EXTEND     = -1;

// ------------------------------------------------------------------------------------ //

//
// verify correctness
//
bool verify_correctness(string file1, string file2);

// ------------------------------------------------------------------------------------ //

//
// main function
//
int main(int argc, char* argv[]){

    bool verify = true;

    //
    // print banner and sanity checks
    //
    sync_wait(then(just(), [&]()
    {
        std::cout <<                               std::endl;
        std::cout << "-----------------------" << std::endl;
        std::cout << "       MULTI GPU       " << std::endl;
        std::cout << "-----------------------" << std::endl;
        std::cout <<                               std::endl;

        // check command line arguments
        if (argc < 5)
        {
            cout << "USAGE: multi_gpu <reference_file> <query_file> <output_file> <res_file>" << endl;
            cout << "result file not provided, skipping the correctness check" << endl;
            verify = false;
        }
    }));

    // ------------------------------------------------------------------------------------ //
    string refFile = argv[1];
    string queFile = argv[2];
    string out_file = argv[3];
    string res_file;

    if(verify == true) res_file = argv[4];

    vector<string> ref_sequences, que_sequences;
    string   lineR, lineQ;

    ifstream ref_file(refFile);
    ifstream quer_file(queFile);

    unsigned largestA = 0, largestB = 0;

    int totSizeA = 0, totSizeB = 0;

    int batch_size = -1;
    // ------------------------------------------------------------------------------------ //

    //
    // setup thread pool scheduler
    //

    int num_gpus = ADEPT::getNumGPUs();

    // initialize a thread pool
    exec::static_thread_pool ctx{num_gpus};
    scheduler auto sch = ctx.get_scheduler();
                                  // 2
    sender auto begin = schedule(sch);

    ADEPT::all_alns all_results(num_gpus);

    // ------------------------------------------------------------------------------------ //

    std::cout << "STATUS: Reading ref and query files" << std::endl;

    // read sequences from files
    sender auto multigpu_pipeline = then(begin, [&]()
    {
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

                        if(lineQ.length() > largestB)
                            largestB = lineQ.length();
                    }
                }

                if (ref_sequences.size() == DATA_SIZE)
                    break;
            }

            ref_file.close();
            quer_file.close();
        }

        // initialize batch_size
        batch_size = ADEPT::get_batch_size(GPU_ID, MAX_QUERY_LEN, MAX_REF_LEN, 100);// batch size per GPU

    })
    | bulk(num_gpus, [&](size_t i)
    {
        // initialize and launch the adept driver in bulk. cannot be done
        // in parallel to file reading as total_alignments is needed.

        // chunk out total_alignments per-pgu
        const int total_alignments = ref_sequences.size();

        int alns_per_gpu = total_alignments/num_gpus;
        int left_over = total_alignments%num_gpus;

		std::vector<std::string>::const_iterator start_, end_;
		start_ = ref_sequences.begin() + i * alns_per_gpu;
		if(i == num_gpus -1)
			end_ = ref_sequences.begin() + (i + 1) * alns_per_gpu + left_over;
		else
			end_ = ref_sequences.begin() + (i + 1) * alns_per_gpu;

		std::vector<std::string> lref_sequences(start_, end_);

		start_ = que_sequences.begin() + i * alns_per_gpu;
		if(i == num_gpus - 1)
			end_ = que_sequences.begin() + (i + 1) * alns_per_gpu + left_over;
		else
			end_ = que_sequences.begin() + (i + 1) * alns_per_gpu;

		std::vector<std::string> lque_sequences(start_, end_);

        // print statust at thread_id=0
        if (i == 0)
        {
            std::cout << "STATUS: Launching ADEPT driver on " << num_gpus << " GPUs" << std::endl;

            // no race condition here since these will be used in the next `then` sender
            all_results.per_gpu = alns_per_gpu;
            all_results.left_over = left_over;
            all_results.gpus = num_gpus;
        }

        // initialize scoring
        std::vector<short> scores = {MATCH, MISMATCH};
        ADEPT::gap_scores gaps(GAP_OPEN, GAP_EXTEND);

        // per-thread adept function
        all_results.results[i] = ADEPT::thread_launch(lref_sequences, lque_sequences, ADEPT::options::ALG_TYPE::SW, ADEPT::options::SEQ_TYPE::DNA,
                             ADEPT::options::CIGAR::YES, ADEPT::options::SCORING::ALNS_AND_SCORE, MAX_REF_LEN, MAX_QUERY_LEN, batch_size, i, scores, gaps);
    })
    | then([&](){
        ofstream results_file(out_file);
        int tot_gpus = all_results.gpus;

        std::cout << std::endl << "STATUS: Writing results..." << std::endl;

        // write the results header
        results_file << "alignment_scores\t"     << "reference_begin_location\t" << "reference_end_location\t"
                    << "query_begin_location\t" << "query_end_location"         << std::endl;

        for(int gpus = 0; gpus < tot_gpus; gpus++){
            int this_count = all_results.per_gpu;
            if(gpus == tot_gpus - 1) this_count += all_results.left_over;

            for(int k = 0; k < this_count; k++){
                results_file<<all_results.results[gpus].top_scores[k]<<"\t"<<all_results.results[gpus].ref_begin[k]<<"\t"<<all_results.results[gpus].ref_end[k] - 1 << "\t"<<all_results.results[gpus].query_begin[k]<<"\t"<<all_results.results[gpus].query_end[k] - 1 <<endl;
            }
        }

        for(int i = 0; i < tot_gpus; i++)
            all_results.results[i].free_results(ADEPT::options::SCORING::ALNS_AND_SCORE);

        // flush everything to stdout
        std::cout << "STATUS: Done" << std::endl << std::endl << std::flush;

        results_file.flush();
        results_file.close();
    })
    | then([&]() {

        int return_state = 0;

        if(verify == true){
            if(!verify_correctness(res_file, out_file)) return_state = -1;

        if(return_state == 0){
                cout<< "STATUS: Correctness test passed"<<endl;
            }else{
                cout<< "STATUS: Correctness test failed"<<endl;
            }
        }
        return return_state;
    });

    auto [r] = sync_wait(std::move(multigpu_pipeline)).value();

    return r;
}

// ------------------------------------------------------------------------------------------ //

bool verify_correctness(string file1, string file2){
  ifstream ref_file(file1);
  ifstream test_file(file2);
  string ref_line, test_line;

  // extract reference sequences
  if(ref_file.is_open() && test_file.is_open())
  {
    while(getline(ref_file, ref_line) && getline(test_file, test_line)){
      if(test_line != ref_line){
        return false;
      }
    }
    ref_file.close();
    test_file.close();
  }

  return true;
}

// ------------------------------------------------------------------------------------ //
