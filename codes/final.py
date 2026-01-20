import preprocess
import llm_query
import slice
import get_results
import argparse
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='DualLM: A tool for processing commits and LLM queries')
    parser.add_argument('--not-reliable', type=str, nargs='+', default=[], help='List of commits to be processed')
    parser.add_argument('--step1-out-file', type=str, default='data/results/step1_res.txt', help='Output file for eval step 1 results')
    parser.add_argument('--step2-out-file', type=str, default='data/results/step2_res.txt', help='Output file for eval step 2 results')

   
    # Parse arguments
    args = parser.parse_args()


  

    out_str=get_results.parse_sliceLM_results(args.not_reliable, args.step1_out_file, args.step2_out_file) 
    print("Final results:\n", out_str)



if __name__ == "__main__":
    main()
    
    