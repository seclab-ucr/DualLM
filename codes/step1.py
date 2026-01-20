import preprocess
import llm_query
import slice
import get_results
import argparse
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='DualLM: A tool for processing commits and LLM queries')
    parser.add_argument('--commits', type=str, nargs='+', default=[], help='List of commits to be processed')
    parser.add_argument('--name', type=str, required=True, help='Name of the evaluation dataset')
    parser.add_argument('--out-file1', type=str, default='out1.txt', help='Output file for the first LLM step')
    parser.add_argument('--out-file2', type=str, default='out2.txt', help='Output file for the second LLM step')
    parser.add_argument('--repo-dir', type=str, required=True, help='Path to the repository directory')
    parser.add_argument('--summary-file', type=str, required=True, help='Path to the summary file')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--not-reliable', type=str, required=True, help='Out file of not_reliable commits.')

    # Parse arguments
    args = parser.parse_args()

    # Query the LLM with the commits
    llm_query.llm_query(args.commits, args.out_file1, args.out_file2)
   
    
    
    # Get the results from the LLM query
    not_reliable, out_str = get_results.parse_llm_results(args.out_file2)
    
    print("*not_reliable*") # the commits that are not reliable to determine its bug type.
    print(not_reliable)
    
    print("*reliable*") # the commits that are reliable to determine its bug type.
    print(out_str)
    
    # Write not_reliable to the specified file
    with open(args.not_reliable, "w") as f:
        for item in not_reliable:
            f.write(str(item) + "\n")
    
    # Get the summaries from the LLM
    llm_query.get_summaries(not_reliable, args.summary_file)
    
    preprocess.build_eval_data_for_random_given(args.name,not_reliable, args.summary_file, args.data_dir) 
  



if __name__ == "__main__":
    main()
    
    