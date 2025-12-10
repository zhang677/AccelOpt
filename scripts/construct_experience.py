import json
import random

def main(args):
    with open(args.experience_list_path, "r") as f:
        experience_list = json.load(f)

    if args.is_first:
        # Randomly select n items from experience list (no duplicates)
        # Handle case where experience_list might be shorter than n
        num_to_select = min(args.n, len(experience_list))
        selected_experience_list = random.sample(experience_list, num_to_select)
        with open(args.output_path, "w") as f:
            json.dump(selected_experience_list, f, indent=4)
    else:
        with open(args.original_rewrite_list_path, "r") as f:
            original_rewrite_list = json.load(f)
        
        # Randomly select n - len(original_rewrite_list) items from experience list (no duplicates)
        num_needed = args.n - len(original_rewrite_list)
        # Handle edge cases:
        # - If num_needed <= 0, select nothing
        # - If experience_list is empty, select nothing
        # - If experience_list is shorter than num_needed, select all available
        if num_needed > 0 and len(experience_list) > 0:
            num_to_select = min(num_needed, len(experience_list))
            selected_experience_list = random.sample(experience_list, num_to_select)
        else:
            selected_experience_list = []
        output_list = original_rewrite_list + selected_experience_list
        with open(args.output_path, "w") as f:
            json.dump(output_list, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_first", action="store_true", help="Set if this is the first run")
    parser.add_argument("--original_rewrite_list_path", type=str, default="")
    parser.add_argument("--experience_list_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    args = parser.parse_args()
    main(args)