---
name: accelopt
description: Optimize a kernel and store the optimization trace in a folder
argument-hint: [kernel-path], [supabase-url]
---

# Workflow: Beam search + planner-executor-summarizer agents
Start from the initial kernel at `kernel-path`.

At each iteration $i$ of beam search, the planner agent generates $N$ plans for each kernel in a set of $B$ candidate kernels augmented with experiences from iteration $i-1$. After that, the executor agent implements every plan with $K$ attempts, generating $B\times N\times K$ kernels in total. By sampling multiple plans for the same candidate, the planner explores diverse optimization strategies, and multiple executor attempts increase the robustness of plan implementation against syntactic and semantic errors. From these generated kernels, high-quality optimizations are selected for the summarizer agent to generate experience items, which are used in the curation of the optimization memory. Finally, $B$ kernels are selected to be explored in the next iteration from those $(B + B\times N\times K)$ kernels. 

The optimization memory is maintained as a queue of optimization items with a capacity cap (ExpN). Each new iteration can append up to TopK experience items to the tail, while the oldest entries in the memory will be discarded once ExpN is reached. Each experience item in the optimization memory consists of a slow-fast kernel pair and the corresponding generalizable optimization strategy curated by the summarizer agent. To prevent irrelevant code from distracting the planner, the summarizer should extract the optimized segment of each pair as pseudocode. 

# Kernel Database
Detect if a supabase instance is set up in the environment at `supabase-url`. If not, quit and help the user set it up. `schema_v2.txt` shows the SQL schema to manage generated kernels. `db_utils.py` captures the common usage patterns of this kernel database with .

# Helper functions
- `kernel_wrapper.py`: contains the `NKIKernel` class that wraps around the kernel code and provides utility functions for profiling, such as measuring latency and extracting kernel features.
- `flb_wrapper.py`: similar to `kernel_wrapper.py`, but designed for FlashInfer-Bench kernels.


