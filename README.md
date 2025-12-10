 <div align="center">
    <img src="img/main-method.png" alt="Method" style="width:95%;">
</div> 

# AccelOpt: Self-improving Agents for AI Accelerator Kernel Optimization

[![arXiv](https://img.shields.io/badge/arXiv-2511.15915-b31b1b.svg)](https://arxiv.org/pdf/2511.15915)

AccelOpt is a self-improving large language model (LLM) agentic system that autonomously optimizes
kernels for emerging AI acclerators, eliminating the need for expert-provided hardware-specific optimization
knowledge. AccelOpt explores the kernel optimization space through iterative generation, informed by an
optimization memory that curates experiences and insights from previously encountered slow-fast kernel pairs.

## Setup
### Trainium Machines
EC2 Instance: trn1.32xlarge

AMI: Deep Learning AMI Neuron (Ubuntu 22.04)

### Install
```
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate # Check the PyTorch version of your AMI
pip install logfire
pip install openai-agents
git clone git@github.com:zhang677/AccelOpt.git
cd AccelOpt
python setup.py install
```

## Run
`experiments/full_complete_local` shows how to run AccelOpt with a local served gpt-oss-120b.

# NKIBench: Kernel Benchmark for AWS Trainium accelerators

NKIBench is a new benchmark suite of AWS Trainium accelerator kernels with varying complexity
extracted from real-world LLM workloads to evaluate the effectiveness of AccelOpt. 

## Dataset
All the kernels are under `/NKIBench`. Kernels are
grouped by operator name and configuration in a structured
storage format as shown in the `/NKIBench/summary.json`

## Profiling
AccelOpt provides a `NKIKernel` class that is pluggable to any AI optimizers. `/tests` shows how to use the profiling API for single and a group of NKI kernels. 
```python
from accelopt.kernel_wrapper import NKIKernel
nki_kernel = NKIKernel(program_path, base_numpy_path)
result = nki_kernel.profile(save_fields)
```

NKIBench estimates the best achievable performance offered by the Trainium hardware, which offers additional insights on how effective AccelOpt has been in exploring the entire optimization landscape. The best achievable performance is calculated in `experiments/full_complete_local/calculate_percentage_of_peak.py`



# Citation
```
@article{zhang2025accelopt,
  title={AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization},
  author={Zhang, Genghan and Zhu, Shaowei and Wei, Anjiang and Song, Zhenyu and Nie, Allen and Jia, Zhen and Vijaykumar, Nandita and Wang, Yida and Olukotun, Kunle},
  journal={arXiv preprint arXiv:2511.15915},
  year={2025}
}
```

