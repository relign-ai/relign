# RELIGN

**RELIGN** is a fully open-sourced RL library tailored specifically for the research and development of reasoning engines. It currently supports state-of-the-art reinforcement learning algorithms like **PPO** (and soon **GRPO**!), alongside useful abstractions for Chain of Thought (CoT) and MCTS inference strategies. We also provide a collection of popular reasoning benchmarks.

> **Note:** RELIGN is alpha software—it may be buggy.

---

## Table of Contents
- [Installation](#installation)
- [Example](#example)
- [What's Next](#whats-next)
- [Contributing (Ranked by Urgency)](#contributing-ranked-by-urgency)
- [Acknowledgements](#acknowledgements)

---

## Installation

1. **Create and activate a conda environment**:

    ```bash
    conda create -n relign python=3.10 -y
    conda activate relign
    ```

2. **Install RELIGN in editable mode**:

    ```bash
    pip install -e .
    ```

---

## Example

In the `example` folder, we provide code to fine-tune a 1B SFT model via PPO on a gsm8k-math benchmark using a Chain of Thought inference strategy. This example demonstrates the different abstraction layers that relign offers. A blog post detailing exactly what’s happening here, why it is important and where we see this going will follow soon.

> The example runs on two A6000 GPUs (96GB VRAM total).
---

## What's Next

- **Docs Page & Project Layout**  
  Comprehensive documentation about the features and classes.

- **Unit Tests**  
  A codebase without tests is hard to maintain and improve.

- **Trainig run Tooling**
  Evaluations, Checkpointing, Metric monitoring(wandb/tensorflow) and Reasoning trace analysis are not yet supported but will be supported soon!

- **New Algorithms (e.g., GRPO)**  
  [Deepseek-r1](https://github.com/deepseek-ai/DeepSeek-R1) introduced a new RL algorithm, GRPO, which will soon be available in RELIGN (we plan to evaluate it as well).

- **More Memory-Efficient Algorithm Runners**  
  Some runs require a lot of memory; we aim to support significant training runs on large but more affordable GPUs. (Including efficient single gpu training)

---

## Contributing (Ranked by Urgency)

1. **Bug Fixes**  
   - Poor memory scheduling (vllM server shutdowns when switching between episode generation and policy training)

2. **Refactors**  
   - Some files exceed 1000 lines, especially in episode generation and inference strategies. Any obvious 
   and simple refractors are always welcome

3. **Episode Generators / Tasks**  
   - We would encourge everyone to add new (novel) tasks and environments to the library on which we can test post training methods. Some inspiration below: 
   - Coding 
   - MLE-bench  
   - Trading  
   - General/Scientific Q&A  

4. **Solving Bounties**  
   - We will soon introduce bounties for new algorithms, environments, inference strategies, and other stack layers. Please follow our Discord for updates.  
   - Because RELIGN is a crypto token project and financial markets offer a verifiable reward signal (returns on trades), our first task/environment objective is to implement a sentiment trading bot for crypto markets.  
   - We also plan to tackle other challenging reasoning benchmarks.

---

## Acknowledgements

RELIGN builds upon and is inspired by the following works:

- [**Guidance**](https://github.com/guidance-ai/guidance)
- [**DeepSeek-Math**](https://github.com/deepseek-ai/DeepSeek-Math)
- **Framework structure inspired by [Stable Baselines 3]**
- **Special acknowledgement to [VinePPO](https://arxiv.org/abs/2410.01679)** for the MCTS + CoT approach and Deepspeed policy abstractions.

> Thank you to all contributors for your open-source efforts!
