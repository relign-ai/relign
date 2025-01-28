<!-- Center the banner at the top -->
<p align="center">
  <img src="assets/banner.jpg" alt="RELIGN Banner" />
</p>

<!-- Center the Discord (server invite) and Twitter badges -->
<p align="center">
  <a href="https://discord.com/invite/9mrHbusc">
    <img src="https://img.shields.io/discord/1333056593880682508?label=Join%20our%20Discord" alt="Discord Server Invite" />
  </a>
  <a href="https://x.com/relignai">
    <img src="https://img.shields.io/twitter/follow/relignai?style=social" alt="Twitter" />
  </a>
</p>

relign is a fully open-sourced RL library tailored specifically for the research and development of reasoning engines. It currently supports state-of-the-art reinforcement learning algorithms like **PPO** (and soon **GRPO**!), alongside useful abstractions for Chain of Thought (CoT) and MCTS inference strategies. All these can be evaluated on popular reasoning benchmarks. 

> **Note:** relign is alpha software—it may be buggy.

---

## Table of Contents
- [Installation](#installation)
- [Example](#example)
- [Bounties](#bounties)
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

In the `example` folder, we provide code to fine-tune a 1B SFT model via PPO on a gsm8k-math benchmark using a Chain of Thought inference strategy. This example demonstrates the different abstraction layers that relign offers. A blog post detailing exactly what's happening here, why it is important, and where we see this going will follow soon.

> The example runs on two A6000 GPUs (96GB VRAM total).

### How to run

```bash
deepspeed --num_gpus=2 examples/ppo_gsm.py
```

If you have a custom DeepSpeed config file (e.g., ds_config.json), you can also specify it:

---

## Bounties

Below is the current bounty task with its reward in RELIGN:

| Description                                                                                                   | Reward in RELIGN  |
|---------------------------------------------------------------------------------------------------------------|-------------------|
| Bounty 1. GRPO: implement DeepSeek's GRPO in RELIGN and train it with standard CoT inference on gsm8k math    | 250k RELIGN       |

### Submit My Own Bounty
If you'd like to propose a new challenge or feature and set your own reward, go to: 
[www.relign.ai/bounties](https://www.relign.ai/bounties).

Bounty Instructions (common GitHub approach):
1. Fork the repository.  
2. Make your changes in a new branch.  
3. Submit a Pull Request referencing the bounty issue.  
4.  We will review your PR and, if merged, send the funds to your wallet.

---

## What's Next
- **Docs Page & Project Layout**  
  Comprehensive documentation about the features and classes.

- **Unit Tests**  
  A codebase without tests is hard to maintain and improve.

- **Training run Tooling**  
  Evaluations, Checkpointing, Metric monitoring (wandb/tensorflow), and Reasoning trace analysis are not yet supported but will be supported soon!

- **New Algorithms (e.g., GRPO)**  
  [Deepseek-r1](https://github.com/deepseek-ai/DeepSeek-R1) introduced a new RL algorithm, GRPO, which will soon be available in RELIGN (we plan to evaluate it as well).

- **More Memory-Efficient Algorithm Runners**  
  Some runs require a lot of VRAM. We aim to setup smaller  
  scale experiments such that developers can run and train 
  models on single-gpu machines

---

## Contributing (Ranked by Urgency)

1. **Bug Fixes**  
   - Poor memory scheduling (vLLM server shutdowns when switching between episode generation and policy training)

2. **Refactors**  
   - Some files exceed 1000 lines, especially in episode generation and inference strategies. Any obvious and simple refactors are always welcome.

3. **Episode Generators / Tasks**  
   - We would encourage everyone to add new (novel) tasks and environments to the library on which we can test post-training methods. Some inspiration below: 
     - Coding  
     - MLE-bench  
     - Trading  
     - General/Scientific Q&A  


---

## Acknowledgements

RELIGN builds upon and is inspired by the following works:

- [**Guidance**](https://github.com/guidance-ai/guidance)
- [**DeepSeek-Math**](https://github.com/deepseek-ai/DeepSeek-Math)
- **Framework structure inspired by [Stable Baselines 3]**
- **Special acknowledgement to [VinePPO](https://arxiv.org/abs/2410.01679)** for the MCTS + CoT approach and Deepspeed policy abstractions.

> Thank you to all contributors for your open-source efforts!
