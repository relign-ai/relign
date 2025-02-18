<!-- Center the banner at the top -->
<p align="center">
  <img src="assets/banner.jpg" alt="RELIGN Banner" />
</p>

<!-- Center the Discord (server invite), Twitter badges, and Hugging Face link -->
<p align="center">
  <a href="https://discord.com/invite/9mrHbusc">
    <img src="https://img.shields.io/discord/1333056593880682508?label=Join%20our%20Discord" alt="Discord Server Invite" />
  </a>
  <a href="https://x.com/relignai">
    <img src="https://img.shields.io/twitter/follow/relignai?style=social" alt="Twitter" />
  </a>
  <a href="https://huggingface.co/relign">
    <img src="https://img.shields.io/badge/Hugging%20Face-Join%20us-yellow" alt="Hugging Face" />
  </a>
</p>

relign is a fully open-sourced RL library tailored specifically for the research and development of reasoning engines. It currently supports state-of-the-art reinforcement learning algorithms like **PPO** and **GRPO**, alongside useful abstractions for Chain of Thought (CoT) and MCTS inference strategies. All these can be evaluated on popular reasoning benchmarks.

> **Note:** relign is alpha software—it may be buggy.

---

## Table of Contents
- [Installation](#installation)
- [Example](#example)
- [Bounties](#bounties)
- [What's Next](#whats-next)
- [Training Runs](#training-runs)
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

Not just the models will be rewarded for their work, but more importantly, our contributors. Implement bounties and we will send you relign

| Description                                                                                                                         | Reward in RELIGN  |
|-------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| ✅ Completed: GRPO – Implement DeepSeek's GRPO in RELIGN and train it with standard CoT inference on gsm8k math                     | 250k              |
| Create a complex medical reasoning task specification + verifier based on HuatuoGPT-o1                                             | 250k              |
| Implement a multi-step learning inference strategy                                                                                 | 250k              |

### Submit My Own Bounty
If you'd like to propose a new challenge or feature and set your own reward, go to: 
[www.relign.ai/bounties](https://www.relign.ai/bounties).

Bounty Instructions (common GitHub approach):
1. Fork the repository.  
2. Make your changes in a new branch.  
3. Submit a Pull Request referencing the bounty issue.  
4. We will review your PR and, if merged, send the funds to your wallet.

---

## What's Next
- **Docs Page & Project Layout**  
  Comprehensive documentation about the features and classes.

- **More Memory-Efficient Algorithm Runners**  
  Some runs require a lot of VRAM. We aim to set up smaller scale experiments such that developers can run and train models on single-GPU machines.
---

## Training Runs

We recently benchmarked **GRPO** in RELIGN. You can view the detailed training report [here](https://wandb.ai/darrynbiervliet/relign-02/reports/Model-realignment-with-GRPO--VmlldzoxMTM5OTYxOA?accessToken=cvgxqwdrxvfyd041j92snl69qi7di49zs26ir72g208dwmps4xdjmmuzrazbyxq6).

---

## Contributing (Ranked by Urgency)


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
- **Special acknowledgement to [VinePPO](https://arxiv.org/abs/2410.01679)** for the MCTS + CoT approach and DeepSpeed policy abstractions.

> Thank you to all contributors for your open-source efforts!
