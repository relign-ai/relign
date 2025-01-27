
# Overview
relign is a fully opensourced RL-library tailored specifically towards 
the research and development of reasoning engines. As of this moment it supports 
state of the art reinfrocemnet learning algortihms like (PPO) (soon(GRPO))!
along side useful abstractions for chain of thought(CoT) and MCTS inference strategies. 
we also provide a collection of popular and reasoning benchmarks.

relign is alpha software!!, and buggy!!... but 


## Installation
To get started, we recommend creating a new conda environment and installing Relign in editable mode:
1. Create and activate a conda environment:  
   ```bash
   conda create -n relign python=3.9 -y
   conda activate relign
   ```
2. Install Relign in editable mode:  
   ```bash
   pip install -e .
   ```

## Example
In the example, we provide code to posttrain a 1b sft model in the via PPO on a math bench via chain of thought. It provides a clear overview of the different layers of abstraction relign provides
A blog on what exactly is happning here will soon follow
The example runs on 2X A6000 Gpu's (96 VRAM total)


# Whats Next: 
- Docs Page, project layout:
  - Comprehensive documentation about the features and classes 
- Unit tests
  - A code base without tests is impossible to progress in. 
- Better Checkpointing/Evaluating, Training Monitoring/Analysis
- New algorithms (like GRPO):
  - Deepseek introduced a new RL algorithm, GRPO, which will soon be available in relign. (and evald) 
- More memory efficient algorithm runners
  - These runs take up alot of memory. I want significant runs on large retail gpu's  


# Contributing (ranked on urgency):
- Bug fixes! 
  Relign has a bunch of bugs that need fixing
  - Poor memory scheduling (vllM server shutdowns when switching between Episode Generation and Policy Training)
- Refactors
  - Lots of low hanging fruits here to be picked
  - Some files are 1000 lines long, especially in the epiosde genration and inference strategies. 
  
- Episode Generators / Tasks!
  - CodeInterpreter
  - MLE-bench 
  - Trading 
  - General Q/A

- Solving Bounties!
  - Bounties for new algorithms, environments, inference strategies
    and other layers of the stack will soon be made available 
    - please follow the discord for updates

- Because we are a crypto token, and because the financial markets offer a beautiful fully veriviable reward feedback signal, (i.e., returns on trades), our first task/environment objective will be to implement a sentiment trading bot for crypto markets.

- Other challenging reasoning benchmarks
- Along side these, we will also continue the refractor and simplify COT and MSTC episode generatos as these will play a crucial role in the long term future of reasoning.

# Acknoledgements
Our framwork and builds upon the following works: 
- This code uses code pieces of: [guidance], [deepseekmath]
 https://github.com/guidance-ai/guidance
 https://github.com/deepseek-ai/DeepSeek-Math
- framework structure inspired by [Stable Baseline 3]
- Special acknlowdegement to [VinePPO], of whom we used the code for MTCS method COT among Deepspeed policy abstractions. 
  Their paper ultimatly inspired me to build this
https://arxiv.org/abs/2410.01679

To all above, thank u for ur opensource contributions. 
And took inspiration from 
- [StableBaseline 3] in the structuring of its RL classes 


# References
- Vineppo: https://arxiv.org/abs/2409.12917
    - A special thanks to vinePPO for laying out an excellent base on which we can continue our work
      do check out their paper & results


