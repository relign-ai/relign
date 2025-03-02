{
  type: "grpo_trainer",
  target_batch_size: 16,
  gradient_accumulation_steps: 1,
  num_epochs_per_iteration: 2,
  num_episodes_per_iteration: 512,
  num_iterations: 650,
  max_seq_length: 2048,
  dataloader_num_workers: 2,
  dataloader_pin_memory: false,
}