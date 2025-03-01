{
  local model_name = "realtreetune/rho-1b-sft-GSM8K",
  
  tokenizer: {
    type: "di_pretrained_tokenizer",
    hf_model_name: model_name,
  },
  
  actor: {
    type: "pretrained_model_for_casual_lm",
    hf_model_name: model_name,
    disable_dropout: true,
    pretrained_args: {
      use_flash_attention_2: true,
    },
  },
  
  critic: {
    type: "pretrained_model_for_value_network",
    pretrained_backbone: {
      type: "pretrained_model_for_casual_lm",
      hf_model_name: model_name,
      disable_dropout: true,
      pretrained_args: {
        use_flash_attention_2: true,
      },
    },
  },
  
  reference: {
    type: "pretrained_model_for_casual_lm",
    hf_model_name: model_name,
    disable_dropout: true,
    pretrained_args: {
      use_flash_attention_2: true,
    },
  },
}
