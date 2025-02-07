{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/packages/
  packages = with pkgs; [ 
    git 
    cudatoolkit
    cudaPackages_11.cuda_nvcc
    mpich
  ];

  languages.python = {
    enable = true;
    uv.enable = true;
  };
}  


