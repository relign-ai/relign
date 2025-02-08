{
  description = "pixi env";
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    { flake-utils, nixpkgs, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        fhs = pkgs.buildFHSEnv {
          name = "pixi-env";

          targetPkgs = _: [ 
            pkgs.pixi 
            pkgs.git
            pkgs.ninja
          ];

          # Needed for dynamically optaining compiler CPU flags
          extraShellHook = ''
            export NIX_ENFORCE_NO_NATIVE=0
          '';
        };
      in
      {
        devShell = fhs.env;
      }
    );
}
