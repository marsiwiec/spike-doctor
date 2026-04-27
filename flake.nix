{
  description = "Spike Doctor — Shiny web app for electrophysiology analysis";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];

      perSystem = { config, self', inputs', pkgs, system, ... }: {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            uv
            python312
            ruff
            ty
            zlib
            stdenv.cc.cc.lib
          ];

          env = {
            UV_PYTHON = "${pkgs.python312}/bin/python3";
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ];
          };

          shellHook = ''
            echo "Spike Doctor dev shell"
            echo "  uv version: $(uv --version)"
            echo "  python version: $(python3 --version)"
          '';
        };
      };
    };
}
