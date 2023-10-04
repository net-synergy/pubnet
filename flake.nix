{
  description =
    "A python package for storing and working with publication data in graph form.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-23.05";
    flake-utils = { url = "github:numtide/flake-utils"; };
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pubnetEnv = pkgs.poetry2nix.mkPoetryEnv {
          projectDir = ./.;
          editablePackageSources = { pubnet = ./pubnet; };
          preferWheels = true;
          groups = [ "test" "dev" "benchmark" ];
          extraPackages = (ps: with ps; [ ipdb ]);
        };
        pubnet = pkgs.poetry2nix.mkPoetryPackages { projectDir = ./.; };
      in {
        packages.pubnet = pubnet;
        packages.default = self.packages.${system}.pubnet;
        devShells.default = pkgs.mkShell {
          packages = [ pubnetEnv pkgs.astyle pkgs.bear pkgs.poetry ];
        };
      });
}
