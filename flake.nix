{
  description =
    "A python package for storing and working with publication data in graph form.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ poetry2nix.overlays.default ];
        };

        pubnetEnv = pkgs.poetry2nix.mkPoetryEnv {
          projectDir = ./.;
          editablePackageSources = { pubnet = ./pubnet; };
          preferWheels = true;
          groups = [ "test" "dev" ];
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
