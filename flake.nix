{
  description =
    "A python package for storing and working with publication data in graph form.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-22.11";
    flake-utils = { url = "github:numtide/flake-utils"; };
    pubmedparser = {
      url = "gitlab:net-synergy/pubmedparser/major-version-1";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = { self, nixpkgs, flake-utils, pubmedparser }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pubnetEnv = pkgs.poetry2nix.mkPoetryEnv {
          projectDir = ./.;
          editablePackageSources = { pubnet = ./pubnet; };
          extraPackages = (ps:
            with ps; [
              ipython
              python-lsp-server
              pyls-isort
              python-lsp-black
              pylsp-mypy
            ]);
          groups = [ "test" ];
        };
        pubnet = pkgs.poetry2nix.mkPoetryPackage { projectDir = ./.; };
      in {
        packages.pubnet = pubnet;
        packages.default = self.packages.${system}.pubnet;
        devShells.default = pkgs.mkShell {
          packages = [
            pubnetEnv
            pkgs.astyle
            pkgs.bear
            pubmedparser.defaultPackage.${system}
          ];
        };
      });
}
