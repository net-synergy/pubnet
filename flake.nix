{
  description = "Package for disambiguating ondes in a graph";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-22.05";
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pubmedparser = {
      url = "gitlab:DavidRConnell/pubmedparser/major-version-1";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = { self, nixpkgs, flake-utils, pubmedparser }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python3;
      in {
        devShell = pkgs.mkShell {
          packages = [
            (python.withPackages (p:
              with p; [
                # development dependencies
                ipython
                pytest
                python-lsp-server
                pyls-isort
                python-lsp-black
                pylsp-mypy

                # runtime dependencies
                numpy
              ]))
            pkgs.astyle
            pkgs.bear
            pubmedparser.defaultPackage.${system}
          ];
          shellHook = ''
            export PYTHONPATH=.
            export C_INCLUDE_PATH=${python}/include/python3.9
          '';
        };
      });
}
