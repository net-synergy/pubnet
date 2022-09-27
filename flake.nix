{
  description =
    "A python package for storing and working with publication data in graph form.";

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
        pubnet = python.pkgs.buildPythonPackage rec {
          pname = "pubnet";
          version = "0.1.0";
          src = ./.;
          propagatedBuildInputs =
            (with python.pkgs; [ numpy pandas scipy pytest pytest-snapshot ]);
          preBuild = ''
            cat >setup.py <<_EOF_
            from setuptools import setup, find_packages
            setup(
                name='${pname}',
                version='${version}',
                license='MIT',
                description="A package for storing and working with publication data in graph form.",
                packages=find_packages(include=['${pname}', '${pname}.*']),
                install_requires=[
                'numpy',
                'pandas',
                'scipy'
                ],
                tests_require=['pytest', 'pytest-snapshot']
            )
            _EOF_
          '';
          checkPhase = ''
            python -m pytest
          '';
        };
      in {
        packages.pubnet = pubnet;
        defaultPackage = self.packages.${system}.pubnet;
        devShell = pkgs.mkShell {
          packages = [
            (python.withPackages (p:
              with p;
              [
                ipython
                python-lsp-server
                pyls-isort
                python-lsp-black
                pylsp-mypy
              ] ++ pubnet.propagatedBuildInputs))
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
