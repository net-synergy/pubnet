{
  description =
    "A python package for storing and working with publication data in graph form.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-22.11";
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
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
        python = pkgs.python3;
        # Nix does not expose `checkInputs` attribute.
        pubnetCheckInputs =
          (with python.pkgs; [ pytest pytest-snapshot mypy black lxml ]);
        pubnet = python.pkgs.buildPythonPackage rec {
          pname = "pubnet";
          version = "0.5.0";
          src = ./.;
          format = "pyproject";
          buildInputs = (with python.pkgs; [ poetry ]);
          propagatedBuildInputs = (with python.pkgs; [
            numpy
            pandas
            scipy
            matplotlib
            igraph
            pyarrow
          ]);
          checkInputs = pubnetCheckInputs;
          authors = [ "David Connell <davidconnell12@gmail.com>" ];
          keywords = [ "publication" "network" ];
          repository = "https://gitlab.com/net-synergy/pubnet";
          checkPhase = ''
            python -m pytest
          '';
        };
        nix2poetryDependency = list:
          builtins.concatStringsSep "\n" (builtins.map (dep:
            let
              pname = if dep.pname == "python3" then "python" else dep.pname;
              versionList = builtins.splitVersion dep.version;
              major = builtins.elemAt versionList 0;
              minor = builtins.elemAt versionList 1;
              version = if pname == "python" then
                ''\"~${major}.${minor}\"''
              else
                ''\"^${major}.${minor}\"'';
            in pname + " = " + version) list);
      in {
        packages.pubnet = pubnet;
        packages.default = self.packages.${system}.pubnet;
        devShells.default = pkgs.mkShell {
          packages = [
            (python.withPackages (p:
              with p;
              [
                ipython
                python-lsp-server
                pyls-isort
                python-lsp-black
                pylsp-mypy
              ] ++ pubnet.propagatedBuildInputs ++ pubnetCheckInputs))
            pkgs.astyle
            pkgs.bear
            pubmedparser.defaultPackage.${system}
          ];
          shellHook = ''
            export PYTHONPATH=.
            export C_INCLUDE_PATH=${python}/include/python3.9

            if [ ! -f pyproject.toml ] || \
               [ $(date +%s -r flake.nix) -gt $(date +%s -r pyproject.toml) ]; then
               pname=${pubnet.pname} \
               version=${pubnet.version} \
               description='A python package for storing and working with publication data in graph form.' \
               license=MIT \
               authors="${
                 builtins.concatStringsSep ",\n    "
                 (builtins.map (name: ''\"'' + name + ''\"'') pubnet.authors)
               }" \
               keywords="${
                 builtins.concatStringsSep ", "
                 (builtins.map (name: ''\"'' + name + ''\"'') pubnet.keywords)
               }" \
               repository=${pubnet.repository} \
               dependencies="${
                 nix2poetryDependency pubnet.propagatedBuildInputs
               }" \
               testDependencies="${nix2poetryDependency pubnetCheckInputs}" \
               ./.pyproject.toml.template
            fi
          '';
        };
      });
}
