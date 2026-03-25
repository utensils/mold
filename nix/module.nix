{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.mold;
in
{
  options.services.mold = {
    enable = lib.mkEnableOption "mold AI image generation server";

    package = lib.mkOption {
      type = lib.types.package;
      description = "The mold package to use. Set to inputs.mold.packages.\${system}.default in your flake.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 7680;
      description = "Port for the mold HTTP server.";
    };

    bindAddress = lib.mkOption {
      type = lib.types.str;
      default = "0.0.0.0";
      description = "Address to bind the server to.";
    };

    modelsDir = lib.mkOption {
      type = lib.types.str;
      default = "/var/lib/mold/models";
      description = "Directory for storing downloaded models.";
    };

    logLevel = lib.mkOption {
      type = lib.types.enum [
        "trace"
        "debug"
        "info"
        "warn"
        "error"
      ];
      default = "info";
      description = "Log level for the mold server.";
    };

    corsOrigin = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Restrict CORS to a specific origin. Null means permissive.";
    };

    environment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      description = "Extra environment variables for the mold server.";
      example = {
        MOLD_EAGER = "1";
        MOLD_T5_VARIANT = "q4";
      };
    };

    hfTokenFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Path to a file containing the HuggingFace API token (e.g. an agenix secret). The token is loaded at service start via EnvironmentFile.";
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Whether to open the firewall port for the mold server.";
    };
  };

  config = lib.mkIf cfg.enable {
    users.users.mold = {
      isSystemUser = true;
      group = "mold";
      home = "/var/lib/mold";
    };
    users.groups.mold = { };

    systemd.tmpfiles.rules = [
      "d ${cfg.modelsDir} 0755 mold mold -"
    ];

    systemd.services.mold = {
      description = "mold AI image generation server";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];

      environment = {
        MOLD_PORT = toString cfg.port;
        MOLD_MODELS_DIR = cfg.modelsDir;
        MOLD_LOG = cfg.logLevel;
        LD_LIBRARY_PATH = "/run/opengl-driver/lib";
      }
      // lib.optionalAttrs (cfg.corsOrigin != null) {
        MOLD_CORS_ORIGIN = cfg.corsOrigin;
      }
      // cfg.environment;

      serviceConfig = {
        Type = "simple";
        User = "mold";
        Group = "mold";
        ExecStartPre = lib.optionals (cfg.hfTokenFile != null) [
          "+${pkgs.writeShellScript "mold-env" ''
            echo "HF_TOKEN=$(cat ${cfg.hfTokenFile})" > /run/mold/env
            chown mold:mold /run/mold/env
            chmod 600 /run/mold/env
          ''}"
        ];
        ExecStart = "${lib.getExe cfg.package} serve --bind ${cfg.bindAddress} --port ${toString cfg.port}";
        Restart = "on-failure";
        RestartSec = 5;

        RuntimeDirectory = "mold";
        StateDirectory = "mold";
        CacheDirectory = "mold";
      }
      // lib.optionalAttrs (cfg.hfTokenFile != null) {
        EnvironmentFile = "-/run/mold/env";
      }
      // {

        # Hardening
        NoNewPrivileges = true;
        ProtectSystem = "full";
        ProtectHome = true;
        PrivateTmp = true;
        PrivateDevices = false;
        ReadWritePaths = [ cfg.modelsDir ];

        # GPU access
        SupplementaryGroups = [
          "video"
          "render"
        ];
      };
    };

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.port ];
  };
}
