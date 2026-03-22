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

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Whether to open the firewall port for the mold server.";
    };
  };

  config = lib.mkIf cfg.enable {
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
        ExecStartPre = "${pkgs.coreutils}/bin/mkdir -p ${cfg.modelsDir}";
        ExecStart = "${lib.getExe cfg.package} serve --bind ${cfg.bindAddress} --port ${toString cfg.port}";
        Restart = "on-failure";
        RestartSec = 5;

        DynamicUser = true;
        StateDirectory = "mold";
        CacheDirectory = "mold";

        # Hardening
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        ReadWritePaths = [ cfg.modelsDir ];

        # GPU access
        SupplementaryGroups = [
          "video"
          "render"
        ];
        DeviceAllow = [
          "/dev/nvidia0"
          "/dev/nvidiactl"
          "/dev/nvidia-uvm"
          "/dev/nvidia-uvm-tools"
          "/dev/dri/renderD128"
        ];
      };
    };

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.port ];
  };
}
