{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.mold;
  discordCfg = cfg.discord;
in
{
  options.services.mold = {
    enable = lib.mkEnableOption "mold AI image generation server";

    package = lib.mkOption {
      type = lib.types.package;
      description = "The mold package to use. Set to inputs.mold.packages.\${system}.default in your flake. Use mold-sm120 for Blackwell GPUs. The module cannot auto-select the package; set it explicitly to match your GPU.";
    };

    cudaArch = lib.mkOption {
      type = lib.types.nullOr (
        lib.types.enum [
          "ada"
          "blackwell"
        ]
      );
      default = null;
      description = ''
        CUDA GPU architecture hint. Emits a warning if set to "blackwell"
        to remind you to use the matching package variant.
        - "ada" — RTX 40-series (Ada Lovelace, sm_89): use packages.''${system}.mold
        - "blackwell" — RTX 50-series (Blackwell, sm_120): use packages.''${system}.mold-sm120
        The module cannot auto-select flake packages; you must set `package` to match.
      '';
      example = "blackwell";
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

    homeDir = lib.mkOption {
      type = lib.types.str;
      default = "/var/lib/mold";
      description = "Base mold directory (MOLD_HOME). Config, cache, and default model storage live under this path.";
    };

    modelsDir = lib.mkOption {
      type = lib.types.str;
      defaultText = lib.literalExpression ''"''${config.services.mold.homeDir}/models"'';
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

    logToFile = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Enable file logging to logDir. When false, logs go to journal only.";
    };

    logDir = lib.mkOption {
      type = lib.types.str;
      default = "${cfg.homeDir}/logs";
      description = "Directory for log files when logToFile is enabled.";
    };

    logRetentionDays = lib.mkOption {
      type = lib.types.int;
      default = 7;
      description = "Number of days to retain rotated log files.";
    };

    corsOrigin = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Restrict CORS to a specific origin. Null means permissive.";
    };

    apiKeyFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Path to a file containing the API key(s) for authentication (e.g. an agenix secret). Loaded at service start. When set, all API requests require an X-Api-Key header.";
    };

    rateLimit = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Per-IP rate limit for generation endpoints (e.g. '10/min', '5/sec', '100/hour'). Read endpoints get 10x the limit. Null disables rate limiting.";
      example = "10/min";
    };

    rateLimitBurst = lib.mkOption {
      type = lib.types.nullOr lib.types.int;
      default = null;
      description = "Override burst allowance for rate limiting. Defaults to 2x the rate limit when null.";
    };

    environment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      description = "Extra environment variables for the mold server.";
      example = {
        MOLD_EAGER = "1";
        MOLD_T5_VARIANT = "q4";
        MOLD_THUMBNAIL_WARMUP = "1";
      };
    };

    hfTokenFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Path to a file containing the HuggingFace API token (e.g. an agenix secret). The token is loaded at service start via EnvironmentFile.";
    };

    outputDir = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Optional directory to persist copies of server-generated images. Null means disabled (default).";
    };

    defaultModel = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Default model for generation. Null uses built-in default (flux2-klein) with smart fallback to the only downloaded model.";
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Whether to open the firewall port for the mold server.";
    };

    discord = {
      enable = lib.mkEnableOption "mold Discord bot (uses the main mold package with discord feature)";

      package = lib.mkOption {
        type = lib.types.package;
        default = cfg.package;
        defaultText = lib.literalExpression "config.services.mold.package";
        description = "The mold package to use for the Discord bot. Defaults to the main mold package (which includes the discord subcommand).";
      };

      tokenFile = lib.mkOption {
        type = lib.types.path;
        description = "Path to a file containing the Discord bot token (e.g. an agenix secret). Loaded via EnvironmentFile as MOLD_DISCORD_TOKEN.";
      };

      moldHost = lib.mkOption {
        type = lib.types.str;
        default = "http://localhost:${toString cfg.port}";
        description = "URL of the mold server to connect to.";
      };

      cooldownSeconds = lib.mkOption {
        type = lib.types.int;
        default = 10;
        description = "Per-user cooldown between generation requests, in seconds.";
      };

      allowedRoles = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        description = "Comma-separated Discord role names or IDs that can use generation commands. Null means unrestricted (all users can generate).";
        example = "artist, 1234567890";
      };

      dailyQuota = lib.mkOption {
        type = lib.types.nullOr lib.types.int;
        default = null;
        description = "Maximum generations per user per UTC day. Null means unlimited.";
        example = 20;
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
        description = "Log level for the Discord bot.";
      };

      environment = lib.mkOption {
        type = lib.types.attrsOf lib.types.str;
        default = { };
        description = "Extra environment variables for the Discord bot.";
      };
    };
  };

  config = lib.mkIf cfg.enable {
    warnings = lib.optionals (cfg.cudaArch == "blackwell") [
      ''
        services.mold.cudaArch is "blackwell" — make sure you set the matching package:
          services.mold.package = inputs.mold.packages.''${system}.mold-sm120;
        The module cannot auto-select flake packages; cudaArch is advisory.
      ''
    ];

    services.mold.modelsDir = lib.mkDefault "${cfg.homeDir}/models";

    users.users.mold = {
      isSystemUser = true;
      group = "mold";
      home = cfg.homeDir;
    };
    users.groups.mold = { };

    systemd.tmpfiles.rules = [
      "d ${cfg.homeDir} 0775 mold mold -"
      "d ${cfg.modelsDir} 0775 mold mold -"
    ]
    ++ lib.optionals (cfg.outputDir != null) [
      "d ${cfg.outputDir} 0775 mold mold -"
    ]
    ++ lib.optionals cfg.logToFile [
      "d ${cfg.logDir} 0775 mold mold -"
    ];

    systemd.services.mold = {
      description = "mold AI image generation server";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];

      environment = {
        MOLD_HOME = cfg.homeDir;
        MOLD_PORT = toString cfg.port;
        MOLD_MODELS_DIR = cfg.modelsDir;
        MOLD_LOG = cfg.logLevel;
        LD_LIBRARY_PATH = "/run/opengl-driver/lib";
      }
      // lib.optionalAttrs (cfg.corsOrigin != null) {
        MOLD_CORS_ORIGIN = cfg.corsOrigin;
      }
      // lib.optionalAttrs (cfg.outputDir != null) {
        MOLD_OUTPUT_DIR = cfg.outputDir;
      }
      // lib.optionalAttrs (cfg.defaultModel != null) {
        MOLD_DEFAULT_MODEL = cfg.defaultModel;
      }
      // lib.optionalAttrs (cfg.rateLimit != null) {
        MOLD_RATE_LIMIT = cfg.rateLimit;
      }
      // lib.optionalAttrs (cfg.rateLimitBurst != null) {
        MOLD_RATE_LIMIT_BURST = toString cfg.rateLimitBurst;
      }
      // cfg.environment;

      serviceConfig = {
        Type = "simple";
        User = "mold";
        Group = "mold";
        UMask = "0002";
        ExecStartPre = lib.optionals (cfg.hfTokenFile != null || cfg.apiKeyFile != null) [
          "+${pkgs.writeShellScript "mold-env" ''
            : > /run/mold/env
            ${lib.optionalString (cfg.hfTokenFile != null) ''
              echo "HF_TOKEN=$(cat ${cfg.hfTokenFile})" >> /run/mold/env
            ''}
            ${lib.optionalString (cfg.apiKeyFile != null) ''
              echo "MOLD_API_KEY=@${cfg.apiKeyFile}" >> /run/mold/env
            ''}
            chown mold:mold /run/mold/env
            chmod 600 /run/mold/env
          ''}"
        ];
        ExecStart = "${lib.getExe cfg.package} serve --bind ${cfg.bindAddress} --port ${toString cfg.port}${lib.optionalString cfg.logToFile " --log-file"}";
        Restart = "on-failure";
        RestartSec = 5;

        RuntimeDirectory = "mold";
        # StateDirectory and CacheDirectory omitted — homeDir is created
        # by tmpfiles.rules and may not be under /var/lib/.
      }
      // lib.optionalAttrs (cfg.hfTokenFile != null || cfg.apiKeyFile != null) {
        EnvironmentFile = "-/run/mold/env";
      }
      // {

        # Hardening
        NoNewPrivileges = true;
        ProtectSystem = "full";
        ProtectHome = true;
        PrivateTmp = true;
        PrivateDevices = false;
        ReadWritePaths = [
          cfg.homeDir
          cfg.modelsDir
        ]
        ++ lib.optionals (cfg.outputDir != null) [ cfg.outputDir ]
        ++ lib.optionals cfg.logToFile [ cfg.logDir ];

        # GPU access
        SupplementaryGroups = [
          "video"
          "render"
        ];
      };
    };

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.port ];

    # ── Discord bot service ─────────────────────────────────────────────
    systemd.services.mold-discord = lib.mkIf discordCfg.enable {
      description = "mold Discord bot";
      after = [
        "network.target"
      ]
      ++ lib.optionals cfg.enable [ "mold.service" ];
      wants = lib.optionals cfg.enable [ "mold.service" ];
      wantedBy = [ "multi-user.target" ];

      environment = {
        MOLD_HOST = discordCfg.moldHost;
        MOLD_DISCORD_COOLDOWN = toString discordCfg.cooldownSeconds;
        MOLD_LOG = discordCfg.logLevel;
      }
      // lib.optionalAttrs (discordCfg.allowedRoles != null) {
        MOLD_DISCORD_ALLOWED_ROLES = discordCfg.allowedRoles;
      }
      // lib.optionalAttrs (discordCfg.dailyQuota != null) {
        MOLD_DISCORD_DAILY_QUOTA = toString discordCfg.dailyQuota;
      }
      // discordCfg.environment;

      serviceConfig = {
        Type = "simple";
        User = "mold";
        Group = "mold";
        EnvironmentFile = discordCfg.tokenFile;
        ExecStart = "${lib.getExe discordCfg.package} discord";
        Restart = "on-failure";
        RestartSec = 10;

        # Hardening — no GPU access needed
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        PrivateDevices = true;
      };
    };
  };
}
