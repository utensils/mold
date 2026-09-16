# Mold for macOS. `make` is the only entry point -- Mold.xcodeproj is
# generated from project.yml and is never checked in.

SHELL := /bin/bash
.DEFAULT_GOAL := help

XCODEGEN ?= xcodegen
PROJECT  := Mold.xcodeproj
SCHEME   := Mold
CONFIG   ?= Debug
BUILD    := $(CURDIR)/build
APP      := $(BUILD)/$(CONFIG)/Mold.app
XCB      := xcodebuild -project $(PROJECT) -scheme $(SCHEME) -configuration $(CONFIG) \
            -derivedDataPath $(BUILD)/DerivedData CONFIGURATION_BUILD_DIR=$(BUILD)/$(CONFIG)

.PHONY: help gen build run run-fresh test lint lint-layers lint-color lint-size clean

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

gen: ## Regenerate Mold.xcodeproj from project.yml
	@command -v $(XCODEGEN) >/dev/null || { echo "xcodegen not found -- run inside 'nix develop'"; exit 1; }
	@$(XCODEGEN) generate --spec project.yml --quiet

build: gen ## Build the app
	@$(XCB) build | grep -Ev '^\s*$$' | tail -5

run: build ## Build and launch the app
	@open $(APP)

run-fresh: build ## Launch with a throwaway preferences domain
	@defaults delete io.utensils.mold.native.fresh 2>/dev/null || true
	@MOLD_NATIVE_FRESH=1 open -n $(APP)

test: ## Run package tests
	@cd Packages/MoldClient && swift test

lint: lint-layers lint-color lint-size ## Run every architecture lint

lint-layers: ## MoldClient stays UI-free; only the root may build a backend
	@# MoldStyle is the design system and of course imports SwiftUI. MoldClient
	@# is the wire layer: it must stay usable from tests and from a CLI, so a UI
	@# import there is the violation worth catching.
	@! grep -rlnE '^import (SwiftUI|AppKit)' Packages/MoldClient/Sources \
	  || { echo "LAYER VIOLATION: MoldClient imports a UI framework"; exit 1; }
	@! grep -rln 'HTTPBackend(' Sources --include='*.swift' | grep -v 'HostStore.swift' \
	  || { echo "LAYER VIOLATION: a backend is constructed outside the composition root"; exit 1; }
	@echo "  layers ok"

lint-color: ## No literal colors -- the system owns the palette
	@! grep -rnE 'Color\((red|hue|white):|Color\(hex' Sources Packages/*/Sources --include='*.swift' \
	  || { echo "COLOR VIOLATION: use a semantic color or a material"; exit 1; }
	@echo "  color ok"

lint-size: ## Advisory: flag files over 150 lines
	@find Sources Packages/*/Sources -name '*.swift' -exec wc -l {} + \
	  | awk '$$1 > 150 && $$2 != "total" { print "  large: " $$2 " (" $$1 " lines)" }'

clean: ## Remove build output and the generated project
	@rm -rf $(BUILD) $(PROJECT) Packages/*/.build
