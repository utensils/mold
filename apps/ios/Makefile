# Mold Studio Companion (iOS / iPadOS). `make help` lists the targets.
SHELL := /bin/bash
.DEFAULT_GOAL := help

XCODEGEN ?= xcodegen
PROJECT  := MoldCompanion.xcodeproj
SCHEME   := MoldCompanion
CONFIG   ?= Debug
BUILD    ?= $(CURDIR)/build
PACKAGES := ../shared/Packages
# Any available iOS 26 simulator by name; CI and a laptop differ in which
# runtimes they carry, so no OS is pinned.
ifeq ($(origin SIM),undefined)
SIM := $(shell scripts/pick-simulator.sh)
endif
DEST     := platform=iOS Simulator,name=$(SIM)

MARKETING_VERSION ?= $(shell ../macos/scripts/workspace-version.sh)
BUILD_NUMBER ?= $(shell git rev-list --count HEAD 2>/dev/null || echo 1)

# Simulator builds sign ad hoc: no Apple account is needed (a hosted runner
# has none), and the entitlements -- App Group, Keychain group -- are still
# embedded, which the simulator honours. Device and App Store builds sign
# with the team in project.yml (`make archive`).
SIM_SIGN := CODE_SIGN_STYLE=Manual CODE_SIGN_IDENTITY=- DEVELOPMENT_TEAM=

XCB := xcodebuild -project $(PROJECT) -scheme $(SCHEME) -configuration $(CONFIG) \
       -destination '$(DEST)' -derivedDataPath $(BUILD)/DerivedData $(SIM_SIGN) \
       CURRENT_PROJECT_VERSION=$(BUILD_NUMBER) MARKETING_VERSION=$(MARKETING_VERSION)

.PHONY: help gen build run test packages-test uitest lint lint-script lint-layers lint-color lint-type lint-widgets lint-a11y lint-size clean

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

gen: ## Regenerate MoldCompanion.xcodeproj from project.yml
	@command -v $(XCODEGEN) >/dev/null || { echo "xcodegen not found -- run inside 'nix develop'"; exit 1; }
	@printf 'settings:\n  base:\n    MARKETING_VERSION: "%s"\n' "$(MARKETING_VERSION)" > Version.yml
	@$(XCODEGEN) generate --spec project.yml --quiet

build: gen ## Build for the simulator
	@$(XCB) build | grep -E 'error:|warning: .*MoldCompanion|BUILD (SUCCEEDED|FAILED)' ; exit $${PIPESTATUS[0]}

APP_PATH = $(BUILD)/DerivedData/Build/Products/$(CONFIG)-iphonesimulator/Mold Studio.app

run: build ## Build, install and launch in the simulator
	@xcrun simctl boot "$(SIM)" 2>/dev/null || true
	@open -a Simulator
	@xcrun simctl install "$(SIM)" "$(APP_PATH)"
	@xcrun simctl launch "$(SIM)" io.utensils.mold.companion >/dev/null && echo "  launched in $(SIM)"

test: gen ## Unit tests on the simulator (the shared packages' own suites run from their Makefiles)
	@$(XCB) test -only-testing:MoldCompanionTests 2>&1 \
	  | grep -E '✘|recorded an issue|error:|Test run with|TEST (SUCCEEDED|FAILED)' ; exit $${PIPESTATUS[0]}

packages-test: ## MoldClient and MoldStyle's own suites on the iOS simulator
	@for pkg in MoldClient MoldStyle; do \
	  echo "  $$pkg"; \
	  (cd $(PACKAGES)/$$pkg && xcodebuild test -scheme $$pkg -destination '$(DEST)' \
	    -derivedDataPath $(BUILD)/PackagesData 2>&1 \
	    | grep -E '✘|recorded an issue|error:|Test run with|TEST (SUCCEEDED|FAILED)'; \
	    exit $${PIPESTATUS[0]}) || exit 1; \
	done

uitest: gen ## The accessibility audit: every destination at xSmall, Large and AX5, light AND dark
	@# The audit sees whatever appearance the simulator is in -- contrast
	@# passing in one says nothing about the other -- so it runs in both and
	@# puts the simulator back the way it found it.
	@xcrun simctl boot "$(SIM)" 2>/dev/null || true
	@was=$$(xcrun simctl ui "$(SIM)" appearance); status=0; \
	for mode in light dark; do \
	  echo "  appearance: $$mode"; \
	  xcrun simctl ui "$(SIM)" appearance $$mode; \
	  $(XCB) test -only-testing:MoldCompanionUITests 2>&1 \
	    | grep -E 'error:|TEST (SUCCEEDED|FAILED)'; \
	  [ $${PIPESTATUS[0]} -eq 0 ] || status=1; \
	done; \
	xcrun simctl ui "$(SIM)" appearance $$was; exit $$status

SWIFT_LINT := ../shared/scripts/swift-lint.sh

lint: lint-script lint-layers lint-color lint-type lint-widgets lint-a11y lint-size ## Run every architecture lint

lint-script: ## The shared lint script still catches what it exists to catch
	@../shared/scripts/tests/swift-lint.sh

lint-layers: ## MoldClient stays UI-free
	@! grep -rlnE '^import (SwiftUI|AppKit|UIKit)' $(PACKAGES)/MoldClient/Sources \
	  || { echo "LAYER VIOLATION: MoldClient imports a UI framework"; exit 1; }
	@echo "  layers ok"

lint-color: ## No literal colors -- the system owns the palette
	@$(SWIFT_LINT) color Sources

lint-type: ## Text styles only -- a fixed point size breaks Dynamic Type (DESIGN.md §6)
	@! grep -rnE '\.font\(\.system\(size:|\.font\(\.custom\(|UIFont\.systemFont\(ofSize:|UIFont\(name:' Sources --include='*.swift' \
	  || { echo "TYPE VIOLATION: use a text style (.body, .headline, ...), never a point size"; exit 1; }
	@! grep -rn 'minimumScaleFactor' Sources --include='*.swift' \
	  || { echo "TYPE VIOLATION: never shrink text to fit -- wrap or restack it"; exit 1; }
	@! grep -rnE 'foregroundStyle\(\.(secondary|tertiary)\)' Sources/Companion Sources/Share --include='*.swift' \
	  || { echo "CONTRAST VIOLATION: use .secondaryText (the system .secondary is ~4.4:1 on white)"; exit 1; }
	@echo "  type ok"

lint-widgets: ## The widget extension reads the App Group only: no network, no Keychain
	@! grep -rnE 'URLSession|HTTPBackend|SecItem|KeychainCredentialStore|import MoldClient' Sources/Widgets --include='*.swift' \
	  || { echo "WIDGET VIOLATION: widgets read the App Group snapshot, nothing else"; exit 1; }
	@! grep -rnE 'URLSession|HTTPBackend|SecItem' Sources/Share --include='*.swift' \
	  || { echo "SHARE VIOLATION: the Share extension stages in the App Group and never networks"; exit 1; }
	@echo "  widgets ok"

lint-a11y: ## Every file with a systemName image must name or opt out of it
	@$(SWIFT_LINT) a11y Sources

lint-size: ## Advisory: flag files over 150 lines
	@$(SWIFT_LINT) size Sources

clean: ## Remove build output and the generated project
	@rm -rf $(BUILD) $(PROJECT) Version.yml
