# Mold Studio Companion (iOS / iPadOS). `make help` lists the targets.
SHELL := /bin/bash
.DEFAULT_GOAL := help

XCODEGEN ?= xcodegen
PROJECT  := MoldCompanion.xcodeproj
SCHEME   := MoldCompanion
CONFIG   ?= Debug
BUILD    ?= $(CURDIR)/build
PACKAGES := ../shared/Packages
# Simulators are addressed by UDID, never by name: a runner carries several
# iOS 26.x runtimes with the same device names, and xcodebuild and simctl
# would each resolve a name on their own. Picked lazily and once, so `lint`,
# `help` and `clean` never touch simctl. Override with SIM=<udid>.
SIM_OR_PICK = $(or $(SIM),$(shell scripts/pick-simulator.sh iPhone))
SIM_ID = $(eval SIM_ID := $(SIM_OR_PICK))$(SIM_ID)
IPAD_ID = $(eval IPAD_ID := $(or $(IPAD),$(shell scripts/pick-simulator.sh iPad)))$(IPAD_ID)
# `UITEST_DEVICES` overrides which simulators the audit sweeps (CI: iPhones).
# The smallest screen, when an iPhone SE simulator is installed (not created).
SMALL_ID = $(eval SMALL_ID := $(or $(SMALL),$(shell scripts/pick-simulator.sh small)))$(SMALL_ID)
DEST = platform=iOS Simulator,id=$(SIM_ID)

MARKETING_VERSION ?= $(shell ../macos/scripts/workspace-version.sh)
BUILD_NUMBER ?= $(shell git rev-list --count HEAD 2>/dev/null || echo 1)

# Simulator builds sign ad hoc: no Apple account is needed (a hosted runner
# has none), and the entitlements -- App Group, Keychain group -- are still
# embedded, which the simulator honours. Device and App Store builds sign
# with the team in project.yml (`make archive`).
SIM_SIGN := CODE_SIGN_STYLE=Manual CODE_SIGN_IDENTITY=- DEVELOPMENT_TEAM=

XCB_BASE = xcodebuild -project $(PROJECT) -scheme $(SCHEME) -configuration $(CONFIG) \
       -derivedDataPath "$(BUILD)/DerivedData" $(SIM_SIGN) \
       CURRENT_PROJECT_VERSION=$(BUILD_NUMBER) MARKETING_VERSION=$(MARKETING_VERSION)
XCB = $(XCB_BASE) -destination '$(DEST)'

# Fails loudly rather than handing xcodebuild an empty `id=`.
require-sim:
	@[ -n "$(SIM_ID)" ] || { echo "no iOS 26 simulator found -- install an iOS 26 runtime"; exit 1; }

.PHONY: help require-sim gen build archive run test packages-test uitest lint lint-script lint-layers lint-color lint-type lint-widgets lint-a11y lint-size clean

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

gen: ## Regenerate MoldCompanion.xcodeproj from project.yml
	@command -v $(XCODEGEN) >/dev/null || { echo "xcodegen not found -- run inside 'nix develop'"; exit 1; }
	@printf 'settings:\n  base:\n    MARKETING_VERSION: "%s"\n' "$(MARKETING_VERSION)" > Version.yml
	@$(XCODEGEN) generate --spec project.yml --quiet

build: require-sim gen ## Build for the simulator
	@$(XCB) build | grep -E 'error:|warning: .*MoldCompanion|BUILD (SUCCEEDED|FAILED)' ; exit $${PIPESTATUS[0]}

APP_PATH = $(BUILD)/DerivedData/Build/Products/$(CONFIG)-iphonesimulator/Mold Studio.app

run: require-sim build ## Build, install and launch in the simulator
	@xcrun simctl boot $(SIM_ID) 2>/dev/null || true
	@open -a Simulator
	@xcrun simctl install $(SIM_ID) "$(APP_PATH)"
	@xcrun simctl launch --terminate-running-process $(SIM_ID) io.utensils.mold.companion >/dev/null && echo "  launched in $(SIM_ID)"

test: require-sim gen ## Unit tests on the simulator (the shared packages' own suites run from their Makefiles)
	@$(XCB) test -only-testing:MoldCompanionTests 2>&1 \
	  | grep -E '✘|recorded an issue|error:|Test run with|TEST (SUCCEEDED|FAILED)' ; exit $${PIPESTATUS[0]}

packages-test: require-sim ## MoldClient, MoldStyle and MoldMesh's own suites on the iOS simulator
	@# A package with more than one product (MoldClient + MoldClientTesting)
	@# keeps its tests on the "-Package" scheme.
	@for pkg in MoldClient MoldStyle MoldMesh; do \
	  echo "  $$pkg"; \
	  (cd $(PACKAGES)/$$pkg && scheme=$$pkg && \
	    { xcodebuild -list 2>/dev/null | grep -q "$$pkg-Package" && scheme=$$pkg-Package; true; } && \
	    xcodebuild test -scheme $$scheme -destination '$(DEST)' \
	    -derivedDataPath "$(BUILD)/PackagesData" 2>&1 \
	    | grep -E '✘|recorded an issue|error:|Test run with|TEST (SUCCEEDED|FAILED)'; \
	    exit $${PIPESTATUS[0]}) || exit 1; \
	done

uitest: require-sim gen ## The accessibility audit: every destination at xSmall, Large and AX5, light AND dark, iPhone AND iPad
	@# The audit sees whatever appearance the simulator is in -- contrast
	@# passing in one says nothing about the other -- so it runs in both, on
	@# an iPhone (tab bar) and an iPad (sidebar, where Models lives), and
	@# puts each simulator back the way it found it, even on Ctrl-C.
	@# Retry only failed tests, once in a fresh process. Keep full diagnostics
	@# and preserve xcodebuild's exit status through both output filters.
	@status=0; \
	for dev in $(or $(UITEST_DEVICES),$(SIM_ID) $(SMALL_ID) $(IPAD_ID)); do \
	  xcrun simctl boot $$dev 2>/dev/null || true; \
	  was=$$(xcrun simctl ui $$dev appearance); \
	  trap "xcrun simctl ui $$dev appearance $$was" EXIT INT TERM; \
	  for mode in light dark; do \
	    echo "  $$(xcrun simctl list devices | grep $$dev | sed -E 's/ \(.*//;s/^ *//') · $$mode"; \
	    xcrun simctl ui $$dev appearance $$mode; \
	    results="$(BUILD)/UITestResults/$$dev-$$mode"; \
	    mkdir -p "$(BUILD)/UITestResults"; rm -rf "$$results.xcresult"; \
	    $(XCB_BASE) -destination "platform=iOS Simulator,id=$$dev" test -only-testing:MoldCompanionUITests \
	      -retry-tests-on-failure -test-iterations 2 -test-repetition-relaunch-enabled YES \
	      -resultBundlePath "$$results.xcresult" 2>&1 \
	      | tee "$$results.log" | grep -E 'error:|TEST (SUCCEEDED|FAILED)'; \
	    [ $${PIPESTATUS[0]} -eq 0 ] || status=1; \
	  done; \
	  xcrun simctl ui $$dev appearance $$was; trap - EXIT INT TERM; \
	done; exit $$status

ARCHIVE ?= $(BUILD)/MoldCompanion.xcarchive

archive: gen ## Unsigned Release archive for App Store Connect (signed at export)
	@xcodebuild -project $(PROJECT) -scheme $(SCHEME) -configuration Release \
	  -destination 'generic/platform=iOS' -archivePath "$(ARCHIVE)" \
	  CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO \
	  CURRENT_PROJECT_VERSION=$(BUILD_NUMBER) MARKETING_VERSION=$(MARKETING_VERSION) \
	  archive | grep -E 'error:|ARCHIVE (SUCCEEDED|FAILED)'; exit $${PIPESTATUS[0]}
	@test -f "$(ARCHIVE)/Products/Applications/Mold Studio.app/AppIcon60x60@2x.png" \
	  || { echo "the archive has no app icon -- App Store Connect would reject it"; exit 1; }

SWIFT_LINT := ../shared/scripts/swift-lint.sh

lint: lint-script lint-layers lint-color lint-type lint-widgets lint-a11y lint-size ## Run every architecture lint

lint-script: ## The shared lint script still catches what it exists to catch
	@../shared/scripts/tests/swift-lint.sh
	@../../scripts/tests/companion-helper.sh

lint-layers: ## MoldClient stays UI-free; one composition root; keys only in the Keychain
	@! grep -rlnE '^import (SwiftUI|AppKit|UIKit)' $(PACKAGES)/MoldClient/Sources \
	  || { echo "LAYER VIOLATION: MoldClient imports a UI framework"; exit 1; }
	@! grep -rlnE 'HTTPBackend(\(|\.claimPairing)' Sources --include='*.swift' \
	  | grep -v 'CompanionStores+Backend.swift' \
	  || { echo "LAYER VIOLATION: a backend is constructed outside the composition root"; exit 1; }
	@! grep -rnE 'X-Api-Key|api_key=' Sources --include='*.swift' \
	  || { echo "KEY VIOLATION: a key never goes in a URL or a hand-built header"; exit 1; }
	@! grep -rn 'SecretStore' Sources --include='*.swift' \
	  || { echo "KEY VIOLATION: the phone keeps keys in the Keychain, never SecretStore's file"; exit 1; }
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
	@! grep -rnE 'URLSession|HTTPBackend|SecItem|KeychainCredentialStore|import MoldClient' Sources/Widgets Sources/Shared --include='*.swift' \
	  || { echo "WIDGET VIOLATION: widgets read the App Group snapshot, nothing else"; exit 1; }
	@! grep -rnE 'URLSession|HTTPBackend|SecItem' Sources/Share Sources/Shared --include='*.swift' \
	  || { echo "SHARE VIOLATION: the Share extension stages in the App Group and never networks"; exit 1; }
	@echo "  widgets ok"

lint-a11y: ## Every file with a systemName image must name or opt out of it
	@$(SWIFT_LINT) a11y Sources

lint-size: ## Advisory: flag files over 150 lines
	@$(SWIFT_LINT) size Sources

clean: ## Remove build output and the generated project
	@rm -rf $(BUILD) $(PROJECT) Version.yml
