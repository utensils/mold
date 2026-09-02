#!/usr/bin/env python3
"""Validate MiniMax H3 pins and compare per-layer conformance evidence.

The checked-in contract is intentionally weight-free. Real checkpoint evidence
must live outside the repository and is accepted only with a separately managed
authorization record whose source document is content-addressed.
"""

from __future__ import annotations

import argparse
import heapq
import hashlib
import importlib.util
import json
import math
import pathlib
import struct
import subprocess
import sys
from typing import Any


MAX_COMPARISON_DIAGNOSTICS = 128
MAX_INDEX_DIAGNOSTICS = 16


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "minimax_h3"
MANIFEST_PATH = FIXTURE_DIR / "conformance-manifest.json"
SYNTHETIC_PATH = FIXTURE_DIR / "synthetic-v1.json"
MANIFEST_SCHEMA_PATH = (
    REPO_ROOT / "docs" / "qualification" / "minimax-h3-conformance-manifest.schema.json"
)
BUNDLE_SCHEMA_PATH = (
    REPO_ROOT / "docs" / "qualification" / "minimax-h3-fixture-bundle.schema.json"
)
LAYER_OUTPUT_SCHEMA_PATH = (
    REPO_ROOT / "docs" / "qualification" / "minimax-h3-layer-output.schema.json"
)
SYNTHETIC_ORACLE_PATH = FIXTURE_DIR / "synthetic-oracle-v1.json"
SYNTHETIC_MOLD_PATH = FIXTURE_DIR / "synthetic-mold-v1.json"

SCHEMA_VERSION = "mold.minimax-h3.conformance-manifest.v1"
SYNTHETIC_SCHEMA_VERSION = "mold.minimax-h3.synthetic.v1"
BUNDLE_SCHEMA_VERSION = "mold.minimax-h3.fixture-bundle.v1"
AUTHORIZATION_SCHEMA_VERSION = "mold.minimax-h3.authorization.v1"
LAYER_OUTPUT_SCHEMA_VERSION = "mold.minimax-h3.layer-output.v1"
LAYER_ADAPTER_SCHEMA_VERSION = "mold.minimax-h3.layer-adapter.v1"
COMPONENT_AUTHORITY_SET_SCHEMA = "mold.minimax-h3.component-authority-set.v1"

# The checked synthetic candidate records the Mold tree from which the original
# H3 contract was extended. Real captures record their exact candidate commit.
SYNTHETIC_MOLD_REVISION = "ea3d1d86fbd09eb6d47848649168c70392db9c63"

EXPECTED_REVISIONS = {
    "minimax-official-code": "8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea",
    "minimax-official-model": "bfc8ed0353f5a9733be73e6b2c98ec0948195b86",
    "diffusers": "9c6a68c32b3b2a64db91800b624d33cec6e25ab8",
    "transformers": "42f189ded85d18d00b51161d694cafd325e32b91",
    "comfyui": "a464ac33588ae182f81a090d910cfbf21e255b73",
    "comfy-checkpoints": "eb8a16107c595128b3a578f82d2ce2f75920c355",
    # Third-party pruned NVFP4 transformers (#1319): downloadable, no runtime arm.
    "nvfp4-checkpoints": "908eccad7e68751190d04c171956f163bfeed741",
    # Third-party Turbo LoRA adapters (v1.1 4-step 768p, v1.0 8-step 768p):
    # runnable on the FL2VA compact stack.
    "lightx2v-turbo-adapters": "05ef678438e84933c406131b59abbf86919b3aac",
    # Third-party SVD-resized Turbo LoRA adapters (avg rank 21, lossy):
    # derivatives of three adapters already pinned above, runnable on the
    # compact stacks.
    "drbaph-resized-loras": "be8eb3ea3466cbb7def202ffec0d2fdc054256ac",
    "sglang": "0c3a76fa0a5bfab410b645f4143e7e8e3cc25c77",
    "vllm-omni": "3d7fc3b9ba3cac88d579d4dc35b78b0b641675fc",
}

EXPECTED_ORACLE_RUNTIME = {
    "python": "3.13.13",
    "torch": "2.13.0+cu130",
    "numpy": "2.5.1",
    "cuda": "13.0",
    "transformers_revision": EXPECTED_REVISIONS["transformers"],
}
EXPECTED_LICENSE_SHA256 = (
    "59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44"
)

EXPECTED_EXCLUDED_ACCELERATIONS = {
    "torch-compile",
    "fp8",
    "cache-dit",
    "sageattention",
    "stochastic-sampling",
    "approximate-attention",
    "tf32",
}
EXPECTED_EXCLUDED_ACCELERATION_ALIASES = {
    "torch-compile": {"torch-compile", "torch-inductor", "inductor"},
    "fp8": {"fp8", "float8", "e4m3", "e4m3fn", "e5m2"},
    "cache-dit": {"cache-dit"},
    "sageattention": {"sageattention", "sage-attn"},
    "stochastic-sampling": {"stochastic-sampling", "stochastic", "ancestral"},
    "approximate-attention": {
        "approximate-attention",
        "approximate",
        "approx-attn",
        "approx",
    },
    "tf32": {"tf32", "tensorfloat32"},
}

EXPECTED_COMPONENT_INDEXES = {
    "official-license": (
        "LICENSE",
        EXPECTED_LICENSE_SHA256,
    ),
    "official-model-index": (
        "model_index.json",
        "5a587fe13b2371427415ac892463142683aefcd8d322e274a3a095eac37ac7d2",
    ),
    "official-modular-index": (
        "modular_model_index.json",
        "a2b6a210e482ffb78e613b553f570c44e101afce6741bd4ed91429d0559af031",
    ),
    "official-fl2va-transformer-index": (
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "ac30a3b58963f2e735d493475fbb81853a5735ec947619648b3e045acda6783e",
    ),
    "official-ref2va-transformer-index": (
        "transformer_ref/diffusion_pytorch_model.safetensors.index.json",
        "ac30a3b58963f2e735d493475fbb81853a5735ec947619648b3e045acda6783e",
    ),
    "official-fl2va-transformer-config": (
        "transformer/config.json",
        "74c11bff524336576096993cbfcdcdc2ef4fa2fa4409df693bdcbc6c666282ae",
    ),
    "official-ref2va-transformer-config": (
        "transformer_ref/config.json",
        "74c11bff524336576096993cbfcdcdc2ef4fa2fa4409df693bdcbc6c666282ae",
    ),
    "official-fl2va-transformer-shard-00001-of-00014": (
        "transformer/diffusion_pytorch_model-00001-of-00014.safetensors",
        "2d847200c45c09dd7f973c1b096663068408ef851ee0b3711d059b6dc5dcd028",
    ),
    "official-fl2va-transformer-shard-00002-of-00014": (
        "transformer/diffusion_pytorch_model-00002-of-00014.safetensors",
        "2c4d362eddd2802180ac9c744849eb9ba8d9c8b984bdf9822cb02ed004b29184",
    ),
    "official-fl2va-transformer-shard-00003-of-00014": (
        "transformer/diffusion_pytorch_model-00003-of-00014.safetensors",
        "949c5aafbbfa5654da730a6a7fafd75adb164d0857b095a30e8bb6d390887d69",
    ),
    "official-fl2va-transformer-shard-00004-of-00014": (
        "transformer/diffusion_pytorch_model-00004-of-00014.safetensors",
        "eef7616790105ee839766bb2027203bf2c0d87c6aa038dca84145a8675f5ce28",
    ),
    "official-fl2va-transformer-shard-00005-of-00014": (
        "transformer/diffusion_pytorch_model-00005-of-00014.safetensors",
        "43fdf42d638e8bc6745f713fae80c93bb301807a1a5ae7249344ce28e202a494",
    ),
    "official-fl2va-transformer-shard-00006-of-00014": (
        "transformer/diffusion_pytorch_model-00006-of-00014.safetensors",
        "6442510b34d173653f0cce5c964b935395a8f7accf0b9cc0aa31aec59805239d",
    ),
    "official-fl2va-transformer-shard-00007-of-00014": (
        "transformer/diffusion_pytorch_model-00007-of-00014.safetensors",
        "29f48f535c91dac76496ca821eeb16ca24bc4caf3f0cae8b920a89b1f966da6d",
    ),
    "official-fl2va-transformer-shard-00008-of-00014": (
        "transformer/diffusion_pytorch_model-00008-of-00014.safetensors",
        "c711b096c764bd60f0b8b6ad49518bfab6d614fb788c725add8741c0674a4cd8",
    ),
    "official-fl2va-transformer-shard-00009-of-00014": (
        "transformer/diffusion_pytorch_model-00009-of-00014.safetensors",
        "44428defe3976cbb87635ad200b958199e739986697cd29fdf27aeb7294b5944",
    ),
    "official-fl2va-transformer-shard-00010-of-00014": (
        "transformer/diffusion_pytorch_model-00010-of-00014.safetensors",
        "3d44939c374c9da382e9c6877e1946adf7b84e08c7a881c068f228d6849411c9",
    ),
    "official-fl2va-transformer-shard-00011-of-00014": (
        "transformer/diffusion_pytorch_model-00011-of-00014.safetensors",
        "224d24430b58127a5577721084e0e704a0e74ec96dd7c35bc6fc0994ebd87c33",
    ),
    "official-fl2va-transformer-shard-00012-of-00014": (
        "transformer/diffusion_pytorch_model-00012-of-00014.safetensors",
        "48fa2bd8fe134eef565ab2464f1c2589a6657cba0d14283dfc06b532f8961f3c",
    ),
    "official-fl2va-transformer-shard-00013-of-00014": (
        "transformer/diffusion_pytorch_model-00013-of-00014.safetensors",
        "be5b4b1809f9d546ffd4b3fcf41e5c1e02b819125caa6bc105c109b04c051bd3",
    ),
    "official-fl2va-transformer-shard-00014-of-00014": (
        "transformer/diffusion_pytorch_model-00014-of-00014.safetensors",
        "8fbd5e6c1fb1df7ce988ca90f3d59e7610e465c7517e4b344eda4a214ba4b97d",
    ),
    "official-ref2va-transformer-shard-00001-of-00014": (
        "transformer_ref/diffusion_pytorch_model-00001-of-00014.safetensors",
        "7a3fcad885f51560e550b2e84c9a8d8b35e62996cfd9076937e992bd23478df9",
    ),
    "official-ref2va-transformer-shard-00002-of-00014": (
        "transformer_ref/diffusion_pytorch_model-00002-of-00014.safetensors",
        "1638ae1dc8ae26c4ba43ad28a6d851ad8983847324bb2b468719c7c81f219706",
    ),
    "official-ref2va-transformer-shard-00003-of-00014": (
        "transformer_ref/diffusion_pytorch_model-00003-of-00014.safetensors",
        "1ef3c4954ffe5a664c2e3028e2a3241190d9c159dce6ba1136002c6af1db5353",
    ),
    "official-ref2va-transformer-shard-00004-of-00014": (
        "transformer_ref/diffusion_pytorch_model-00004-of-00014.safetensors",
        "12d92f2975cfd5c5b786126385c52e5bf64884d4b4d6e60c3ef5d857c3f7469f",
    ),
    "official-ref2va-transformer-shard-00005-of-00014": (
        "transformer_ref/diffusion_pytorch_model-00005-of-00014.safetensors",
        "304d41ce03d59ac94bceb055935bf4e034df0badf8b0df4ded327c08a288a4cc",
    ),
    "official-ref2va-transformer-shard-00006-of-00014": (
        "transformer_ref/diffusion_pytorch_model-00006-of-00014.safetensors",
        "12a134b7c76d86edbe8fa2dc315f6cdaf4e1aca1b6ea4dfe4cad92df03d42eeb",
    ),
    "official-ref2va-transformer-shard-00007-of-00014": (
        "transformer_ref/diffusion_pytorch_model-00007-of-00014.safetensors",
        "b96395261359937c00fb42f4eb29306dc59b1a3368eeba52af4fb66e3e142c69",
    ),
    "official-ref2va-transformer-shard-00008-of-00014": (
        "transformer_ref/diffusion_pytorch_model-00008-of-00014.safetensors",
        "1897a6bf3b4fc834bb82d73ca02a7afc7d38c07f50ec5382cd54cd2f91b604d1",
    ),
    "official-ref2va-transformer-shard-00009-of-00014": (
        "transformer_ref/diffusion_pytorch_model-00009-of-00014.safetensors",
        "edfb38235adc96b99f55a401849befce59075a745e99c2d8c63ff358dd36443d",
    ),
    "official-ref2va-transformer-shard-00010-of-00014": (
        "transformer_ref/diffusion_pytorch_model-00010-of-00014.safetensors",
        "f8710775cf3413670edd7e23861b650a3431a71a6cc14cb1080623ab6b052385",
    ),
    "official-ref2va-transformer-shard-00011-of-00014": (
        "transformer_ref/diffusion_pytorch_model-00011-of-00014.safetensors",
        "9e18acc09f84edb5b34df9628efa15cfcab8bb76e8e20c1c2e979a107a0f7215",
    ),
    "official-ref2va-transformer-shard-00012-of-00014": (
        "transformer_ref/diffusion_pytorch_model-00012-of-00014.safetensors",
        "ea2e18228f8bdba1a4e0f32b155e4586df055997c45356213d05b971ba13e2f4",
    ),
    "official-ref2va-transformer-shard-00013-of-00014": (
        "transformer_ref/diffusion_pytorch_model-00013-of-00014.safetensors",
        "1e12083b1875678f7414ff55b09cd8bb1c30b861243f9bb7ff1e75b6ad3f1bdc",
    ),
    "official-ref2va-transformer-shard-00014-of-00014": (
        "transformer_ref/diffusion_pytorch_model-00014-of-00014.safetensors",
        "b340f44b5690cc745d48ae399381ec15b26a4fe25d483f677ccb4960dadb50d4",
    ),
    "official-text-encoder-index": (
        "text_encoder/model.safetensors.index.json",
        "06c952c569285870b811989b794b9766493e280fb77fbcb957fc4e5fcf25403a",
    ),
    "official-text-encoder-shard-00001-of-00014": (
        "text_encoder/model-00001-of-00014.safetensors",
        "6b9dfbc930e505402ae9d7e5091a9d7d656cda5f34614f01cfe70bfb0cca27cb",
    ),
    "official-text-encoder-shard-00002-of-00014": (
        "text_encoder/model-00002-of-00014.safetensors",
        "d8bb44b4ff303fe76fe9e894022fb3dc71b15a2e716592790fe0e3c3e60478fa",
    ),
    "official-text-encoder-shard-00003-of-00014": (
        "text_encoder/model-00003-of-00014.safetensors",
        "54f22e8b3168f8dc962fac0d313607ebf52a12b433d4cf3098a0d82d9f042940",
    ),
    "official-text-encoder-shard-00004-of-00014": (
        "text_encoder/model-00004-of-00014.safetensors",
        "ad09c74d3c13ee29b5d0d84548fd8a3424a651564eaccd519946c296e59c557f",
    ),
    "official-text-encoder-shard-00005-of-00014": (
        "text_encoder/model-00005-of-00014.safetensors",
        "fc993c8a0e2a5b0570f383e1a95dc3a1281d1b224b6f3ee908f4827941e1dfc2",
    ),
    "official-text-encoder-shard-00006-of-00014": (
        "text_encoder/model-00006-of-00014.safetensors",
        "82f05620d1f718a90c362b221d6a184ff1a0f53301d706882d3df49695fa1974",
    ),
    "official-text-encoder-shard-00007-of-00014": (
        "text_encoder/model-00007-of-00014.safetensors",
        "fb91da8cb01ff4de3eef0eab1c3e769a734b3a1aafc61734068638a0d6c86934",
    ),
    "official-text-encoder-shard-00008-of-00014": (
        "text_encoder/model-00008-of-00014.safetensors",
        "431ca56535c8781944ce3801f5eb61c45531e853ecc5846d936ebaf4761b764f",
    ),
    "official-text-encoder-shard-00009-of-00014": (
        "text_encoder/model-00009-of-00014.safetensors",
        "3825e3f4302f4d2f7d76aa7430d2ce0864fde6b9e540a5806bf0d8e38e4d9f47",
    ),
    "official-text-encoder-shard-00010-of-00014": (
        "text_encoder/model-00010-of-00014.safetensors",
        "aded5a4d1d5e22dbd8b6f79266b6eb88c840411b09527c53917a1419ace22e2f",
    ),
    "official-text-encoder-shard-00011-of-00014": (
        "text_encoder/model-00011-of-00014.safetensors",
        "3820ffe8d8d6477f6fe8d614ef3c87abb264ee39accebf43a1507b970d80946f",
    ),
    "official-text-encoder-shard-00012-of-00014": (
        "text_encoder/model-00012-of-00014.safetensors",
        "05ad2d08ce71963121c9b03f1d9ec5d7641052f4b23c6c12b80d71065eb8e98e",
    ),
    "official-text-encoder-shard-00013-of-00014": (
        "text_encoder/model-00013-of-00014.safetensors",
        "b64f2289871261fdd1abbd3b78bcd66011b341de3dc8eeb2ed1a473ee7c8d95c",
    ),
    "official-text-encoder-shard-00014-of-00014": (
        "text_encoder/model-00014-of-00014.safetensors",
        "e45b6c9998c77ee5a6577f9f47bc76416c1d4d387169e50c4c9d3134ea51b13b",
    ),
    "official-video-vae-config": (
        "vae/config.json",
        "78f67deec3d63aae807f2bfe7154bc1e26f6372cb20b63265fcbae1b62bb5745",
    ),
    "official-video-vae-index": (
        "vae/diffusion_pytorch_model.safetensors.index.json",
        "15f6d44553c3c616b0dc999920aa784f92ecee7e4201f1f99ac405cfbf3061ca",
    ),
    "official-video-vae-shard-00001-of-00003": (
        "vae/diffusion_pytorch_model-00001-of-00003.safetensors",
        "72f4c6be84ac0674f27398cde991dd9d719762f3952c4921aa66b2ce542f6374",
    ),
    "official-video-vae-shard-00002-of-00003": (
        "vae/diffusion_pytorch_model-00002-of-00003.safetensors",
        "2e05e8bc23fa4071043e17fd242be8acd0685e781a43987432b2eae925be4198",
    ),
    "official-video-vae-shard-00003-of-00003": (
        "vae/diffusion_pytorch_model-00003-of-00003.safetensors",
        "c05d6ac4b1a33de372799d708531da6320f6a3ce6d1ce6d895e770988e004a39",
    ),
    "official-audio-vae-config": (
        "audio_vae/config.json",
        "9a3c645ff892b376c6f5f4c8685964cd75474731af594ff058492a0000caabb6",
    ),
    "official-tokenizer-json": (
        "tokenizer/tokenizer.json",
        "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
    ),
    "official-tokenizer-config": (
        "tokenizer/tokenizer_config.json",
        "a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68",
    ),
    "official-tokenizer-merges": (
        "tokenizer/merges.txt",
        "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    ),
    "official-tokenizer-vocab": (
        "tokenizer/vocab.json",
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
    "official-processor-config": (
        "processor/preprocessor_config.json",
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    ),
    "official-processor-video-config": (
        "processor/video_preprocessor_config.json",
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    ),
    "official-processor-chat-template": (
        "processor/chat_template.json",
        "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce",
    ),
    "official-processor-tokenizer-json": (
        "processor/tokenizer.json",
        "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
    ),
    "official-processor-tokenizer-config": (
        "processor/tokenizer_config.json",
        "a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68",
    ),
    "official-processor-merges": (
        "processor/merges.txt",
        "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    ),
    "official-processor-vocab": (
        "processor/vocab.json",
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
    "official-video-scheduler-config": (
        "scheduler/scheduler_config.json",
        "8fa6c3aa70dc9e691e1a6df899fd1b6f75f70481a27cee6e18a303817075c304",
    ),
    "official-audio-scheduler-config": (
        "audio_scheduler/scheduler_config.json",
        "804780f7133477067bd6bbfbc02dc8b3cf9feeb400f97c08f5b1d5f6cbab3840",
    ),
}

EXPECTED_LAYERS = {
    "tokenizer-processor",
    "qwen-layer-50",
    "visual-vae",
    "audio-vae",
    "token-refiner",
    "transformer-block",
    "packed-layout",
    "dual-sampler",
    "noise-allocation",
    "end-to-end-t2va",
    "end-to-end-fl2va",
    "end-to-end-ref2va",
}

AUDIO_VAE_CHECKPOINT_SHA256_BY_ROLE = {
    "oracle": "52c59e67ba8de5477c81bfbced0327aabf500f1bfdeefd5ee754529241cb26cb",
    "mold": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
}

FORBIDDEN_FIXTURE_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".flac",
    ".gguf",
    ".jpeg",
    ".jpg",
    ".mp3",
    ".mp4",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".wav",
    ".webp",
}


class ConformanceFailure(Exception):
    """A fail-closed conformance contract violation."""


def fail(message: str) -> None:
    raise ConformanceFailure(message)


def load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read JSON {path}: {error}")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        fail(f"cannot hash {path}: {error}")
    return digest.hexdigest()


def checked_repository_implementation_path(value: str) -> pathlib.Path:
    relative = pathlib.PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.parts[0] != "scripts"
        or ".." in relative.parts
    ):
        fail("oracle adapter implementation path escapes the reviewed checkout")
    current = REPO_ROOT
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            current.lstat()
        except OSError:
            fail("oracle adapter implementation path is unavailable")
        if pathlib.Path(current).is_symlink():
            fail("oracle adapter implementation path may not traverse a symbolic link")
        if index < len(relative.parts) - 1 and not current.is_dir():
            fail("oracle adapter implementation path has a non-directory parent")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(REPO_ROOT.resolve(strict=True))
    except (OSError, ValueError):
        fail("oracle adapter implementation path escapes the reviewed checkout")
    if not resolved.is_file():
        fail("oracle adapter implementation path is not a regular file")
    return resolved


def schema_helper() -> Any:
    helper_path = REPO_ROOT / "scripts" / "validate-cuda-qualification-report.py"
    spec = importlib.util.spec_from_file_location("mold_schema_helper", helper_path)
    if spec is None or spec.loader is None:
        fail(f"cannot load hermetic schema helper: {helper_path}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def validate_schema(instance: Any, schema_path: pathlib.Path) -> None:
    helper = schema_helper()
    schema = load_json(schema_path)
    try:
        helper.audit_schema_keywords(schema)
        helper.validate_schema(instance, schema, schema)
    except helper.ValidationFailure as error:
        fail(f"schema validation failed for {schema_path.name}: {error}")


def f32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def shifted_schedule(grid_points: int, shift: float) -> list[float]:
    values: list[float] = []
    for index in range(grid_points):
        base = f32(1.0 - index / (grid_points - 1))
        numerator = f32(f32(shift) * base)
        denominator = f32(1.0 + f32(f32(shift - 1.0) * base))
        shifted = f32(numerator / denominator)
        if not values or shifted != values[-1]:
            values.append(shifted)
    return values


def euler_update(
    sample: list[float], velocity: list[float], sigma: float, sigma_next: float
) -> list[float]:
    timestep = f32(1.0 - sigma)
    sigma_from_timestep = f32(1.0 - timestep)
    ratio = f32(sigma_next / sigma)
    one_minus_ratio = f32(1.0 - ratio)
    output = []
    for sample_value, velocity_value in zip(sample, velocity, strict=True):
        sample_value = f32(sample_value)
        velocity_value = f32(velocity_value)
        denoised = f32(sample_value + f32(sigma_from_timestep * velocity_value))
        output.append(f32(f32(ratio * sample_value) + f32(one_minus_ratio * denoised)))
    return output


def align_frames(requested: int) -> int:
    while requested % 17 != 5:
        requested += 1
    return requested


def synthetic_fixture() -> dict[str, Any]:
    frame_grid = []
    for nominal_seconds in (5, 10, 15):
        requested = nominal_seconds * 24
        aligned = align_frames(requested)
        frame_grid.append(
            {
                "nominal_seconds": nominal_seconds,
                "requested_frames": requested,
                "aligned_frames": aligned,
                "video_latent_frames": ((aligned - 5) // 17) * 5 + 2,
                "audio_latents_per_channel": round(aligned / 24 * 40),
                "actual_duration_seconds": aligned / 24,
            }
        )

    video_sigmas = shifted_schedule(4, 12.0)
    audio_sigmas = shifted_schedule(4, 3.0)
    sample = [0.25, -0.5, 1.0, -2.0]
    video_velocity = [0.5, 0.25, -1.5, 2.0]
    audio_velocity = [-0.25, 0.75, 0.5, -1.0]
    step_index = 1

    spans = [5.0 / 3.0 * value for value in (1, 4, 4, 4, 4, 1)]
    temporal_positions = [3.0]
    for span in spans:
        temporal_positions.append(temporal_positions[-1] + span)

    return {
        "schema_version": SYNTHETIC_SCHEMA_VERSION,
        "authority": {
            "source": "diffusers-minimax-h3",
            "revision": EXPECTED_REVISIONS["diffusers"],
            "execution": "weight-free-python-float32-emulation",
        },
        "frame_grid": frame_grid,
        "scheduler": {
            "grid_points": 4,
            "transformer_evaluations": 3,
            "video_shift": 12.0,
            "audio_shift": 3.0,
            "video_sigmas": video_sigmas,
            "audio_sigmas": audio_sigmas,
            "video_timesteps": [f32(1.0 - value) for value in video_sigmas[:-1]],
            "audio_timesteps": [f32(1.0 - value) for value in audio_sigmas[:-1]],
            "terminal_zero_included": True,
            "ancestral_noise_reinjected": False,
        },
        "coupled_update": {
            "step_index": step_index,
            "sample": sample,
            "video_velocity": video_velocity,
            "audio_velocity": audio_velocity,
            "video_next": euler_update(
                sample,
                video_velocity,
                video_sigmas[step_index],
                video_sigmas[step_index + 1],
            ),
            "audio_next": euler_update(
                sample,
                audio_velocity,
                audio_sigmas[step_index],
                audio_sigmas[step_index + 1],
            ),
            "velocity_sign": "data-ward-plus",
            "update_dtype": "float32",
        },
        "packed_layout": {
            "order": [
                "text",
                "reference-audio",
                "reference-video",
                "target-audio",
                "target-video",
            ],
            "row_counts": [3, 4, 6, 8, 10],
            "row_offsets": [0, 3, 7, 13, 21, 31],
            "modality_tags": {
                "video": 0,
                "text": 1,
                "audio": 2,
            },
            "video_soundtrack_precedes_visual_rows": True,
        },
        "temporal_rope": {
            "origin": 3.0,
            "frame_rescale": 5.0 / 3.0,
            "frames_per_latent_pattern": [1, 4, 4, 4, 4],
            "positions_for_seven_latents": temporal_positions,
        },
        "noise_allocation_order": [
            {"domain": "condition-posterior", "shape": [1, 24, 1, 2, 2]},
            {"domain": "condition-noise", "shape": [1, 24, 1, 2, 2]},
            {"domain": "target-video", "shape": [1, 24, 2, 2, 2]},
            {"domain": "target-audio", "shape": [2, 3, 32]},
        ],
    }


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def tensor_content_sha256(dtype: str, shape: list[int], values: list[float]) -> str:
    """Hash the checked synthetic tensor's canonical numeric representation."""
    payload = {"dtype": dtype, "shape": shape, "values": values}
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def component_authority_set_sha256(
    component_indexes: list[dict[str, str]],
) -> str:
    payload = {
        "schema_version": COMPONENT_AUTHORITY_SET_SCHEMA,
        "components": [
            {"id": component["id"], "sha256": component["sha256"]}
            for component in sorted(component_indexes, key=lambda value: value["id"])
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_statistics(values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "std": math.sqrt(variance),
    }


def synthetic_tensor(key: str, values: list[float]) -> dict[str, Any]:
    shape = [len(values)]
    return {
        "key": key,
        "shape": shape,
        "dtype": "float32",
        "content_sha256": tensor_content_sha256("float32", shape, values),
        "statistics": tensor_statistics(values),
        "samples": [
            {"index": [index], "value": value} for index, value in enumerate(values)
        ],
    }


def synthetic_layer_outputs() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the checked weight-free producer/adaptor comparison pair."""
    coupled_update = synthetic_fixture()["coupled_update"]
    video_values = coupled_update["video_next"]
    oracle_audio_values = coupled_update["audio_next"]
    mold_audio_values = [f32(value + 0.000001) for value in oracle_audio_values]
    input_record = {
        "id": "synthetic-v1",
        "sha256": sha256_file(SYNTHETIC_PATH),
        "component_index_sha256": EXPECTED_COMPONENT_INDEXES["official-modular-index"][
            1
        ],
    }
    environment = {
        "device": "cpu-synthetic",
        "dtype": "float32",
        "attention_backend": "scalar-python",
        "forbidden_accelerations_disabled": True,
    }

    def document(
        role: str,
        implementation: str,
        source_id: str,
        revision: str,
        outputs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "schema_version": LAYER_OUTPUT_SCHEMA_VERSION,
            "family": "minimax-h3",
            "case_id": "synthetic-coupled-update-v1",
            "layer": "dual-sampler",
            "authority_tier": "synthetic",
            "authorization_document_sha256": None,
            "input": dict(input_record),
            "producer": {
                "role": role,
                "implementation": implementation,
                "source_id": source_id,
                "revision": revision,
            },
            "adapter": {
                "schema_version": LAYER_ADAPTER_SCHEMA_VERSION,
                "id": f"{implementation}-synthetic-dual-sampler-v1",
                "command": (
                    "python3 scripts/minimax-h3-conformance.py "
                    f"print-synthetic-output --role {role}"
                ),
                "tensor_hash_encoding": "canonical-json-numeric-v1",
            },
            "environment": dict(environment),
            "outputs": outputs,
        }

    oracle = document(
        "oracle",
        "diffusers",
        "diffusers",
        EXPECTED_REVISIONS["diffusers"],
        [
            synthetic_tensor("video_next", video_values),
            synthetic_tensor("audio_next", oracle_audio_values),
        ],
    )
    oracle["comparison"] = [
        {
            "key": "video_next",
            "absolute": 0.0,
            "relative": 0.0,
            "metric": "elementwise-atol-plus-rtol",
            "hash_policy": "exact",
        },
        {
            "key": "audio_next",
            "absolute": 0.000002,
            "relative": 0.000001,
            "metric": "elementwise-atol-plus-rtol",
            "hash_policy": "record-only",
        },
    ]
    mold = document(
        "mold",
        "mold-synthetic-adapter",
        "mold",
        SYNTHETIC_MOLD_REVISION,
        [
            synthetic_tensor("video_next", video_values),
            synthetic_tensor("audio_next", mold_audio_values),
        ],
    )
    return oracle, mold


def non_finite_name(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    return "Inf" if value > 0 else "-Inf"


def reject_non_finite(value: Any, label: str, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        fail(f"{label} {location} is {non_finite_name(value)}")
    if isinstance(value, dict):
        for key, child in value.items():
            reject_non_finite(child, label, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_non_finite(child, label, f"{location}[{index}]")


def keyed_records(
    records: list[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record["key"]
        if key in keyed:
            fail(f"{label} contains duplicate key {key!r}")
        keyed[key] = record
    return keyed


def sample_records(
    output: dict[str, Any], label: str
) -> dict[tuple[int, ...], dict[str, Any]]:
    samples: dict[tuple[int, ...], dict[str, Any]] = {}
    shape = output["shape"]
    for sample in output["samples"]:
        index = tuple(sample["index"])
        if len(index) != len(shape):
            fail(
                f"{label} sample index {list(index)} has rank {len(index)}, "
                f"expected {len(shape)}"
            )
        if any(offset >= extent for offset, extent in zip(index, shape, strict=True)):
            fail(f"{label} sample index {list(index)} is outside shape {shape}")
        if index in samples:
            fail(f"{label} contains duplicate sample index {list(index)}")
        samples[index] = sample
    return samples


def validate_layer_output(value: Any, label: str) -> dict[str, Any]:
    reject_non_finite(value, label)
    validate_schema(value, LAYER_OUTPUT_SCHEMA_PATH)
    if value["schema_version"] != LAYER_OUTPUT_SCHEMA_VERSION:
        fail(f"{label} uses an unsupported layer-output schema")
    if value["family"] != "minimax-h3":
        fail(f"{label} targets a different model family")
    if value["layer"] not in EXPECTED_LAYERS:
        fail(f"{label} names unknown layer {value['layer']!r}")

    component_indexes = value["input"].get("component_indexes")
    if component_indexes is None:
        component_hashes = {
            component_sha for _, component_sha in EXPECTED_COMPONENT_INDEXES.values()
        }
        if value["input"]["component_index_sha256"] not in component_hashes:
            fail(f"{label} component index hash is not in the pinned authority set")
    else:
        component_ids: set[str] = set()
        for component in component_indexes:
            identifier = component["id"]
            if identifier in component_ids:
                fail(f"{label} component indexes contain duplicate id {identifier!r}")
            component_ids.add(identifier)
            expected = EXPECTED_COMPONENT_INDEXES.get(identifier)
            if expected is None or component["sha256"] != expected[1]:
                fail(f"{label} component index authority is not pinned")
        expected_summary = (
            component_indexes[0]["sha256"]
            if len(component_indexes) == 1
            else component_authority_set_sha256(component_indexes)
        )
        if value["input"]["component_index_sha256"] != expected_summary:
            fail(f"{label} component index summary does not match its authority list")

    producer = value["producer"]
    if producer["role"] == "oracle":
        source_id = producer["source_id"]
        expected_revision = EXPECTED_REVISIONS.get(source_id)
        if expected_revision is None or producer["revision"] != expected_revision:
            fail(f"{label} oracle source revision is not pinned")
    elif producer["source_id"] != "mold":
        fail(f"{label} Mold producer source_id must be 'mold'")

    authorization_hash = value["authorization_document_sha256"]
    if value["authority_tier"] == "synthetic":
        if authorization_hash is not None:
            fail(f"{label} synthetic evidence must not claim real authorization")
    elif authorization_hash is None:
        fail(f"{label} real evidence lacks its authorization document hash")

    outputs = keyed_records(value["outputs"], f"{label} outputs")
    for key, output in outputs.items():
        sample_records(output, f"{label} output {key!r}")
        statistics = output["statistics"]
        if statistics["min"] > statistics["max"]:
            fail(f"{label} output {key!r} has min greater than max")
        if not statistics["min"] <= statistics["mean"] <= statistics["max"]:
            fail(f"{label} output {key!r} mean lies outside min/max")

    comparison = value.get("comparison")
    if producer["role"] == "oracle":
        if comparison is None:
            fail(f"{label} oracle lacks comparison policies")
        policies = keyed_records(comparison, f"{label} comparison policies")
        missing = sorted(set(outputs) - set(policies))
        extra = sorted(set(policies) - set(outputs))
        if missing or extra:
            fail(
                f"{label} comparison policy key mismatch: "
                f"missing={missing}, extra={extra}"
            )
        piecewise_keys = {
            "magnitude_threshold",
            "large_absolute",
            "large_relative",
        }
        for key, policy in policies.items():
            has_piecewise_fields = piecewise_keys & set(policy)
            if policy["metric"] == "piecewise-magnitude-atol-plus-rtol":
                if has_piecewise_fields != piecewise_keys:
                    fail(f"{label} comparison policy {key!r} is incomplete")
            elif has_piecewise_fields:
                fail(f"{label} comparison policy {key!r} has unexpected fields")
    elif comparison is not None:
        fail(f"{label} Mold producer must not declare oracle comparison policies")
    return value


def tolerance_issue(
    context: str,
    oracle_value: float,
    mold_value: float,
    policy: dict[str, Any],
) -> str | None:
    absolute = policy["absolute"]
    relative = policy["relative"]
    if (
        policy["metric"] == "piecewise-magnitude-atol-plus-rtol"
        and abs(oracle_value) >= policy["magnitude_threshold"]
    ):
        absolute = policy["large_absolute"]
        relative = policy["large_relative"]
    try:
        error = abs(mold_value - oracle_value)
        allowed = absolute + relative * abs(oracle_value)
    except OverflowError:
        return f"{context} values are outside the finite comparison domain"
    if error <= allowed:
        return None
    return (
        f"{context} tolerance exceeded: oracle={oracle_value!r}, mold={mold_value!r}, "
        f"abs_error={error!r}, allowed={allowed!r}, atol={absolute!r}, "
        f"rtol={relative!r}, metric={policy['metric']!r}"
    )


def compare_layer_outputs(oracle: Any, mold: Any) -> list[str]:
    """Compare one Mold producer document with its numerical oracle."""
    oracle = validate_layer_output(oracle, "oracle")
    mold = validate_layer_output(mold, "Mold")
    if oracle["producer"]["role"] != "oracle":
        fail("--oracle document producer role is not oracle")
    if mold["producer"]["role"] != "mold":
        fail("--mold document producer role is not mold")

    issues: list[str] = []
    issue_count = 0

    def report_issue(issue: str) -> None:
        nonlocal issue_count
        issue_count += 1
        if len(issues) < MAX_COMPARISON_DIAGNOSTICS:
            issues.append(issue)
    for field in ("case_id", "layer", "authority_tier"):
        if oracle[field] != mold[field]:
            report_issue(
                f"comparison {field} mismatch: oracle={oracle[field]!r}, "
                f"mold={mold[field]!r}"
            )
    oracle_input = dict(oracle["input"])
    mold_input = dict(mold["input"])
    if oracle["layer"] == "audio-vae" and mold["layer"] == "audio-vae":
        for role, evidence in (("oracle", oracle_input), ("mold", mold_input)):
            expected_checkpoint = AUDIO_VAE_CHECKPOINT_SHA256_BY_ROLE[role]
            if evidence.pop("checkpoint_sha256", None) != expected_checkpoint:
                report_issue(
                    f"audio-vae {role} checkpoint authority differs from the "
                    "reviewed role-specific checkpoint"
                )
    if oracle_input != mold_input:
        report_issue(
            f"comparison input mismatch: oracle={oracle_input!r}, mold={mold_input!r}"
        )
    if (
        oracle["adapter"]["tensor_hash_encoding"]
        != mold["adapter"]["tensor_hash_encoding"]
    ):
        report_issue(
            "tensor hash encoding mismatch: "
            f"oracle={oracle['adapter']['tensor_hash_encoding']!r}, "
            f"mold={mold['adapter']['tensor_hash_encoding']!r}"
        )
    if oracle["authorization_document_sha256"] != mold["authorization_document_sha256"]:
        report_issue("authorization document hash mismatch between producers")

    layer = oracle["layer"]
    case_id = oracle["case_id"]
    oracle_outputs = keyed_records(oracle["outputs"], "oracle outputs")
    mold_outputs = keyed_records(mold["outputs"], "Mold outputs")
    missing_outputs = sorted(set(oracle_outputs) - set(mold_outputs))
    extra_outputs = sorted(set(mold_outputs) - set(oracle_outputs))
    if missing_outputs or extra_outputs:
        report_issue(
            f"layer={layer} case={case_id} output key mismatch: "
            f"missing Mold outputs={missing_outputs}, extra Mold outputs={extra_outputs}"
        )

    policies = keyed_records(oracle["comparison"], "oracle comparison policies")
    notes: list[str] = []
    for key in sorted(set(oracle_outputs) & set(mold_outputs)):
        expected = oracle_outputs[key]
        actual = mold_outputs[key]
        policy = policies[key]
        context = f"layer={layer} case={case_id} output={key}"
        if expected["shape"] != actual["shape"]:
            report_issue(
                f"{context} shape mismatch: oracle={expected['shape']}, "
                f"mold={actual['shape']}"
            )
        if expected["dtype"] != actual["dtype"]:
            report_issue(
                f"{context} dtype mismatch: oracle={expected['dtype']!r}, "
                f"mold={actual['dtype']!r}"
            )
        if expected["content_sha256"] != actual["content_sha256"]:
            hash_message = (
                f"{context} hash mismatch: oracle={expected['content_sha256']}, "
                f"mold={actual['content_sha256']}, policy={policy['hash_policy']}"
            )
            if policy["hash_policy"] == "exact":
                report_issue(hash_message)
            else:
                notes.append(hash_message)

        for statistic in ("min", "max", "mean", "std"):
            issue = tolerance_issue(
                f"{context} statistic={statistic}",
                expected["statistics"][statistic],
                actual["statistics"][statistic],
                policy,
            )
            if issue is not None:
                report_issue(issue)

        expected_samples = sample_records(expected, f"oracle output {key!r}")
        actual_samples = sample_records(actual, f"Mold output {key!r}")
        missing_samples = set(expected_samples) - set(actual_samples)
        extra_samples = set(actual_samples) - set(expected_samples)
        if missing_samples or extra_samples:
            missing_preview = heapq.nsmallest(MAX_INDEX_DIAGNOSTICS, missing_samples)
            extra_preview = heapq.nsmallest(MAX_INDEX_DIAGNOSTICS, extra_samples)
            report_issue(
                f"{context} sample key mismatch: missing Mold indexes "
                f"(first {len(missing_preview)} of {len(missing_samples)})="
                f"{[list(index) for index in missing_preview]}, extra Mold indexes "
                f"(first {len(extra_preview)} of {len(extra_samples)})="
                f"{[list(index) for index in extra_preview]}"
            )
        for index in sorted(set(expected_samples) & set(actual_samples)):
            issue = tolerance_issue(
                f"{context} sample={list(index)}",
                expected_samples[index]["value"],
                actual_samples[index]["value"],
                policy,
            )
            if issue is not None:
                report_issue(issue)

    if issues:
        if issue_count > len(issues):
            issues.append(
                f"{issue_count - len(issues)} additional comparison issues omitted"
            )
        fail("Mold-vs-oracle comparison failed:\n- " + "\n- ".join(issues))
    return notes


def validate_manifest() -> dict[str, Any]:
    manifest = load_json(MANIFEST_PATH)
    validate_schema(manifest, MANIFEST_SCHEMA_PATH)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        fail("unexpected conformance manifest schema version")

    revisions = {source["id"]: source["revision"] for source in manifest["sources"]}
    if (
        len(manifest["sources"]) != len(EXPECTED_REVISIONS)
        or revisions != EXPECTED_REVISIONS
    ):
        fail("source revisions drifted from the reviewed H3 authority set")
    if manifest["numerical_authority"].get("oracle_runtime") != EXPECTED_ORACLE_RUNTIME:
        fail("oracle runtime identity drifted from the reviewed H3 authority set")

    excluded = set(manifest["numerical_authority"]["excluded_accelerations"])
    if (
        len(manifest["numerical_authority"]["excluded_accelerations"])
        != len(EXPECTED_EXCLUDED_ACCELERATIONS)
        or excluded != EXPECTED_EXCLUDED_ACCELERATIONS
    ):
        fail("ground-truth acceleration exclusions drifted")
    raw_aliases = manifest["numerical_authority"]["excluded_acceleration_aliases"]
    aliases = {identifier: set(values) for identifier, values in raw_aliases.items()}
    if (
        set(raw_aliases) != excluded
        or any(len(values) != len(set(values)) for values in raw_aliases.values())
        or aliases != EXPECTED_EXCLUDED_ACCELERATION_ALIASES
    ):
        fail("ground-truth acceleration exclusion aliases drifted")

    indexes = {
        item["id"]: (item["relative_path"], item["sha256"])
        for item in manifest["component_indexes"]
    }
    if (
        len(manifest["component_indexes"]) != len(EXPECTED_COMPONENT_INDEXES)
        or indexes != EXPECTED_COMPONENT_INDEXES
    ):
        fail("official component index fingerprints drifted")

    layers = {layer["id"] for layer in manifest["fixture_layers"]}
    if (
        len(manifest["fixture_layers"]) != len(EXPECTED_LAYERS)
        or layers != EXPECTED_LAYERS
    ):
        fail("fixture layer coverage is incomplete or contains an unreviewed layer")
    for layer in manifest["fixture_layers"]:
        oracle_adapter = layer.get("oracle_adapter")
        if oracle_adapter is not None:
            implementation_path = checked_repository_implementation_path(
                oracle_adapter["implementation_path"]
            )
            if oracle_adapter["implementation_sha256"] != sha256_file(
                implementation_path
            ):
                fail(f"fixture layer {layer['id']} adapter authority drifted")
            expected_command_start = f"python3 {oracle_adapter['implementation_path']} "
            if not oracle_adapter["command_prefix"].startswith(expected_command_start):
                fail(f"fixture layer {layer['id']} adapter command is not path-bound")
        component_ids = layer["required_component_indexes"]
        if len(component_ids) != len(set(component_ids)) or not set(
            component_ids
        ).issubset(EXPECTED_COMPONENT_INDEXES):
            fail(f"fixture layer {layer['id']} has unreviewed component authority")
        provenance = layer["required_provenance"]
        pinned_provenance = layer["pinned_provenance"]
        pinned_keys = [record["key"] for record in pinned_provenance]
        pinned_components = [
            component_id
            for record in pinned_provenance
            for component_id in record["component_indexes"]
        ]
        invariants = layer["role_invariant_provenance"]
        if (
            len(provenance) != len(set(provenance))
            or len(pinned_keys) != len(set(pinned_keys))
            or not set(pinned_keys).issubset(provenance)
            or any(
                len(record["component_indexes"])
                != len(set(record["component_indexes"]))
                or not set(record["component_indexes"]).issubset(component_ids)
                for record in pinned_provenance
            )
            or not set(pinned_components).issubset(EXPECTED_COMPONENT_INDEXES)
            or not set(pinned_keys).issubset(invariants)
            or len(invariants) != len(set(invariants))
            or not set(invariants).issubset(provenance)
        ):
            fail(f"fixture layer {layer['id']} has invalid provenance authority")
        input_contract = layer["input_contract"]
        expected_task = {
            "end-to-end-t2va": "t2va",
            "end-to-end-fl2va": "fl2va",
            "end-to-end-ref2va": "ref2va",
        }.get(layer["id"])
        if expected_task is None:
            if input_contract != {"kind": "identity-sha256-v1"}:
                fail(f"fixture layer {layer['id']} has an invalid input contract")
        elif input_contract != {
            "kind": "canonical-e2e-json-sha256-v1",
            "task": expected_task,
            "algorithm": "minimax-h3-flow-euler-v1",
            "guidance": "0",
            "video_shift": "12",
            "audio_shift": "3",
            "dimension_multiple": 32,
            "frame_step": 17,
            "frame_offset": 5,
            "min_frames": 124,
            # 345, not 362: #985 lowered the ceiling because the next grid
            # value is 15.083 s and the diffusers path rejects it. The fixture
            # manifest and `mold_core::minimax_h3::MAX_FRAMES` moved together;
            # this copy did not, and the contract has failed ever since.
            "max_frames": 345,
            "max_pixels": 1_069_056,
            "min_aspect_ratio": "0.25",
            "max_aspect_ratio": "4",
            "fps": 24,
            "video_scheduler_component": "official-video-scheduler-config",
            "audio_scheduler_component": "official-audio-scheduler-config",
        } or not {
            input_contract["video_scheduler_component"],
            input_contract["audio_scheduler_component"],
        }.issubset(component_ids):
            fail(f"fixture layer {layer['id']} has an invalid input contract")

    policy = manifest["capture_policy"]
    if not all(
        policy[field]
        for field in (
            "external_fixture_root_required",
            "authorization_record_required",
            "weights_in_repository_forbidden",
            "generated_media_in_repository_forbidden",
        )
    ):
        fail("real H3 capture policy is not fail closed")

    expected_synthetic = synthetic_fixture()
    checked_synthetic = load_json(SYNTHETIC_PATH)
    if checked_synthetic != expected_synthetic:
        fail("synthetic fixture does not match the deterministic generator")
    synthetic = manifest["synthetic_fixture"]
    if synthetic["relative_path"] != "synthetic-v1.json":
        fail("synthetic fixture path is not repository-relative and fixed")
    if synthetic["sha256"] != sha256_file(SYNTHETIC_PATH):
        fail("synthetic fixture hash drifted")

    expected_oracle, expected_mold = synthetic_layer_outputs()
    checked_oracle = load_json(SYNTHETIC_ORACLE_PATH)
    checked_mold = load_json(SYNTHETIC_MOLD_PATH)
    if checked_oracle != expected_oracle:
        fail("synthetic oracle layer output does not match the deterministic generator")
    if checked_mold != expected_mold:
        fail("synthetic Mold layer output does not match the deterministic generator")
    compare_layer_outputs(checked_oracle, checked_mold)

    tracked = []
    try:
        output = subprocess.run(
            ["git", "ls-files", "-z", "--", str(FIXTURE_DIR.relative_to(REPO_ROOT))],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        tracked = [REPO_ROOT / path.decode() for path in output.split(b"\0") if path]
    except (OSError, subprocess.CalledProcessError):
        tracked = [path for path in FIXTURE_DIR.rglob("*") if path.is_file()]
    forbidden = [
        path for path in tracked if path.suffix.lower() in FORBIDDEN_FIXTURE_SUFFIXES
    ]
    if forbidden:
        fail(
            f"restricted model/media artifacts are present in the fixture tree: {forbidden}"
        )
    return manifest


def canonical_external_directory(label: str, value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if not path.is_absolute():
        fail(f"{label} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        fail(f"cannot resolve {label}: {error}")
    if not resolved.is_dir():
        fail(f"{label} is not a directory")
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        pass
    else:
        fail(f"{label} must live outside the Mold repository")
    return resolved


def validate_authorization(record_path: pathlib.Path) -> dict[str, Any]:
    if not record_path.is_absolute():
        fail("authorization record path must be absolute")
    record = load_json(record_path)
    required = {
        "schema_version",
        "family",
        "decision",
        "license_revision",
        "license_sha256",
        "approved_scopes",
        "source_document_path",
        "source_document_sha256",
        "review_reference",
    }
    if not isinstance(record, dict) or set(record) != required:
        fail("authorization record fields are incomplete or unreviewed")
    if record["schema_version"] != AUTHORIZATION_SCHEMA_VERSION:
        fail("authorization record schema is unsupported")
    if record["family"] != "minimax-h3" or record["decision"] != "approved":
        fail("authorization record does not approve MiniMax H3")
    if record["license_revision"] != EXPECTED_REVISIONS["minimax-official-model"]:
        fail("authorization record covers a different H3 license revision")
    if record["license_sha256"] != EXPECTED_LICENSE_SHA256:
        fail("authorization record covers different H3 license bytes")
    scopes = record["approved_scopes"]
    if not isinstance(scopes, list) or not {
        "checkpoint-execution",
        "fixture-capture",
        "generated-output-retention",
    }.issubset(scopes):
        fail("authorization record does not cover every conformance activity")
    source_document = pathlib.Path(record["source_document_path"])
    if not source_document.is_absolute() or not source_document.is_file():
        fail("authorization source document must be an existing absolute file")
    try:
        source_document.resolve().relative_to(REPO_ROOT)
    except ValueError:
        pass
    else:
        fail("authorization source document must not be committed to the repository")
    if sha256_file(source_document) != record["source_document_sha256"]:
        fail("authorization source document hash does not match its record")
    if (
        not isinstance(record["review_reference"], str)
        or not record["review_reference"].strip()
    ):
        fail("authorization record lacks an external review reference")
    return record


def resolve_comparison_path(label: str, value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        fail(f"cannot resolve {label}: {error}")
    if not resolved.is_file():
        fail(f"{label} is not a file")
    return resolved


def compare_output_files(
    oracle_value: str,
    mold_value: str,
    fixture_root_value: str | None = None,
    authorization_value: str | None = None,
) -> list[str]:
    oracle_path = resolve_comparison_path("oracle output", oracle_value)
    mold_path = resolve_comparison_path("Mold output", mold_value)
    checked_pair = (
        oracle_path == SYNTHETIC_ORACLE_PATH.resolve()
        and mold_path == SYNTHETIC_MOLD_PATH.resolve()
    )

    authorization: dict[str, Any] | None = None
    if not checked_pair:
        if fixture_root_value is None or authorization_value is None:
            fail(
                "non-checked-in comparison requires an approved external fixture root "
                "and authorization record"
            )
        fixture_root = canonical_external_directory("fixture root", fixture_root_value)
        for label, path in (("oracle output", oracle_path), ("Mold output", mold_path)):
            try:
                path.relative_to(fixture_root)
            except ValueError:
                fail(f"{label} must be inside the approved external fixture root")
        authorization = validate_authorization(pathlib.Path(authorization_value))
    elif fixture_root_value is not None or authorization_value is not None:
        if fixture_root_value is None or authorization_value is None:
            fail("--fixture-root and --authorization-record must be supplied together")
        fail(
            "checked-in synthetic comparison does not accept real authorization evidence"
        )

    oracle = load_json(oracle_path)
    mold = load_json(mold_path)
    validate_layer_output(oracle, "oracle")
    validate_layer_output(mold, "Mold")
    if authorization is not None:
        authorization_hash = authorization["source_document_sha256"]
        for label, document in (("oracle", oracle), ("Mold", mold)):
            recorded = document["authorization_document_sha256"]
            if (
                document["authority_tier"] != "synthetic"
                and recorded != authorization_hash
            ):
                fail(f"{label} output is not bound to the authorization evidence")
    return compare_layer_outputs(oracle, mold)


def parse_sources(values: list[str]) -> dict[str, pathlib.Path]:
    sources: dict[str, pathlib.Path] = {}
    for value in values:
        identifier, separator, raw_path = value.partition("=")
        if not separator or identifier in sources:
            fail(f"--source must be a unique id=/absolute/path pair, got {value!r}")
        sources[identifier] = pathlib.Path(raw_path)
    return sources


def verify_source_checkouts(manifest: dict[str, Any], values: list[str]) -> None:
    sources = parse_sources(values)
    if set(sources) != set(EXPECTED_REVISIONS):
        missing = sorted(set(EXPECTED_REVISIONS) - set(sources))
        extra = sorted(set(sources) - set(EXPECTED_REVISIONS))
        fail(f"source checkout set differs; missing={missing}, extra={extra}")
    for identifier, expected_revision in EXPECTED_REVISIONS.items():
        root = sources[identifier]
        if not root.is_absolute() or not root.is_dir():
            fail(f"source {identifier} is not an existing absolute directory")
        try:
            observed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            fail(f"cannot read source revision for {identifier}: {error}")
        if observed != expected_revision:
            fail(f"source {identifier} is {observed}, expected {expected_revision}")

    model_root = sources["minimax-official-model"]
    for item in manifest["component_indexes"]:
        path = model_root / item["relative_path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            fail(f"component metadata drifted: {item['id']} ({path})")


def validate_bundle(
    manifest: dict[str, Any],
    fixture_root_value: str,
    bundle_value: str,
    authorization_value: str,
) -> None:
    fixture_root = canonical_external_directory("fixture root", fixture_root_value)
    authorization_path = pathlib.Path(authorization_value)
    authorization = validate_authorization(authorization_path)
    bundle_path = pathlib.Path(bundle_value).resolve(strict=True)
    try:
        bundle_path.relative_to(fixture_root)
    except ValueError:
        fail("fixture bundle must be inside the approved external fixture root")
    bundle = load_json(bundle_path)
    validate_schema(bundle, BUNDLE_SCHEMA_PATH)
    if bundle["schema_version"] != BUNDLE_SCHEMA_VERSION:
        fail("external fixture bundle schema is unsupported")
    if bundle["manifest_sha256"] != sha256_file(MANIFEST_PATH):
        fail("external fixture bundle targets a different conformance manifest")
    if (
        bundle["authorization_document_sha256"]
        != authorization["source_document_sha256"]
    ):
        fail("external fixture bundle is not bound to the authorization evidence")
    fixture_layers = {fixture["layer"] for fixture in bundle["fixtures"]}
    unknown_layers = fixture_layers - EXPECTED_LAYERS
    if unknown_layers:
        fail(
            f"external fixture bundle contains unknown layers: {sorted(unknown_layers)}"
        )
    if not bundle["fixtures"]:
        fail("external fixture bundle contains no evidence")
    for fixture in bundle["fixtures"]:
        evidence_path = (fixture_root / fixture["relative_path"]).resolve(strict=True)
        try:
            evidence_path.relative_to(fixture_root)
        except ValueError:
            fail(f"fixture evidence escapes its external root: {fixture['id']}")
        if (
            not evidence_path.is_file()
            or sha256_file(evidence_path) != fixture["sha256"]
        ):
            fail(f"fixture evidence hash mismatch: {fixture['id']}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the weight-free H3 contract, pinned checkouts, or a separately "
            "authorized external fixture bundle."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "check-contract", help="validate checked-in pins and synthetic fixtures"
    )

    verify = subparsers.add_parser(
        "verify-sources", help="verify every pinned local source checkout"
    )
    verify.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="ID=/ABSOLUTE/PATH",
        help="repeat once for every source id in the conformance manifest",
    )

    bundle = subparsers.add_parser(
        "validate-bundle", help="validate authorized evidence outside the repository"
    )
    bundle.add_argument("--fixture-root", required=True)
    bundle.add_argument("--bundle", required=True)
    bundle.add_argument("--authorization-record", required=True)

    compare = subparsers.add_parser(
        "compare", help="compare one Mold per-layer output with its oracle"
    )
    compare.add_argument("--oracle", required=True, help="oracle layer-output JSON")
    compare.add_argument("--mold", required=True, help="Mold layer-output JSON")
    compare.add_argument(
        "--fixture-root",
        help="approved external root; required except for the checked synthetic pair",
    )
    compare.add_argument(
        "--authorization-record",
        help="external authorization record; required with --fixture-root",
    )

    subparsers.add_parser(
        "print-synthetic", help="print the deterministic synthetic fixture"
    )
    synthetic_output = subparsers.add_parser(
        "print-synthetic-output",
        help="print one deterministic synthetic producer/adaptor document",
    )
    synthetic_output.add_argument("--role", choices=("oracle", "mold"), required=True)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        if args.command == "print-synthetic":
            sys.stdout.buffer.write(canonical_bytes(synthetic_fixture()))
            return 0
        if args.command == "print-synthetic-output":
            oracle, mold = synthetic_layer_outputs()
            sys.stdout.buffer.write(
                canonical_bytes(oracle if args.role == "oracle" else mold)
            )
            return 0
        manifest = validate_manifest()
        if args.command == "verify-sources":
            verify_source_checkouts(manifest, args.source)
        elif args.command == "validate-bundle":
            validate_bundle(
                manifest,
                args.fixture_root,
                args.bundle,
                args.authorization_record,
            )
        elif args.command == "compare":
            notes = compare_output_files(
                args.oracle,
                args.mold,
                args.fixture_root,
                args.authorization_record,
            )
            for note in notes:
                print(f"MiniMax H3 conformance note: {note}")
        print(f"MiniMax H3 conformance {args.command} passed")
        return 0
    except (ConformanceFailure, OSError, OverflowError) as error:
        print(f"invalid MiniMax H3 conformance evidence: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
