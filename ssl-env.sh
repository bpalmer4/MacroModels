#!/usr/bin/env bash
# Environment for the run-*.sh scripts, which all source this file. Two
# unrelated macOS workarounds live here: the TLS trust store, and PyTensor's
# C compilation.
#
# --- 1. TLS: point Python's trust store at certifi ---
#
# Python verifies certificates against OpenSSL's default store
# (/private/etc/ssl/cert.pem on macOS). That store does not carry the "Sectigo
# Public Server Authentication Root R46" root that rba.gov.au now chains to, so
# RBA downloads fail with:
#
#   urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]
#   certificate verify failed: self-signed certificate in certificate chain>
#
# curl is unaffected because it uses the macOS keychain. The venv's certifi
# bundle does carry the root, so point Python at that instead.
#
# An SSL_CERT_FILE already set in the environment wins; if certifi cannot be
# resolved this is a no-op rather than an error.

if [[ -z "${SSL_CERT_FILE:-}" ]]; then
    if _certifi_pem="$(uv run --quiet python -c 'import certifi; print(certifi.where())' 2>/dev/null)" \
        && [[ -f "${_certifi_pem}" ]]; then
        export SSL_CERT_FILE="${_certifi_pem}"
        export REQUESTS_CA_BUNDLE="${_certifi_pem}"
    fi
    unset _certifi_pem
fi

# --- 2. PyTensor: disable C compilation ---
#
# PyTensor's `GCC_compiler.compile_args` appends "-ld64" on any macOS whose
# major version is >= 15, meaning to select Apple's classic linker. clang reads
# a bare "-ld64" as "link library d64", and Xcode 27 removed the classic linker,
# so every C compile dies with:
#
#   ld: library 'd64' not found
#
# The same code is in the latest PyTensor (3.3.2 as at 2026-09-15), so this is
# not fixed by upgrading. PyTensor's own comment on the version test reads
# "This might be incorrect."
#
# Setting cxx empty is PyTensor's supported way to fall back to the Python
# implementations. It costs nothing here because every model these scripts run
# samples with nuts_sampler="numpyro", so the work is done by JAX and the C
# backend is incidental: measured at 35.9s against 42.2s on rstar_bonds.
# Results are not bit-identical (float ordering moves the RNG stream) but differ
# by <= 0.024 on the state paths, which is Monte Carlo noise.
#
# Remove this block once PyTensor ships a fix. A PYTENSOR_FLAGS already set in
# the environment wins, so a sweep can still override it.
if [[ -z "${PYTENSOR_FLAGS:-}" ]]; then
    export PYTENSOR_FLAGS="cxx="
fi
