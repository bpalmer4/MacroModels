#!/usr/bin/env bash
# Point Python's TLS trust store at certifi. Sourced by the run-*.sh scripts.
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
