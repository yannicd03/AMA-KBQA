#!/usr/bin/env bash
#
# Render the deployment config files from their templates.
#
#     ./deploy/hetzner/render-deploy-config.sh
#
# Reads deploy/hetzner/deploy.env (gitignored, host-specific) and writes:
#
#     deploy/hetzner/cloudflared-amakbqa.yml       (gitignored)
#     deploy/hetzner/cloudflared-amakbqa.service   (gitignored)
#
# Both outputs are gitignored on purpose: they contain the origin host, the
# tunnel id and the deploy user, none of which belong in a public repository.
# Copy the rendered files to the server with the deployment SOP
# (.agent/SOP/hetzner_demo_deployment.md).

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="${here}/deploy.env"

if [[ ! -f "${env_file}" ]]; then
  echo "error: ${env_file} not found." >&2
  echo "       cp deploy/hetzner/deploy.env.example deploy/hetzner/deploy.env" >&2
  echo "       then fill in the values for your host." >&2
  exit 1
fi

if ! command -v envsubst >/dev/null 2>&1; then
  echo "error: envsubst not found (install the gettext package)." >&2
  exit 1
fi

# shellcheck source=/dev/null
set -a
source "${env_file}"
set +a

required=(
  DEPLOY_USER
  CLOUDFLARE_TUNNEL_ID
  CLOUDFLARE_CREDENTIALS_DIR
  DEMO_HOSTNAME
  DEMO_STAGING_HOSTNAME
  DEMO_PORT
  DEMO_STAGING_PORT
)

missing=()
for var in "${required[@]}"; do
  [[ -n "${!var:-}" ]] || missing+=("${var}")
done

if (( ${#missing[@]} > 0 )); then
  echo "error: deploy.env is missing required values: ${missing[*]}" >&2
  exit 1
fi

if [[ "${CLOUDFLARE_TUNNEL_ID}" == "00000000-0000-0000-0000-000000000000" ]]; then
  echo "error: CLOUDFLARE_TUNNEL_ID is still the placeholder value." >&2
  exit 1
fi

# Substitute only the variables we own, so anything else surviving in a
# template is left alone rather than silently blanked.
substitute='${DEPLOY_USER} ${CLOUDFLARE_TUNNEL_ID} ${CLOUDFLARE_CREDENTIALS_DIR} ${DEMO_HOSTNAME} ${DEMO_STAGING_HOSTNAME} ${DEMO_PORT} ${DEMO_STAGING_PORT}'

for name in cloudflared-amakbqa.yml cloudflared-amakbqa.service; do
  template="${here}/${name}.template"
  output="${here}/${name}"
  if [[ ! -f "${template}" ]]; then
    echo "error: missing template ${template}" >&2
    exit 1
  fi
  envsubst "${substitute}" < "${template}" > "${output}"
  echo "rendered ${output#"${PWD}"/}"
done

echo
echo "Both outputs are gitignored. Next steps are in"
echo ".agent/SOP/hetzner_demo_deployment.md."
