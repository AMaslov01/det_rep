#!/usr/bin/env bash
set -euo pipefail

project_id="${1:-project-fe2f39ea-f456-4e8f-8e8}"
region="europe-west4"
service_name="gemini-openai-gateway"
service_account_name="gemini-gateway-runtime"
secret_id="gemini-gateway-keys"
budget_name="Gemini gateway monthly EUR 50"
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
release="${GATEWAY_RELEASE:-$(git -C "$root_dir" rev-parse --short HEAD)}"
if [[ -n "$(git -C "$root_dir" status --porcelain)" ]]; then
  release="${release}-dirty"
fi

gcloud projects describe "$project_id" --format='value(projectId)' >/dev/null
gcloud services enable \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  billingbudgets.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com \
  --project="$project_id"

project_number="$(gcloud projects describe "$project_id" --format='value(projectNumber)')"
build_service_account="${project_number}-compute@developer.gserviceaccount.com"
# New projects use the Compute Engine default account for source builds. It needs
# Cloud Run Builder, but no runtime Vertex or secret permissions.
gcloud projects add-iam-policy-binding "$project_id" \
  --member="serviceAccount:${build_service_account}" \
  --role="roles/run.builder" \
  --quiet

service_account_email="${service_account_name}@${project_id}.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe "$service_account_email" --project="$project_id" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$service_account_name" --project="$project_id" --display-name="Gemini gateway runtime"
fi
gcloud projects add-iam-policy-binding "$project_id" \
  --member="serviceAccount:${service_account_email}" \
  --role="roles/aiplatform.user" \
  --quiet

if ! gcloud secrets describe "$secret_id" --project="$project_id" >/dev/null 2>&1; then
  gcloud secrets create "$secret_id" --project="$project_id" --replication-policy="automatic"
  printf '%s' '{"version":1,"keys":[]}' | gcloud secrets versions add "$secret_id" --project="$project_id" --data-file=-
fi
gcloud secrets add-iam-policy-binding "$secret_id" \
  --project="$project_id" \
  --member="serviceAccount:${service_account_email}" \
  --role="roles/secretmanager.secretAccessor" \
  --quiet

gcloud run deploy "$service_name" \
  --project="$project_id" \
  --region="$region" \
  --source="$root_dir" \
  --service-account="$service_account_email" \
  --allow-unauthenticated \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=${project_id},GATEWAY_KEY_SECRET=${secret_id},GATEWAY_KEY_CACHE_TTL_SECONDS=60,GATEWAY_RELEASE=${release}" \
  --timeout=600 \
  --quiet

billing_account="$(gcloud billing projects describe "$project_id" --format='value(billingAccountName)' | sed 's#billingAccounts/##')"
if [[ -z "$billing_account" ]]; then
  echo "No billing account is linked to ${project_id}; Cloud Run was deployed but no budget was created." >&2
  exit 1
fi
existing_budget="$(gcloud billing budgets list --billing-account="$billing_account" --format='value(displayName,name)' | awk -F '\t' -v expected="$budget_name" '$1 == expected { print $2; exit }')"
if [[ -z "$existing_budget" ]]; then
  gcloud billing budgets create \
    --billing-account="$billing_account" \
    --display-name="$budget_name" \
    --budget-amount=50EUR \
    --calendar-period=month \
    --filter-projects="projects/${project_id}" \
    --credit-types-treatment=include-all-credits \
    --threshold-rule=percent=0.50 \
    --threshold-rule=percent=0.90 \
    --threshold-rule=percent=1.00 \
    --quiet
fi

service_url="$(gcloud run services describe "$service_name" --project="$project_id" --region="$region" --format='value(status.url)')"
printf 'Gateway URL: %s\n' "$service_url"
printf 'Create a recipient key: python3.12 -m gemini_gateway.manage_keys --project %s create <recipient-id>\n' "$project_id"
