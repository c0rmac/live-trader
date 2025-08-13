#!/bin/bash

# A script to securely add exchange API keys to Google Cloud Secret Manager.

# First, ensure the Secret Manager API is enabled.
echo "Ensuring Secret Manager API is enabled..."
gcloud services enable secretmanager.googleapis.com --quiet

while true; do
  # --- Prompt for user input ---
  read -p "Enter the exchange name (e.g., binance, coinbase): " exchange_name

  # Exit if the user just presses Enter
  if [ -z "$exchange_name" ]; then
    echo "No exchange name entered. Exiting."
    break
  fi

  # Use -s flag to hide the input for keys
  read -sp "Enter the API Key for $exchange_name: " api_key
  echo "" # Newline after hidden input
  read -sp "Enter the Secret Key for $exchange_name: " secret_key
  echo "" # Newline after hidden input

  # --- Validate input ---
  if [ -z "$api_key" ] || [ -z "$secret_key" ]; then
    echo "API Key or Secret Key cannot be empty. Please try again."
    continue
  fi

  # --- Format names and run gcloud commands ---
  # Convert exchange name to uppercase for the secret name
  base_name=$(echo "$exchange_name" | tr 'a-z' 'A-Z')
  api_key_name="${base_name}_API_KEY"
  secret_key_name="${base_name}_SECRET_KEY"

  echo "--------------------------------------------------"
  echo "Preparing to add the following secrets:"
  echo "1. Name: $api_key_name"
  echo "2. Name: $secret_key_name"
  echo "--------------------------------------------------"

  # Create the API Key secret
  echo "Creating secret: $api_key_name..."
  gcloud secrets create "$api_key_name" --replication-policy="automatic" --quiet || {
    echo "Secret '$api_key_name' may already exist. Attempting to add a new version."
  }
  echo -n "$api_key" | gcloud secrets versions add "$api_key_name" --data-file=- --quiet

  # Create the Secret Key secret
  echo "Creating secret: $secret_key_name..."
  gcloud secrets create "$secret_key_name" --replication-policy="automatic" --quiet || {
    echo "Secret '$secret_key_name' may already exist. Attempting to add a new version."
  }
  echo -n "$secret_key" | gcloud secrets versions add "$secret_key_name" --data-file=- --quiet

  echo ""
  echo "✅ Successfully added secrets for $exchange_name."
  echo ""

  # --- Ask to continue ---
  read -p "Do you want to add another key? (yes/no): " continue_choice
  if [[ ! "$continue_choice" =~ ^[Yy]es$ ]]; then
    break
  fi
done

echo "Script finished."