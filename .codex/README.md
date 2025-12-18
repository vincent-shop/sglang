# Codex CLI Setup

## 1. Config Location

```bash
# Option A: Use default location
mkdir -p ~/.codex
cp .codex/config.toml ~/.codex/

# Option B: Set custom location
export CODEX_HOME=/path/to/.codex
```

## 2. Set API Key

```bash
export SHOPIFY_PROXY_API_KEY="your-key-here"
unset OPENAI_API_KEY OPENAI_BASE_URL  # avoid conflicts
```

## 3. Run

```bash
codex
```

## Config Reference

```toml
model_provider = "shopify_openai_proxy"
model = "gpt-5.1-codex"

[model_providers.shopify_openai_proxy]
name = "Shopify OpenAI Proxy"
base_url = "https://proxy.shopify.ai/v1"
env_key = "SHOPIFY_PROXY_API_KEY"
wire_api = "responses"
```

