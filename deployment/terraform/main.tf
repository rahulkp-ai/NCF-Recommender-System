# Placeholder — intentionally minimal until this project actually needs
# provider-managed infrastructure (see deployment/terraform/README.md).

terraform {
  required_version = ">= 1.7"
  required_providers {
    # Example: uncomment and configure when moving beyond free-tier PaaS.
    # render = {
    #   source  = "render-oss/render"
    #   version = "~> 1.0"
    # }
  }
}
