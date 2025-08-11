variable "project_id" {
  description = "The Google Cloud project ID where resources will be deployed."
  type        = string
  default     = "live-trader-468613"
}

variable "region" {
  description = "The GCP region to deploy the resources in."
  type        = string
  default     = "us-central1"
}

variable "git_repo_url" {
  description = "The URL of the Git repository containing your Python application."
  type        = string
  default     = "https://github.com/c0rmac/live-trader.git"
  # Example: "https://github.com/your-username/your-repo.git"
}