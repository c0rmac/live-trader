# Enables the Compute Engine API for your project
resource "google_project_service" "compute_api" {
  service                    = "compute.googleapis.com"
  disable_on_destroy         = false
  disable_dependent_services = true
}

# Defines the Compute Engine Virtual Machine
resource "google_compute_instance" "python_app_vm" {
  project      = var.project_id
  name         = "python-app-server"
  machine_type = "e2-small" # 0.5 vCPU, 2GB RAM
  zone         = "${var.region}-a"

  # Configures the boot disk with a Debian 11 OS image
  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
    }
  }

  # Configures the network to allow internet access and assign an external IP
  network_interface {
    network = "default"
    access_config {}
  }

  # This shell script runs once when the VM starts for the first time
  metadata_startup_script = <<-EOT
    #!/bin/bash
    # Wait for the system to settle
    sleep 10

    # Log all output from this script for debugging
    exec > >(tee /var/log/startup-script.log|logger -t startup-script -s 2>/dev/console) 2>&1

    echo "Updating system and installing dependencies..."
    apt-get update
    apt-get install -y python3-pip git

    echo "Cloning application from Git repository..."
    git clone "${var.git_repo_url}" /opt/app
    cd /opt/app

    echo "Installing Python requirements..."
    pip3 install -r requirements.txt

    echo "Starting the main Python script in the background..."
    # 'nohup' keeps the script running even if the session ends
    # '&' runs the command in the background
    nohup python3 main.py &
  EOT

  # The service account defines the VM's permissions within GCP
  service_account {
    scopes = ["cloud-platform"]
  }

  # Ensures the Compute API is enabled before trying to create a VM
  depends_on = [google_project_service.compute_api]
}