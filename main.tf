# Enables the Compute Engine API for your project
resource "google_project_service" "compute_api" {
  service                    = "compute.googleapis.com"
  disable_on_destroy         = false
  disable_dependent_services = true
}

# Defines the Compute Engine Virtual Machine
resource "google_compute_instance" "python_app_vm" {
  project      = var.project_id
  name         = "live-trader-vm" # Changed from python-app-server
  machine_type = "e2-small"
  zone         = "${var.region}-a"

  # Configures the boot disk with a Debian 11 OS image
  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
    }
  }

  # Configures the network to allow internet access
  network_interface {
    network = "default"
    access_config {}
  }

  # Startup script to install dependencies and run your Python application
  metadata_startup_script = <<-EOT
    #!/bin/bash
    sleep 10
    exec > >(tee /var/log/startup-script.log|logger -t startup-script -s 2>/dev/console) 2>&1

    echo "Updating system and installing dependencies..."
    apt-get update
    apt-get install -y python3-pip git wget unzip build-essential git-lfs

    echo "Downloading and installing TA-Lib C library..."
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr
    make
    make install
    cd ..

    echo "Cloning application from Git repository..."
    git clone "${var.git_repo_url}" /opt/app
    cd /opt/app

    # --- ADD LFS PULL STEPS ---
    echo "Setting up Git LFS and pulling large files..."
    git lfs install
    git lfs pull
    # --- END LFS PULL STEPS ---

    echo "Installing Python requirements..."
    cd src
    pip3 install -r requirements.txt

    echo "Starting the main Python script from the src directory..."
    nohup python3 main.py &
  EOT

  # Defines the VM's permissions
  service_account {
    scopes = ["cloud-platform"]
  }

  depends_on = [google_project_service.compute_api]
}