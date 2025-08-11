output "vm_name" {
  description = "The name of the deployed Compute Engine VM."
  value       = google_compute_instance.python_app_vm.name
}

output "vm_external_ip" {
  description = "The external IP address of the deployed Compute Engine VM."
  value       = google_compute_instance.python_app_vm.network_interface[0].access_config[0].nat_ip
}