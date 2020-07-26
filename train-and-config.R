options(repos="https://cran.revolutionanalytics.com/")

# install the latest version from CRAN
install.packages(pkgs="azuremlsdk", repo="https://ftp.osuosl.org/pub/cran/")
install.packages(pkgs="optparse", repo="https://ftp.osuosl.org/pub/cran/")

library_dependencies <- c(
    "optparse", 
    "azuremlsdk"
)

library_dependencies;
length(library_dependencies)

for (i in 1:length(library_dependencies)) {
    dependency <- library_dependencies[i]
    myInstalledPackages <- library()$results[,1]


    if (!(dependency %in% myInstalledPackages)) {
      paste("Installing: ", dependency)
      install.packages(dependency)
    }
}


azuremlsdk::install_azureml(envname = 'r-reticulate')
library(azuremlsdk)

# Settings
experiment_name <- "Iris-Class-on-R"
target_path <- "training-data"
training_data <- "IrisDataset.csv"
job_script <- "Classify.R"
training_target_name <- "train-vm"
deployment_target_name <- "deploy-vm"

# Load workspace
ws <- load_workspace_from_config()

# Create experiment
exp <- experiment(ws, experiment_name)

# Create compute target
compute_target <- get_compute(ws, cluster_name = training_target_name)
if (is.null(compute_target)) {
  vm_size <- "STANDARD_D2_V2" 
  compute_target <- create_aml_compute(workspace = ws,
                                       cluster_name = training_target_name,
                                       vm_size = vm_size,
                                       min_nodes = 1,
                                       max_nodes = 1)
}

wait_for_provisioning_completion(compute_target)

# Upload data to Datastore
datastore <- get_default_datastore(ws)

upload_files_to_datastore(datastore,
                          list(training_data),
                          target_path = target_path,
                          overwrite = TRUE)

# Create an Estimator
est <- estimator(source_directory = ".",
                 entry_script = job_script,
                 script_params = list("--data_folder" = datastore$path(target_path)),
                 compute_target = compute_target
                 )

# Submit the Job
run <- submit_experiment(exp, est)
view_run_details(run)
wait_for_run_completion(run, show_output = TRUE)

# Fetch Results
metrics <- get_run_metrics(run)
metrics

# Get trained model
download_files_from_run(run, prefix="outputs/")
classification_model <- readRDS("outputs/model.rds")
summary(classification_model)

print("Training complete!")
