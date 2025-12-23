library(MASS)

# set up dimensions
args <- commandArgs(trailingOnly=TRUE)
start <- as.integer(args[1])
start <- start * 8
nsim <- 8

for (i in 1:nsim) {
  print(i)
  idx <- start + i

  # Run Python under the ltcinf conda env so torch loads correctly.
  # Requires that "conda" is available in PATH (run_all.sh sources conda.sh).
  conda_exe <- Sys.getenv("CONDA_EXE", "conda")
  py_env <- Sys.getenv("PYTHON_CONDA_ENV", "ltcinf")
  system2(conda_exe, c("run", "-n", py_env, "python", "surrogate.py", as.character(idx)))
}
