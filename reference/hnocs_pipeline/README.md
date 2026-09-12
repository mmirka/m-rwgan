# HNOCS data-pipeline scripts (reference only — not runnable in this repo)

These 8 scripts document how the NoC training data was originally simulated.
They are not runnable here (they need HNOCS + Orion3.0 installed), but they
show the data provenance behind `data/reference/experimental_data/` and the
reward-network training data.

They implement the glue between M-RWGAN and **HNOCS**, an OMNeT++-based
NoC simulator (`HNOCS_README.md`), plus the **Orion3.0** power/area model
(non-commercial UC San Diego academic license) — neither of which is
vendored here. None of these scripts will run without both installed.

| Script | What it does |
|---|---|
| `routers.py` | Defines router-architecture codes (the "Big/Medium/Small" = 112/104/102 classes used throughout the project) and per-type NED-generation helpers. |
| `ned_writer.py` | Writes OMNeT++ `.ned` topology files for a given router-class assignment (`X`) so HNOCS can simulate it. |
| `dataset_creator_dyn.py` | Randomly samples router-class assignments (`X` matrices) for a fixed mesh and drives dataset generation end-to-end. |
| `save_dataset.py` | Orchestrates: write NED file → invoke HNOCS/OMNeT++ → collect stats → append to the pickled `Data_AX` dataset used everywhere else in this project. |
| `latency_FIR_extract.py` | Parses HNOCS `.sca`/`.stat` simulator output into per-injection-rate latency curves. |
| `power_FIR_extract.py` | Extracts power figures from simulator output via the Orion3.0 model. |
| `jouls_FIR_compute.py` | Computes energy (joules) from the extracted power curves. |
| `plot_results.py` | Ad-hoc plotting helper for inspecting simulator output during data collection (not a paper/thesis figure script). |

If you need to regenerate training data from scratch, start with HNOCS
itself (see `HNOCS_README.md`), then adapt `save_dataset.py` as the
top-level driver — but this is a multi-day undertaking (simulator setup +
thousands of simulations).

**See `REPRODUCTION.md`** in this directory for concrete build/run steps
and exactly how these 8 scripts chain together with a HNOCS+Orion3.0
install.
