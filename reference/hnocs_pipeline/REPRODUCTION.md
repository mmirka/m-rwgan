# Reproducing the HNOCS + Orion3.0 data-generation pipeline

This is a concrete, from-scratch guide to running the simulator side of the
project: turning a router-class assignment (an `X` matrix, as used
throughout `src/mrwgan/`) into a simulated `Data_AX` sample. It complements
`README.md` in this directory (which documents the 8 glue scripts in
isolation) by documenting the piece those scripts assume already exists:
the simulator itself.

**Status:** everything below was checked against a real HNOCS + Orion3.0
tree — build recipe, `.ned` module definitions, a heterogeneous
`omnetpp.ini`, and real simulator output files.

This is genuinely a multi-day undertaking if you're doing a full 10k-sample
dataset per traffic pattern — the paper itself puts dataset generation +
Reward training at "approx. 3 days" (DATE2022, `evaluation.tex`). Running
*one* simulation end-to-end, once the simulator is built, is minutes of
work; the cost is in scale.

## 0. What HNOCS and Orion3.0 are

- **HNOCS** ("HNoCS: Modular Open-Source Simulator for Heterogeneous
  NoCs", Ben-Itzhak et al., SAMOS XII) is a NoC simulation framework built
  as an OMNeT++/OMNEST project. It provides parametrized router/port/core
  modules (wormhole switching, round-robin or winner-takes-all
  arbitration) that support per-link, per-VC heterogeneity — the feature
  this project's router-class assignments depend on.
- **Orion3.0** ("ORION 3.0: A Power and Area Simulator for On-Chip
  Networks", Kahng et al., UC San Diego) is a power/area estimation
  library, linked into HNOCS to turn simulated activity (toggle rates) and
  router microarchitecture parameters into power (mW) and area (µm²)
  figures.
- Both are **not vendored in this repo**. Orion3.0 in particular carries a
  non-commercial UC San Diego academic license (see `ORION3_0/README` in
  the upstream tree) — copying it into a git repo that could be
  shared/published would violate that license. This matches the framing in
  `reference/hnocs_pipeline/README.md`.

## 1. Where to get them

HNOCS is distributed via `http://hnocs.eew.technion.ac.il/` (per its
`README.md`); Orion3.0 is available from UC San Diego's VLSI CAD lab (see
the license contact in `ORION3_0/README`). The rest of this guide refers to
files in a checked-out, built HNOCS + Orion3.0 tree. A working tree of that
kind contains, among other things:

- `Docker.memo` — the build/run recipe (reproduced verbatim below).
- `GETTING STARTED`, `README.md`, `NEWS` — upstream HNOCS docs.
- `examples/sync/8x8/`, `examples/async/8x8/`, `.../4x4/`,
  `.../uniform_eval/` — stock demo configs.
- `src/` — the OMNeT++ NED + C++ source (cores, routers, topologies).
- `ORION3_0/` — the power/area library source, with `ORION3_0/README`,
  `README.ORION1.0`, `README.ORION2.0`. Build it (section 2) to produce
  `libOrion3.a` / `libOrion3.so`.
- `simulations/simu<N>/` — per-thread run directories from dataset
  generation. The heterogeneous `omnetpp.ini` files here are the reference
  for what a real heterogeneous config looks like — the stock `examples/`
  configs are homogeneous and don't use the machinery this project needs.

Exact file layouts may differ by HNOCS/Orion3.0 version, but the general
shape of the pipeline below should still apply.

## 2. Build

`Docker.memo` (24 lines, quoted verbatim):

```
1) Install Docker

$>sudo apt-get install docker.io
$>sudo chmod 666 /var/run/docker.sock

2) Go to HNOCS main directory

3) Run opp_docker_shell

$>./opp_docker_shell

4) Go to src directory

$> cd src

5) Compile the project

$> make

(May be required to add Orion libs path to $LD_LIBRARY_PATH environment varaible)

6) Go to the simulation directory (where is the file .ini)
7) Run the simulation
$> opp_run -r 0 -m -u Cmdenv -c General -n ../..:../../../src -l ../../../src/hnocs omnetpp.ini
```

`opp_docker_shell` (the script `./opp_docker_shell` runs) is:

```sh
#!/bin/sh
docker run --rm -it -v "$(pwd):/root/models" -u "$(id -u):$(id -g)" omnetpp/omnetpp:u18.04-5.6.2
```

So concretely: it drops you into a container built from the
`omnetpp/omnetpp:u18.04-5.6.2` image (OMNeT++ 5.6.2 on Ubuntu 18.04) with
the HNOCS directory bind-mounted at `/root/models`. `make` inside `src/`
is a standard OMNeT++ `opp_makemake`-generated Makefile (see
`src/Makefile`'s header comment — it was generated with
`opp_makemake --nolink -f --deep -O out -I. ... -L../ORION3_0/libs
-Lout/$(CONFIGNAME)/src -d src -X.`), i.e. it links against
`../ORION3_0/libs/libOrion3.{a,so}`.

**Gap in Docker.memo:** it does not mention building Orion3.0 itself —
only linking against `../ORION3_0/libs`. If that directory already contains
prebuilt `libOrion3.a` / `libOrion3.so`, `Docker.memo`'s `cd src && make`
works as-is against them. To build from a clean Orion3.0 checkout instead,
build it first:

```sh
cd ORION3_0
make          # builds orion_router (standalone CLI) from the Makefile's `all` target
make libs     # builds libOrion3.a / libOrion3.so and copies them into libs/
```

(`ORION3_0/Makefile`'s `libs` target: `libOrion3.a` and `libOrion3.so` are
linked from `orion_interface.o router.o router_power.o router_area.o
SWVC.o CLKCTRL.o NONPR.o INBUF.o OUTBUF.o XBAR.o`, then copied to `libs/`.)

Orion3.0's technology parameters are compile-time, set in
`ORION3_0/includes/SIM_port.h` (and `ORION3_0/SIM_port.h`, kept in sync)
*before* building the library — they are not exposed as OMNeT++ `.ini`
parameters. The values that match the paper/thesis are:

```c
#define PARM_TECH_POINT       45        // 45nm
#define PARM_TRANSISTOR_TYPE  NVT
#define PARM_Vdd              1.0       // 1.0V
#define PARM_Freq             0.650e9   // 650MHz
```

(DATE2022 `evaluation.tex`: "45nm manufacturing, a Vdd of 1.0V, and a
650MHz frequency"; thesis `Contribution3.tex` Table `orion_param`: same,
plus "Crossbar type: Matrix"; the reference `SIM_port.h` sets
`PARM_crossbar_model TRISTATE_CROSSBAR`, which is what to match if you want
the reference build.) If you rebuild
Orion3.0 from a clean checkout, copy these `#define`s into your
`SIM_port.h` first.

After `make` in `src/`, per Docker.memo's parenthetical, you may need
`LD_LIBRARY_PATH` to include Orion3.0's lib directory at run time:

```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/../ORION3_0/libs
```

## 3. Run one simulation from the stock 8x8 example

`examples/sync/8x8/` is the closest stock config to this project's
headline 8x8 mesh, and is a good smoke test that the build works before
touching anything heterogeneous. Its `run` script is one line:

```sh
#!/bin/sh
../../../src/run_nocs $*
```

Its `omnetpp.ini` (quoted, key lines):

```ini
[General]
network = hnocs.topologies.Mesh
**.routerType = "hnocs.routers.hier.Router"
**.coreType   = "hnocs.cores.NI"
**.sourceType = "hnocs.cores.sources.PktFifoSrc"
**.sinkType   = "hnocs.cores.sinks.InfiniteBWMultiVCSink"
**.portType   = "hnocs.routers.hier.Port"
**.inPortType = "hnocs.routers.hier.inPort.InPortSync"
**.OPCalcType = "hnocs.routers.hier.opCalc.static.XYOPCalc"
**.VCCalcType = "hnocs.routers.hier.vcCalc.free.FLUVCCalc"
**.schedType  = "hnocs.routers.hier.sched.wormhole.SchedSync"
sim-time-limit = 2ms
**.numVCs = 2
**.rows = 8
**.columns = 8
**.heterogeneous = false
**.tClk = 2ns
```

From `examples/sync/8x8/`, run it directly with the command Docker.memo
gives (adjusted for the relative depth from this directory, matching
`GETTING STARTED`'s directory convention):

```sh
cd examples/sync/8x8
opp_run -r 0 -m -u Cmdenv -c General -n ../../..:../../../src -l ../../../src/hnocs omnetpp.ini
```

This is a **homogeneous** 8x8 mesh (`heterogeneous = false`, uniform
`dstId` traffic via `(id + intuniform(1, 63)) % 64`, no Orion output
wired up in this example's `.ini`) — it validates the build and basic
mesh wiring, but it is *not* the mechanism this project actually uses for
per-router-class heterogeneity or for Orion power/area extraction. That
mechanism is described next.

## 4. What this project's real heterogeneous setup actually is

The repo's `ned_writer.py` (see its `write_buffers()`, which emits
`router[i].bufN = ...; router[i].bufS = ...; ...`) targets a specific NED
network defined in `src/topologies/myMesh_template.ned`:

```ned
network myMesh_RouterNSEWL_template
{
    parameters:
        string routerType = "RouterNSEWL";
        ...
        // bufs :
    submodules:
        router[columns*rows]: RouterNSEWL {
            parameters:
                numPorts = 5;
                id = index;
                bufN = default(1); bufS = default(1); bufE = default(1);
                bufW = default(1); bufL = default(1);
        }
        core[columns*rows]: <coreType> like NI_Ifc { ... }
    connections allowunconnected:
        // standard 2D mesh wiring, ports 0=N,1=W,2=S,3=E,4=core
}
```

`RouterNSEWL` (defined in `src/routers/hier/Router.ned`, alongside
`Router`/`Router2`/`Router3`) is the module that actually supports
**per-port, per-router buffer sizes** — its 5 integer parameters
(`bufN/bufS/bufE/bufW/bufL`) are exactly what `ned_writer.write_buffers()`
overrides per router. It hardcodes its ports to
`hnocs.routers.hier.Port_bufSize` (also in `Router.ned`/`Port.ned`), which
is the port module carrying a `bufSize` parameter. Both `RouterNSEWL` and
`Port_bufSize` are already present in the vendored `src/` — no source
modification is needed to get heterogeneous per-router buffers.

`ned_writer.py`'s pipeline (`write()`/`write_simuID()` etc.):
1. Copies `src/topologies/myMesh_template.ned` to a working file
   (`myMesh_new.ned` or `myMesh_simu<N>.ned`).
2. `write_dimension()` fills in `rows`/`columns`.
3. `write_buffers()` fills in each router's 5 buffer sizes, computed by
   `routers.X2buffers(X)` from the router-class codes in `X` (e.g. 112 ->
   `HOMO` type, buffer size 12 on all 5 ports — see `routers.py`'s
   `full_DIC_R`/`create_Router_HOMO`).
4. `write_NetworkName()` renames the network to `my_Mesh_toTest` or
   `my_Mesh_simu<N>`.

The `.ini` file must then reference that generated network name. A
reference `simulations/simu1/omnetpp.ini` from an actual past run, for a
3x4 mesh — the reference `.ini` files in this tree are 3x4 and 4x4, so the
8x8 configuration described in sections 4 to 6 is built by analogy from them —
shows the working pattern, annotated with the author's own `# MMirka` comments:

```ini
[General]
network = hnocs.topologies.my_Mesh_simu1
**.routerType = "hnocs.routers.hier.Router2"  # MMirka - unused with myMesh*
**.portType   = "hnocs.routers.hier.Port_bufSize" # MMirka _ prev: hnocs.routers.hier.Port
**.powerFilePath = "../HNOCS/simulations/simu1/results/power"
**.areaFilePath  = "../HNOCS/simulations/simu1/results/area"
**.IsMeshNetwork = true
**.probabilisticBench = xmldoc("../HNOCS/simulations/simu1/edit_synthetic_traffic.xml")
**.useRoutingTable = false
sim-time-limit = 70us
**.numVCs = 1
**.rows = 3
**.columns = 4
**.statStartTime = 20us
**.flitArrivalDelay = ${FID=16.6666666666667,14.8148148148148,...,4.3010752688172}ns
**.heterogeneous = false
**.tClk = 2ns
warmup-period = 20us
repeat = 1
```

Notes on the fields that matter and aren't obvious from the stock example:

- `**.routerType`/`**.portType` are recorded (and show up in the `.sca`
  output, confirmed below) but functionally irrelevant once `network =
  hnocs.topologies.my_Mesh_simuN` is used — that network hardcodes
  `RouterNSEWL`/`Port_bufSize` directly in its submodule declaration
  (`Router.ned`), not via the ini's type strings. This is exactly what the
  `# MMirka - unused with myMesh*` comment says.
- `**.powerFilePath` / `**.areaFilePath` are consumed by
  `InputBufferedRouterPowerAssess` and `AreaAssess` (submodules wired into
  every router in `Router.ned`) to write the raw `power.*` files and
  `area.area` that this repo's `power_FIR_extract.py`/`save_dataset.py`
  later parse.
- `**.probabilisticBench = xmldoc(...)` points at a small XML describing
  traffic — this is the "hotspot"/"uniform" traffic mechanism, backed by
  `src/cores/sources/ProbabilisticModel/SpatialDistribution/{SDUniform,SDHotspot}.cc`
  (both present in the vendored `src/`). A hotspot XML
  (`simulations/simu0/edit_synthetic_traffic.xml`):
  ```xml
  <synthetic_traffic>
    <nodes></nodes>
    <default>
        <spatial_distribution type="hotspot" pos_x="6" pos_y="6" proba="0.1"/>
        <temporal_distribution type="constant" interval="10"/>
        <packets_size_distribution type="constant" size="16"/>
    </default>
  </synthetic_traffic>
  ```
  and a uniform one (`simulations/simu1/edit_synthetic_traffic.xml`) just
  uses `<spatial_distribution type="uniform"/>`. The paper's hotspot setup
  is `pos_x=6, pos_y=6, proba=0.3` (30% of traffic to the hotspot router,
  DATE2022 `evaluation.tex`) — note the `simu0` example uses
  `proba="0.1"` and, being paired with a 4x4 mesh (`rows=4`), has a
  `pos_x/pos_y=6,6` that is out of range for that mesh (max valid
  coordinate would be 3), so don't copy it verbatim for a 4x4 run.
- `**.useRoutingTable = false` means `RT.txt` (present alongside each
  `simu<N>/omnetpp.ini`) is a **vestigial, unused file** for this static-XY
  routing configuration (`OPCalcType = hnocs.routers.hier.opCalc.static.XYOPCalc`)
  — don't spend time reconciling its dimensions with the mesh size.
- `**.flitArrivalDelay = ${FID=...}ns` is an OMNeT++ parameter sweep: one
  run of `opp_run` executes the simulation once per FID value in the list,
  producing one `General-FID=<value>-#0.sca` scalar file per injection
  rate. This is exactly what `latency_FIR_extract.py` /
  `power_FIR_extract.py` glob for (`General-FID=*.sca`) to build the
  latency/power-vs-injection-rate curves stored in `Data_AX`.
- `**.heterogeneous = false` even in a genuinely heterogeneous run is not
  a contradiction: per `GETTING STARTED`/`Router.ned`, this flag controls
  an unrelated internal switch (whether `SchedSync`'s clock is derived
  automatically); the actual heterogeneity comes entirely from the
  per-router buffer sizes baked into the `.ned` file by `ned_writer.py`.

## 5. Running one simulation and extracting results, end to end

Given an `X` (list of 64 router-class codes, e.g. mostly `104` with some
`112`/`102`) and the fixed 8x8 adjacency matrix (`A_mesh8x8`, built in
`dataset_creator_dyn.py`), the real pipeline (as implemented by
`dataset_creator_dyn.py`'s `thread_function()` / `save_dataset.py`'s
`extract_results()`) is:

1. **Write the topology.** `ned_writer.write_simuID(X, simuID)` (or the
   3x4-specific `write_simuID_3x4`, or a new `write_simuID_8x8`-style call
   you add analogously — see `write_simuID_3x4`'s body, it's a 3-line
   specialization of `write_simuID`) writes
   `HNOCS/src/topologies/myMesh_simu<simuID>.ned`.

2. **Set up the run directory.** Under `HNOCS/simulations/simu<simuID>/`,
   you need an `omnetpp.ini` modeled on section 4 above (network name
   `my_Mesh_simu<simuID>`, `rows=8`/`columns=8`, correct
   `powerFilePath`/`areaFilePath`/`probabilisticBench` pointing at that
   same `simu<simuID>` directory) plus an `edit_synthetic_traffic.xml` for
   the traffic pattern you want.

3. **Run it.** From `dataset_creator_dyn.py`'s `thread_function()`,
   the actual invocation used by this project (not Docker.memo's — that's
   the generic per-example one; this is the project-specific one that
   points at the generated topology and the project's own `src/hnocs`
   library):
   ```sh
   opp_run -m -u Cmdenv -c General \
     -n ../HNOCS/examples:../HNOCS/src \
     -l ../HNOCS/src/hnocs \
     ../HNOCS/simulations/simu<simuID>/omnetpp.ini
   ```
   (paths are relative to wherever the Python driver script runs from —
   `hnocs_path = "../HNOCS/"` at the top of `dataset_creator_dyn.py`;
   adjust to your actual working directory.) With the `${FID=...}` sweep
   in the `.ini`, this single command runs one simulation per injection
   rate and produces one `General-FID=<value>-#0.sca` per rate, plus
   (via the power/area submodules) `power.*` and `area.area` files under
   the paths given by `powerFilePath`/`areaFilePath`, all landing in
   `simu<simuID>/results/`.

4. **Move results into a per-sample folder.** `extract_results(simuID,
   folder)` in `dataset_creator_dyn.py`/`save_dataset.py` does exactly:
   ```sh
   mkdir <folder>
   mv HNOCS/simulations/simu<simuID>/results/* <folder>/
   mkdir <folder>/area && mv <folder>/area.area <folder>/area/
   ```
   This is the `sim_<i>/` layout you'll see referenced everywhere in this
   project's real data (confirmed against a real per-sample directory,
   `sim_318/`):
   ```
   sim_318/
     General-FID=9.30232558139534-#0.sca   # one per FID/injection rate
     General-FID=10-#0.sca
     ... (24 of these for a 24-point FID sweep)
     area/area.area
   ```
   Real `area.area` content (12-router 3x4 example — same format for
   8x8, just 64 rows):
   ```
   #Router Id;Buffer Area;Arbiter Area;Crossbar Area; Total Area
   0;8087.94;520.38;11289.6;19897.9
   1;17225.5;520.38;11289.6;29035.4
   ...
   ```
   A real `.sca` file's header records every `.ini` param verbatim
   (confirms section 4's `routerType`/`portType` values actually land in
   the output), followed by per-statistic blocks:
   ```
   version 2
   run General-20-20210928-19:09:08-17971
   attr network hnocs.topologies.my_Mesh_simu32
   itervar FID 10
   param **.routerType "\"hnocs.routers.hier.Router2\""
   param **.rows 3
   param **.columns 4
   ...
   statistic my_Mesh_simu32.router[0].port[2].sched link-utilization
   field count 1
   field mean 8.96
   ...
   ```
   (raw `power.*` files are deleted by `extract_results()`/`save_dataset_inter()`
   after extraction to save space — they don't survive; only
   the derived `.stat` files below do.)

5. **Parse latency.** `python latency_FIR_extract.py -i <folder>/ -o
   <folder>/latencies/` globs `<folder>/General-FID=*.sca`, extracts the
   `end-to-end-latency-ns` and `network-latency-ns` statistic means per
   FID, converts FID (ns) to FIR (injection rate, `FIR = clock_ns/FID`
   with `clock_ns = 2.0`), and writes semicolon-separated
   `<folder>/latencies/__end2endLatency.stat` /
   `__networkLatency.stat`, one `FIR;value` line per injection rate. Real
   example:
   ```
   0.1;25.46670571811967
   0.105;24.98213123314787
   ...
   0.215;1244.968078721521
   ```
   This `[injection_rates, latencies]` pair (as two parallel lists) is
   exactly `Data_AX.latency`.

6. **Parse power.** `python power_FIR_extract.py -i <folder>/ -o
   <folder>/powers/` globs `<folder>/power.*` (the raw Orion output before
   it's deleted), sums per-router power, and writes
   `<folder>/powers/__router_<j>_Power.stat` (one per router) plus
   `__globalPower.stat` (sum across all routers), same `FIR;value` format.
   `Data_AX.powers` is the list of per-router `[FIR, power]` pairs;
   `Data_AX.total_power` is the global one.

7. **Compute energy.** `mkdir <folder>/jouls/ && python
   jouls_FIR_compute.py -s <folder>/ -p <folder>/powers/__globalPower.stat
   -o <folder>/jouls/__efficiency.stat` re-reads the `.sca` files for
   `number-sent-packets`, converts to bytes sent
   (`flits_in_packet=8, bytes_in_flit=4`), and divides power-over-time
   (`simu_time = 50us`) by bytes to get pJ/byte. This becomes
   `Data_AX.total_jouls`.

8. **Assemble and pickle.** This is what `save_dataset.py`/
   `save_dataset_inter()` do per sample — read the four kinds of `.stat`
   files above plus `area.area`, build one
   `Data_AX(A, X, latency, total_power, powers, total_area, areas,
   total_jouls)` (class now canonically defined in
   `src/mrwgan/data_types.py`, not redefined ad hoc — import it from
   there for any new code), append to a list, and `pickle.dump()` the
   list. This pickle — a `list[Data_AX]` — is the format every dataset
   under `figures/data/baseline_topology_dataset/*/dataset` and
   `data/raw/generated/*` is in, and what
   `mrwgan.data_types.load_data_ax_dataset()` reads.

## 6. Worked example: one new GAN-generated config to one `Data_AX` sample

This closes the specific gap between "the GAN just generated a router
config" and "that config has simulated performance data usable by
`figures/pareto_figures.py --generated-dataset`".

**Starting point — raw, unsimulated generator output.** Pull a generator's
raw output out of a GAN training-history pickle and reshape it:
```python
X_list = history_fake[100]                                   # one training epoch's generated batch
X_list = X_list.reshape((BATCHSIZE, nb_routers, nb_routers + nb_classes))
# BATCHSIZE=100, nb_routers=64, nb_classes=3
```
i.e. one generated NoC is a `(64, 67)` array: for each of the 64 routers,
a 3-value softmax/Gumbel-softmax over `{Big, Medium, Small}` (columns
0–2) followed by a 64-dim one-hot node-ID (columns 3–66) — the
`NodeIDTransform` concatenation described in `src/mrwgan/data_types.py`.
This shape is confirmed against a real already-simulated sample in
`data/raw/generated/10-50/hotspot30_SatAndPow_GCN-GCN_70-30_dataset`
(`Data_AX.X.shape == (64, 67)` there too — same representation, just
already carried all the way through simulation). Some generator variants
instead output plain `(64, 3)` (no node ID) — `pareto_figures.py`'s
`normalized_area()` handles both by only ever reading the first
`n_classes` columns.

**Step 1 — discretize to router codes.** `ned_writer.py`/`routers.py`
need raw integer codes (`112`/`104`/`102`), not soft class vectors:
```python
import numpy as np

ROUTER_CLASS_MAP_3 = {112: 0, 104: 1, 102: 2}          # from src/mrwgan/data_types.py
INV_CLASS_MAP = {v: k for k, v in ROUTER_CLASS_MAP_3.items()}

def discretize(x_router_feature_matrix):    # shape (64, 3) or (64, 67)
    return [INV_CLASS_MAP[int(np.argmax(row[:3]))] for row in x_router_feature_matrix]

X_codes = discretize(one_generated_noc)     # e.g. [104, 104, 112, 102, ...], len 64
```

**Step 2 — write the topology and run HNOCS.** Using the fixed 8x8
adjacency (`A_mesh8x8` from `dataset_creator_dyn.py`) and a `simuID` of
your choosing:
```python
import ned_writer
ned_writer.write_simuID(X_codes, simuID)   # writes HNOCS/src/topologies/myMesh_simu<simuID>.ned
```
Then, having pre-created `HNOCS/simulations/simu<simuID>/omnetpp.ini` (see
section 4 — 8x8, `network = hnocs.topologies.my_Mesh_simu<simuID>`, correct
`powerFilePath`/`areaFilePath`/`probabilisticBench` for the traffic
pattern you're targeting — `hotspot30` means `pos_x=6, pos_y=6,
proba=0.3`, `uniform` means `<spatial_distribution type="uniform"/>`),
run it per section 5, step 3.

**Step 3 — extract and assemble.** Run `latency_FIR_extract.py`,
`power_FIR_extract.py`, `jouls_FIR_compute.py` per section 5 steps 5–7,
then build the sample directly against the canonical `Data_AX`:
```python
import pickle, sys
sys.path.insert(0, "src")   # repo root's src/, for mrwgan.data_types
from mrwgan.data_types import Data_AX

# parse_stat_file(path) reads a "FIR;value" file into [[FIR...], [value...]]
# — same parsing loop as save_dataset.py's inline while(1)/readline() blocks.
latency = parse_stat_file(f"{folder}/latencies/__end2endLatency.stat")
total_power = parse_stat_file(f"{folder}/powers/__globalPower.stat")
powers = [parse_stat_file(f"{folder}/powers/__router_{j}_Power.stat") for j in range(64)]
total_jouls = parse_stat_file(f"{folder}/jouls/__efficiency.stat")
areas = parse_area_file(f"{folder}/area/area.area")   # column 4 of each line
total_area = sum(areas)

A = A_mesh8x8   # same fixed adjacency as every other 8x8 sample in this project
sample = Data_AX(A, X_codes, latency, total_power, powers, total_area, areas, total_jouls)

with open("my_new_generated_sample_dataset", "wb") as f:
    pickle.dump([sample], f)   # a *list* of Data_AX, matching every other dataset file's format
```
(You could instead store `X` as the original `(64, 67)` feature array
rather than discretized codes — `pareto_figures.py` accepts either
representation, per its `normalized_area()`. Discretized codes are
simpler and match the baseline-topology datasets' convention.)

The resulting `my_new_generated_sample_dataset` file is then a drop-in
argument to:
```sh
python figures/pareto_figures.py \
    --generated-dataset my_new_generated_sample_dataset \
    --generated-label my_new_sample
```
For a statistically meaningful "generated" series (average over ~40–100
samples per weight-ratio point, per `data/raw/generated/README.md`), repeat
steps 1–3 for a batch of generated
configs from the same GAN checkpoint and accumulate them into one list
before pickling.
