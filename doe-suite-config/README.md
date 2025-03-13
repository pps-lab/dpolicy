<a name="readme-top"></a>
# Reproduce Experiments

<!-- ABOUT THE PROJECT -->
## About The Project

We provide the necessary commands to reproduce the entire evaluation of the paper.
The evaluation is built using the [DoE-Suite](https://github.com/nicolas-kuechler/doe-suite), making it the most straightforward way to reproduce the results.
However, it is also possible to obtain the individual commands used to invoke the [DPolicy dp-planner](../cohere-dpolicy) and run them manually.


### Built With

* [![Poetry][poetry-shield]][poetry-url]
* [![DoE-Suite][doesuite-shield]][doesuite-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

* [Python Poetry](https://python-poetry.org/)
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

* [Make](https://www.gnu.org/software/make/)

* Local clone of the repository (with submodules)
    ```sh
    git clone --recurse-submodules git@github.com:pps-lab/dpolicy.git
    ```


### Installation

#### DoE-Suite

1. Setup environment variables for the Doe-Suite:
    ```sh
    # root project directory (expects the doe-suite-config dir in this folder)
    export DOES_PROJECT_DIR=<PATH>

    #  Your unique short name, such as your organization's acronym or your initials.
    export DOES_PROJECT_ID_SUFFIX=<SUFFIX>
    ```

    Additional variables for Docker-based execution:
    ```sh
    export DOES_CLOUD=docker

    export DOES_DOCKER_USER=ubuntu
    export DOES_DOCKER_SSH_PUBLIC_KEY=<YOUR-SSH-PUBLIC-KEY>
    export DOCKER_HOST=<YOUR-DOCKER-HOST>    #e.g., unix:///home/ubuntu/.docker/desktop/docker.sock

    ```

    Additional variables for ETHZ Euler (Slurm-based Scientific Compute Cluster):
    ```sh
    export DOES_CLOUD=euler

    # Replace <YOUR-NETHZ> with your NETHZ username
    export DOES_EULER_USER=<YOUR-NETHZ>
    ```


2. Configure SSH for Docker:
If using Docker, ensure your SSH configuration (`~/.ssh/config`) allows access to the Docker container. For detailed instructions, refer to the DoE-Suite [documentation](https://nicolas-kuechler.github.io/doe-suite/installation.html#base-installation).


[!TIP]
To aid in debugging, consider commenting out the `stdout_callback = community.general.selective line` in the [doe-suite/ansible.cfg](../doe-suite/ansible.cfg) file. This will provide more verbose output.




#### Paper Results and Workload Data

1. Download the raw results of the DPolicy evaluation: [Download (546 MB)](https://drive.google.com/file/d/1mP8cYAjpczndLAoHj6G0YfuVPgwN2CuE/view?usp=sharing)


2. Unarchive the file: `sp25-dpolicy-results.zip`


3. Move the extracted result folders to [doe-suite-results](../doe-suite-results/):

    ```sh
    # the directory should look like:

    doe-suite-results/
    ├─ trap_1731489444
    ├─ trap-relax_1731504884
    └─ trap-time-threads_1731532838
    ```

4. The `sp25-dpolicy-results.zip` includes also an archive labeled `applications.zip`, containing the original workloads. Alternatively, these workloads can be newly sampled with the [workload-simulator](../workload-simulator/).


5. The [doe-suite-config/roles/data-setup-local](roles/data-setup-local) Ansible role is responsible for making the `applications.zip` file available within the remote experiment environment. Place your `applications.zip` (either the one from the download or newly generated) into the `doe-suite-config/upload/` directory, replacing the placeholder `applications.zip` if present.



#### Gurobi License

The [dp-planner](../cohere-dpolicy) relies on [Gurobi](https://www.gurobi.com/) for solving resource allocation problems, which requires a Gurobi license.
Free academic licenses can be obtained from the following link: https://www.gurobi.com/downloads/end-user-license-agreement-academic/

The acquired license keys still need to be activated on the designated machine.

For Docker Environments: If you are using Docker, obtain a `gurobi.lic` WSL license file and place it in the `doe-suite-config/upload/` directory. The DoE-Suite will then handle its placement within the Docker container.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Experiments

After the installation, the original experiment outcomes referenced in the paper will be accessible at [doe-suite-results](../doe-suite-results).
Any additional experiments conducted will also be stored in the same location.


For simplifying the reproduction of results, we provide a custom [Makefile](../Makefile) to simplify the interface.
All further commands available within the `doe-suite` can be accessed via the [Makefile](../doe-suite/Makefile) located in [doe-suite](../doe-suite/).
For further details, please refer to the [doe-suite documentation](https://nicolas-kuechler.github.io/doe-suite/).


We can reconstruct all evaluation figures in the paper with:
```
make plot-all
```

### Context Scenario

The suite design [doe-suite-config/designs/trap-relax.yml](designs/trap-relax.yml) defines the experiment.


To obtain all individual commands, use:
```
make cmd-context
```


For rerunning all the experiments, execute:
```
make run-context
```


The generation of the context plot (Figure 3 left) relies on the super ETL config [doe-suite-config/super_etl/trap-combined.yml](super_etl/trap-combined.yml).

To regenerate the plot, use:
```
make plot-context
```

By default, the figure is derived from the experiment results shown in the paper.
However, it is possible to provide a custom results directory obtained from `run-context`.

<div align="center">
    <img src=./../.github/resources/context-fig3.png width="600">
</div>


<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Scope Scenario

The suite designs [doe-suite-config/designs/trap.yml](designs/trap.yml) and [doe-suite-config/designs/trap.yml](designs/trap.yml) define the experiments.


To obtain all individual commands, use:
```
make cmd-scope
```


For rerunning all the experiments, execute:
```
make run-scope
```


The generation of the scope plots (Figure 3 middle + Figure 4) relies on the super ETL config [doe-suite-config/super_etl/trap-combined.yml](super_etl/trap-combined.yml).

To regenerate the plots, use:
```
make plot-scope
```

By default, the figures are derived from the experiment results shown in the paper.
However, it is possible to provide a custom results directory obtained from `run-scope`.


<div align="center">
    <img src=./../.github/resources/scope-category-fig3.png width="600">
</div>

<div align="center">
    <img src=./../.github/resources/scope-attribute-fig4.png width="600">
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Time-based Privacy Unit

The suite design [doe-suite-config/designs/trap-time-threads.yml](designs/trap-time-threads.yml) defines the experiment.


To obtain all individual commands, use:
```
make cmd-timeunit
```


For rerunning all the experiments, execute:
```
make run-timeunit
```

The generation of the time-based privacy unit plot (Figure 3 right) relies on the super ETL config [doe-suite-config/super_etl/trap-combined.yml](super_etl/trap-combined.yml).

To regenerate the plot, use:
```
make plot-timeunit
```

By default, the figure is derived from the experiment results shown in the paper.
However, it is possible to provide a custom results directory obtained from `run-timeunit`.



<div align="center">
    <img src=./../.github/resources/timeunit-fig3.png width="600">
</div>


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->

[poetry-shield]: https://img.shields.io/badge/poetry-grey?style=for-the-badge&logo=poetry
[poetry-url]: https://python-poetry.org/


[doesuite-shield]: https://img.shields.io/badge/doe--suite-grey?style=for-the-badge&logo=github
[doesuite-url]: https://github.com/nicolas-kuechler/doe-suite
