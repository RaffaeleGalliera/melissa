# [WIP] A Multi Agent Reinforcement Learning Environment for Message Dissemination
 
The framework is written in Python and is based on PyTorch. It implements a customized extension of the Tianshou framework and defines the MARL environment following the PettingZoo API. The GAT and global max pooling employ the implementation provided by PyTorch Geometric. Training and testing graphs were generated with the aid of the NetworkX library.

## Implementation Details

### Framework Components
- The main components of our framework can be found in the `graph_env/env` folder.
- Environment dynamics are defined in:
  - `graph_env/env/graph.py`
  - `graph_env/env/utils/core.py`
- Networks and policy implementations can be found in:
  - `graph_env/env/utils/networks`
  - `graph_env/env/utils/policies`


### Training and Testing

1. Build the Docker image:
   ```bash
   docker build -t marl_mpr .
2. For machines without a GPU or Apple Mac devices (including ones employing Apple MX SoC):
    ```bash
     docker build -t marl_mpr -f Dockefile.cpu .
    ```
3. Run the container:
    ```bash
    docker run --ipc=host --gpus all -v ${PWD}:/home/devuser/dev:Z -it --rm marl_mpr
    ```
4. (Optional) Visualization:

    To visualize agents in action on testing graphs, use the following command:
    ```bash
   docker run --ipc=host --gpus all -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ${PWD}:/home/devuser/dev:Z -it --rm marl_mpr
    ```
    Alternatively, you can install the requirements in a Python (≥ 3.8) virtual environment:
    ```bash 
    pip install -r requirements.txt
    ```

### Training Models

Trained models are saved in `log/algorithm_name/weights` as `model_name_last.pth`. Before training, you'll be prompted to log training data using the Weight and Biases (WANDB) logging tool.

- DGN-R 
  ```bash 
  python train_dgn_r.py --model-name DGN-R
  ```
- L-DGN 
  ```bash 
  python train_l_dgn.py --model-name L-DGN
  ```
- HL-DGN 
  ```bash 
  python train_hl_dgn.py --model-name hL-DGN
  ```

#### Seeding

By default, the seed is set to 9. You can change this value using `--seed X` where X is your chosen seed.

### Testing Models

Trained models can be found in their respective subfolders of the `/log` folder. To reproduce results:

- DGN-R:
```bash
python train_dgn_r.py --watch --model-name DGN-R.pth
```

- DGN-R:
    ```bash
    python train_dgn_r.py --watch --model-name DGN-R.pth
    ```

- L-DGN:
    ```bash
    python train_l_dgn.py --watch --model-name L-DGN.pth
    ```

- MPR Heuristic:
    
    ```bash
    python train_hl_dgn.py --watch --model-name HL-DGN.pth --mpr-policy
    ```
### Topologies Dataset

We've generated two datasets containing connected graph topologies. The first has 20 nodes per graph, and the second has 50 nodes. Training and testing sets for each dataset contain 50K and 100 graphs, respectively.  You can switch between different numbers of agents (20/50) using the `--n-agents` argument (default is 20). 
All topologies can be download [here](https://drive.google.com/file/d/1Osnw_jqmIOjTqH6i2Zt8352J2LhE3w8O/view?usp=sharing). The compressed folder should be unzipped in the root directory of the project.

