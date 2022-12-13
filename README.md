# [WIP] MARL-mpr
 While on root:
- Build the Dockerfile `docker build -t raffaele/dancing_bee:0.1 .` folder
- Run `docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/home/devuser/dev:Z  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -it --rm  raffaele/dancing_bee:0.1
`