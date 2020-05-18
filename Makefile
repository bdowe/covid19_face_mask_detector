help:
	@cat Makefile

IMAGE_NAME=tensorflow
IMAGE_TAG=covid19-face-mask-detector
DATA?=`pwd`
DOCKER=docker
ifdef GPU
	DOCKER=nvidia-docker
endif

build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

bash: build
	$(DOCKER) run -it -ti --net=host --ipc=host -v $(DATA):/data -e DISPLAY=$(DISPLAY) -v /tmp/.X11-unix:/tmp/.X11-unix --env="QT_X11_NO_MITSHM=1" $(IMAGE_NAME):$(IMAGE_TAG) bash

notebook: build
	$(DOCKER) run -it -v $(DATA):/data --net=host $(IMAGE_NAME):$(IMAGE_TAG)