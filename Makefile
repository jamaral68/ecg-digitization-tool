IMAGE ?= ecg-digitization-tool:rocm

.PHONY: build run shell

build:
	docker build -t $(IMAGE) .

run:
	docker run -it --cap-add=SYS_PTRACE \
	--security-opt \
	seccomp=unconfined \
	--device=/dev/kfd \
	--device=/dev/dri \
	--group-add video \
	--ipc=host \
	--shm-size=8G \
	$(IMAGE)

shell:
	docker run --rm -it $(IMAGE) bash

