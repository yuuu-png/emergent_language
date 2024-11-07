include .env

ARGS =
GPU = 0
allargs = $(ARGS) --gpu $(GPU)

def = docker/singularity/image.def
sif = image.sif
tag = 0.0.6

singularity = singularity exec --fakeroot --nv --bind $(PWD):/work,$(HOST_DATA_DIR):/data --pwd /work --env-file .env $(sif)
remote = library://atahatah/simple_caption/runner:$(tag)

ulimit = ulimit -n 1048576

image.sif:
	singularity pull $(sif) $(remote)
	# singularity verify $(sif)

singularity-build:
	singularity build --fakeroot --notest $(sif) $(def)

singularity-push:
	singularity push -U $(sif) $(remote)

singularity-test: image.sif
	singularity test $(sif)

continuous_simclr = /work/src/wandb/continuous_simclr

continuous_simclr: image.sif .env
	$(ulimit)
	$(singularity) python3 $(continuous_simclr)/train.py $(allargs)

generate = /work/src/wandb/generate

generate: image.sif .env
	$(ulimit)
	$(singularity) python3 $(generate)/train.py $(allargs)

generate-copy: image.sif .env
	$(ulimit)
	$(singularity) python3 $(generate)/copy_with.py $(allargs)

generate-sweep: image.sif .env
	$(ulimit)
	$(singularity) python3 $(generate)/sweep.py $(allargs)

generate-%: image.sif .env
	$(ulimit)
	$(singularity) python3 $(generate)/sweep.py --sweep_id ${@:generate-%=%} $(allargs)

generate-classify: image.sif .env
	$(ulimit)
	$(singularity) python3 $(generate)/classify.py $(allargs)

pretrain = /work/src/wandb/pretrain

pretrain: image.sif .env
	$(ulimit)
	$(singularity) python3 $(pretrain)/train.py $(allargs)

pretrain-sweep: image.sif .env
	$(ulimit)
	$(singularity) python3 $(pretrain)/sweep.py $(allargs)

pretrain-%: image.sif .env
	$(ulimit)
	$(singularity) python3 $(pretrain)/sweep.py --sweep_id ${@:pretrain-%=%} $(allargs)

pretrain-classify: image.sif .env
	$(ulimit)
	$(singularity) python3 $(pretrain)/classify.py $(allargs)

instance = /work/src/wandb/instance
instance: image.sif .env
	$(ulimit)
	$(singularity) python3 $(instance)/train.py $(allargs)

simclr = /work/src/wandb/simclr

simclr: image.sif .env
	$(ulimit)
	$(singularity) python3 $(simclr)/train.py $(allargs)

simclr-copy: image.sif .env
	$(ulimit)
	$(singularity) python3 $(simclr)/copy_with.py $(allargs)

simclr-sweep: image.sif .env
	$(ulimit)
	$(singularity) python3 $(simclr)/sweep.py $(allargs)

simclr-%: image.sif .env
	$(ulimit)
	$(singularity) python3 $(simclr)/sweep.py --sweep_id ${@:simclr-%=%} $(allargs)

simclr-classify: image.sif .env
	$(ulimit)
	$(singularity) python3 $(simclr)/classify.py $(allargs)

translate = /work/src/wandb/simclr

translate: image.sif .env
	$(ulimit)
	$(singularity) python3 $(translate)/train.py $(allargs)

# Docker

docker_image = pytorch_env
docker_container = pytorch_env

# Dockerfile -> docker image
docker-build:
	docker build --progress=plain -t $(docker_image) . ;

docker-build-no-cache:
	docker build --no-cache --progress=plain -t $(docker_image) . ;

docker_run_args = -dit --name $(docker_container) \
	--volume $(PWD):/work \
	--volume $(HOST_DATA_DIR):/data \
	--shm-size=16g \
	--gpus all \
	--env-file .env

# docker image -> docker container
docker-run:
	docker run $(docker_run_args) $(docker_image) ;

# docker container -> docker exec
docker-start:
	docker start $(docker_container) ;

# stop docker exec
docker-stop:
	docker stop $(docker_container) ;

docker-restart:
	docker stop $(docker_container) ;
	docker start $(docker_container) ;

# remove docker container
docker-rm:
	docker rm $(docker_container) ;

# remove docker container and run again
docker-rerun:
	docker stop $(docker_container) ;
	docker rm $(docker_container) ;
	docker run $(docker_run_args) $(docker_image) ;

test_script = /work/bin/test.sh

# run test script in docker container
docker-test:
	docker exec -it $(docker_container) $(test_script) ;

.PHONY: test
test:
	$(test_script)
