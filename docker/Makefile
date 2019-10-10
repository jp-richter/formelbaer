build: check-environment
	find .. -size +50M | xargs -i% echo "WARNING: Not sending large file to Docker %"
	docker rmi s876cnsm:5000/${USER}-arxiv
	cd .. && find . -size +50M > .dockerignore && \
		docker build -t s876cnsm:5000/${USER}-arxiv . \
		-f docker/Dockerfile \
		--build-arg USER=${USER} \
		--build-arg USERID=${UID} \
		--build-arg "constraint:nodetype!=phi"
	docker push s876cnsm:5000/${USER}-arxiv

build-unsafe: check-environment
	cd .. && rm -f .dockerignore && \
		docker build -t s876cnsm:5000/${USER}-arxiv . \
		-f docker/Dockerfile \
		--build-arg USER=${USER} \
		--build-arg USERID=${UID} \
		--build-arg "constraint:nodetype!=phi"
	docker push s876cnsm:5000/${USER}-arxiv
run:
	docker run --rm -ti \
		-c20 -m128g \
		--name ${USER}-arxiv \
		-v/rdata/s01c_a1_001:/rdata/ \
		-e "constraint:nodetype!=phi" \
		--tmpfs /ramdisk \
		s876cnsm:5000/${USER}-arxiv
run-gpu:
	docker run --rm -ti \
		-c20 -m128g \
		--name ${USER}-arxiv \
		-v/rdata/s01c_a1_001:/rdata/ \
		-e "constraint:nodetype!=phi" \
		-e "constraint:node==s876gn02" \
		--device /dev/nvidia0 \
		--device /dev/nvidia1 \
		--device /dev/nvidiactl \
		--device /dev/nvidia-uvm \
		--tmpfs /ramdisk \
		--env CUDA_VISIBLE_DEVICES=1 \
		s876cnsm:5000/${USER}-arxiv
		
compile-pngs:
	echo "USAGE: make compile-pngs PART=x"
	docker run --rm -ti -d\
		-c20 -m128g \
		--name ${USER}-arxiv-$(PART) \
		-v/rdata/s01c_a1_001:/rdata/ \
		-e constraint:nodetype!=phi \
		--tmpfs /ramdisk \
		s876cnsm:5000/${USER}-arxiv \
		python3 -u extract_png_formulas.py --total_partitions 10 --partition $(PART) 1

check-environment:
	[ -z "${UID}" ] && { echo "UID not set, run 'export UID' "; exit 1; } || true
	[ -z "${USER}" ] && { echo "USER not set, run 'export USER' "; exit 1; } || true
