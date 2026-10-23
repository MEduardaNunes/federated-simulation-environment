run:
	flwr run . --stream

run-simple:
	flwr run . --stream \
	--run-config="num-server-rounds=5 fraction-train=0.1" \
    --federation-config="num-supernodes=100 init-args-num-cpus=4 init-args-num-gpus=0 client-resources-num-cpus=2 client-resources-num-gpus=0.1"

run-complete:
	flwr run . --stream \
    --run-config="num-server-rounds=200 fraction-train=0.1" \
    --federation-config="num-supernodes=100 init-args-num-cpus=4 init-args-num-gpus=0 client-resources-num-cpus=1 client-resources-num-gpus=0.1"

clean-data:
	rm -r data/*

clean-results:
	rm -r results/*