run:
	flwr run . --stream

clean-data:
	rm -r data/*

clean-results:
	rm -r results/*