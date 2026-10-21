run:
	flwr run . --stream

clean-data:
	rm -f data/*

clean-results:
	rm -f results/*