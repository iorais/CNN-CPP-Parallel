all:
	nvcc -lcuda -lcublas *.cu -o CNN  -arch=sm_70 -O3

run:
	./CNN
clean:
	rm CNN
