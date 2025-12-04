CFLAGS = -O
CC = g++

peace: directory out/main.o out/flesch.o
	$(CC) $(CFLAGS) -o peace out/main.o out/flesch.o

directory:
	mkdir -p out

out/main.o: src/main.cpp
	$(CC) $(CFLAGS) -o out/main.o -c src/main.cpp

out/flesch.o: src/flesch/flesch.cpp src/flesch/flesch.hpp
	$(CC) $(CFLAGS) -o out/flesch.o -c src/flesch/flesch.cpp

clean:
	rm -fr out
	rm peace