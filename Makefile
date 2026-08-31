CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -Iinclude

SRC = src/order_book.cpp src/lobster_parser.cpp src/replay.cpp

all: test_reconstruction test_csv_replay

test_reconstruction: tests/test_reconstruction.cpp $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

test_csv_replay: tests/test_csv_replay.cpp $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

test: all
	./test_reconstruction
	@echo "----------------------------------------"
	./test_csv_replay

order_book_bench: bench/order_book_bench.cpp src/order_book.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

bench: order_book_bench
	./order_book_bench

clean:
	rm -f test_reconstruction test_csv_replay order_book_bench

.PHONY: all test bench clean
