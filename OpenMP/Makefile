CC := clang
CFLAGS := -g -O2 -Wall -Wextra -Isrc -Iexternal -I/opt/homebrew/opt/libomp/include -Xclang -fopenmp
LDFLAGS := -L/opt/homebrew/opt/libomp/lib -lomp

SRC_DIR := src
BUILD_DIR := build

SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
TARGET := main

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LDFLAGS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(TARGET).dSYM

run: $(TARGET)
	./$(TARGET)

-include $(OBJS:.o=.d)
