NVCC := nvcc

# Architettura della GPU target: verifica con `nvidia-smi` sull'istanza AWS e
# aggiusta se necessario (T4 = sm_75, V100 = sm_70, A10G = sm_86, A100 = sm_80).
ARCH := -arch=sm_75

NVCCFLAGS := -O2 -std=c++14 -Iinclude -Iexternal $(ARCH)
LDFLAGS :=

SRC_DIR := src
APP_DIR := app
BUILD_DIR := build

C_SRCS := $(wildcard $(SRC_DIR)/*.c)
CU_SRCS := $(wildcard $(SRC_DIR)/*.cu)
APP_SRCS := $(wildcard $(APP_DIR)/*.c)

OBJS := $(C_SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o) \
        $(CU_SRCS:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o) \
        $(APP_SRCS:$(APP_DIR)/%.c=$(BUILD_DIR)/%.o)

TARGET := main

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) $(LDFLAGS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -x c -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: $(APP_DIR)/%.c | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -x c -MMD -MP -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

run: $(TARGET)
	./$(TARGET)

-include $(OBJS:.o=.d)
