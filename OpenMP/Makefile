CC := clang
CFLAGS := -g -O2 -Wall -Wextra -Iinclude -Iexternal -I/opt/homebrew/opt/libomp/include -Xclang -fopenmp
LDFLAGS := -L/opt/homebrew/opt/libomp/lib -lomp

SRC_DIR := src
APP_DIR := app
TEST_DIR := tests
BUILD_DIR := build

# Flag che disattivano la vettorizzazione: servono per il baseline scalare.
# Scrivere C puro senza pragma NON basta, clang auto-vettorizza comunque a -O2.
NOVEC_FLAGS := -fno-vectorize -fno-slp-vectorize

# sequential.c e' compilato due volte, con suffissi di simbolo diversi, per
# produrre la variante scalare e quella auto-vettorizzata nello stesso binario.
SRCS := $(filter-out $(SRC_DIR)/sequential.c, $(wildcard $(SRC_DIR)/*.c))
SEQ_OBJS := $(BUILD_DIR)/sequential_scalar.o $(BUILD_DIR)/sequential_simd.o
APP_SRCS := $(wildcard $(APP_DIR)/*.c)
LIB_OBJS := $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o) $(SEQ_OBJS)
APP_OBJS := $(APP_SRCS:$(APP_DIR)/%.c=$(BUILD_DIR)/%.o)
OBJS := $(LIB_OBJS) $(APP_OBJS)

TARGET := main
TEST_TARGET := $(BUILD_DIR)/test_correctness

.PHONY: all clean run test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LDFLAGS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: $(APP_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/sequential_simd.o: $(SRC_DIR)/sequential.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DSEQ_SUFFIX=_simd -MMD -MP -c $< -o $@

$(BUILD_DIR)/sequential_scalar.o: $(SRC_DIR)/sequential.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(NOVEC_FLAGS) -DSEQ_SUFFIX=_scalar -MMD -MP -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Validazione della versione parallela contro il baseline sequenziale.
$(TEST_TARGET): $(BUILD_DIR)/test_correctness.o $(LIB_OBJS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

test: $(TEST_TARGET)
	./$(TEST_TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(TARGET).dSYM

run: $(TARGET)
	./$(TARGET)

-include $(OBJS:.o=.d)
-include $(BUILD_DIR)/test_correctness.d
