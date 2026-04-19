PKG_CONFIG ?= pkg-config
CC ?= cc
CARGO ?= cargo

ROOT_DIR := ..
BUILD_DIR := build
TARGET_DIR := $(ROOT_DIR)/target/debug
EXAMPLE := $(BUILD_DIR)/render_714_file
RUST_STATICLIB := $(TARGET_DIR)/libstarmine_ad.a
RUST_NATIVE_STATIC_LIBS := $(shell cd $(ROOT_DIR) && $(CARGO) rustc -p libstarmine_ad --lib --crate-type staticlib -- --print native-static-libs 2>&1 | sed -n 's/^note: native-static-libs: //p')

CFLAGS += -std=c11 -O2 -Wall -Wextra -I$(ROOT_DIR)/include
CFLAGS += $(shell $(PKG_CONFIG) --cflags libavformat libavcodec libavutil)
LDLIBS += -L$(TARGET_DIR) -lstarmine_ad
LDLIBS += $(shell $(PKG_CONFIG) --libs libavformat libavcodec libavutil)
LDLIBS += $(RUST_NATIVE_STATIC_LIBS)

.PHONY: all clean

all: $(EXAMPLE)

$(RUST_STATICLIB):
	cd $(ROOT_DIR) && $(CARGO) build -p libstarmine_ad

$(EXAMPLE): render_714_file.c $(RUST_STATICLIB) $(ROOT_DIR)/include/starmine_ad.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ render_714_file.c $(LDLIBS)

clean:
	rm -rf $(BUILD_DIR)
