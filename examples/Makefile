PKG_CONFIG ?= pkg-config
CC ?= cc
CARGO ?= cargo

ROOT_DIR := ..
MODE ?= debug

ifeq ($(MODE),release)
BUILD_DIR := build/release
TARGET_DIR := $(ROOT_DIR)/target/release
CARGO_BUILD_ARGS := --release
OPT_CFLAGS := -O3 -DNDEBUG -flto
OPT_LDFLAGS := -flto
else ifeq ($(MODE),debug)
BUILD_DIR := build/debug
TARGET_DIR := $(ROOT_DIR)/target/debug
CARGO_BUILD_ARGS :=
OPT_CFLAGS := -O2
OPT_LDFLAGS :=
else
$(error unsupported MODE '$(MODE)'; expected 'debug' or 'release')
endif

EXAMPLE := $(BUILD_DIR)/render_714_file
RUST_STATICLIB := $(TARGET_DIR)/libstarmine_ad.a
RUST_NATIVE_STATIC_LIBS := $(shell cd $(ROOT_DIR) && $(CARGO) rustc $(CARGO_BUILD_ARGS) -p libstarmine_ad --lib --crate-type staticlib -- --print native-static-libs 2>&1 | sed -n 's/^note: native-static-libs: //p')

CFLAGS += -std=c11 -Wall -Wextra -I$(ROOT_DIR)/include $(OPT_CFLAGS)
CFLAGS += $(shell $(PKG_CONFIG) --cflags libavformat libavcodec libavutil)
LDFLAGS += $(OPT_LDFLAGS)
LDLIBS += -L$(TARGET_DIR) -lstarmine_ad
LDLIBS += $(shell $(PKG_CONFIG) --libs libavformat libavcodec libavutil)
LDLIBS += $(RUST_NATIVE_STATIC_LIBS)

.PHONY: all clean debug release

all: $(EXAMPLE)

$(RUST_STATICLIB):
	cd $(ROOT_DIR) && $(CARGO) build $(CARGO_BUILD_ARGS) -p libstarmine_ad

$(EXAMPLE): render_714_file.c $(RUST_STATICLIB) $(ROOT_DIR)/include/starmine_ad.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ render_714_file.c $(LDLIBS)

debug:
	$(MAKE) MODE=debug all

release:
	$(MAKE) MODE=release all

clean:
	rm -rf $(BUILD_DIR)
