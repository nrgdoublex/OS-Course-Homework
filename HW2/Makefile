CC=gcc
CFLAGS=-g -Wall
LDFLAGS=-lpthread -lncurses -lpanel -lmenu -lform
CFILES=$(shell ls|grep .c)
PROGS=$(patsubst %.c,%,$(CFILES))

all:$(PROGS)

%:%.c
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS)

clean:$(PROGS)
	rm -f $(PROGS)
