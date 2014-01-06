all: library train_ae

library:
	( cd lib ; $(MAKE) )

train_ae: 
	( cd demo ; $(MAKE) )

clean:
	( cd lib ; $(MAKE) clean )
	( cd demo ; $(MAKE) clean )
