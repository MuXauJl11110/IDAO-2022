all: build

build:
	@echo 'starting build....'
	unzip -q data.zip
	@echo 'ending build....'
run:	
	@echo 'starting run....'
	bash train.sh
	bash run.sh
	@echo 'ending run....'