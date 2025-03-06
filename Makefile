#ZIP_PROGRAMME=zip -r
ZIP_PROGRAMME=7z a -tzip -mx=1

all: FMU/Controller.fmu FMU/KalmanFilter.fmu FMU/Plant.fmu FMU/Room.fmu FMU/Supervisor.fmu


FMU/Controller.fmu: $(shell find FMU/Controller -print)
	@rm -f $@ 
	@cd FMU/Controller && $(ZIP_PROGRAMME) ../../$@ ./* > /dev/null


FMU/KalmanFilter.fmu: $(shell find FMU/KalmanFilter -print)
	@rm -f $@ 
	@cd FMU/KalmanFilter && $(ZIP_PROGRAMME) ../../$@ ./* > /dev/null


FMU/Plant.fmu: $(shell find FMU/Plant -print)
	@rm -f $@ 
	@cd FMU/Plant && $(ZIP_PROGRAMME) ../../$@ ./* > /dev/null


FMU/Room.fmu: $(shell find FMU/Room -print)
	@rm -f $@ 
	@cd FMU/Room && $(ZIP_PROGRAMME) ../../$@ ./*  > /dev/null


FMU/Supervisor.fmu: $(shell find FMU/Supervisor -print)
	@rm -f $@ 
	@cd FMU/Supervisor && $(ZIP_PROGRAMME) ../../$@ ./* > /dev/null


launch: all
	docker build . --tag lausdahl/maestro:2.3.0-model-swap
	docker run -it -v $(PWD):/work/model/post  lausdahl/maestro:2.3.0-model-swap

# TODO test
