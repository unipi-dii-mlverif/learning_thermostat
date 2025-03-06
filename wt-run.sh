#!/bin/sh

#echo Configuring extension
export MAESTRO_EXT_CP=$(ls -p /work/model/classpath/*.jar | tr '\n' ':')
#export MAESTRO_EXT_CP="/work/model/classpath/commons-lang-2.6.jar:/work/model/classpath/exp4j-0.4.8.jar"
echo "\t Extension cp $MAESTRO_EXT_CP"

echo "Generating Mabl specifications"

mkdir -p transition

#maestro import --help

maestro import sg1 ./FaultInject.mabl mm1.json simulation-config.json -fsp . -output stage1/
#maestro import --verbose sg1 mm1.json simulation-config.json -fsp . -output stage1/

EXITCODE=$?
test $EXITCODE -eq 0 && echo "ok" || exit $EXITCODE;


maestro import sg1 ./FaultInject.mabl mm2.json simulation-config.json -fsp . -output transition/stage2/
#maestro import sg1 mm2.json simulation-config.json -fsp . -output transition/stage2/

EXITCODE=$?
test $EXITCODE -eq 0 && echo "ok" || exit $EXITCODE;

echo "Simulating specifications"

maestro  interpret stage1/spec.mabl -tms 220 -transition transition -output stage1 2>&1 | tee out.txt

EXITCODE=$?
test $EXITCODE -eq 0 && echo "ok" || exit $EXITCODE;
