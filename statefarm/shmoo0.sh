#!/bin/sh

LOGFILE=shmoo0.log

./statefarm.py -v --ep0 3 --lr 1e-1 > ${LOGFILE}
./statefarm.py -v --ep0 3 --lr 1e-2 >> ${LOGFILE}
./statefarm.py -v --ep0 3 --lr 1e-3 >> ${LOGFILE}
./statefarm.py -v --ep0 3 --lr 1e-4 >> ${LOGFILE}
./statefarm.py -v --ep0 3 --lr 1e-5 >> ${LOGFILE}
./statefarm.py -v --ep0 3 --lr 1e-6 >> ${LOGFILE}
./statefarm.py -v --ep0 3 --lr 1e-7 >> ${LOGFILE}
./statefarm.py -v --ep0 3 --lr 1e-8 >> ${LOGFILE}
./statefarm.py -v --ep0 3 --lr 1e-9 >> ${LOGFILE}
