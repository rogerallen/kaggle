#!/bin/sh
#
#   1e-1 std 807.678804# histo [   0    0 2698   57    0    0    0    0    0    0]
#   1e-2 std 646.498917# histo [ 429    0  120   29    0    0 2177    0    0    0]
#   1e-3 std 702.191035# histo [2377    0    0  101  126    9  114   19    0    9]
#   1e-4 std 335.245358# histo [277 841  27   5 176  67 993  59  43 267]
#   1e-5 std 288.689193# histo [ 29 613  20 148  94 771  67 208 733  72]
# * 1e-6 std 213.648894# histo [276 157 411 417  88 395  62 166  24 759]
#   1e-7 std 301.634298# histo [ 26  76  54 157 177   1 744 584 100 836]
#   1e-8 std 374.224598# histo [ 103   15    3  305  454  297   53    4 1309  212]
#   1e-9 std 420.059103# histo [ 406   53   10   21 1427    1  313  467   54    3]

LOGFILE=logs/shmoo1_0227_1.log

./statefarm.py -v --ep0 5 --lr 1e-6 --save 0227_1 > ${LOGFILE}
./statefarm.py -v --ep0 5 --lr 5e-7 --save 0227_2 >> ${LOGFILE}
./statefarm.py -v --ep0 5 --lr 1e-7 --save 0227_3 >> ${LOGFILE}
./statefarm.py -v --ep0 5 --lr 5e-8 --save 0227_4 >> ${LOGFILE}
./statefarm.py -v --ep0 5 --lr 1e-8 --save 0227_5 >> ${LOGFILE}
