abi_changes_YYYY_dmspXX.txt refers to the data from year YYYY and DMSPXX.

The header for each file is 
"DD-Mth-YYYY hh:mm:ss     MLAT (deg)     MLOC_TIME     Code\n"

The data is formatted using the following string format in Matlab:
"%s     %.4f        %.5f       %d\n", datetime, mlat, mloc_time, code

Codes:
1: ascending crossing of equatorial boundary (flag 1->2)
2: descending crossing of equatorial boundary (flag 2->1)
3: ascending crossing of poleward boundary (flag 2->3)
4: descending crossing of poleward boundary (flag 3->2)