export OMP_NUM_THREADS
export GOMP_CPU_AFFINITY

ITE=$(seq 10) # nombre de mesures
THREADS="1 2 4 6 8" # nombre de threads
SIZES="64 128 256 512 1024 2048 4096" # taille de plateau
GOMP_CPU_AFFINITY=$(seq 0 8) # vérifier à l'aide de lstopo la bonne alternance des processeurs
PARAM="../prog -n -k vie -i 1000" # parametres commun à toutes les executions

execute (){
EXE="$PARAM $*"
OUTPUT="$OMP_SCHEDULE-$(echo $* | tr -d ' ')"
#for nb in $ITE; do for OMP_NUM_THREADS in $THREADS ; do echo -n "$OUTPUT $OMP_NUM_THREADS " >> ALL ;  $EXE | tail -n1 >> ALL; done; done
for SIZE in $SIZES; do for nb in $ITE ; do echo -n "$OUTPUT $SIZE " >> ALL_omp_schedule ;  $EXE -s $SIZE | tail -n1 >> ALL_omp_schedule; done; done
}

#for i in 256 512 1024 2048;
#do
export OMP_SCHEDULE=static
execute  -v omp
execute  -v omp_tile
execute  -v omp_tile_opt

export OMP_SCHEDULE=dynamic
execute  -v omp
execute  -v omp_tile
execute  -v omp_tile_opt
#done
