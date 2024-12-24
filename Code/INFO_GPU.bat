nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits -l 5 | while read line; do
  used_mem=$(echo $line | cut -d ',' -f1)
  free_mem=$(echo $line | cut -d ',' -f2)
  gpu_util=$(echo $line | cut -d ',' -f3)
  temp=$(echo $line | cut -d ',' -f4)
  
  echo "Mémoire utilisée: ${used_mem} MiB, Mémoire libre: ${free_mem} MiB"
  echo "Utilisation du GPU: ${gpu_util} %"
  echo "Température du GPU: ${temp} C"
  echo "-----------------------------"
done
