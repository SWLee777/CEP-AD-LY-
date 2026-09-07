device=0

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)

for epoch in {1..25}; do
  for i in "${!depth[@]}"; do
    for j in "${!n_ctx[@]}"; do
        base_dir=trained_on_mvtecad
        save_dir=./checkpoints/${base_dir}/
        LOG=${save_dir}"res.log"
        echo ${LOG}


        checkpoint_path=C:/Users/Administrator/Desktop/CEP-AD/checkpoints/${base_dir}/epoch_${epoch}.pth

        CUDA_VISIBLE_DEVICES=${device} python C:/Users/Administrator/Desktop/CEP-AD/test.py --dataset btad \
        --data_path C:/Users/Administrator/Desktop/BTAD --save_path ./results/${base_dir}/zero_shot  --img_path imgs/headct \
        --checkpoint_path ${checkpoint_path} \
        --features_list 6 12 18 24 --image_size 518 --seed 111 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --metrics image-pixel-level
    done
  done
done

