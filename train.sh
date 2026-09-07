device=0

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
        base_dir=trained_on_mvtecad
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python C:/Users/Administrator/Desktop/CEP-AD/train.py --dataset mvtecad --train_data_path C:/Users/Administrator/Desktop/MVTecAD \
        --save_path ${save_dir} \
        --features_list 6 12 18 24 --image_size 518  --batch_size 8 --print_freq 1 \
        --epoch 25 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
    wait
    done
done

