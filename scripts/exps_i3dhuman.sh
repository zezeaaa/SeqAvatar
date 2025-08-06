GPU_id=0
SEQUENCES=(ID1_1 ID1_2 ID2_1 ID3_1)

# <your I3D-Human path>
data_path=mydata/I3D-Human/
iter=15000
densify_until_iter=1800

seq_len=8
seq_xyz_knn=8
time_step_num=3
max_time_step=42
minimal_time_step=24

l1_loss_w=1.0
ssim_loss_w=0.1
lpips_loss_w=0.1

for SEQUENCE in ${SEQUENCES[@]}; do
    exp_name=I3D-Human/${SEQUENCE}/
    dataset_path=$data_path/$SEQUENCE\-train/
    model_path=output/$exp_name
    mkdir -p "$model_path/logs"

    # Train
    echo "Training on GPU $GPU_id for sequence $SEQUENCE"
    CUDA_VISIBLE_DEVICES=$GPU_id python train.py -s $dataset_path --eval --exp_name $exp_name \
        --motion_offset_flag --smpl_type smpl --actor_gender neutral \
        --iterations $iter --densify_until_iter $densify_until_iter \
        --seq_len $seq_len --seq_xyz_knn $seq_xyz_knn \
        --time_step_num $time_step_num   --max_time_step $max_time_step  --minimal_time_step $minimal_time_step \
        --l1_loss_w $l1_loss_w --ssim_loss_w $ssim_loss_w --lpips_loss_w $lpips_loss_w \
        2>&1 | tee "$model_path/logs/train_${SEQUENCE}.log"

    # Evaluation
    echo "Evaluating on GPU $GPU_id for sequence $SEQUENCE"
    CUDA_VISIBLE_DEVICES=$GPU_id python render.py -s $dataset_path -m $model_path \
        --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration $iter --skip_train \
        --seq_len $seq_len  --seq_xyz_knn $seq_xyz_knn \
        --time_step_num $time_step_num   --max_time_step $max_time_step  --minimal_time_step $minimal_time_step \
        2>&1 | tee "$model_path/logs/render_${SEQUENCE}.log"
done