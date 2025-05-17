export OPENAI_LOGDIR=./sample_results  # 设置模型保存路径

# 设置模型参数
MODEL_FLAGS="--num_channels 128 --num_res_blocks 3"

# 设置图像尺寸（可以根据需要修改）
# 对于正方形图片：
# IMAGE_FLAGS="--image_size 64"
# 对于矩形图片：
IMAGE_FLAGS="--image_height 16 --image_width 64"

# 设置扩散模型参数
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"

# 设置训练参数
TRAIN_FLAGS="--batch_size 32"

# 运行采样脚本
python scripts/channel_sample.py \
    --model_path ./checkpoints/model010000.pt \
    $MODEL_FLAGS \
    $IMAGE_FLAGS \
    $DIFFUSION_FLAGS \
    $TRAIN_FLAGS 