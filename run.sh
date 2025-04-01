


#!/bin/bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# 运行 whu_STIL-KDNet 项目训练


# 运行第1个训练任务
echo "正在启动 levir_STIL-KDNet 项目训练..."
nohup python main_train.py --project_name levir_STIL-KDNet --data_name LEVIR --batch_size 1  > /dev/null 2>&1 &
pid1=$!
echo "levir_STIL-KDNet 项目已开始，进程 ID 为 $pid1"

# 等待第1个训练任务完成
wait $pid1
echo "levir_STIL-KDNet 项目训练已完成，正在启动 sysu_STIL-KDNet 项目训练..."

# 运行第2个训练任务
echo "正在启动 sysu_STIL-KDNet 项目训练..."
nohup python main_train.py --project_name sysu_STIL-KDNet --data_name SYSU --batch_size 1 > /dev/null 2>&1 &
pid2=$!
echo "sysu_STIL-KDNet 项目已开始，进程 ID 为 $pid2"

# 等待第2个训练任务完成
wait $pid2


# 运行第3个训练任务
echo "正在启动 whu_STIL-KDNet 项目训练..."
nohup python main_train.py --project_name whu_STIL-KDNet --data_name WHU --batch_size 1 > /dev/null 2>&1 &
pid3=$!
echo "whu_STIL-KDNet 项目已开始，进程 ID 为 $pid3"

echo "所有训练项目均已按顺序完成。"
