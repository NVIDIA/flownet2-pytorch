USAGE EXAMPLE:

python main.py --skip_training --skip_validation --inference_dataset ImagesFromFolder --inference_dataset_root /home/tomrunia/data/RepMeasureDataset/videos_raw/images/  --inference_dataset_iext jpg --inference --save_flow --model FlowNet2 --resume /home/tomrunia/dev/lib/flownet2-pytorch/pretrained/FlowNet2_checkpoint.pth.tar --inference_batch_size 2
==> UPDATE: don't use the above instruction, but instead use flow_extractor.py
