python saliency_map_pointnet.py --gpu 1 --dump_dir='dump/low25_truelabel' --total_droppoints 25 --lowdrop
python saliency_map_pointnet.py --gpu 1 --dump_dir='dump/low50_truelabel' --total_droppoints 50 --lowdrop
python saliency_map_pointnet.py --gpu 1 --dump_dir='dump/low75_truelabel' --total_droppoints 75 --lowdrop
python saliency_map_pointnet.py --gpu 1 --dump_dir='dump/low100_truelabel' --total_droppoints 100 --lowdrop
python saliency_map_pointnet.py --gpu 1 --dump_dir='dump/low150_truelabel' --total_droppoints 150 --lowdrop
python saliency_map_pointnet.py --gpu 1 --dump_dir='dump/low200_truelabel' --total_droppoints 200 --lowdrop

python saliency_map_pointnet.py --gpu 1 --dump_dir='dump/low25_predlabel' --total_droppoints 25 --lowdrop --pred_labels
python saliency_map_pointnet.py --gpu 1 --dump_dir='dump/low50_predlabel' --total_droppoints 50 --lowdrop --pred_labels
python saliency_map_pointnet.py --gpu 1 --dump_dir='dump/low75_predlabel' --total_droppoints 75 --lowdrop --pred_labels
python saliency_map_pointnet.py --gpu 1 --dump_dir='dump/low100_predlabel' --total_droppoints 100 --lowdrop --pred_labels
python saliency_map_pointnet.py --gpu 1 --dump_dir='dump/low150_predlabel' --total_droppoints 150 --lowdrop --pred_labels
python saliency_map_pointnet.py --gpu 1 --dump_dir='dump/low200_predlabel' --total_droppoints 200 --lowdrop --pred_labels

