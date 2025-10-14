set -e

GPU=$1
NODE=$2
PORT=$3
CONFIG=$4

# Manipulation method. One of ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'].
METHOD=Deepfakes
COMPRESSION=c23
methods=("ALL")
# methods=("FF-ALL" "Celeb-DF" "DFDC"  FaceForensics_c23)
for scene in ${methods[@]}; do
	CUDA_VISIBLE_DEVICES=${GPU} \
        python -m torch.distributed.launch --nproc_per_node=${NODE} --master_port ${PORT} \
        final_image_test.py -c ${CONFIG} --method ${METHOD} --compression ${COMPRESSION} 
done
