import torch
import argparse
from pytorchyolo.utils.utils import load_classes, print_environment_info
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.test import evaluate_model_file  # add parameter device

print_environment_info()
parser = argparse.ArgumentParser(description="Evaluate validation data.")
parser.add_argument("-m", "--model", type=str, default="config/yolov3-voc.cfg", help="Path to model definition file (.cfg)")
parser.add_argument("-w", "--weights", type=str, default="/home/xinrui/federated-yolov3/checkpoints/voc/yolov3_ckpt_60.pth", help="Path to weights or checkpoint file (.weights or .pth)")
parser.add_argument("-d", "--data", type=str, default="config/voc.data", help="Path to data config file (.data)")
parser.add_argument("-b", "--batch_size", type=int, default=8, help="Size of each image batch")
parser.add_argument("-v", "--verbose", action='store_true', help="Makes the validation more verbose")
parser.add_argument('--gpu', type=int, default=1, help="GPU ID, -1 for CPU")
parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
parser.add_argument("--iou_thres", type=float, default=0.5, help="IOU threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.1, help="Object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.5, help="IOU threshold for non-maximum suppression")
args = parser.parse_args()
print(f"Command line arguments: {args}")

# Load configuration from data file
data_config = parse_data_config(args.data)
# Path to file containing all images for validation
valid_path = data_config["valid"]
class_names = load_classes(data_config["names"])  # List of class names

args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

precision, recall, AP, f1, ap_class = evaluate_model_file(
    args.model,
    args.device,
    args.weights,
    valid_path,
    class_names,
    batch_size=args.batch_size,
    img_size=args.img_size,
    n_cpu=args.n_cpu,
    iou_thres=args.iou_thres,
    conf_thres=args.conf_thres,
    nms_thres=args.nms_thres,
    verbose=True)

print("precision: {:.10f}".format(precision.mean()))
print("recall: {:.10f}".format(recall.mean()))
print("mAP: {:.10f}".format(AP.mean()))
print("f1: {:.10f}".format(f1.mean()))