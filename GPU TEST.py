import torch
import torchvision


def main():
    print("PyTorch:", torch.__version__)
    print("TorchVision:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_properties(0))
        return 0
    print("未检测到可用 CUDA，请运行 CHECK_ENVIRONMENT.cmd 查看完整原因。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
