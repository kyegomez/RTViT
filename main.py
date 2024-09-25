import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List
import timm
from torchvision import transforms
import matplotlib.pyplot as plt
from collections import deque
from loguru import logger
import os

# Type aliases
ImageType = np.ndarray
TensorType = torch.Tensor

# Define where to save the checkpoints
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,  checkpoint_path: str) -> None:
    """
    Save the model and optimizer state to a checkpoint file.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved at epoch to {checkpoint_path}")

def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str) -> int:
    """
    Load the model and optimizer state from a checkpoint file.
    Returns the epoch to resume from.
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        logger.info(f"Resumed training from checkpoint at epoch {epoch}")
        return epoch
    else:
        logger.info("No checkpoint found, starting training from scratch")
        return 0

def get_camera_feed(camera_index: int = 0) -> cv2.VideoCapture:
    """
    Access the camera feed.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Cannot open camera")
        raise IOError("Cannot open camera")
    logger.info("Camera opened successfully")
    return cap


def preprocess_frame(frame: ImageType) -> TensorType:
    """
    Preprocess the image frame for the model.
    """
    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225],  # ImageNet stds
            ),
        ]
    )
    tensor = preprocess(frame)
    return tensor.unsqueeze(0)  # Add batch dimension


def load_student_model(num_classes: int) -> nn.Module:
    """
    Load the student model (Swin Transformer) with the correct number of classes and additional improvements for faster learning.
    """
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=True,
        num_classes=num_classes,
        drop_rate=0.0,  # Disable dropout for faster learning
        drop_path_rate=0.1,  # Adjust drop path rate for better regularization
        global_pool='avg',  # Use average pooling for better feature aggregation
    )
    model.train()
    # Apply additional improvements
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()  # Freeze batch normalization layers for faster learning
    logger.info(
        f"Loaded student Swin Transformer model with num_classes={num_classes} and additional improvements for faster learning"
    )
    return model


def load_teacher_model(num_classes: int) -> nn.Module:
    """
    Load the teacher model (ConvNeXt Large) with the correct number of classes.
    """
    model = timm.create_model(
        "convnext_large", pretrained=True, num_classes=num_classes
    )
    model.eval()
    logger.info(
        f"Loaded teacher ConvNeXt model with num_classes={num_classes}"
    )
    return model


def get_loss_function() -> nn.Module:
    """
    Define the loss function for knowledge distillation.
    """
    return nn.KLDivLoss(reduction="batchmean")


def get_optimizer(
    model: nn.Module, learning_rate: float = 1e-4
) -> torch.optim.Optimizer:
    """
    Define the optimizer.
    """
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def plot_loss(loss_history: deque) -> None:
    """
    Plot the loss history in real-time.
    """
    plt.figure(1)
    plt.clf()
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Real-Time Training Loss")
    plt.legend()
    plt.pause(0.001)  # Pause to update the plot


def train_real_time(
    student_model: nn.Module,
    teacher_model: nn.Module,
    cap: cv2.VideoCapture,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    num_classes: int,
    class_names: List[str],
    checkpoint_path: str = "./checkpoint"
) -> None:
    """
    Train the student model in real-time using the camera feed, guided by the teacher model.
    """
    use_amp = (
        torch.cuda.is_available()
    )  # Use AMP only if CUDA is available

    if use_amp:
        scaler = (
            torch.cuda.amp.GradScaler()
        )  # For mixed-precision training
        logger.info(
            "Using torch.cuda.amp for mixed-precision training"
        )
    else:
        logger.info(
            "CUDA is not available. Running on CPU without AMP."
        )

    loss_history = deque(
        maxlen=100
    )  # Store recent loss values for plotting

    plt.ion()  # Turn on interactive mode for real-time plotting

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame")
                break

            # Preprocess the frame
            input_tensor = preprocess_frame(frame).to(device)

            # Get teacher model's output
            with torch.no_grad():
                teacher_output = teacher_model(input_tensor)
                teacher_probs = nn.functional.softmax(
                    teacher_output / 1.0, dim=1
                )  # Temperature=1.0

            # Forward pass
            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    student_output = student_model(input_tensor)
                    student_log_probs = nn.functional.log_softmax(
                        student_output / 1.0, dim=1
                    )  # Temperature=1.0
                    loss = loss_function(
                        student_log_probs, teacher_probs
                    )
                # Backward pass and optimization
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                student_output = student_model(input_tensor)
                student_log_probs = nn.functional.log_softmax(
                    student_output / 1.0, dim=1
                )  # Temperature=1.0
                loss = loss_function(student_log_probs, teacher_probs)
                # Backward pass and optimization
                loss.backward()
                optimizer.step()

            # Update loss history and plot
            loss_value = loss.item()
            loss_history.append(loss_value)
            plot_loss(loss_history)

            # Get predicted class
            _, predicted_idx = torch.max(student_output, dim=1)
            predicted_class = class_names[predicted_idx.item()]

            # Overlay predicted class and loss on frame
            cv2.putText(
                frame,
                f"Predicted: {predicted_class}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Loss: {loss_value:.4f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            # Display the frame
            cv2.imshow("Real-Time Training", frame)
            if cv2.waitKey(1) == ord("q"):
                logger.info("Exit signal received")
                break

            # Logging
            logger.info(
                f"Loss: {loss_value}, Predicted: {predicted_class}"
            )
            try:
                # Auto-save the model after every 'save_interval' iterations
                # if epoch % save_interval == 0:
                save_checkpoint(student_model, optimizer, checkpoint_path=checkpoint_path)
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

    except Exception:
        logger.exception("An error occurred during training")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.close()
        logger.info(
            "Released camera, closed plots, and destroyed all windows"
        )


def main() -> None:
    """
    Main function to set up and start the real-time training.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Define your classes
    import requests
    class_names = requests.get('https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json').json()
    print(class_names)
    num_classes = len(class_names)

    student_model = load_student_model(num_classes=num_classes).to(
        device
    )
    teacher_model = load_teacher_model(num_classes=num_classes).to(
        device
    )

    optimizer = get_optimizer(student_model)
    loss_function = get_loss_function()
    cap = get_camera_feed()

    train_real_time(
        student_model,
        teacher_model,
        cap,
        optimizer,
        loss_function,
        device,
        num_classes,
        class_names,
    )


if __name__ == "__main__":
    main()