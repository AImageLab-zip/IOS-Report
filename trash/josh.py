import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import torchvision.datasets.utils as data_utils
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ---------------------------
# Metric Approximations
# ---------------------------

def compute_true_metric(model, x, eta):
    """Compute the 'true' metric using model(x+eta)."""
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)  # (1, C)
        p, y = torch.max(probs, dim=1)    # scalar predicted prob and class index

        logits_t = model(x + eta)
        probs_t = F.softmax(logits_t, dim=1)
        p_t, y_t = torch.max(probs_t, dim=1)

        same_class_mask = (y_t == y).float()
        metric_true = torch.sqrt(p * p_t) * same_class_mask
    return metric_true.squeeze()


def jacobian_linear_approx(model, x, eta):
    """First-order (Jacobian) approximation for the metric."""
    if x.grad is not None:
        x.grad.zero_()

    logits = model(x)
    probs = F.softmax(logits, dim=1)
    p, y = torch.max(probs, dim=1)
    p_pred = probs[0, y]  # scalar prob for the predicted class

    # Compute gradient of that prob w.r.t. input
    p_pred.backward(retain_graph=True)
    grad = x.grad.detach()

    grad_term = torch.dot(grad.flatten(), eta.flatten())
    metric_approx = p_pred + 0.5 * grad_term  # small eta linearization
    return metric_approx.squeeze()


def fisher_empirical_approx(model, x, eta):
    """Second-order (empirical Fisher) approximation."""
    if x.grad is not None:
        x.grad.zero_()

    logits = model(x)
    probs = F.softmax(logits, dim=1)
    p, y = torch.max(probs, dim=1)
    p_pred = probs[0, y]

    # gradient of log-prob of predicted class
    log_p = torch.log(p_pred)
    log_p.backward(retain_graph=True)
    g = x.grad.detach()

    dir_dot = torch.dot(g.flatten(), eta.flatten())
    dir_sq = dir_dot ** 2

    fisher_correction = 0.5 * p_pred * dir_sq
    metric_fisher = p_pred + fisher_correction
    return metric_fisher.squeeze()

# ---------------------------
# Analysis with ResNet18
# ---------------------------

H, W = 224, 224
torch.manual_seed(0)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

url1 = "https://i.guim.co.uk/img/media/327aa3f0c3b8e40ab03b4ae80319064e401c6fbc/377_133_3542_2834/master/3542.jpg?width=1900&dpr=1&s=none&crop=none"
filename1 = "cat.jpg"
data_utils.download_url(url1, '.', filename=filename1, md5=None)
image = Image.open(filename1).convert('RGB')
x = transform(image).unsqueeze(0)

# resize
x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
x.requires_grad_(True)

sigma_values = np.logspace(-5, 0, 10)  # fewer points for speed
jacobian_approx_values, true_metric_values, fisher_metric_values = [], [], []

print("Analyzing with ResNet18...")

for sigma in sigma_values:
    torch.manual_seed(42)
    eta = torch.randn_like(x) * sigma

    metric_true = compute_true_metric(model, x, eta)
    metric_jac = jacobian_linear_approx(model, x, eta)
    metric_fisher = fisher_empirical_approx(model, x, eta)

    true_metric_values.append(metric_true.item())
    jacobian_approx_values.append(metric_jac.item())
    fisher_metric_values.append(metric_fisher.item())

    print(f"σ={sigma:.1e} | True={metric_true.item():.6f} | Jacobian={metric_jac.item():.6f} | Fisher={metric_fisher.item():.6f}")

print("\nAnalysis complete.")
