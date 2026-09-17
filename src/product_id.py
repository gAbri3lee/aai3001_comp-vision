"""
On-demand currency/product identification stage (CLAUDE.md pipeline stage
2, secondary track, optional). Uses a generic ImageNet-pretrained
classifier as a placeholder — it recognizes general object categories,
not actual currency denominations, since no currency-specific dataset
exists yet. Swap in a fine-tuned classifier the same way the detector
gets swapped (see CLAUDE.md's model swap strategy) once one exists.
"""
import cv2
import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ProductIdentifier:
    def __init__(self):
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.model = mobilenet_v3_small(weights=weights)
        self.model.eval()
        self.transform = weights.transforms()
        self.categories = weights.meta["categories"]

    @torch.no_grad()
    def identify(self, frame_bgr, topk: int = 1):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        batch = self.transform(tensor).unsqueeze(0)
        probs = torch.nn.functional.softmax(self.model(batch)[0], dim=0)
        top_probs, top_idx = probs.topk(topk)
        return [(self.categories[i], float(p)) for p, i in zip(top_probs, top_idx)]
