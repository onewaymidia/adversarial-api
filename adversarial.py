import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import sys
import os

# ══════════════════════════════════════════
IMAGEM_ENTRADA = "foto.jpg"
EPSILON = 0.03
# ══════════════════════════════════════════

if not os.path.exists(IMAGEM_ENTRADA):
    print(f"ERRO: arquivo '{IMAGEM_ENTRADA}' não encontrado.")
    print("Coloque sua imagem na pasta e renomeie para 'foto.jpg'")
    sys.exit(1)

print("Carregando modelo (primeira vez demora ~1 minuto)...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

print("Processando sua imagem...")
img_pil = Image.open(IMAGEM_ENTRADA).convert("RGB")
tamanho_original = img_pil.size

img_tensor = transform(img_pil).unsqueeze(0)
img_tensor.requires_grad = True

output = model(norm(img_tensor))
pred_antes = output.argmax(dim=1).item()

loss = torch.nn.CrossEntropyLoss()(output, output.argmax(dim=1))
loss.backward()

img_adv = torch.clamp(img_tensor + EPSILON * img_tensor.grad.sign(), 0, 1)

with torch.no_grad():
    pred_depois = model(norm(img_adv)).argmax(dim=1).item()

img_adv_pil = transforms.ToPILImage()(img_adv.squeeze(0))
img_adv_pil = img_adv_pil.resize(tamanho_original, Image.LANCZOS)
img_adv_pil.save("adversarial.png")

diff = torch.clamp(torch.abs(img_adv - img_tensor) * 15, 0, 1)
diff_pil = transforms.ToPILImage()(diff.squeeze(0))
diff_pil.save("diferenca_ampliada.png")

print()
print("══════════════════════════════")
print(f"  Pronto!")
print(f"  Classe ANTES:  {pred_antes}")
print(f"  Classe DEPOIS: {pred_depois}")
if pred_antes != pred_depois:
    print("  A IA foi enganada com sucesso!")
else:
    print("  Tente aumentar o EPSILON para 0.05 ou mais.")
print(f"  Arquivos gerados:")
print(f"    adversarial.png")
print(f"    diferenca_ampliada.png")
print("══════════════════════════════")
