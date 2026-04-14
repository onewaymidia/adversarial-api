import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from typing import List, Optional
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import io
import zipfile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

print("Carregando modelo ResNet50...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()
print("Modelo pronto!")

norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

def processar_imagem(contents, epsilon, cover_contents=None, cover_opacity=0.02):
    img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    tamanho_original = img_pil.size

    img_tensor = transform(img_pil).unsqueeze(0)
    img_tensor.requires_grad = True

    output = model(norm(img_tensor))
    loss = torch.nn.CrossEntropyLoss()(output, output.argmax(dim=1))
    loss.backward()

    img_adv = torch.clamp(img_tensor + (epsilon/100.0) * img_tensor.grad.sign(), 0, 1)
    img_adv_arr = img_adv.squeeze(0).permute(1,2,0).detach().numpy()

    if cover_contents:
        cover_pil = Image.open(io.BytesIO(cover_contents)).convert("RGB")
        cover_pil = cover_pil.resize((224,224), Image.LANCZOS)
        cover_arr = np.array(cover_pil, dtype=np.float32) / 255.0
        img_adv_arr = np.clip(img_adv_arr * (1 - cover_opacity) + cover_arr * cover_opacity, 0, 1)

    img_adv_pil = Image.fromarray((img_adv_arr * 255).astype(np.uint8))
    img_adv_pil = img_adv_pil.resize(tamanho_original, Image.LANCZOS)

    buf = io.BytesIO()
    img_adv_pil.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/adversarial")
async def gerar_adversarial(
    file: UploadFile = File(...),
    cover: Optional[UploadFile] = File(default=None),
    epsilon: float = Form(default=8.0),
    cover_opacity: float = Form(default=0.02)
):
    contents = await file.read()
    cover_contents = await cover.read() if cover else None
    resultado = processar_imagem(contents, epsilon, cover_contents, cover_opacity)
    return Response(content=resultado, media_type="image/png")

@app.post("/adversarial-lote")
async def gerar_adversarial_lote(
    files: List[UploadFile] = File(...),
    cover: Optional[UploadFile] = File(default=None),
    epsilon: float = Form(default=8.0),
    cover_opacity: float = Form(default=0.02),
    nome_zip: str = Form(default="imagens_camufladas")
):
    cover_contents = await cover.read() if cover else None
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in files:
            contents = await file.read()
            resultado = processar_imagem(contents, file.filename, cover_contents, cover_opacity)
            zip_file.writestr(f"camuflada_{file.filename}", resultado)
    zip_buffer.seek(0)
    return Response(
        content=zip_buffer.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={nome_zip}.zip"}
    )
