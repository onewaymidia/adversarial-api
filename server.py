import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from typing import List
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
import zipfile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Carregando modelo...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.eval()
print("Modelo pronto!")

norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

def processar_imagem(contents, epsilon):
    img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    tamanho_original = img_pil.size
    img_tensor = transform(img_pil).unsqueeze(0)
    img_tensor.requires_grad = True
    output = model(norm(img_tensor))
    loss = torch.nn.CrossEntropyLoss()(output, output.argmax(dim=1))
    loss.backward()
    img_adv = torch.clamp(img_tensor + epsilon * img_tensor.grad.sign(), 0, 1)
    img_adv_pil = transforms.ToPILImage()(img_adv.squeeze(0))
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
    epsilon: float = Form(default=0.03)
):
    contents = await file.read()
    resultado = processar_imagem(contents, epsilon)
    return Response(content=resultado, media_type="image/png")

@app.post("/adversarial-lote")
async def gerar_adversarial_lote(
    files: List[UploadFile] = File(...),
    epsilon: float = Form(default=0.03),
    nome_zip: str = Form(default="imagens_camufladas")
):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in files:
            contents = await file.read()
            resultado = processar_imagem(contents, epsilon)
            nome_arquivo = f"camuflada_{file.filename}"
            zip_file.writestr(nome_arquivo, resultado)
    zip_buffer.seek(0)
    return Response(
        content=zip_buffer.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={nome_zip}.zip"}
    )
