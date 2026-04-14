from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from typing import List
from PIL import Image
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

print("Servidor pronto!")

def processar_imagem(contents, epsilon):
    img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    tamanho_original = img_pil.size
    img_arr = np.array(img_pil, dtype=np.float32) / 255.0
    gray = 0.299*img_arr[:,:,0] + 0.587*img_arr[:,:,1] + 0.114*img_arr[:,:,2]
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    sinal = np.sign(gx + gy)
    sinal_3ch = np.stack([sinal]*3, axis=2)
    img_adv = np.clip(img_arr + (epsilon/100.0) * sinal_3ch, 0, 1)
    img_adv_pil = Image.fromarray((img_adv * 255).astype(np.uint8))
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
    epsilon: float = Form(default=8.0)
):
    contents = await file.read()
    resultado = processar_imagem(contents, epsilon)
    return Response(content=resultado, media_type="image/png")

@app.post("/adversarial-lote")
async def gerar_adversarial_lote(
    files: List[UploadFile] = File(...),
    epsilon: float = Form(default=8.0),
    nome_zip: str = Form(default="imagens_camufladas")
):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in files:
            contents = await file.read()
            resultado = processar_imagem(contents, epsilon)
            zip_file.writestr(f"camuflada_{file.filename}", resultado)
    zip_buffer.seek(0)
    return Response(
        content=zip_buffer.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={nome_zip}.zip"}
    )
