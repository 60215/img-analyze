from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import uuid
import json
import os
from img_tagger import get_image_tags

app = FastAPI()

os.makedirs("storage/images", exist_ok=True)
os.makedirs("storage/metadata", exist_ok=True)

@app.post("/upload/")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()

    image_id = str(uuid.uuid4())
    ext = file.filename.split(".")[-1]
    image_path = f"storage/images/{image_id}.{ext}"

    with open(image_path, "wb") as f:
        f.write(contents)

    vibe_tags = get_image_tags(contents, is_vibe_tags=True)
    object_tags = get_image_tags(contents, is_vibe_tags=False)

    metadata = {
        "id": image_id,
        "filename": file.filename,
        "vibe_tags": vibe_tags,
        "object_tags": object_tags
    }

    with open(f"storage/metadata/{image_id}.json", "w") as f:
        json.dump(metadata, f)

    return {"message": "Image successfully processed!", "id": image_id}


@app.get("/images/")
async def list_images():
    imgs = []

    for filename in os.listdir("storage/metadata"):
        with open(os.path.join("storage/metadata", filename), "r") as f:
            meta = json.load(f)
            imgs.append({
                "id": meta["id"],
                "filename": meta["filename"]
            })
    return imgs

@app.get("/analyze/{image_id}")
async def get_analysis(image_id: str):
    meta_path = f"storage/metadata/{image_id}.json"
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Analysis not found")
    with open(meta_path, "r") as f:
        return json.load(f)


@app.get("/image/{image_id}")
async def get_image(image_id: str):
    for ext in ["jpg", "jpeg", "png"]:
        path = f"storage/images/{image_id}.{ext}"
        if os.path.exists(path):
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)