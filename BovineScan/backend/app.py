from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
import json

from backend.api.routes import router as api_router

app = FastAPI(title="Bovine Classifier")

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Templates
templates = Jinja2Templates(directory="frontend/templates")

# Include api
app.include_router(api_router, prefix="/api")

@app.get("/", response_class=HTMLResponse)
async def serve_spa(request: Request):
    # Pass gallery items to the template
    gallery_dir = "frontend/static/gallery"
    items = []
    if os.path.exists(gallery_dir):
        files = os.listdir(gallery_dir)
        for f in files:
            if f.endswith(('.jpg', '.png', '.jpeg')):
                breed_name = os.path.splitext(f)[0]
                items.append({
                    "name": breed_name,
                    "image": f"/static/gallery/{f}"
                })
        # Sort items for consistent display
        items = sorted(items, key=lambda x: x["name"])
        
    return templates.TemplateResponse(
        request=request,
        name="index.html", 
        context={"gallery_items": items}
    )
