from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
from datetime import datetime
from pathlib import Path

app = FastAPI(title="Memory App Backend")

# Ensure uploads directory exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "Memory App Backend is running",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    Receive audio file uploads from iOS app
    """
    try:
        # Validate file type
        if not file.filename.endswith('.m4a'):
            raise HTTPException(status_code=400, detail="Only .m4a files are accepted")
        
        # Save file
        file_path = UPLOAD_DIR / file.filename
        
        # Read and write file
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Get file size
        file_size = len(contents)
        
        print(f"✓ Received: {file.filename} ({file_size} bytes)")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "filename": file.filename,
                "size": file_size,
                "message": "File uploaded successfully"
            }
        )
        
    except Exception as e:
        print(f"✗ Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/files")
async def list_files():
    """
    List all uploaded files
    """
    try:
        files = []
        for file_path in UPLOAD_DIR.glob("*.m4a"):
            stat = file_path.stat()
            files.append({
                "filename": file_path.name,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
            })
        
        files.sort(key=lambda x: x['created'], reverse=True)
        
        return {
            "count": len(files),
            "files": files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)