import time

import uvicorn
from fastapi import FastAPI, Depends
from starlette.requests import Request

app = FastAPI(title="My generator",
              summary="generator",
              description="Generates response")


async def get_body(request: Request):
    return await request.body()

@app.post("/generate/", tags=["Answer Generation"])
async def get_response(body: bytes = Depends(get_body)):
    print(body.decode("utf-8"))
    time.sleep(5)
    return body

if __name__=="__main__":
    uvicorn.run(host="127.0.0.1",port=8000,app=app)
