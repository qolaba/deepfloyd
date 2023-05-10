from huggingface_hub import login
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch
from fastapi import Security, Depends, FastAPI, HTTPException
from typing import List, Optional, Union
import io,os
from fastapi.responses import StreamingResponse
import uvicorn
import time
from fastapi.security.api_key import APIKeyQuery, APIKeyCookie, APIKeyHeader, APIKey
from starlette.status import HTTP_403_FORBIDDEN
from starlette.responses import RedirectResponse, JSONResponse
import zipfile

login("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")

stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
stage_1.enable_model_cpu_offload()

stage_2 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16)
stage_2.enable_model_cpu_offload()


API_KEY = ["1234567"]
API_KEY_NAME = "access_token"
COOKIE_DOMAIN = "localtest.me"

api_key_query = APIKeyQuery(name=API_KEY_NAME, auto_error=False)
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_cookie = APIKeyCookie(name=API_KEY_NAME, auto_error=False)



app = FastAPI()


async def get_api_key(api_key_query: str = Security(api_key_query),api_key_header: str = Security(api_key_header),api_key_cookie: str = Security(api_key_cookie)):
    if api_key_query in API_KEY:
        return api_key_query
    elif api_key_header in API_KEY:
        return api_key_header
    elif api_key_cookie in API_KEY:
        return api_key_cookie
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
        )

        
def get_image_buffer(image):
    img_buffer = io.BytesIO()
    image.save(img_buffer, 'PNG')
    img_buffer.seek(0)
    
    return img_buffer
def get_zip_buffer(list_of_tuples):
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for file_name, data in list_of_tuples:
            zip_file.writestr(file_name, data.read())
    zip_buffer.seek(0)
    return zip_buffer
    
@app.post("/getimage")
def get_image(
    #prompt: Union[str, List[str]],
    prompt: Optional[str] = "dog",
    ratio : Optional[str]= "1:1",
    hrns: Optional[int] = 30,
    imns: Optional[int] = 50,
    guidance_scale: Optional[float] = 7.5,
    negative_prompt: Optional[str] = "extra limbs",
    num_out:  Optional[int] = 1,
    api_key: APIKey = Depends(get_api_key)):
    
    if ratio == "1:1":
        height, width = 64, 64
    elif ratio == "3:2":
        height, width = 96, 64
    elif ratio == "2:1":
        height, width = 64, 32
    elif ratio == "2:3":
        height, width = 64,96
    elif ratio == "1:2":
        height, width = 32, 64
    else :
        height, width = 64, 64
        

    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt=[prompt]*num_out, negative_prompt=[negative_prompt]*num_out)
   
    image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt", num_inference_steps=imns, guidance_scale=guidance_scale, height=height, width=width).images

    image = stage_2(image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt", num_inference_steps=hrns, guidance_scale=guidance_scale).images
    image=pt_to_pil(image)    
    
    for i in range(0,len(image)):
        image[i]=image[i].resize((width*4,height*4))
        
    if(len(image)==1):
        filtered_image = io.BytesIO()
        image[0].save(filtered_image, "JPEG")
        filtered_image.seek(0)
        return StreamingResponse(filtered_image, media_type="image/jpeg")
    else:
        list_of_tuples=[]
        for i in range(0,len(image)):
            list_of_tuples.append((str(i)+'.png', get_image_buffer(image[i])))
        buff=get_zip_buffer(list_of_tuples) 
        response = StreamingResponse(buff, media_type="application/zip")
        response.headers["Content-Disposition"] = "attachment; filename=images.zip"
        return response




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)    
