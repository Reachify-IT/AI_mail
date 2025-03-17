from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import re
import asyncio
from .llm import (
    process_input, train_model, train_model_2, generate_response,
    extract_email_parts, process_video
)
import os
import shutil

router = APIRouter()

# ✅ Define request model
class RequestData(BaseModel):
    my_company: str
    my_designation: str
    my_name: str
    my_mail: str
    my_work: str
    my_cta_link: str
    client_name: str
    client_company: str
    client_designation: str
    client_website: str
    video_path: str


@router.post("/process-email")
async def process_email(data: RequestData):
    try:
        # ✅ Run these tasks concurrently
        client_about_website, client_website_issue= await asyncio.gather(
            process_input(data.client_website),  # ✅ Directly await the async function
            asyncio.to_thread(process_video, data.video_path, "local")  # ✅ Keep this since it's not async
        )


        system_prompt = train_model(
            data.my_company, data.my_designation, data.my_name, data.my_mail,
            data.my_work, data.client_name, data.client_company, 
            data.client_designation, data.client_website, 
            client_website_issue, client_about_website, data.video_path, data.my_cta_link
        )

        # ✅ Run LLM call in a separate thread
        # response = await asyncio.to_thread(generate_response, system_prompt)
        response = await generate_response(system_prompt)

        
        response = re.sub(r"\*\*", "", response)
        my_subject_text, email_body_text = extract_email_parts(response)
        email_body_text = re.sub(r"\[\s*Recipient\s*\]", f"{data.my_name}", email_body_text)
        print("\n=== client ===")
        print(client_about_website)
        print("\n=== client ===")
        print(client_about_website)
        print("\n=== emaillk ===")
        print(response)
        print("\n=== Email Subject ===")
        print(my_subject_text)
        print("\n=== Email Body ===")
        print(email_body_text)


        final_prompt = train_model_2(
            data.my_company, data.my_designation, data.my_name, data.my_mail,
            data.my_work, data.client_name, data.client_company,
            data.client_designation, data.client_website,
            client_website_issue, client_about_website, data.my_cta_link, email_body_text, data.video_path
        )

        # final_response = await asyncio.to_thread(generate_response, final_prompt)
        final_response = await generate_response(final_prompt)
        final_response = re.sub(r"\}\}", "}", re.sub(r"\{\{", "{", final_response))

        # ✅ Extract HTML content if available
        matches = re.findall(r"```(.*?)```", final_response, re.DOTALL)
        cleaned_html = re.sub(r"^.*?<\s*!DOCTYPE\s+html.*?>\s*", "", matches[0], flags=re.DOTALL | re.IGNORECASE) if matches else ""
        cleaned_html = re.sub(r"\\\*", "*", cleaned_html)

        if not re.search(r"<\s*html\b.*?>", cleaned_html, re.IGNORECASE | re.DOTALL):
            cleaned_html = ""

        return {
            "subject": my_subject_text,
            "cleaned_html": cleaned_html
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
