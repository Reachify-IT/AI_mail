from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import re
from .llm import (
    process_input, train_model, train_model_2, generate_response,
    extract_email_parts, process_video
)

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
def process_email(data: RequestData):
    try:
        # ✅ Extract website information
        client_about_website = process_input(data.client_website)
        client_website_issue = process_video(data.video_path, "local")

        # ✅ Generate the first model response
        system_prompt = train_model(
            data.my_company, data.my_designation, data.my_name, data.my_mail,
            data.my_work, data.client_name, data.client_company, 
            data.client_designation, data.client_website, 
            client_website_issue, client_about_website
        )
        response = re.sub(r"\*\*", "", generate_response(system_prompt))
        my_subject_text, my_body_text = extract_email_parts(response)

        # ✅ Generate the final response
        final_prompt = train_model_2(
            data.my_company, data.my_designation, data.my_name, data.my_mail,
            data.my_work, data.client_name, data.client_company,
            data.client_designation, data.client_website,
            client_website_issue, client_about_website, data.my_cta_link, my_body_text, data.video_path
        )
        final_response = re.sub(r"\}\}", "}", re.sub(r"\{\{", "{", generate_response(final_prompt)))

        
        # ✅ Extract HTML content if available
        matches = re.findall(r"```(.*?)```", final_response, re.DOTALL)
        cleaned_html = re.sub(r"^.*?<\s*!DOCTYPE\s+html.*?>\s*", "", matches[0], flags=re.DOTALL | re.IGNORECASE) if matches else ""

        cleaned_html = re.sub(r"\\\*", "*", cleaned_html)

        if not re.search(r"<\s*html\b.*?>", cleaned_html, re.IGNORECASE | re.DOTALL):
            cleaned_html = ""

        # print(cleaned_html)

        return {
            "subject": my_subject_text,
            "cleaned_html": cleaned_html
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
