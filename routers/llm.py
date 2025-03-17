## import library
import os
import re
import shutil
import ollama
import requests
import tempfile
import asyncio
import chromadb
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from langchain_community import embeddings
# from langchain_community.llms import Ollama
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings

import requests
import time
import whisper
# from moviepy import *
# from moviepy.editor import VideoFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip

# os.environ["OLLAMA_HOST"] = "http://172.31.46.239:11434"

# ollama_model = ollama.Ollama(model="llama3")
CHROMA_DB_PATH = "./chromaa_db"
app = FastAPI()


# Add CORS support
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Change this to specific domains for security
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# Define request model
# class RequestData(BaseModel):
#     my_company: str
#     my_designation: str
#     my_name: str
#     my_mail: str
#     my_work: str
#     my_cta_link: str
#     client_name: str
#     client_company: str
#     client_designation: str
#     client_website: str
#     client_website_issue: str

def reset_chroma():
    try:
        # ✅ Delete ChromaDB directory if it exists
        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH)
            print("✅ ChromaDB directory deleted successfully.")

        # ✅ Wait before initializing a new database
        time.sleep(2)

        # ✅ Reinitialize ChromaDB
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        client.get_or_create_collection("rag-chroma")

    except Exception as e:
        print(f"⚠️ ChromaDB Reset Error: {e}")
reset_chroma()  # Call function before initializing the database








def download_video(url):
    """Download video from a direct URL and save it locally."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        with open(temp_file.name, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        return temp_file.name
    else:
        raise Exception(f"Failed to download video from URL: {url}")

# def extract_audio(video_path, audio_path="temp_audio.wav"):
#     clip = VideoFileClip(video_path)
#     if clip.audio is None:
#         raise ValueError("No audio found in the video file. Please upload a valid video with sound.")

#     clip.audio.write_audiofile(audio_path, codec="pcm_s16le")
#     return audio_path


import uuid

def extract_audio(video_path):
    unique_id = uuid.uuid4().hex  # Generate a unique filename
    audio_path = f"temp_audio_{unique_id}.wav"
    
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        raise ValueError("No audio found in the video file. Please upload a valid video with sound.")
    
    clip.audio.write_audiofile(audio_path, codec="pcm_s16le")
    return audio_path


def transcribe_audio(audio_path, model_size="base"):  # Change model to "medium"
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language="en")
    return result["text"]


def process_video(video_source, source_type="local", model_size="base"):
    
    video_path = download_video(video_source)
    # video_path = video_source  # Local file

    audio_path = extract_audio(video_path)
    # st.audio(audio_path, format="audio/wav")
    text = transcribe_audio(audio_path, model_size)

    # Cleanup
    os.remove(audio_path)
    os.remove(video_path)

    return text








# ✅ Use a global Ollama model instance
async def generate_response(prompt):
    response = await asyncio.to_thread(
        ollama.chat,
        model="llama3",
        messages=[{"role": "system", "content": prompt}]
    )
    return response["message"]["content"]

async def process_input(urls):
    try:
        urls_list = [url.strip() for url in urls.split("\n") if url.strip()]
        if not urls_list:
            return "Error: No valid URLs provided."

        # ✅ Load documents concurrently
        async def load_document(url):
            return await asyncio.to_thread(WebBaseLoader(url).load)

        doc_tasks = [load_document(url) for url in urls_list]
        docs = await asyncio.gather(*doc_tasks)
        docs_list = [item for sublist in docs for item in sublist]

        # ✅ Split text
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=10000, chunk_overlap=100)
        doc_splits = await asyncio.to_thread(text_splitter.split_documents, docs_list)
    
        # ✅ Store in ChromaDB
        vectorstore = await asyncio.to_thread(
            Chroma.from_documents,
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=OllamaEmbeddings(model='nomic-embed-text'),
            persist_directory=CHROMA_DB_PATH
        )
        retriever = vectorstore.as_retriever()

        # ✅ Retrieve relevant content
        retrieved_docs = await asyncio.to_thread(retriever.invoke, "Summarize this website")
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # ✅ Enhanced RAG prompt
        summary_template = """
        Summarize the website by describing:
        - What the website is about
        - What the website does
        
        Context:
        {context}
        """
        summary_prompt = ChatPromptTemplate.from_template(summary_template)
        response_prompt = summary_prompt.format(context=context)

        # ✅ Generate summary
        response = await generate_response(response_prompt)

        return response

    except Exception as e:
        return f"Error: {str(e)}"




def extract_email_parts(email_text):
    if not email_text:
        return "No Subject Found", "No Body Found"

    # Extract subject
    subject_pattern = r"(?i)^Subject:\s*(.+)"
    subject_match = re.search(subject_pattern, email_text, re.MULTILINE)
    subject_text = subject_match.group(1).strip() if subject_match else "Not found"

    # Extract body (everything after subject line)
    body_pattern = r"(?i)^Subject:.*?\n([\s\S]+)"
    body_match = re.search(body_pattern, email_text, re.MULTILINE)
    body_text = body_match.group(1).strip() if body_match else "No body found"

    return subject_text, body_text



# system_prompt


def train_model(my_company, my_designation, my_name, my_mail, my_work, client_name, client_company, client_designation,
                client_website, client_website_issue, client_about_website, video_path, my_cta_link):
    system_prompt = f"""You are an expert cold email writer at {my_company}. Your task is to generate a NEW, ORIGINAL email (subject line + body) that is FULLY PERSONALIZED using ONLY the provided input variables.

### INPUT VARIABLES (THESE ARE THE ONLY VALUES YOU SHOULD USE):
- SENDER: {my_name} = "[provided name]"
- SENDER: {my_designation} = "[provided designation]"
- SENDER: {my_company} = "[provided company]"
- SENDER: {my_mail} = "[provided email]"
- SENDER: {my_work} = "[provided work type]"
- SENDER: {my_cta_link} = "[provided CTA link]"
- RECIPIENT: {client_name} = "[provided client name]"
- RECIPIENT: {client_company} = "[provided client company]"
- RECIPIENT: {client_designation} = "[provided client designation]"
- RECIPIENT: {client_website} = "[provided client website]"
- RECIPIENT: {video_path} = "[provided video path]"
- RECIPIENT: {client_website_issue} = "[provided client website issues]"
- RECIPIENT: {client_about_website} = "[provided summary over client website]"

### IMPORTANT RULES:
1. DO NOT USE ANY EXAMPLE CONTENT - create completely original text
2. DO NOT MENTION "BizFlow", "CRM" or any other business not specified in the input
3. ONLY reference the specific client company and website provided in the input
4. ONLY offer services related to the specific {my_work} value provided
5. NEVER copy text from examples - all content must be new and original

### EMAIL STRUCTURE:
1. SUBJECT LINE: Attention-grabbing, specific to {client_company}'s actual business

2. GREETING: Address {client_name} directly

3. OPENING: 
   - Reference {client_company} and {client_website} specifically
   - Mention you've created a video analysis about their website

4. ISSUES (3-4 bullet points):
   - Highlight specific website issues related to {my_work}
   - Ensure they're relevant to the client's actual business
   - Ensure some specific issues like {client_website_issue}

5. VALUE PROPOSITION:
   - Explain how {my_company} and {my_work} can solve their problems
   - Mention the video you've created at {video_path}

6. CALL-TO-ACTION:
   - Invite them to watch the video at {video_path}
   - Suggest using {my_cta_link} to schedule a call

7. SIGNATURE:
   - {my_name}
   - {my_designation}, {my_company}
   - {my_mail}

### FINAL CHECK:
- Verify your email ONLY mentions the specific client company provided
- Confirm you're ONLY offering services related to the provided {my_work}
- Ensure all content is completely original and not copied from examples

### FORMAT YOUR OUTPUT EXACTLY LIKE THIS:

**Subject:** [Your subject line here]

Hey {client_name},

[Your opening paragraph showing personalization and research]

Here's what I noticed:
✅ [Specific issue #1 with business impact]
✅ [Specific issue #2 with business impact]
✅ [Specific issue #3 with business impact]
✅ [Optional issue #4 with business impact]

[Value proposition paragraph explaining how you can help]

[Clear, casual call-to-action]

Best,
{my_name}
{my_designation}, {my_company}
{my_mail}


---
### **Example 1:

- **Issue Identified**: Content issues, navigation problems, slow performance. 
- **Your Work**: Web developer offering improvements. 


**Generated Email:** 

**Subject:** Your Website Deserves Better—Here’s How

Hey {client_name}, 

I just checked out your website, and while it looks great, I noticed a few tweaks that could **seriously boost user experience and conversions**. Think of it like giving your site a little **makeover for speed, clarity, and engagement**—small changes, BIG impact. 

Here’s what caught my eye: 
✅ **Navigation could be smoother**—users might be getting lost. 
✅ **Speed issues**—slow pages = lost visitors (and revenue). 
✅ **Design consistency**—fonts and spacing could use a little love. 
✅ **Contact info isn’t super clear**—you want leads to reach you fast! 

I’d love to help refine these areas so your site works **as hard as you do**. If you’re open to it, let’s chat—I can show you quick, actionable fixes! 

Let me know your thoughts. 

**Best,**


{my_name} 
{my_designation}, {my_company} 
{my_mail}


---
### **Example 2:

- **Issue Identified**: Low website traffic, weak SEO strategy. 
- **Your Work**: SEO specialist optimizing search rankings. 

**Generated Email:** 

**Subject:** Your Website Deserves More Traffic (Let’s Fix That!) 


Hey {client_name}, 


I came across your website and noticed it’s **got huge potential**—but it looks like Google isn’t showing it enough love. Right now, you might be **missing out on tons of free organic traffic** simply because of a few SEO blind spots. 


Here’s what I spotted: 
🔍 **Keyword gaps**—your competitors are ranking for terms you should own. 
⚡ **On-page SEO**—small tweaks to meta tags and headers could boost visibility. 
🚀 **Slow load times**—Google hates slow sites (and so do users). 


Good news? These are **easy fixes**, and I’d love to help. Let’s chat about how I can get **more eyes (and leads!) on your site.** 


Think it’s worth a quick call? Let me know! 


**Best,**


{my_name} 
{my_designation}, {my_company} 
{my_mail}


---
### **Example 3:

- **Client Company**: {client_company} 
- **Issue Identified**: Low user retention, slow app performance. 
- **Your Work**: Mobile app developer fixing performance issues. 


**Generated Email:** 

**Subject:** Your App Shouldn’t Be Losing Users—Let’s Fix That 


Hey {client_name}, 


I love what you’re building with {client_company}! A **fitness app that motivates users? Genius.** But I noticed something that might be holding you back—users aren’t sticking around, and I think I know why. 


Common app pain points I see: 
📉 **Performance issues**—slow load times make users bounce. 
🖌 **UI tweaks**—a smoother design could improve user experience. 
📊 **Engagement features**—gamification & push notifications can boost retention. 


I specialize in **making apps faster, smoother, and stickier** so users keep coming back. If you’re open to it, let’s chat—I’ve got a few ideas that could make a big difference. 


What do you think? 


**Best,** 


{my_name} 
{my_designation}, {my_company} 
{my_mail}




---
### **Example 4:

- **Client Company**: {client_company} 
- **Issue Identified**: Low engagement on social media ads. 
- **Your Work**: Social media ads expert improving conversions. 


**Generated Email:** 

**Subject:** Let’s Make Your Social Ads Work 10x Harder


Hey {client_name}, 


I love your brand—your meal service looks **delicious AND convenient**! But I noticed your social ads **aren’t getting the engagement they deserve** (which means wasted ad spend). 


What’s likely happening: 
❌ **Audience mismatch**—your ads might be showing to the wrong people. 
❌ **Creative fatigue**—same visuals = lower click-through rates. 
❌ **Landing page disconnect**—are users dropping off after clicking? 


The good news? **I fix these problems for a living.** Let’s fine-tune your ad targeting, refresh your creatives, and optimize your funnels so you get **more conversions for the same budget**. 


Interested? Let’s chat—I’d love to help. 


**Best,**


{my_name} 
{my_designation}, {my_company} 
{my_mail}


---
### **Example 5:

- **Client Company**: {client_company}
- **Issue Identified**: Outdated CRM software, user complaints. 
- **Your Work**: Software developer upgrading outdated systems. 


**Generated Email:** 

**Subject:** Time for a CRM Upgrade? Let’s Talk


Hey {client_name}, 


I know how frustrating it can be when your CRM **starts slowing things down instead of speeding them up**. I took a look at {client_company}, and I think a few strategic upgrades could **massively improve efficiency and user experience.** 


Some quick wins we could tackle: 
🛠 **Bug fixes & performance boosts**—say goodbye to glitches. 
🚀 **Feature enhancements**—automation tools to streamline workflow. 
📈 **UI/UX improvements**—modern design = happier users. 


I’ve helped other businesses **upgrade without disrupting operations**, and I’d love to do the same for {client_company}. Let’s talk? 


**Best,**


{my_name} 
{my_designation}, {my_company} 
{my_mail}

"""

    # print(system_prompt)

    return system_prompt


def train_model_2(my_company, my_designation, my_name, my_mail, my_work, client_name, client_company,
                  client_designation, client_website, client_website_issue, client_about_website,
                  my_cta_link, email_body_text, video_path):
    system_prompt_1 = f"""You are an expert HTML email designer specializing in creating professional, responsive emails for business outreach. Your task is to convert the provided text into a polished HTML email that effectively utilizes all client and sender information.

### INPUT VARIABLES:
- SENDER: {my_name}, {my_designation}, {my_company}, {my_mail}, {my_work}
- RECIPIENT: {client_name}, {client_designation}, {client_company}, {client_website}
- CONTENT: {email_body_text}, {client_website_issue}, {client_about_website}
- ACTION ELEMENTS: {my_cta_link}, {video_path}

### REQUIREMENTS:
1. CREATE A FULLY RESPONSIVE HTML EMAIL that:
   - Maintains the original message intent in {email_body_text} without omitting or altering key details.
   - Addresses specific issues mentioned in {client_website_issue}
   - Incorporates insights from {client_about_website}
   - Presents a professional appearance aligned with {my_work} industry standards

2. PERSONALIZATION:
   - Directly address {client_name} and reference {client_company} naturally
   - Reference {client_website} specifically when discussing improvements
   - Tailor content to match the sender's expertise ({my_work})
   - Ensure all sender details appear in a professional signature

3. VISUAL STRUCTURE:
   - Use clear headings, concise paragraphs, and bullet points for readability
   - Include properly styled buttons for {my_cta_link} and {video_path}
   - Maintain consistent spacing, fonts, and color scheme throughout
   - Optimize for both desktop and mobile viewing

4. TECHNICAL REQUIREMENTS:
   - Include all necessary HTML structure (DOCTYPE, head, body)
   - Use inline CSS for maximum email client compatibility
   - Ensure all links are properly formatted and functioning
   - Create a professional signature block with all sender details

5. AVOID:
   - Omitting or modifying key details from {email_body_text}.
   - Complex HTML elements that may break in email clients
   - Unnecessary images or large file elements
   - Generic copy that doesn't utilize the specific variables provided
   - Outdated design patterns or poor mobile responsiveness

OUTPUT: Provide a complete, ready-to-use HTML email that seamlessly integrates all variables and meets professional standards for business communication.

    ### **Output Format:**
    Generate a **fully formatted HTML email** with External styles, ensuring `{email_body_text}` remains **unaltered** while being properly structured for readability. The footer should be correctly generated at end  contain my_name, my_designation, my_company and my email if provided.

    ---
** Use the below html templates as example and generate a customized html email according to the body text given also make sure to customize the email content according to "About Website Analysis":


### **💡 Example 1: 

```html

<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            width: 100%;
            max-width: 600px;
            margin: auto;
            padding: 20px;
        }}
        .btn {{
            display: inline-block;
            background-color: #007bff;
            color: white !important;
            padding: 12px 20px;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Hi {client_name},</h1>
        <p>I recently checked out your website, and while it’s already impressive, there are some areas that could be optimized for a <strong>better user experience and conversions<strong>.</p>

        <h2>Key Issues Noticed:</h2>
        <ul>
            <li><strong>Navigation challenges</strong> – Users might find it hard to move around.</li>
            <li><strong>Performance issues</strong> – Slow loading times could impact engagement.</li>
            <li><strong>Design inconsistencies</strong> – A few areas could be refined for better branding.</li>
            <li><strong>Contact info placement</strong> – Making it clearer could boost leads.</li>
        </ul>

        <p>I've created a short video explaining the possible improvements. You can watch it below:</p>

        <div style="text-align: center;">
            
            <p><a href="{video_path}" class="btn">🎥 Watch Video</a></p>
        </div>

        <p>I'd love to collaborate with you and help improve these aspects. Let’s explore some <strong>quick and actionable solutions</strong> tailored for {client_company}.</p>

        <div style="text-align: center;">
        <p><a href="{my_cta_link}" class="btn">Let’s Discuss the Fixes</a></p>
        </div>
        <p>Looking forward to your thoughts!</p>

        <p>Best,<br>{my_name} <br> {my_designation} <br> {my_company} <br> <a href="mailto:{my_mail}">{my_mail}</a></p>
    </div>
</body>
</html>
```

---


### **💡 Example 2: 

```html
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            color: #444;
            background-color: #f4f4f4;
            padding: 20px;
        }}
        .container {{
            max-width: 600px;
            background: white !important;
            padding: 30px;
            margin: auto;
            border-radius: 10px;
       
        }}
        .btn {{
            display: block;
            text-align: center;
            background-color: #007bff;
            color: white !important;
            padding: 14px;
            margin-top: 20px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h2>🚀 Let’s Optimize Your Website, {client_name}!</h2>
        <p>Hey {client_name}, I took a look at <strong>{client_company}’s website</strong>, and I see great potential! A few targeted tweaks could <strong>significantly improve user experience and engagement</strong>.</p>

        <h3>Quick Wins We Can Implement:</h3>
        <ul>
            <li>📌 <strong>Smoother navigation</strong> to improve user flow</li>
            <li>⚡ <strong>Speed optimization</strong> to reduce page load time</li>
            <li>🎨 <strong>Refined design elements</strong> for brand consistency</li>
            <li>📞 <strong>Better contact placement</strong> to increase conversions</li>
        </ul>

        <p>I've created a short video explaining the possible improvements. You can watch it below:</p>

        <div style="text-align: center;">
            
            <p><a href="{video_path}" class="btn">🎥 Watch Video</a></p>
        </div>

        <p>I’d love to share some quick strategies that can <strong>deliver results without disrupting your current setup.</strong></p>

        <a href="{my_cta_link}" class="btn">Let’s Chat About It</a>

        <p>Looking forward to your thoughts!</p>

        <p>Best, <br><br> {my_name} <br> {my_designation} <br> {my_company} <br> <a href="mailto:{my_mail}">{my_mail}</a></p>
    </div>
</body>
</html>
```

---


### **💡 Example 3: 

```html

<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            background-color: #222;
            color: white !important;
            padding: 20px;
        }}
        .container {{
            max-width: 600px;
            background: #333;
            padding: 30px;
            margin: auto;
            border-radius: 10px;
    
        }}
        .btn {{
            display: block;
            text-align: center;
            background-color: #007bff;
            color: white !important;
            padding: 14px;
            margin-top: 20px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h2>🚀 Time to Supercharge Your Website, {client_name}!</h2>
        <p>Hey {client_name}, I was checking out <strong>{client_company}</strong>’s site, and I noticed some easy <strong>performance and UX improvements</strong> that could take your brand to the next level.</p>

        <h3>Here’s What We Can Optimize:</h3>
        <ul>
            <li>💡 <strong>Better Navigation</strong> – Ensure a seamless experience</li>
            <li>⚡ <strong>Faster Load Times</strong> – Speed = higher engagement</li>
            <li>🎨 <strong>Sleek & Modern Design Enhancements</strong></li>
            <li>📞 <strong>Contact Form Fixes</strong> – Make it easier for leads to reach you</li>
        </ul>

        
        <p>I've created a short video explaining the possible improvements. You can watch it below:</p>

        <div style="text-align: center;">
            
            <p><a href="{video_path}" class="btn">🎥 Watch Video</a></p>
        </div>


        <p>Let’s make <strong>small changes for big results</strong>! I’d love to share how we can get started.</p>

        <a href="{my_cta_link}" class="btn">Let’s Optimize Together</a>

        <p>Best, <br><br> {my_name} <br> {my_designation} <br> {my_company} <br> <a href="mailto:{my_mail}">{my_mail}</a></p>
    </div>
</body>
</html>
```

---


### **💡 Example 4: 

```html

<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: 'Helvetica', sans-serif;
            color: #222;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        .container {{
            max-width: 600px;
            background: white !important;
            padding: 30px;
            margin: auto;
            border-radius: 10px;

        }}
        .btn {{
            display: block;
            text-align: center;
            background-color: #007bff;
            color: white !important;
            padding: 14px;
            margin-top: 20px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h2>Hi {client_name}, Let’s Elevate {client_company}’s Website! 🎯</h2>
        <p>Your website has great potential, but a few refinements could significantly <strong>enhance performance and user experience.</strong></p>

        <h3>Key Enhancements:</h3>
        <ul>
            <li>✔️ Faster load times for <strong>better engagement</strong></li>
            <li>✔️ Enhanced design consistency <strong>for brand trust</strong></li>
            <li>✔️ Improved contact forms <strong>for more leads</strong></li>
        </ul>

        
        <p>I've created a short video explaining the possible improvements. You can watch it below:</p>

        <div style="text-align: center;">
            
            <p><a href="{video_path}" class="btn">🎥 Watch Video</a></p>
        </div>

        <p>Let’s chat about <strong>simple, high impact changes</strong> that can help {client_company} thrive online.</p>

        

        <a href="{my_cta_link}" class="btn">Let’s Connect & Improve</a>

        <p>Best, <br><br> {my_name} <br> {my_designation} <br> {my_company} <br> <a href="mailto:{my_mail}">{my_mail}</a></p>
    </div>
</body>
</html>
```




### **💡 Example 5: 

---html

<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            color: #333;
            line-height: 1.6;
            background-color: #f9f9f9;
            padding: 20px;
        }}
        .container {{
            max-width: 600px;
            background: white !important;
            padding: 30px;
            margin: auto;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        .btn {{
            display: block;
            background-color: #007bff;
            color: white !important;
            text-align: center;
            padding: 14px;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h2>Hi {client_name}, Let’s Elevate {client_company}’s Website! 🎯</h2>
        <p>Your website has great potential, but a few refinements could significantly <strong>enhance performance and user experience.</strong></p>

        <h3>Key Enhancements:</h3>
        <ul>
            <li> Faster load times for <strong>better engagement</strong></li>
            <li> Enhanced design consistency <strong>for brand trust</strong></li>
            <li> Improved contact forms <strong>for more leads</strong></li>
        </ul>

        <p>I’ve created a short video explaining the possible improvements. You can watch it below:</p>
        <p style="text-align: center;"><a href="{video_path}" class="btn">🎥 Watch Video</a></p>
        
        <p>Let’s chat about <strong>simple, high impact changes</strong> that can help {client_company} thrive online.</p>
        <p><a href="{my_cta_link}" class="btn">Let’s Connect & Improve</a></p>
        
        <p>Best,<br><br>{my_name}<br>{my_designation}<br>{my_company}<br><a href="mailto:{my_mail}">{my_mail}</a></p>
    </div>
</body>
</html>

---


### **💡 Example 6: 

```html

<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            color: #333;
            line-height: 1.6;
            background-color: #f9f9f9;
            padding: 20px;
        }}
        .container {{
            max-width: 600px;
            background: white !important;
            padding: 30px;
            margin: auto;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        .btn {{
            display: block;
            background-color: #007bff;
            color: white !important;
            text-align: center;
            padding: 14px;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h2>Hi {client_name}, Let’s Optimize {client_company}’s Software! 🚀</h2>
        <p>Your software has immense potential, and with a few strategic upgrades, we can <strong>enhance performance, security, and user retention.</strong></p>

        <h3>Key Areas for Improvement:</h3>
        <ul>
            <li> Code optimization for <strong>faster load times</strong></li>
            <li> Security patches to <strong>safeguard user data</strong></li>
            <li> UX/UI enhancements for <strong>seamless navigation</strong></li>
        </ul>

        <p>I’ve put together a quick analysis video with tailored insights for {client_company}. Check it out below:</p>
        <p style="text-align: center;"><a href="{video_path}" class="btn">🎥 Watch Analysis</a></p>
        
        <p>Let’s explore simple, high-impact upgrades that will maximize {client_company}’s efficiency and growth.</p>
        <p><a href="{my_cta_link}" class="btn">Let’s Connect & Improve</a></p>
        
        <p>Best,<br><br>{my_name}<br>{my_designation}<br>{my_company}<br><a href="mailto:{my_mail}">{my_mail}</a></p>
    </div>
</body>
</html>


---


### **💡 Example 7: 

```html

<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            color: #333;
            line-height: 1.6;
            background-color: #f9f9f9;
            padding: 20px;
        }}
        .container {{
            max-width: 600px;
            background: white !important;
            padding: 30px;
            margin: auto;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        .btn {{
            display: block;
            background-color: #007bff;
            color: white !important;
            text-align: center;
            padding: 14px;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h2>Hi {client_name}, Let’s Supercharge {client_company}’s Ads! 📈</h2>
        <p>Your ads are performing well, but a few strategic tweaks can <strong>significantly boost conversions and ROI.</strong></p>

        <h3>Key Improvement Areas:</h3>
        <ul>
            <li>✔️ Ad copy refinements for <strong>higher engagement</strong></li>
            <li>✔️ Advanced audience targeting for <strong>better lead quality</strong></li>
            <li>✔️ Landing page optimizations to <strong>increase conversions</strong></li>
        </ul>

        <p>I’ve put together a quick breakdown video outlining the biggest opportunities for {client_company}. Check it out below:</p>
        <p style="text-align: center;"><a href="{video_path}" class="btn">🎥 Watch My Breakdown</a></p>
        
        <p>Let’s refine your ads and maximize returns!</p>
        <p><a href="{my_cta_link}" class="btn">Let’s Optimize Your Ads</a></p>
        
        <p>Best,<br><br>{my_name}<br>{my_designation}<br>{my_company}<br><a href="mailto:{my_mail}">{my_mail}</a></p>
    </div>
</body>
</html>

---


### **💡 Example 8: 

```html

<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            color: #333;
            line-height: 1.6;
            background-color: #f9f9f9;
            padding: 20px;
        }}
        .container {{
            max-width: 600px;
            background: white !important;
            padding: 30px;
            margin: auto;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        .btn {{
            display: block;
            background-color: #007bff;
            color: white !important;
            text-align: center;
            padding: 14px;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h2>Hi {client_name}, Let’s Perfect {client_company}’s Landing Page! 🚀</h2>
        <p>Your landing page has strong potential, and a few strategic tweaks can <strong>boost conversions significantly.</strong></p>

        <h3>Key Optimization Areas:</h3>
        <ul>
            <li> Stronger CTA placements for <strong>higher engagement</strong></li>
            <li> Faster load speed to <strong>reduce drop-offs</strong></li>
            <li> Clearer messaging for <strong>better user understanding</strong></li>
        </ul>

        <p>I’ve put together a quick video breakdown with insights tailored to {client_company}. Check it out below:</p>
        <p style="text-align: center;"><a href="{video_path}" class="btnn">🎥 See My Suggestions</a></p>
        
        <p>Let’s fine-tune your page and increase conversions!</p>
        <p><a href="{my_cta_link}" class="btn">Let’s Talk Optimization</a></p>
        
        <p>Best,<br><br>{my_name}<br>{my_designation}<br>{my_company}<br><a href="mailto:{my_mail}">{my_mail}</a></p>
    </div>
</body>
</html>

        



---


### **💡 Example 9: 

```html

<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            color: #333;
            line-height: 1.6;
            background-color: #f9f9f9;
            padding: 20px;
        }}
        .container {{
            max-width: 600px;
            background: white !important;
            padding: 30px;
            margin: auto;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        .btn {{
            display: block;
            background-color: #007bff;
            color: white !important;
            text-align: center;
            padding: 14px;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        
        <h2>Hi {client_name}, Let’s Boost {client_company}’s SEO! 🚀</h2>
        <p>I performed a quick SEO audit on {client_company} and found some <strong>key areas for improvement</strong> that can enhance your search rankings.</p>

        <h3>Key Opportunities:</h3>
        <ul>
            <li>✔️ Addressing keyword gaps for <strong>higher visibility</strong></li>
            <li>✔️ Building backlinks to <strong>boost domain authority</strong></li>
            <li>✔️ On-page SEO fixes for <strong>better indexing</strong></li>
        </ul>

        <p>I’ve created a short video explaining the audit insights. You can watch it below:</p>
        <p style="text-align: center;"><a href="{video_path}" class="btn">🎥 Watch My Audit</a></p>
        
        <p>Let’s chat about an <strong>SEO strategy tailored</strong> to help {client_company} rank higher and get more organic traffic.</p>
        <p><a href="{my_cta_link}" class="btn">Schedule a Strategy Call</a></p>
        
        <p>Best,<br><br>{my_name}<br>{my_designation}<br>{my_company}<br><a href="mailto:{my_mail}">{my_mail}</a></p>
    </div>
</body>
</html>

---


### **💡 Example 10: automobile brand

```html
<!DOCTYPE html>
<html>
<head>
<style>
body {{
  font-family: 'Arial', sans-serif;
  color: #444;
  background-color: #f4f4f4;
  padding: 20px;
}}
.container {{
  max-width: 600px;
  background: white !important;
  padding: 30px;
  margin: auto;
  border-radius: 10px;
  box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}}
.header {{
  color: #333;
  margin-bottom: 20px;
}}
.btn {{
  display: block;
  text-align: center;
  background-color: #007bff;
  color: white !important;
  padding: 14px;
  margin-top: 20px;
  border-radius: 5px;
  text-decoration: none;
  font-weight: bold;
}}
.video-container {{
  margin: 25px 0;
  text-align: center;
}}
.footer {{
  margin-top: 30px;
  border-top: 1px solid #eee;
  padding-top: 15px;
}}
ul {{
  padding-left: 20px;
}}
li {{
  margin-bottom: 10px;
}}
</style>
</head>
<body>
<div class="container">
  <h2 class="header">Elevate {client_company} Digital Experience</h2>
  
  <p>Hello Raghav,</p>
  
  <p>I hope this email finds you well. I'm {my_name}, {my_designation} at {my_company}, and I recently spent some time analyzing <strong>{client_company} website</strong>.</p>
  
  <p>As passionate advocates for premium user experiences, we believe your iconic brand deserves a digital presence that matches the power and elegance of your product.</p>
  
  <h3>Areas of Opportunity I've Identified:</h3>
  <ul>
    <li>🚀 <strong>Performance optimization</strong> - Reducing load times for a smoother browsing experience</li>
    <li>🎨 <strong>UI refinements</strong> - Enhancing visual hierarchy to better highlight your premium products</li>
    <li>📱 <strong>Mobile responsiveness</strong> - Creating a seamless experience across all devices</li>
    <li>🔍 <strong>User journey mapping</strong> - Streamlining the path from discovery to purchase</li>
  </ul>
  
  <p>I've created a brief video analysis outlining specific improvements that could help increase engagement and conversions:</p>
  
  <div class="video-container">
    <a href="{video_path}" class="btn">🎥 Watch Your Website Analysis</a>
  </div>
  
  <p>These enhancements could significantly impact your customer experience while maintaining the premium feel of the {client_company} brand. I'd be happy to discuss how we can implement these changes with minimal disruption to your current operations.</p>
  
  <a href="{my_cta_link}" class="btn">Schedule a 15-Minute Discovery Call</a>
  
  <p>Looking forward to the possibility of helping {client_company} achieve an even more impressive digital presence.</p>
  
  <div class="footer">
    <p>Best regards,<br><br>
    {my_name}<br>
    {my_designation}<br>
    <a href="mailto:{my_mail}">{my_mail}</a></p>
  </div>
</div>
</body>
</html>

---

**  IMPORTANT **

Generate only one custom html email on the basis of body text provided.
"""
    return system_prompt_1












# ✅ Create a global Ollama instance
# ollama_model = ollama.Ollama(model="llama3")


# def generate_response(system_prompt):
#     response = ollama.chat(
#         model="llama3",
#         messages=[{"role": "system", "content": system_prompt}]
#     )
#     return response["message"]["content"]  # ✅ Extract content properly



