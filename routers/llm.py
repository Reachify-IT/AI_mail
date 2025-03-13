## import library
import os
import re
import shutil
import ollama
import requests
import tempfile
import chromadb
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from langchain_community import embeddings
from langchain_community.llms import Ollama
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

import requests

import whisper
# from moviepy import *
# from moviepy.editor import VideoFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip

# os.environ["OLLAMA_HOST"] = "http://172.31.46.239:11434"


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
        # Close any existing ChromaDB instances
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        client.delete_collection("rag-chroma")
        del client  # Release resources

        # Ensure files are closed before deleting
        import time
        time.sleep(2)  # Wait for 2 seconds

        # Delete the database folder
        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH)
            print("✅ ChromaDB directory deleted successfully.")
    except Exception as e:
        print(f"⚠️ ChromaDB Error: {e}")



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

def extract_audio(video_path, audio_path="temp_audio.wav"):
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
    # if source_type == "youtube":
    #     video_path = download_youtube_video(video_source)
    # # elif source_type == "drive":
    # #     video_path = download_drive_video(video_source)
    # else:
    # just in case

    
    video_path = download_video(video_source)
    # video_path = video_source  # Local file

    audio_path = extract_audio(video_path)
    # st.audio(audio_path, format="audio/wav")
    text = transcribe_audio(audio_path, model_size)

    # Cleanup
    os.remove(audio_path)
    # os.remove(video_path)

    return text







# Function to process input
def process_input(urls):
    try:
        model_local = Ollama(model="llama3")

        # Convert string of URLs to list and filter out empty ones
        urls_list = [url.strip() for url in urls.split("\n") if url.strip()]
        if not urls_list:
            return "Error: No valid URLs provided."

        docs = [WebBaseLoader(url).load() for url in urls_list]
        docs_list = [item for sublist in docs for item in sublist]

        # Split the text into chunks
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=10000, chunk_overlap=100)
        doc_splits = text_splitter.split_documents(docs_list)

        # Convert text chunks into embeddings and store in vector database
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=embeddings.OllamaEmbeddings(model='nomic-embed-text'),
            persist_directory=CHROMA_DB_PATH
        )
        retriever = vectorstore.as_retriever()

        # Enhanced RAG prompt for extracting a complete website summary
        after_rag_template = """
        Generate a comprehensive summary of the website content. The summary should include:
        - Key topics and themes covered.
        - Overall purpose and main objectives of the website.
        - Notable sections or pages that provide valuable information.
        - Any important insights derived from the content.

        Provide a clear and structured summary that captures the essence of the website.
        Context:
        {context}
        """

        after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
        after_rag_chain = (
                {"context": retriever}
                | after_rag_prompt
                | model_local
                | StrOutputParser()
        )
        return after_rag_chain.invoke("")

    except Exception as e:
        return f"Error: {str(e)}"





def extract_email_parts(email_text):
    if not email_text:
        return "No Subject Found", "No Body Found"

    subject_pattern = r"(?i)Subject:\s*(.+)"
    subject_match = re.search(subject_pattern, email_text)
    subject_text = subject_match.group(1).strip() if subject_match else "Not found"

    pattern = re.compile(r"Subject:.*?\n(.*\n\S+@\S+)", re.DOTALL)
    result = pattern.search(email_text)

    if result:
        body_text = result.group(1).strip()
    else:
        body_text = "No body found"

    return subject_text, body_text



# system_prompt


def train_model(my_company, my_designation, my_name, my_mail, my_work, client_name, client_company, client_designation,
                client_website, client_website_issue, client_about_website):
    system_prompt = f"""You are an outreach expert at {my_company} specializing in **custom, high-converting cold emails**. Your goal is to generate a **single compelling email** that feels natural, persuasive, and engaging—without being overly formal or robotic.  

Use the following **context**:  
- **Company Name**: {my_company}  
- **Your Role**: {my_designation} ({my_name})  
- **Your Contact**: {my_mail}  
- **Your Work**: {my_work}  

- **Client Name**: {client_name}  
- **Client Company**: {client_company}  
- **Client Role**: {client_designation}   
- **Client Website**: [{client_company} Website]({client_website})  
- **Issue Identified**: {client_website_issue}  
- **Insights from Website**: {client_about_website}  

**Tone**: Conversational, yet professional. Make it feel like a **helpful, friendly expert reaching out**, not a hard sales pitch.  

**Your task:**  
Generate **both a subject** and **a full email body**.  

**Structure:**  
1️ **Subject Line**:  
   - Must be attention-grabbing but **not clickbait**.  
   - Should immediately highlight value or relevance.  
   - Keep it **short and engaging** (max 10 words).  

2️ **Email Body**:  
  - **Opening**: Personal, warm, and engaging. Show you did your research.  
  - **Pain Points**: Highlight the company’s challenges in a **human** way.  
  - **Solution**: Clearly explain how {my_company} and {my_work} can **fix the problem**.  
  - **Call-to-Action**: Casual but persuasive nudge to continue the conversation.  



### **Example Output**  

**Subject:** Let’s Unlock {client_company}’s Website Potential

**Email Body:**  

Hey {client_name},  

I checked out [{client_company}’s website]({client_website}), and I love what you’re building! But I noticed a few areas where small changes could make a **huge difference** in conversions and user experience.  

Here’s what stood out:  
✅ **Navigation issues** – Some sections feel tricky to access.  
✅ **Performance optimizations** – Faster loading = happier visitors.  
✅ **Accessibility improvements** – Let’s ensure a smooth experience for all users.  
✅ **Contact info visibility** – Potential customers should find you easily!  

At {my_company}, we specialize in **{my_work}**, helping brands turn their websites into **high-performing assets**. A few quick optimizations could **boost engagement, usability, and business impact.**  

Would love to share some quick wins—open to a quick chat?  

**Best,**  
{my_name}  
{my_designation}, {my_company}  
{my_mail}  



---
### **Example 1: Website Optimization (Lively & Conversational)** 

- **Client Company**: Reachify Innovations 
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
### **Example 2: SEO & Digital Marketing (Lively & Engaging)** 

- **Client Company**: GrowthEdge Solutions 
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
### **Example 3: Mobile App Development (Energetic & Human-like)** 

- **Client Company**: FitGo App 
- **Issue Identified**: Low user retention, slow app performance. 
- **Your Work**: Mobile app developer fixing performance issues. 


**Generated Email:** 

**Subject:** Your App Shouldn’t Be Losing Users—Let’s Fix That 


Hey {client_name}, 


I love what you’re building with FitGo! A **fitness app that motivates users? Genius.** But I noticed something that might be holding you back—users aren’t sticking around, and I think I know why. 


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
### **Example 4: Social Media Ad Optimization (High-Energy, Sales-Driven)** 

- **Client Company**: FreshBites Meal Service 
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
### **Example 5: Software Development (Engaging & Persuasive)** 

- **Client Company**: BizFlow CRM 
- **Issue Identified**: Outdated CRM software, user complaints. 
- **Your Work**: Software developer upgrading outdated systems. 


**Generated Email:** 

**Subject:** Time for a CRM Upgrade? Let’s Talk


Hey {client_name}, 


I know how frustrating it can be when your CRM **starts slowing things down instead of speeding them up**. I took a look at BizFlow, and I think a few strategic upgrades could **massively improve efficiency and user experience.** 


Some quick wins we could tackle: 
🛠 **Bug fixes & performance boosts**—say goodbye to glitches. 
🚀 **Feature enhancements**—automation tools to streamline workflow. 
📈 **UI/UX improvements**—modern design = happier users. 


I’ve helped other businesses **upgrade without disrupting operations**, and I’d love to do the same for BizFlow. Let’s talk? 


**Best,**


{my_name} 
{my_designation}, {my_company} 
{my_mail}

"""

    # print(system_prompt)

    return system_prompt


def train_model_2(my_company, my_designation, my_name, my_mail, my_work, client_name, client_company,
                  client_designation, client_website, client_website_issue, client_about_website,
                  my_cta_link, my_body_text, video_path):
    system_prompt_1 = f"""You are an expert in generating precise, structured, and visually appealing **HTML email**. Your primary task is to **convert the provided email body text (`{my_body_text}`) into a clean, responsive HTML email** with proper formatting while ensuring no alterations to the content.

    ### **User Inputs:**
    - **Sender Details**:
      - Name: {my_name}
      - Designation: {my_designation}
      - Company: {my_company}
      - Email: {my_mail}
      - Work Type: {my_work}
      - CTA Link: {my_cta_link}
      - Video Path: {video_path}
    - **Recipient Details**:
      - Name: {client_name}
      - Designation: {client_designation}
      - Company: {client_company}
      - Website: {client_website}
      - Website Issues: {client_website_issue}
      - About Website Analysis: {client_about_website}

    - **Email Body**:
      - The **entire email body (`{my_body_text}`) must be inserted as-is** into a properly structured HTML format.
      - **No edits, rewording, or restructuring**—just **pure formatting**.
      - Headings, paragraphs, bullet points, and key highlights should **follow the existing structure**.

    ### **HTML Template Requirements:**
    1. **Content Formatting**:
       - Convert `{my_body_text}` directly into **HTML with proper `<h1>`, `<h2>`, `<p>`, `<ul>`, `<strong>`, and `<em>` tags** where needed.
       - Maintain line breaks, indentation, and spacing **exactly as in the original text**.
       - Use **consistent typography** for readability.
       - 


    2. **Responsive Design**:
       - Ensure the email is **mobile-friendly** and adapts to different screen sizes.
       - Use **External CSS** to maintain compatibility across different email clients.

    3. **Call-to-Action (CTA)**:
       - If `{video_path}` is provided, include a **visually clear btn** styled for engagement.
       - If `{my_cta_link}` is provided, include a **visually clear btn** styled for engagement.
       - The CTA **should match the intent of `{my_body_text}`** without modifying its wording.

    4. **Style Guidelines**:
       - Use a **clean, professional, and minimalistic design**.
       - Ensure **proper spacing** for better readability.
       - The background should be **subtle** to enhance text clarity.
       - **No excessive styling**—focus on clarity and structure.

    ---

    ### **Output Format:**
    Generate a **fully formatted HTML email** with External styles, ensuring `{my_body_text}` remains **unaltered** while being properly structured for readability. The footer should be correctly generated at end  contain my_name, my_designation, my_company and my email ef provided.

    ---
** Use the below formats as example and generate a customized html email according to the body text data given:


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
            color: white;
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
            background: white;
            padding: 30px;
            margin: auto;
            border-radius: 10px;
       
        }}
        .btn {{
            display: block;
            text-align: center;
            background-color: #007bff;
            color: white;
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
            color: white;
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
            color: white;
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
            background: white;
            padding: 30px;
            margin: auto;
            border-radius: 10px;

        }}
        .btn {{
            display: block;
            text-align: center;
            background-color: #007bff;
            color: white;
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

        <p>Let’s chat about <strong>simple, high-impact changes</strong> that can help {client_company} thrive online.</p>

        

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
            background: white;
            padding: 30px;
            margin: auto;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        .btn {{
            display: block;
            background-color: #007bff;
            color: white;
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
            <li>✔️ Faster load times for <strong>better engagement</strong></li>
            <li>✔️ Enhanced design consistency <strong>for brand trust</strong></li>
            <li>✔️ Improved contact forms <strong>for more leads</strong></li>
        </ul>

        <p>I’ve created a short video explaining the possible improvements. You can watch it below:</p>
        <p style="text-align: center;"><a href="{video_path}" class="btn">🎥 Watch Video</a></p>
        
        <p>Let’s chat about <strong>simple, high-impact changes</strong> that can help {client_company} thrive online.</p>
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
            background: white;
            padding: 30px;
            margin: auto;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        .btn {{
            display: block;
            background-color: #007bff;
            color: white;
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
            <li>✔️ Code optimization for <strong>faster load times</strong></li>
            <li>✔️ Security patches to <strong>safeguard user data</strong></li>
            <li>✔️ UX/UI enhancements for <strong>seamless navigation</strong></li>
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
            background: white;
            padding: 30px;
            margin: auto;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        .btn {{
            display: block;
            background-color: #007bff;
            color: white;
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
            background: white;
            padding: 30px;
            margin: auto;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        .btn {{
            display: block;
            background-color: #007bff;
            color: white;
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
            <li>✔️ Stronger CTA placements for <strong>higher engagement</strong></li>
            <li>✔️ Faster load speed to <strong>reduce drop-offs</strong></li>
            <li>✔️ Clearer messaging for <strong>better user understanding</strong></li>
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
            background: white;
            padding: 30px;
            margin: auto;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        .btn {{
            display: block;
            background-color: #007bff;
            color: white;
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
        <!-- Template 5: SEO Strategy Outreach -->
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


**  IMPORTANT **

Generate only one custom html email on the basis of body text provided.
"""
    return system_prompt_1










def generate_response(system_prompt):
    response = ollama.chat(model='llama3', messages=[{"role": "system", "content": system_prompt}])
    return response['message']['content']




