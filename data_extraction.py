import base64
import io
from PIL import Image
import pdf2image
import google.generativeai as genai
import re


def get_gemini_response(prompt, pdf_content=None, input_text=None):
    """Get response from Gemini API with proper error handling"""
    try:
        model_gemini = genai.GenerativeModel('gemini-1.5-flash')

        # Prepare content for the API call
        content = []
        if prompt:
            content.append(prompt)
        if pdf_content:
            content.append(pdf_content[0])
        if input_text:
            content.append(input_text)

        response = model_gemini.generate_content(content)
        return response.text
    except Exception as e :
        return f"Error with Gemini API: {str(e)}"


def input_pdf_setup(uploaded_file):
    """Convert uploaded PDF to format needed for Gemini API"""
    if uploaded_file is not None:
        # Convert the PDF to image
        images = pdf2image.convert_from_bytes(uploaded_file.read())

        # Use all pages from the PDF
        pdf_parts = []
        for page in images :
            img_byte_arr = io.BytesIO()
            page.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

            pdf_parts.append({
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
            })
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")


def extract_resume_data(pdf_content, job_description):
    """Extract all necessary data from resume using LLM"""
    data = {}

    # Extract technologies/skills
    tech_prompt = """
    You are an experienced ATS scanner. Extract ALL technical skills and technologies mentioned in this resume.
    Return ONLY a comma-separated list like: Python, JavaScript, React, etc.
    Do not include explanations or additional text.
    """
    data['skills'] = get_gemini_response(tech_prompt, pdf_content)

    # Extract years of experience
    exp_prompt = """
    Extract the total years of professional experience from this resume.
    Return ONLY a number (can be decimal like 3.5). If unclear, estimate based on work history.
    """
    experience = get_gemini_response(exp_prompt, pdf_content)
    try:
        data['years_experience'] = float(re.search(r'\d+(\.\d+)?', experience).group())
    except (AttributeError, ValueError):
        data['years_experience'] = 0

    # Generate resume summary
    summary_prompt = """
    Generate a concise summary (max 3 paragraphs) of this candidate's profile, focusing on their:
    1. Technical expertise and main skills
    2. Work experience highlights
    3. Education and relevant achievements
    """
    data['summary'] = get_gemini_response(summary_prompt, pdf_content)

    return data


def extract_job_data(job_description):
    """Extract all necessary data from job description using LLM"""
    data = {}

    # Extract required technologies/skills
    tech_prompt = f"""
    You are an experienced ATS scanner. Extract ALL technical skills and technologies required in this job description.
    Return ONLY a comma-separated list like: Python, JavaScript, React, etc.
    Do not include explanations or additional text.

    Job Description:
    {job_description}
    """
    data['required_skills'] = get_gemini_response(tech_prompt, input_text=job_description)

    # Extract required years of experience
    exp_prompt = f"""
    Extract the minimum years of experience required for this job from the description.
    Return ONLY a number (can be decimal like 3.5). If not explicitly stated, return your best estimate.

    Job Description:
    {job_description}
    """
    experience = get_gemini_response(exp_prompt, input_text=job_description)
    try:
        data['required_years'] = float(re.search(r'\d+(\.\d+)?', experience).group())
    except (AttributeError, ValueError):
        data['required_years'] = 0

    # Generate job summary
    summary_prompt = f"""
    Generate a concise summary (max 2 paragraphs) of this job posting, focusing on:
    1. The key responsibilities and expectations
    2. The required technical skills and qualifications
    
    Job Description:
    {job_description}
    """
    data['summary'] = get_gemini_response(summary_prompt, input_text=job_description)

    return data


def extract_technologies(text):
    """Convert comma-separated text to a list of technologies"""
    if not text:
        return []
    # Split by comma and clean up whitespace
    techs = [tech.strip() for tech in text.split(',')]
    # Remove empty items
    return [tech for tech in techs if tech]
